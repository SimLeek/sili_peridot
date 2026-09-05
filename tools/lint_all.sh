#!/usr/bin/env bash
# tools/lint_all.sh — run the full linter suite (pure-Python repo, no C++).
#   Hard gates fail the run (exit 1). Advisory tools report but never fail.
#   --fix   auto-fix what can be auto-fixed (ruff --fix, codespell -w)
set -uo pipefail
cd "$(dirname "$0")/.."

FIX=0
[ "${1:-}" = "--fix" ] && FIX=1

TARGETS="model scripts tests tools"

HARD_FAIL=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; HARD_FAIL=1; }
adv()  { printf '  \033[33mADVIS\033[0m %s\n' "$1"; }

hard() {  # hard <desc> <cmd...>
    local d="$1"; shift
    if "$@" >/tmp/lint_out 2>&1; then ok "$d"; else bad "$d"; sed 's/^/        /' /tmp/lint_out | head -25; fi
}
advisory() {  # advisory <desc> <cmd...>
    local d="$1"; shift
    if "$@" >/tmp/lint_out 2>&1; then ok "$d"; else adv "$d (findings, non-blocking)"; sed 's/^/        /' /tmp/lint_out | head -15; fi
}

echo "── python ──────────────────────────────────────────────"
[ "$FIX" -eq 1 ] && ruff check --fix $TARGETS >/dev/null 2>&1
hard  "ruff check"          ruff check $TARGETS
hard  "ruff format"         ruff format --check $TARGETS
hard  "bandit"              bandit -r model scripts -c .bandit -q
[ "$FIX" -eq 1 ] && codespell -w -q2 $TARGETS README.md >/dev/null 2>&1
hard  "codespell"           codespell --skip "*.sst,*.safetensors,*.log" $TARGETS
# Advisory, not a hard gate: 219 pre-existing errors across 28 files in
# model/ (never annotated before this lint setup existed) would otherwise
# block every future commit touching those files. Promote to hard once
# model/ is actually clean.
advisory "mypy (strict, model/ only)" mypy model
advisory "pydoclint"             pydoclint --style=numpy --check-return-types=False model
advisory "vulture (dead code)"   vulture model scripts tools --min-confidence 80
advisory "radon (complexity)"    radon cc model scripts -s -nb --min B
# Whole-tree baseline scan (NOT scoped to changed files, unlike the hooks
# above) -- pre-commit only ever sees staged files, so pre-existing debt
# elsewhere in the tree would otherwise stay invisible forever. Run this
# manually / in CI to see the real state of the codebase, not just diffs.
advisory "lizard (function length/complexity, whole tree)" lizard $TARGETS -w

echo "── comments ────────────────────────────────────────────"
# Advisory, not a hard gate: 37 of ~90 files in model/+scripts/ exceed the
# 30% comment-density limit (research narrative embedded inline, never
# checked against this limit before this lint setup existed). A dedicated
# comment-to-research-doc reorganization pass is scheduled separately.
advisory "comment density limit" python3 tools/comment_limit.py $TARGETS
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    advisory "comment check (offline, diff-based)" python3 tools/comment_check.py
else
    printf '  \033[90mSKIP\033[0m  comment check (not a git repo)\n'
fi

echo "────────────────────────────────────────────────────────"
if [ "$HARD_FAIL" -eq 0 ]; then
    printf '\033[32mall hard gates passed\033[0m\n'; exit 0
else
    printf '\033[31msome hard gates failed\033[0m\n'; exit 1
fi
