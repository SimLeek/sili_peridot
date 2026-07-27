"""
sili_peridot/tests/conftest.py
─────────────────────────────
Shared pytest fixtures/hooks.
"""
import ctypes
import gc


def trim_memory():
    """gc.collect() + glibc malloc_trim(0). Freed Python/torch/sili
    objects are already reclaimed by refcounting the instant they go
    out of scope -- the problem is one layer below that: glibc's
    allocator keeps freed arena pages for reuse by this SAME process
    rather than returning them to the OS, so RSS stays elevated even
    though nothing is actually leaked (confirmed via a two-round
    load/prune/quantize/free cycle showing bounded, not growing, RSS
    that a malloc_trim(0) call drops by ~90%). Matters here because the
    real-checkpoint tests run close to this machine's 15GB ceiling.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def pytest_runtest_teardown(item, nextitem):
    """Trim after every test. Cheap (a few ms) -- without it, retained
    -but-freed allocator memory from earlier tests/files accumulates
    across a pytest session and can be the difference between passing
    and getting OOM-killed once test_eval_quantization.py's real
    checkpoint loads run."""
    trim_memory()
