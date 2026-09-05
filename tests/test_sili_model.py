import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import scipy.sparse

torch = pytest.importorskip("torch")

from model.config import MiniCPM5Config
from model.sili_model import build_sili_model, compute_logits_sili, evaluate_next_token_prediction_sili


def _tiny_config(n_layers=2) -> MiniCPM5Config:
    return MiniCPM5Config(
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=n_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=15,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )


def _fake_sparse_state(cfg: MiniCPM5Config, seed=3) -> dict:
    torch.manual_seed(seed)
    sd = {}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[p + ".self_attn.q_proj.weight"] = {
            "raw": torch.randn(cfg.q_proj_out, cfg.attn_in),
            "shape": (cfg.q_proj_out, cfg.attn_in),
        }
        sd[p + ".self_attn.k_proj.weight"] = {
            "raw": torch.randn(cfg.kv_proj_out, cfg.attn_in),
            "shape": (cfg.kv_proj_out, cfg.attn_in),
        }
        sd[p + ".self_attn.v_proj.weight"] = {
            "raw": torch.randn(cfg.kv_proj_out, cfg.attn_in),
            "shape": (cfg.kv_proj_out, cfg.attn_in),
        }
        sd[p + ".self_attn.o_proj.weight"] = {
            "raw": torch.randn(cfg.attn_out, cfg.o_proj_in),
            "shape": (cfg.attn_out, cfg.o_proj_in),
        }
        sd[p + ".mlp.gate_proj.weight"] = {
            "raw": torch.randn(cfg.mlp_hidden, cfg.mlp_in),
            "shape": (cfg.mlp_hidden, cfg.mlp_in),
        }
        sd[p + ".mlp.up_proj.weight"] = {
            "raw": torch.randn(cfg.mlp_hidden, cfg.mlp_in),
            "shape": (cfg.mlp_hidden, cfg.mlp_in),
        }
        sd[p + ".mlp.down_proj.weight"] = {
            "raw": torch.randn(cfg.mlp_out, cfg.mlp_hidden),
            "shape": (cfg.mlp_out, cfg.mlp_hidden),
        }
        sd[p + ".input_layernorm.weight"] = {"raw": torch.ones(cfg.hidden_size), "shape": (cfg.hidden_size,)}
        sd[p + ".post_attention_layernorm.weight"] = {"raw": torch.ones(cfg.hidden_size), "shape": (cfg.hidden_size,)}
    sd["model.embed_tokens.weight"] = {
        "raw": torch.randn(cfg.vocab_size, cfg.hidden_size),
        "shape": (cfg.vocab_size, cfg.hidden_size),
    }
    sd["lm_head.weight"] = {
        "raw": torch.randn(cfg.vocab_size, cfg.hidden_size),
        "shape": (cfg.vocab_size, cfg.hidden_size),
    }
    sd["model.norm.weight"] = {"raw": torch.ones(cfg.hidden_size), "shape": (cfg.hidden_size,)}
    return sd


def _fake_sparse_state_csr_embed(cfg: MiniCPM5Config, seed=3) -> dict:
    """Same as _fake_sparse_state, but embed_tokens/lm_head are stored as
    {"csr": ...} (with real structural zeros) instead of {"raw": ...} --
    matches what B3's real pruning produces for these two tensors (both
    2-D, so eligible for CSR) and exercises build_sili_model's scipy-CSR
    path instead of the dense fallback."""
    sd = _fake_sparse_state(cfg, seed=seed)
    torch.manual_seed(seed + 100)
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        dense = torch.randn(cfg.vocab_size, cfg.hidden_size)
        mask = torch.rand(cfg.vocab_size, cfg.hidden_size) < 0.4  # ~40% density
        sparse = dense * mask
        csr = sparse.to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
        sd[name] = {"csr": csr, "shape": (cfg.vocab_size, cfg.hidden_size)}
    return sd


class _FakeTokenizer:
    """Deterministic stand-in: maps each text to a fixed short token-id
    sequence within vocab_size, no real tokenizer/torch dependency needed
    for this structural test."""

    def __init__(self, vocab_size, seed=0):
        self.vocab_size = vocab_size
        self.rng = np.random.RandomState(seed)

    def __call__(self, text, return_tensors="pt"):
        ids = self.rng.randint(0, self.vocab_size, size=(1, 6))
        return {"input_ids": torch.from_numpy(ids)}


class TestBuildSiliModel:
    def test_shapes_and_consumes_sparse_state(self):
        cfg = _tiny_config(n_layers=2)
        sparse_state = _fake_sparse_state(cfg)

        m = build_sili_model(sparse_state, cfg)

        assert m["embed_tokens"].shape == (cfg.vocab_size, cfg.hidden_size)
        assert m["lm_head"].shape == (cfg.vocab_size, cfg.hidden_size)
        assert m["final_norm"].shape == (cfg.hidden_size,)
        assert len(m["step_layers"]) == 2
        assert sparse_state == {}


class TestComputeLogitsSili:
    def test_output_shape(self):
        cfg = _tiny_config(n_layers=2)
        sparse_state = _fake_sparse_state(cfg)
        m = build_sili_model(sparse_state, cfg)

        token_ids = np.array([1, 4, 2, 7], dtype=np.int64)
        logits = compute_logits_sili(token_ids, m, cfg, half_bandwidth=4)

        assert logits.shape == (4, cfg.vocab_size)
        assert np.all(np.isfinite(logits))


class TestEvaluateNextTokenPredictionSili:
    def test_returns_finite_plausible_eval_result(self):
        cfg = _tiny_config(n_layers=2)
        sparse_state = _fake_sparse_state(cfg)
        m = build_sili_model(sparse_state, cfg)
        tok = _FakeTokenizer(cfg.vocab_size)

        result = evaluate_next_token_prediction_sili(m, tok, cfg, half_bandwidth=6, texts=["a", "b"])

        assert len(result.per_text_loss) == 2
        assert len(result.per_text_accuracy) == 2
        assert all(np.isfinite(x) and x >= 0 for x in result.per_text_loss)
        assert all(0.0 <= a <= 1.0 for a in result.per_text_accuracy)
        assert np.isfinite(result.perplexity) and result.perplexity >= 1.0


class TestSparseEmbedAndLmHead:
    """embed_tokens/lm_head are stored as scipy CSR (not dense) whenever
    B3 pruned them to {"csr": ...} -- real checkpoint density is ~20%/
    ~70%, previously always densified regardless (~1.2GB wasted)."""

    def test_build_sili_model_keeps_csr_entries_sparse(self):
        cfg = _tiny_config(n_layers=1)
        sparse_state = _fake_sparse_state_csr_embed(cfg)

        m = build_sili_model(sparse_state, cfg)

        assert scipy.sparse.issparse(m["embed_tokens"])
        assert scipy.sparse.issparse(m["lm_head"])
        assert m["embed_tokens"].shape == (cfg.vocab_size, cfg.hidden_size)
        assert m["lm_head"].shape == (cfg.vocab_size, cfg.hidden_size)

    def test_raw_entries_still_stay_dense(self):
        # The pre-existing fixture never has embed/lm_head pruned to CSR --
        # confirms the dense fallback path is still reachable/correct.
        cfg = _tiny_config(n_layers=1)
        sparse_state = _fake_sparse_state(cfg)

        m = build_sili_model(sparse_state, cfg)

        assert isinstance(m["embed_tokens"], np.ndarray)
        assert isinstance(m["lm_head"], np.ndarray)

    def test_sparse_path_logits_match_dense_reference(self):
        # Build the SAME underlying weights two ways -- once as {"csr":...}
        # (exercises the scipy path), once densified to {"raw":...}
        # ourselves as the reference -- and confirm compute_logits_sili
        # gives numerically identical results either way.
        cfg = _tiny_config(n_layers=1)
        sparse_state_csr = _fake_sparse_state_csr_embed(cfg)

        # Build a dense-equivalent fixture from the exact same CSR tensors.
        # dict(...) is a fresh top-level dict -- build_sili_model only pops
        # top-level keys, never mutates an entry dict in place, so the two
        # calls below don't interfere with each other despite sharing the
        # per-layer entry objects.
        sparse_state_dense = dict(sparse_state_csr)
        for name in ("model.embed_tokens.weight", "lm_head.weight"):
            csr_entry = sparse_state_csr[name]
            sparse_state_dense[name] = {
                "raw": csr_entry["csr"].to_dense(),
                "shape": csr_entry["shape"],
            }

        m_csr = build_sili_model(dict(sparse_state_csr), cfg)
        m_dense = build_sili_model(sparse_state_dense, cfg)

        token_ids = np.array([1, 4, 2], dtype=np.int64)
        logits_csr = compute_logits_sili(token_ids, m_csr, cfg, half_bandwidth=3)
        logits_dense = compute_logits_sili(token_ids, m_dense, cfg, half_bandwidth=3)

        np.testing.assert_allclose(logits_csr, logits_dense, rtol=1e-4, atol=1e-4)
