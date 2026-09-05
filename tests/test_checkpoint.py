import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

torch = pytest.importorskip("torch")

from model.checkpoint import load_minicpm5_checkpoint, verify_checkpoint_matches_config
from model.config import MiniCPM5Config

REAL_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "MiniCPM5-1B-Base")
REAL_CONFIG_PATH = os.path.join(REAL_CHECKPOINT_DIR, "config.json")


def _tiny_config(tie_word_embeddings: bool = False) -> MiniCPM5Config:
    return MiniCPM5Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=10,
        rms_norm_eps=1e-6,
        rope_theta=5000000.0,
        tie_word_embeddings=tie_word_embeddings,
    )


def _fake_state_dict(cfg: MiniCPM5Config) -> dict:
    sd = {
        "model.embed_tokens.weight": torch.randn(cfg.vocab_size, cfg.hidden_size),
        "lm_head.weight": torch.randn(cfg.vocab_size, cfg.hidden_size),
        "model.norm.weight": torch.randn(cfg.hidden_size),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        sd[p + "input_layernorm.weight"] = torch.randn(cfg.hidden_size)
        sd[p + "post_attention_layernorm.weight"] = torch.randn(cfg.hidden_size)
        sd[p + "self_attn.q_proj.weight"] = torch.randn(cfg.q_proj_out, cfg.attn_in)
        sd[p + "self_attn.k_proj.weight"] = torch.randn(cfg.kv_proj_out, cfg.attn_in)
        sd[p + "self_attn.v_proj.weight"] = torch.randn(cfg.kv_proj_out, cfg.attn_in)
        sd[p + "self_attn.o_proj.weight"] = torch.randn(cfg.attn_out, cfg.o_proj_in)
        sd[p + "mlp.gate_proj.weight"] = torch.randn(cfg.mlp_hidden, cfg.mlp_in)
        sd[p + "mlp.up_proj.weight"] = torch.randn(cfg.mlp_hidden, cfg.mlp_in)
        sd[p + "mlp.down_proj.weight"] = torch.randn(cfg.mlp_out, cfg.mlp_hidden)
    return sd


class TestVerifyCheckpointMatchesConfig:
    def test_valid_fake_checkpoint_passes(self):
        cfg = _tiny_config()
        verify_checkpoint_matches_config(_fake_state_dict(cfg), cfg)  # no raise

    def test_wrong_q_proj_shape_raises(self):
        cfg = _tiny_config()
        sd = _fake_state_dict(cfg)
        sd["model.layers.0.self_attn.q_proj.weight"] = torch.randn(3, 3)
        with pytest.raises(AssertionError):
            verify_checkpoint_matches_config(sd, cfg)

    def test_missing_tensor_raises(self):
        cfg = _tiny_config()
        sd = _fake_state_dict(cfg)
        del sd["model.layers.1.mlp.down_proj.weight"]
        with pytest.raises(AssertionError):
            verify_checkpoint_matches_config(sd, cfg)

    def test_wrong_layer_count_raises(self):
        cfg = _tiny_config()
        sd = _fake_state_dict(cfg)
        for k in list(sd):
            if k.startswith("model.layers.1."):
                del sd[k]
        with pytest.raises(AssertionError):
            verify_checkpoint_matches_config(sd, cfg)

    def test_tied_embeddings_when_config_says_untied_raises(self):
        cfg = _tiny_config(tie_word_embeddings=False)
        sd = _fake_state_dict(cfg)
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]  # same tensor object
        with pytest.raises(AssertionError):
            verify_checkpoint_matches_config(sd, cfg)

    def test_untied_embeddings_when_config_says_tied_is_allowed(self):
        # Tying is only enforced one direction: config says untied but
        # checkpoint is tied -> error; config says tied but checkpoint
        # happens to be untied -> not this function's job to catch.
        cfg = _tiny_config(tie_word_embeddings=True)
        sd = _fake_state_dict(cfg)
        verify_checkpoint_matches_config(sd, cfg)  # no raise

    def test_unexpected_attention_bias_raises(self):
        cfg = _tiny_config()
        sd = _fake_state_dict(cfg)
        sd["model.layers.0.self_attn.q_proj.bias"] = torch.zeros(cfg.q_proj_out)
        with pytest.raises(AssertionError):
            verify_checkpoint_matches_config(sd, cfg)

    def test_unexpected_qk_norm_raises(self):
        cfg = _tiny_config()
        sd = _fake_state_dict(cfg)
        sd["model.layers.0.self_attn.q_norm.weight"] = torch.ones(cfg.head_dim)
        with pytest.raises(AssertionError):
            verify_checkpoint_matches_config(sd, cfg)


@pytest.mark.skipif(
    not os.path.isdir(REAL_CHECKPOINT_DIR), reason="MiniCPM5-1B-Base checkpoint not present on this machine"
)
class TestRealCheckpoint:
    """Loads the actual ~2.16GB safetensors shard -- one load per test
    session's worth of tests would be wasteful, but a single memory-mapped
    safetensors load only takes ~2s, so a fresh load per test is fine."""

    def test_loads_expected_tensor_count(self):
        sd = load_minicpm5_checkpoint(REAL_CHECKPOINT_DIR)
        # embed_tokens + lm_head + norm + 24 layers * 9 tensors/layer
        assert len(sd) == 3 + 24 * 9

    def test_passes_verification_against_real_config(self):
        cfg = MiniCPM5Config.from_json(REAL_CONFIG_PATH)
        sd = load_minicpm5_checkpoint(REAL_CHECKPOINT_DIR)
        verify_checkpoint_matches_config(sd, cfg)  # no raise

    def test_embed_and_lm_head_are_not_tied(self):
        sd = load_minicpm5_checkpoint(REAL_CHECKPOINT_DIR)
        assert sd["lm_head.weight"].data_ptr() != sd["model.embed_tokens.weight"].data_ptr()

    def test_weights_are_bfloat16(self):
        sd = load_minicpm5_checkpoint(REAL_CHECKPOINT_DIR)
        assert sd["model.embed_tokens.weight"].dtype == torch.bfloat16
