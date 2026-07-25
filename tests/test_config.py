import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from model.config import MiniCPM5Config

REAL_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'MiniCPM5-1B-Base', 'config.json')


def _write_config(tmp_path, overrides=None):
    raw = {
        "hidden_size": 1536,
        "intermediate_size": 4608,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": 130560,
        "rms_norm_eps": 1e-06,
        "rope_theta": 5000000,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
    }
    raw.update(overrides or {})
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    return str(path)


class TestFromJson:
    def test_loads_all_fields(self, tmp_path):
        cfg = MiniCPM5Config.from_json(_write_config(tmp_path))
        assert cfg.hidden_size == 1536
        assert cfg.intermediate_size == 4608
        assert cfg.num_hidden_layers == 24
        assert cfg.num_attention_heads == 16
        assert cfg.num_key_value_heads == 2
        assert cfg.head_dim == 128
        assert cfg.vocab_size == 130560
        assert cfg.rms_norm_eps == pytest.approx(1e-06)
        assert cfg.rope_theta == pytest.approx(5000000)
        assert cfg.tie_word_embeddings is False
        assert cfg.hidden_act == "silu"

    def test_hidden_act_defaults_to_silu_if_missing(self, tmp_path):
        raw_path = tmp_path / "config.json"
        raw = {
            "hidden_size": 1536, "intermediate_size": 4608, "num_hidden_layers": 24,
            "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 128,
            "vocab_size": 130560, "rms_norm_eps": 1e-6, "rope_theta": 5000000,
            "tie_word_embeddings": False,
        }
        raw_path.write_text(json.dumps(raw))
        cfg = MiniCPM5Config.from_json(str(raw_path))
        assert cfg.hidden_act == "silu"

    def test_is_frozen(self, tmp_path):
        cfg = MiniCPM5Config.from_json(_write_config(tmp_path))
        with pytest.raises(Exception):
            cfg.hidden_size = 999


class TestProjectionShapes:
    """MiniCPM5-1B-Base is NOT hidden_size-square -- these lock that in."""

    def _cfg(self, tmp_path):
        return MiniCPM5Config.from_json(_write_config(tmp_path))

    def test_q_proj_out_is_not_hidden_size(self, tmp_path):
        cfg = self._cfg(tmp_path)
        assert cfg.q_proj_out == 16 * 128 == 2048
        assert cfg.q_proj_out != cfg.hidden_size

    def test_kv_proj_out_uses_kv_heads_not_attention_heads(self, tmp_path):
        cfg = self._cfg(tmp_path)
        assert cfg.kv_proj_out == 2 * 128 == 256

    def test_o_proj_in_matches_q_proj_out(self, tmp_path):
        cfg = self._cfg(tmp_path)
        assert cfg.o_proj_in == cfg.q_proj_out

    def test_attn_and_mlp_boundary_shapes_are_hidden_size(self, tmp_path):
        cfg = self._cfg(tmp_path)
        assert cfg.attn_in == cfg.attn_out == cfg.hidden_size
        assert cfg.mlp_in == cfg.mlp_out == cfg.hidden_size

    def test_mlp_hidden_is_intermediate_size(self, tmp_path):
        cfg = self._cfg(tmp_path)
        assert cfg.mlp_hidden == 4608


@pytest.mark.skipif(not os.path.exists(REAL_CONFIG_PATH),
                    reason="MiniCPM5-1B-Base/config.json not present on this machine")
class TestRealCheckpointConfig:
    """Loads the actual checkpoint's config.json, not a fixture copy --
    catches drift between this module's assumptions and the real file."""

    def test_matches_known_values(self):
        cfg = MiniCPM5Config.from_json(REAL_CONFIG_PATH)
        assert cfg.hidden_size == 1536
        assert cfg.intermediate_size == 4608
        assert cfg.num_hidden_layers == 24
        assert cfg.num_attention_heads == 16
        assert cfg.num_key_value_heads == 2
        assert cfg.head_dim == 128
        assert cfg.vocab_size == 130560
        assert cfg.rms_norm_eps == pytest.approx(1e-6)
        assert cfg.rope_theta == pytest.approx(5000000)
        assert cfg.tie_word_embeddings is False
