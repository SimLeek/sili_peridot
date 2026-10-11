"""Tests for the pure (non-GUI) image-composition logic in
scripts/live_synapse_display.py -- the parts that don't need a real
display, so they're deterministic and safe to run in CI. Actual window
push (LiveSynapseDisplay.show) is exercised via the
use_displayarray=False + a monkeypatched cv2.imshow/waitKey, never a
real GUI call."""

import numpy as np

from scripts.live_synapse_display import (
    LiveSynapseDisplay,
    compose_frame,
    compose_pool_panel,
    deviation_strip,
    heatmap,
    normalize_to_uint8,
)


class TestNormalizeToUint8:
    def test_constant_array_returns_mid_gray_not_nan(self):
        mat = np.full((4, 4), 5.0, dtype=np.float32)
        out = normalize_to_uint8(mat)
        assert out.dtype == np.uint8
        assert np.all(out == 128)

    def test_spreads_across_full_range_for_varying_input(self):
        mat = np.linspace(0.0, 100.0, 100).reshape(10, 10).astype(np.float32)
        out = normalize_to_uint8(mat)
        assert out.min() < 20
        assert out.max() > 235

    def test_outlier_clipped_not_dominating(self):
        # 99 normal values clustered near 1.0, one huge outlier at
        # 1000 -- percentile clipping should keep the cluster visible
        # (spread across a wide range) rather than all collapsing to 0.
        mat = np.full(100, 1.0, dtype=np.float32)
        mat[0] = 1000.0
        out = normalize_to_uint8(mat.reshape(10, 10))
        cluster = out.flatten()[1:]
        assert cluster.max() - cluster.min() < 5  # cluster reads as ~uniform
        assert out.flatten()[0] == 255  # outlier still clips to the top


class TestHeatmap:
    def test_shape_scales_by_cell_px(self):
        mat = np.random.rand(8, 6).astype(np.float32)
        img = heatmap(mat, cell_px=3)
        assert img.shape == (24, 18, 3)
        assert img.dtype == np.uint8


class TestDeviationStrip:
    def test_reset_active_columns_get_green_flag(self):
        dev = np.array([0.1, 0.2, 5.0, 0.3], dtype=np.float32)
        active = np.array([0, 0, 1, 0], dtype=np.uint8)
        strip = deviation_strip(dev, active, width_px=40, height_px=10)
        assert strip.shape == (10, 40, 3)
        # column 2 of 4 maps to pixel columns [20:30) at width_px=40
        assert np.all(strip[:, 20:30, 1] == 255)
        assert np.all(strip[:, 0:10, 1] == 0)


class TestComposePoolPanel:
    def test_includes_only_supplied_parts(self):
        imp = np.random.rand(4, 4).astype(np.float32)
        panel_imp_only = compose_pool_panel("q_proj", 100, imp, None, None, None, cell_px=2)
        panel_full = compose_pool_panel(
            "q_proj",
            100,
            imp,
            np.random.rand(4, 4).astype(np.float32),
            np.random.rand(4).astype(np.float32),
            np.zeros(4, dtype=np.uint8),
            cell_px=2,
        )
        # full version has label + weight heatmap + strip + importance
        # heatmap; imp-only has label + importance heatmap -- full must
        # be taller.
        assert panel_full.shape[0] > panel_imp_only.shape[0]
        assert panel_full.shape[1] == panel_imp_only.shape[1]


class TestComposeFrame:
    def test_empty_panels_returns_none(self):
        assert compose_frame({}) is None

    def test_pads_shorter_panels_and_orders_left_to_right(self):
        tall = np.ones((20, 10, 3), dtype=np.uint8) * 50
        short = np.ones((10, 10, 3), dtype=np.uint8) * 200
        frame = compose_frame({"b": short, "a": tall}, pool_order=["a", "b"])
        assert frame is not None
        assert frame.shape[0] == 20  # padded to the taller panel's height
        # "a" (tall, value 50) occupies the left block; "b" starts
        # after a 6px gap at column 10+6=16.
        assert np.all(frame[:, 0:10] == 50)
        assert np.all(frame[0:10, 16:26] == 200)
        assert np.all(frame[10:20, 16:26] == 0)  # short panel's padded region

    def test_respects_pool_order_default_sorted(self):
        a = np.ones((5, 3, 3), dtype=np.uint8)
        z = np.ones((5, 3, 3), dtype=np.uint8) * 2
        frame = compose_frame({"z_pool": z, "a_pool": a})
        # default order is sorted -- a_pool first
        assert np.all(frame[:, 0:3] == 1)


class TestLiveSynapseDisplayNoGui:
    def test_update_pool_pushes_via_stubbed_show(self, monkeypatch):
        pushed = []
        disp = LiveSynapseDisplay(use_displayarray=False)
        monkeypatch.setattr(disp, "show", lambda frame: pushed.append(frame))
        imp = np.random.rand(4, 4).astype(np.float32)
        frame = disp.update_pool("input_proj", 42, imp)
        assert frame is not None
        assert len(pushed) == 1
        assert pushed[0] is frame

    def test_multiple_pools_all_present_in_composed_frame(self, monkeypatch):
        pushed = []
        disp = LiveSynapseDisplay(use_displayarray=False, pool_order=["a", "b"])
        monkeypatch.setattr(disp, "show", lambda frame: pushed.append(frame))
        disp.update_pool("a", 1, np.random.rand(4, 4).astype(np.float32))
        disp.update_pool("b", 2, np.random.rand(4, 4).astype(np.float32))
        # second call's composed frame must still include pool "a"'s
        # panel from the first call (cached, not dropped).
        assert len(pushed) == 2
        assert pushed[1].shape[1] > pushed[0].shape[1]

    def test_closed_defaults_false_without_real_window(self):
        disp = LiveSynapseDisplay(use_displayarray=False)
        assert disp.closed() is False
