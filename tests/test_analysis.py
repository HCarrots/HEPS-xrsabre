import numpy as np
import pandas as pd

from xrslab import analysis, math_func
from xrslab.xrs_roi import XRSRoi


def test_normalize_detector_array_uses_scan_axis_and_handles_zero_monitor():
    detector = np.full((3, 2, 2), 8.0)
    result = analysis.normalize_detector_array(detector, [2, 4, 0])
    assert np.allclose(result[0], 4)
    assert np.allclose(result[1], 2)
    assert np.all(result[2] == 0)


def test_remove_i0_glitches_does_not_mutate_inputs():
    det1d = [pd.DataFrame({"I0": [10.0, 1.0, 10.0]})]
    detector = [{"lambda": np.array([[[1.0]], [[9.0]], [[3.0]]])}]
    corrected_1d, corrected_2d, counts = analysis.remove_i0_glitches(
        det1d, detector, "I0"
    )
    assert counts == [1]
    assert det1d[0]["I0"].tolist() == [10.0, 1.0, 10.0]
    assert corrected_1d[0]["I0"].tolist() == [10.0, 10.0, 10.0]
    assert corrected_2d[0]["lambda"].ravel().tolist() == [1.0, 2.0, 3.0]


def test_masks_sums_and_auto_adjust_roi():
    stack = np.zeros((3, 8, 8), dtype=float)
    stack[:, 3, 4] = [1, 2, 3]
    item = XRSRoi(1, 5, 1, 5, 0, name="VB-A1")
    masks = analysis.build_roi_masks(stack, [item], filter_value=0.5, auto_adjust=True, roi_size=3)
    sums = analysis.sum_roi_spectra(stack, [item], masks.masks)
    assert item.x_center == 4
    assert item.y_center == 3
    assert masks.pixel_counts == [1]
    assert sums.raw[0].tolist() == [1.0, 2.0, 3.0]
    assert sums.filtered[0].tolist() == [1.0, 2.0, 3.0]


def test_fit_q_and_interpolation_pipeline():
    x = np.linspace(9.4, 9.6, 100)
    y = math_func.gauss(x, 5.0, 9.5, 0.01, 0.2)
    item = XRSRoi(1, 3, 1, 3, 0, name="VB-A1")
    sums = analysis.RoiSums([y], [y])
    summary = analysis.fit_elastic_rois(
        x, [item], sums, [], analysis.RoiSums([], []),
        fit_type=math_func.gauss, energy_range=(9.4, 9.6),
        r_squared_threshold=0.9,
    )
    fit_result = analysis.add_q_columns(
        summary.table, [item], [], [9.8], analysis.DEFAULT_MODULE_ANGLES
    )
    analysis.configure_roi_selection([item], [], modules=["VB"], q_range=(0, 10))
    result = analysis.interpolate_and_sum(
        [[9.4, 9.5, 9.6]],
        [analysis.RoiSums([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]])],
        [analysis.RoiSums([], [])],
        [item], [],
        fit_result.assign(center=[9500.0]),
        step_ev=100,
    )
    assert not summary.bad_rois
    assert result.selected_names == ["VB-A1"]
    assert result.total_intensity.tolist() == [1.0, 2.0, 3.0]
