"""Tests for credit_detector.py.

They cover the pure helper functions, the tracker total logic, a full
end-to-end run of process_frame() on synthetic frames, and the CLI help.

These tests do not need a camera, a display or pypylon.
Run with:  pytest
"""

import os
import subprocess
import sys

import cv2

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import credit_detector as cd  # noqa: E402


# ---------- order_points ----------

def test_order_points_sorts_corners():
    # Shuffled corners of an axis-aligned rectangle.
    pts = [(100, 50), (0, 0), (0, 50), (100, 0)]
    out = cd.order_points(pts)
    expected = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
    assert out.dtype == np.float32
    assert np.allclose(out, expected)


def test_order_points_accepts_cv2_box_shape():
    box = np.array([[[10, 10]], [[60, 10]], [[60, 30]], [[10, 30]]], dtype=np.float32)
    out = cd.order_points(box)
    assert out.shape == (4, 2)
    assert np.allclose(out[0], [10, 10])
    assert np.allclose(out[2], [60, 30])


# ---------- estimate_w_h_from_box ----------

def test_estimate_w_h_from_box_rectangle():
    box = np.array([[0, 0], [200, 0], [200, 80], [0, 80]], dtype=np.float32)
    w, h = cd.estimate_w_h_from_box(box)
    assert w == pytest.approx(200.0)
    assert h == pytest.approx(80.0)
    assert isinstance(w, float) and isinstance(h, float)


def test_estimate_w_h_from_box_order_independent():
    box = np.array([[200, 80], [0, 0], [0, 80], [200, 0]], dtype=np.float32)
    w, h = cd.estimate_w_h_from_box(box)
    assert (w, h) == pytest.approx((200.0, 80.0))


# ---------- rotate90 ----------

def test_rotate90_shapes_and_identity():
    img = np.arange(6 * 4 * 3, dtype=np.uint8).reshape(6, 4, 3)
    assert cd.rotate90(img, 0) is img
    assert cd.rotate90(img, 4) is img
    assert cd.rotate90(img, 1).shape == (4, 6, 3)
    assert cd.rotate90(img, 2).shape == (6, 4, 3)
    assert cd.rotate90(img, 3).shape == (4, 6, 3)


def test_rotate90_matches_numpy():
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert np.array_equal(cd.rotate90(img, 1), np.rot90(img, -1))  # clockwise
    assert np.array_equal(cd.rotate90(img, 2), np.rot90(img, 2))
    assert np.array_equal(cd.rotate90(img, 3), np.rot90(img, 1))   # counter-clockwise
    assert np.array_equal(cd.rotate90(img, -1), cd.rotate90(img, 3))


# ---------- calculate_credit_value ----------

@pytest.mark.parametrize(
    "digits, ctype, expected",
    [
        ([2, 3, 4], "BRONZE", 24),
        ([5], "BRONZE", 5),
        ([2, 0, 7], "BRONZE", 0),
        ([1, 2, 3], "SILVER", 123),
        ([0, 4], "SILVER", 4),
        ([1, 2], "GOLD", 120),
        ([9], "GOLD", 90),
    ],
)
def test_calculate_credit_value_rules(digits, ctype, expected):
    assert cd.calculate_credit_value(digits, ctype) == expected


def test_calculate_credit_value_invalid_inputs():
    assert cd.calculate_credit_value([], "SILVER") is None
    assert cd.calculate_credit_value(None, "GOLD") is None
    assert cd.calculate_credit_value([1, "UNKNOWN"], "SILVER") is None
    assert cd.calculate_credit_value([1, 2], "UNKNOWN") is None


# ---------- detect_fake_credit ----------

@pytest.mark.parametrize(
    "ctype, w, h, fake, reasons",
    [
        ("SILVER", 1100, 400, False, []),
        ("SILVER", 400, 1100, False, []),          # orientation does not matter
        ("SILVER", 1000, 400, True, ["TOO_SMALL_SILVER"]),
        ("SILVER", 1250, 400, True, ["TOO_LARGE_SILVER"]),
        ("SILVER", 1050, 0, False, []),            # boundaries are inclusive
        ("SILVER", 1200, 0, False, []),
        ("GOLD", 700, 300, False, []),
        ("GOLD", 699, 300, True, ["TOO_SMALL_GOLD"]),
        ("GOLD", 5000, 300, False, []),            # GOLD has no max
        ("BRONZE", 800, 300, False, []),
        ("BRONZE", 801, 300, True, ["TOO_LARGE_BRONZE"]),
        ("BRONZE", 10, 10, False, []),             # BRONZE has no min
        ("UNKNOWN", 5000, 5000, False, []),        # unknown type is never flagged
    ],
)
def test_detect_fake_credit(ctype, w, h, fake, reasons):
    assert cd.detect_fake_credit(ctype, w, h) == (fake, reasons)


# ---------- DIGIT_MAP ----------

def test_digit_map_covers_all_digits_once():
    values = sorted(cd.DIGIT_MAP.values())
    assert values == list(range(10))
    assert len(cd.DIGIT_MAP) == 10


def test_digit_map_key_format():
    for (c1, c2) in cd.DIGIT_MAP:
        assert c1 in (1, 2, 3)
        assert len(c2) == 3 and set(c2) <= {0, 1}
        # C2 is always "filled from the top": no gap like (1, 0, 1).
        assert list(c2) == sorted(c2, reverse=True)


def test_digit_map_matches_simple_rule():
    # README: digit = 3 * (C1 class - 1) + number of C2 blocks  (0 = short C1, no C2 blocks)
    for (c1, c2), digit in cd.DIGIT_MAP.items():
        assert digit == 3 * (c1 - 1) + sum(c2)
    # Medium / tall C1 with no C2 blocks is not a valid digit.
    assert (2, (0, 0, 0)) not in cd.DIGIT_MAP
    assert (3, (0, 0, 0)) not in cd.DIGIT_MAP


# ---------- extract_digit_features ----------

WARPED = np.zeros((520, 200, 3), dtype=np.uint8)  # portrait credit size used by the pipeline


def _row(c1_height, n_c2, small_h=20):
    """Build one synthetic digit row: one tall C1 block plus n_c2 small C2 blocks."""
    blocks = [(20, 100, 40, c1_height)]               # column 1 (left)
    for i in range(n_c2):
        blocks.append((120, 100 + i * (small_h + 5), 30, small_h))  # column 2 (right)
    return blocks


@pytest.mark.parametrize(
    "c1_h, n_c2, expected_c1, expected_digit",
    [
        (20, 1, 1, 1),   # ratio 1.0  -> height class 1
        (20, 2, 1, 2),
        (20, 3, 1, 3),
        (40, 1, 2, 4),   # ratio 2.0  -> height class 2
        (40, 2, 2, 5),
        (40, 3, 2, 6),
        (70, 1, 3, 7),   # ratio 3.5  -> height class 3
        (70, 2, 3, 8),
        (70, 3, 3, 9),
    ],
)
def test_extract_digit_features_synthetic(c1_h, n_c2, expected_c1, expected_digit):
    c1, c2, digit = cd.extract_digit_features(_row(c1_h, n_c2), WARPED, "t")
    assert c1 == expected_c1
    assert c2 == [1] * n_c2 + [0] * (3 - n_c2)
    assert digit == expected_digit


def test_extract_digit_features_zero_and_empty():
    # Only a column-1 block and no column-2 blocks -> digit 0.
    assert cd.extract_digit_features([(20, 100, 40, 30)], WARPED, "t") == (1, [0, 0, 0], 0)
    assert cd.extract_digit_features([], WARPED, "t") == (None, None, "UNKNOWN")


def test_extract_digit_features_unmapped_pattern():
    # A tiny column-2 block still sets the reference height (ratio 40/10 = 4 -> class 3),
    # but it is too small to count as a C2 block. (3, [0,0,0]) is not in DIGIT_MAP.
    blocks = [(20, 100, 40, 40), (120, 100, 10, 10)]  # 10x10 = 100 px < 0.1% of 520*200
    c1, c2, digit = cd.extract_digit_features(blocks, WARPED, "t")
    assert c1 == 3 and c2 == [0, 0, 0]
    assert digit == "UNKNOWN"


# ---------- classify_credit_color_from_warp ----------

@pytest.mark.parametrize(
    "bgr, expected",
    [
        ((0, 0, 255), "BRONZE"),     # red / orange-red
        ((255, 0, 0), "SILVER"),     # blue
        ((0, 215, 255), "GOLD"),     # yellow
        ((128, 128, 128), "UNKNOWN"),  # grey: no saturation
    ],
)
def test_classify_credit_color_from_warp(bgr, expected):
    img = np.full((520, 200, 3), bgr, dtype=np.uint8)
    assert cd.classify_credit_color_from_warp(img) == expected


# ---------- remove_green_background ----------

BELT_GREEN = (0, 160, 0)  # BGR, inside GREEN_LOWER..GREEN_UPPER


def test_remove_green_background_keeps_only_non_green():
    frame = np.full((300, 400, 3), BELT_GREEN, dtype=np.uint8)
    assert cv2.countNonZero(cd.remove_green_background(frame)) == 0
    frame[100:200, 150:250] = (255, 255, 255)   # a white card on the belt
    mask = cd.remove_green_background(frame)
    assert mask[150, 200] == 255                # card centre is foreground
    assert mask[10, 10] == 0                    # belt is background


# ---------- detect_blocks_refined / group_blocks_by_digit_row ----------

def test_detect_blocks_refined_unknown_type_returns_nothing():
    blocks, vis = cd.detect_blocks_refined(WARPED, "UNKNOWN", 1)
    assert blocks == [] and vis.shape == WARPED.shape


def test_detect_blocks_refined_ignores_blocks_near_edge():
    img = np.full((520, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (60, 100), (99, 139), (255, 0, 0), -1)   # well inside
    cv2.rectangle(img, (2, 300), (41, 339), (255, 0, 0), -1)    # touches the 10 px margin
    blocks, _ = cd.detect_blocks_refined(img, "SILVER", 1)
    assert blocks == [(60, 100, 40, 40)]


def test_group_blocks_by_digit_row():
    blocks = [(120, 300, 30, 20), (20, 100, 40, 40), (120, 105, 30, 20), (20, 295, 40, 30)]
    rows = cd.group_blocks_by_digit_row(blocks, WARPED, 1)
    assert len(rows) == 2                                  # threshold = 12% of 520 px
    assert sorted(rows[0]) == [(20, 100, 40, 40), (120, 105, 30, 20)]   # top row first
    assert sorted(rows[1]) == [(20, 295, 40, 30), (120, 300, 30, 20)]
    assert cd.group_blocks_by_digit_row([], WARPED, 1) == []


# ---------- process_frame (end to end on synthetic frames) ----------

def _make_card(rows, color_bgr):
    """Draw a 200x520 white credit. rows = [(C1 class, number of C2 blocks), ...]."""
    img = np.full((520, 200, 3), 255, dtype=np.uint8)
    y = 60
    for c1_class, n_c2 in rows:
        c1_h = {1: 20, 2: 45, 3: 70}[c1_class]   # ratio to 20 px C2 blocks: 1.0 / 2.25 / 3.5
        cv2.rectangle(img, (30, y), (79, y + c1_h - 1), color_bgr, -1)
        for i in range(n_c2):
            top = y + i * 25
            cv2.rectangle(img, (120, top), (149, top + 19), color_bgr, -1)
        y += max(c1_h, n_c2 * 25) + 50
    return img


def _frame_with_card(card, card_size, frame_size):
    """Scale the card to card_size (w, h) and place it on a green belt of frame_size (w, h)."""
    fw, fh = frame_size
    frame = np.full((fh, fw, 3), BELT_GREEN, dtype=np.uint8)
    cw, ch = card_size
    big = cv2.resize(card, (cw, ch), interpolation=cv2.INTER_NEAREST)
    x0, y0 = (fw - cw) // 2, (fh - ch) // 2
    frame[y0:y0 + ch, x0:x0 + cw] = big
    return frame


def _summary(results):
    return [(ct, digits, value, fake, reasons) for (ct, digits, value, _box, fake, reasons) in results]


def test_process_frame_bronze_worked_example(capsys):
    # README worked example: BRONZE with digits 2 and 3 -> 2 x 3 = 6. Longer side 780 px <= 800 -> REAL.
    card = _make_card([(1, 2), (1, 3)], (0, 0, 255))
    frame = _frame_with_card(card, (300, 780), (800, 1000))
    assert _summary(cd.process_frame(frame)) == [("BRONZE", [2, 3], 6, False, [])]
    # Upside down gives the same result (orientation is normalised).
    flipped = np.ascontiguousarray(np.rot90(frame, 2))
    assert _summary(cd.process_frame(flipped)) == [("BRONZE", [2, 3], 6, False, [])]


def test_process_frame_silver_real_in_any_orientation(capsys):
    card = _make_card([(2, 1), (1, 3)], (255, 0, 0))   # digits 4, 3
    frame = _frame_with_card(card, (440, 1144), (1600, 1400))
    expected = [("SILVER", [4, 3], 43, False, [])]
    assert _summary(cd.process_frame(frame)) == expected
    landscape = np.ascontiguousarray(np.rot90(frame))
    assert _summary(cd.process_frame(landscape)) == expected


def test_process_frame_small_silver_is_fake(capsys):
    card = _make_card([(2, 1), (1, 3)], (255, 0, 0))
    frame = _frame_with_card(card, (300, 780), (800, 1000))   # longer side 780 < 1050
    assert _summary(cd.process_frame(frame)) == [("SILVER", [4, 3], 43, True, ["TOO_SMALL_SILVER"])]


def test_process_frame_empty_belt(capsys):
    frame = np.full((480, 640, 3), BELT_GREEN, dtype=np.uint8)
    assert cd.process_frame(frame) == []


def test_draw_results_on_frame_does_not_change_input(capsys):
    card = _make_card([(1, 2), (1, 3)], (0, 0, 255))
    frame = _frame_with_card(card, (300, 780), (800, 1000))
    before = frame.copy()
    out = cd.draw_results_on_frame(frame, cd.process_frame(frame))
    assert out.shape == frame.shape
    assert np.array_equal(frame, before)
    assert not np.array_equal(out, frame)


# ---------- box_iou ----------

def test_box_iou():
    a = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    b = np.array([[50, 0], [150, 0], [150, 100], [50, 100]], dtype=np.float32)
    far = a + 1000
    assert cd.box_iou(a, a) == pytest.approx(1.0)
    assert cd.box_iou(a, b) == pytest.approx(1 / 3, abs=0.01)
    assert cd.box_iou(a, far) == pytest.approx(0.0)


# ---------- CreditTracker (total logic) ----------

BOX = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)


def test_tracker_default_parameters():
    t = cd.CreditTracker()
    assert t.iou_threshold == 0.15
    assert t.max_age == 30
    assert t.total_value == 0 and t.tracked == []


def test_tracker_adds_value_after_credit_leaves(capsys):
    t = cd.CreditTracker(max_age=2)
    t.update([("SILVER", [1, 2], 12, BOX)], 1)
    t.update([("SILVER", [1, 3], 13, BOX)], 2)   # same credit (IoU = 1)
    assert len(t.tracked) == 1 and t.total_value == 0
    for f in range(3, 6):                         # credit gone for > max_age frames
        t.update([], f)
    assert t.total_value == 13                    # max of value history
    assert t.tracked == []


def test_tracker_finalizes_only_after_more_than_max_age(capsys):
    t = cd.CreditTracker(max_age=2)
    t.update([("GOLD", [4, 2], 420, BOX)], 1)
    t.update([], 3)                               # missing for exactly max_age frames: still alive
    assert t.total_value == 0 and len(t.tracked) == 1
    t.update([], 4)                               # missing for max_age + 1 frames: finalized
    assert t.total_value == 420 and t.tracked == []


def test_tracker_credits_still_visible_are_not_counted(capsys):
    # Known limitation: nothing is added while the credit is still being seen.
    t = cd.CreditTracker(max_age=2)
    for f in range(1, 10):
        t.update([("SILVER", [5], 5, BOX)], f)
    assert t.total_value == 0 and len(t.tracked) == 1


def test_tracker_does_not_match_different_type(capsys):
    t = cd.CreditTracker()
    t.update([("SILVER", [1], 1, BOX)], 1)
    t.update([("GOLD", [1], 10, BOX)], 2)         # same place, other type -> new track
    assert len(t.tracked) == 2


def test_tracker_low_iou_creates_new_track(capsys):
    t = cd.CreditTracker()
    t.update([("SILVER", [1], 1, BOX)], 1)
    t.update([("SILVER", [1], 1, BOX + [90, 0])], 2)   # IoU about 0.05 < 0.15
    assert len(t.tracked) == 2


def test_tracker_one_detection_per_track_per_frame(capsys):
    t = cd.CreditTracker()
    t.update([("SILVER", [1], 1, BOX)], 1)
    t.update([("SILVER", [1], 1, BOX), ("SILVER", [2], 2, BOX)], 2)   # 2nd cannot reuse the track
    assert len(t.tracked) == 2


def test_tracker_value_rules_on_finalize(capsys):
    t = cd.CreditTracker(max_age=0)
    t.update([("BRONZE", [0], 0, BOX)], 1)                   # only zeros -> adds 0
    t.update([("SILVER", [], None, BOX + 500)], 1)           # no value at all -> never added
    t.update([], 2)
    assert t.total_value == 0
    assert t.tracked == []                                    # both tracks are dropped


def test_tracker_ignores_none_values_in_history(capsys):
    t = cd.CreditTracker(max_age=0)
    t.update([("BRONZE", [], None, BOX)], 1)
    t.update([("BRONZE", [2, 3], 6, BOX)], 2)
    t.update([("BRONZE", [], None, BOX)], 3)
    t.update([], 4)
    assert t.total_value == 6


# ---------- command line ----------

def _run_cli(*args):
    return subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "credit_detector.py"), *args],
        capture_output=True, text=True, timeout=60,
    )


def test_cli_without_arguments_prints_usage():
    r = _run_cli()
    assert r.returncode == 1
    assert "Credit Detector v2.2.0" in r.stdout
    assert "USAGE" in r.stdout and "CONTROLS" in r.stdout


def test_cli_video_mode_needs_filename():
    r = _run_cli("video")
    assert r.returncode == 1
    assert "Video mode requires filename" in r.stdout


def test_cli_missing_video_file_fails_cleanly():
    r = _run_cli("video", "does_not_exist.mp4")
    assert r.returncode == 1
    assert "Could not open" in r.stdout


def test_version_constant():
    assert cd.__version__ == "2.2.0"
