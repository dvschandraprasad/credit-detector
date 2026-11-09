"""
Credit Detector - machine-vision system for counting "credit" cards on a conveyor.

Finds BRONZE, SILVER and GOLD credits on a green conveyor belt, straightens and
orients each one, decodes its digits from the coloured block pattern, checks
the size to flag fakes, tracks credits across frames and keeps a running total.

Input sources: video file, webcam, or Basler camera (via pypylon).

Version: 2.2.0
Date: 2025-01-06
Authors: Group 12 (Credit Detection Team)
License: MIT

Usage:
    python credit_detector.py video <filename> [output]
    python credit_detector.py camera [output]
    python credit_detector.py basler [output]
Run without arguments to see the full help text.
"""

import os
import sys

# CRITICAL: Suppress pypylon logging warnings BEFORE import
os.environ['PYLON_CAMEMU'] = '0'
import warnings
warnings.filterwarnings('ignore')

# Import pypylon FIRST before any other imports to avoid logging conflicts
try:
    from pypylon import pylon
    BASLER_AVAILABLE = True
except ImportError:
    BASLER_AVAILABLE = False
    pylon = None

import cv2
import numpy as np
import time
import uuid

__version__ = "2.2.0"

# ================= CONFIG =================
GREEN_LOWER = np.array([35, 40, 40])
GREEN_UPPER = np.array([90, 255, 255])
GREEN_LOWER_BRIGHT = np.array([35, 20, 80])
GREEN_UPPER_BRIGHT = np.array([90, 255, 255])
MIN_CONTOUR_AREA = 8000
TRACKER_DEBUG = True

COLOR_CONFIG = {
    "BRONZE": {"lower1": np.array([0, 100, 49]), "upper1": np.array([16, 255, 255]),
               "lower2": np.array([170, 100, 49]), "upper2": np.array([180, 255, 255]),
               "display_color": (0, 165, 255)},
    "SILVER": {"lower": np.array([95, 105, 100]), "upper": np.array([128, 255, 255]),
               "display_color": (255, 0, 0)},
    "GOLD": {"lower": np.array([22, 100, 100]), "upper": np.array([35, 255, 255]),
             "display_color": (0, 215, 255)},
}

DIGIT_MAP = {
    (1, (0, 0, 0)): 0, (1, (1, 0, 0)): 1, (1, (1, 1, 0)): 2, (1, (1, 1, 1)): 3,
    (2, (1, 0, 0)): 4, (2, (1, 1, 0)): 5, (2, (1, 1, 1)): 6,
    (3, (1, 0, 0)): 7, (3, (1, 1, 0)): 8, (3, (1, 1, 1)): 9,
}

SHOW_DEBUG_WINDOWS = False
FRAME_SKIP = 0
DISPLAY_SCALE = 0.7

SIZE_THRESHOLDS = {
    "SILVER": {"min": 1050, "max": 1200},
    "GOLD": {"min": 700},
    "BRONZE": {"max": 800}
}

# ================= UTILS =================

# --- show_debug ---
# What it does:
# - Display an image in an OpenCV window only when debug mode is enabled.
# - Used to avoid opening many windows during normal runs.
def show_debug(name, img):
    if SHOW_DEBUG_WINDOWS:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)

# --- remove_green_background ---
# What it does:
# - Segment out the green conveyor background using HSV thresholds.
# - Return a cleaned foreground mask (credits = white, background = black) using morphology.
def remove_green_background(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.bitwise_or(cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER),
                                 cv2.inRange(hsv, GREEN_LOWER_BRIGHT, GREEN_UPPER_BRIGHT))
    fg = cv2.bitwise_not(green_mask)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8), iterations=2)
    return cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

# --- order_points ---
# What it does:
# - Take 4 corner points and return them in a consistent order: top-left, top-right, bottom-right, bottom-left.
# - Required for stable width/height estimation and perspective warping.
def order_points(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)], 
                     pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)

# --- estimate_w_h_from_box ---
# What it does:
# - Estimate the width and height of a rotated rectangle (minAreaRect box).
# - Averages opposite edges to reduce noise and jitter.
def estimate_w_h_from_box(box):
    r = order_points(box)
    tl, tr, br, bl = r
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    return float(w), float(h)

# --- warp_credit_auto ---
# What it does:
# - Perspective-warp a detected credit region into a fixed-size rectangle.
# - Chooses portrait vs landscape output size based on the detected box dimensions.
def warp_credit_auto(img, box):
    w, h = estimate_w_h_from_box(box)
    out_w, out_h = (200, 520) if h > w else (520, 200)
    src = order_points(box)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))

# --- rotate90 ---
# What it does:
# - Rotate an image by 0/90/180/270 degrees using OpenCV fast rotations.
# - k is the number of 90° clockwise steps.
def rotate90(img, k):
    k %= 4
    if k == 0: return img
    if k == 1: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if k == 2: return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# --- classify_credit_color_from_warp ---
# What it does:
# - Classify the credit as BRONZE/SILVER/GOLD from the warped image using HSV color ranges.
# - Uses a saturation mask to ignore low-color pixels (glare/shadows) and votes by pixel count.
def classify_credit_color_from_warp(warped):
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    #checks if the saturation is > 80 and h,s,v = 0,1,2
    sat_mask = (hsv[:, :, 1] > 80).astype(np.uint8) * 255
    scores = {}
    for k, cfg in COLOR_CONFIG.items():
        if k == "BRONZE":
            m = cv2.inRange(hsv, cfg["lower1"], cfg["upper1"]) | cv2.inRange(hsv, cfg["lower2"], cfg["upper2"])
        else:
            m = cv2.inRange(hsv, cfg["lower"], cfg["upper"])
        scores[k] = cv2.countNonZero(cv2.bitwise_and(m, m, mask=sat_mask))
    best = max(scores, key=scores.get)
    return best if scores[best] > 250 else "UNKNOWN"

# --- build_insert_mask_for_type ---
# What it does:
# - Build a binary mask for the colored inserts/digits for a given credit type.
# - Applies saturation filtering and morphology to produce a clean insert mask.
def build_insert_mask_for_type(warped, credit_type):
    if credit_type == "UNKNOWN":
        # warped.shape will extract the width and height leaving the color channels 
        return np.zeros(warped.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    cfg = COLOR_CONFIG[credit_type]
    mask = (cv2.inRange(hsv, cfg["lower1"], cfg["upper1"]) | cv2.inRange(hsv, cfg["lower2"], cfg["upper2"])) \
           if credit_type == "BRONZE" else cv2.inRange(hsv, cfg["lower"], cfg["upper"])
    mask = cv2.bitwise_and(mask, mask, mask=((hsv[:, :, 1] > 60).astype(np.uint8) * 255))
    H, W = warped.shape[:2]
    k = max(1, int(min(H, W) * 0.006))
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

# --- detect_symbol_at_top ---
# What it does:
# - Detect the GOLD '7-like' decoration in the logical top region of the credit.
# - Returns a score (0/300/500) based on size and position of a detected blob.
def detect_symbol_at_top(mask, H, W):
    #rows:    from 0 to 25% of height ,columns: all columns
    top_band = mask[0:int(H*0.25), :].copy()
    k = max(3, int(min(H, W) * 0.02))
    kernel = np.ones((k, k), np.uint8)
    top_band = cv2.morphologyEx(top_band, cv2.MORPH_CLOSE, kernel, iterations=3)
    top_band = cv2.morphologyEx(top_band, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(top_band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0
    
    total_area = H * W
    min_area, strong_area = total_area * 0.005, total_area * 0.015
    center_left, center_right = W * 0.2, W * 0.8
    best_score = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w / 2
        if area >= min_area and center_left < cx < center_right:
            best_score = max(best_score, 500 if area > strong_area else 300)
    
    return best_score

# --- detect_fake_credit ---
# What it does:
# - Apply size-based sanity checks to flag credits that are too small/large for their type.
# - Uses the longer side so the rule works regardless of rotation.
def detect_fake_credit(credit_type, pre_w, pre_h):
    """Check if credit is fake based on longer dimension (works for any orientation)"""
    reasons = []
    longer_side = max(pre_w, pre_h)  # Use longer dimension regardless of orientation
    
    if credit_type == "SILVER":
        if longer_side < SIZE_THRESHOLDS["SILVER"]["min"]:
            reasons.append("TOO_SMALL_SILVER")
        elif longer_side > SIZE_THRESHOLDS["SILVER"]["max"]:
            reasons.append("TOO_LARGE_SILVER")
    elif credit_type == "GOLD":
        if longer_side < SIZE_THRESHOLDS["GOLD"]["min"]:
            reasons.append("TOO_SMALL_GOLD")
    elif credit_type == "BRONZE":
        if longer_side > SIZE_THRESHOLDS["BRONZE"]["max"]:
            reasons.append("TOO_LARGE_BRONZE")
    
    return len(reasons) > 0, reasons

# --- normalize_orientation_grade_c ---
# What it does:
# - Try all 4 rotations of the warped credit and score each to choose the most likely upright orientation.
# - Uses portrait bias, GOLD symbol-at-top bonus, and top-vs-bottom insert/saturation heuristics.
def normalize_orientation_grade_c(warped, credit_id="X"):
    credit_type = classify_credit_color_from_warp(warped)
    H0, W0 = warped.shape[:2]
    
    if SHOW_DEBUG_WINDOWS:
        print(f"\n>>> ORIENTATION INPUT: {W0}×{H0}, type={credit_type}")

    best_score, best_img, best_k = -1e18, warped, 0

    for k in range(4):
        cand = rotate90(warped, k)
        H, W = cand.shape[:2]
        portrait_bonus = 1000.0 if H > W else -1000.0
        hsv = cv2.cvtColor(cand, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        mask = build_insert_mask_for_type(cand, credit_type)
        symbol_bonus = detect_symbol_at_top(mask, H, W) if credit_type == "GOLD" else 0
        
        top_slice, bot_slice = slice(0, int(H * 0.25)), slice(int(H * 0.75), H)
        top_insert = cv2.countNonZero(mask[top_slice, :])
        bot_insert = cv2.countNonZero(mask[bot_slice, :])
        top_sat, bot_sat = float(np.mean(sat[top_slice, :])), float(np.mean(sat[bot_slice, :]))

        score = portrait_bonus + symbol_bonus
        if symbol_bonus == 0:
            score += 0.004 * (top_insert - bot_insert) + 0.8 * (top_sat - bot_sat) - 0.003 * bot_insert

        if SHOW_DEBUG_WINDOWS:
            print(f"  k={k} ({k*90:3d}°): {W:3d}×{H:3d}, score={score:7.1f}")

        if score > best_score:
            best_score, best_img, best_k = score, cand, k

    H, W = best_img.shape[:2]
    if W > H:
        best_img = cv2.rotate(best_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    if SHOW_DEBUG_WINDOWS:
        print(f">>> CHOSEN: {best_k*90}°, final: {best_img.shape[1]}×{best_img.shape[0]}\n")

    return best_img

# --- group_blocks_by_digit_row ---
# What it does:
# - Group detected insert blocks into horizontal rows, where each row corresponds to one digit.
# - Uses vertical center clustering with a height-scaled threshold to be resolution tolerant.
def group_blocks_by_digit_row(blocks, warped, cid):
    if not blocks:
        return []
    items = [(b, b[1] + b[3] / 2) for b in blocks]
    items.sort(key=lambda x: x[1])
    rows, cur, cy = [], [items[0][0]], items[0][1]
    thresh = max(12, int(warped.shape[0] * 0.12))

    for b, y in items[1:]:
        if abs(y - cy) <= thresh:
            cur.append(b)
            cy = (cy + y) / 2
        else:
            rows.append(cur)
            cur, cy = [b], y
    rows.append(cur)
    return rows

# --- extract_digit_features ---
# What it does:
# - Convert a row of blocks into digit features (C1 height class + C2 presence pattern).
# - Maps those features through DIGIT_MAP to produce the final digit label.
def extract_digit_features(blocks, warped, label):
    if not blocks:
        return None, None, "UNKNOWN"

    H, W = warped.shape[:2]
    max_area = max(b[2] * b[3] for b in blocks)
    big_blocks = [b for b in blocks if b[2] * b[3] >= max_area * 0.9]
    big_block = min(big_blocks, key=lambda b: b[0]) if len(big_blocks) > 1 else big_blocks[0]
    bx, by, bw, bh = big_block
    bcx = bx + bw / 2

    col1_blocks, col2_blocks = [], []
    grouping_threshold = bw * 0.8

    for (x, y, w, h) in blocks:
        cx = x + w / 2
        (col1_blocks if abs(cx - bcx) < grouping_threshold else col2_blocks).append((x, y, w, h))

    if not col1_blocks:
        return None, None, "UNKNOWN"

    c1_h = max(col1_blocks, key=lambda b: b[3])[3]
    ref_h = np.median([b[3] for b in col2_blocks]) if col2_blocks else c1_h
    ratio = c1_h / max(1.0, ref_h)
    C1_height = 1 if ratio < 1.6 else (2 if ratio < 3.2 else 3)

    if not col2_blocks:
        C2 = [0, 0, 0]
    else:
        MIN_C2_AREA = H * W * 0.001
        valid_c2 = sorted([b for b in col2_blocks if b[2] * b[3] >= MIN_C2_AREA], 
                         key=lambda b: b[1] + b[3] / 2)
        C2 = [0, 0, 0]
        for i in range(min(3, len(valid_c2))):
            C2[i] = 1

    digit = DIGIT_MAP.get((C1_height, tuple(C2)), "UNKNOWN")
    return C1_height, C2, digit

# --- detect_blocks_refined ---
# What it does:
# - Detect candidate digit blocks by color masking, morphology, and contour filtering.
# - Filters by area, rectangularity (area/hull_area), and border proximity; returns blocks + visualization.
def detect_blocks_refined(warped, credit_type, cid):
    if credit_type == "UNKNOWN":
        return [], warped.copy()
    
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    cfg = COLOR_CONFIG[credit_type]
    mask = (cv2.inRange(hsv, cfg["lower1"], cfg["upper1"]) | cv2.inRange(hsv, cfg["lower2"], cfg["upper2"])) \
           if credit_type == "BRONZE" else cv2.inRange(hsv, cfg["lower"], cfg["upper"])
    mask &= (hsv[:, :, 1] > 60)
    
    k = max(1, int(min(warped.shape[:2]) * 0.006))
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks, vis = [], warped.copy()
    H, W = warped.shape[:2]

    for c in contours:
        area = cv2.contourArea(c)
        if not (H * W * 0.002 <= area <= H * W * 0.08):
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0 or area / hull_area < 0.85:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if x < 10 or y < 10 or x + w > W - 10 or y + h > H - 10:
            continue
        blocks.append((x, y, w, h))
        cv2.rectangle(vis, (x, y), (x + w, y + h), cfg["display_color"], 2)

    if SHOW_DEBUG_WINDOWS:
        show_debug(f"Stage 4 - Blocks {cid}", vis)
    return blocks, vis

# --- calculate_credit_value ---
# What it does:
# - Compute the final credit value from decoded digits based on the credit type rules.
# - BRONZE: multiply digits; SILVER: concatenate; GOLD: concatenate then ×10.
def calculate_credit_value(digits, credit_type):
    if not digits or "UNKNOWN" in digits:
        return None
    if credit_type == "BRONZE":
        value = 1
        for d in digits:
            value *= int(d)
        return value
    elif credit_type == "SILVER":
        return int(''.join(map(str, digits)))
    elif credit_type == "GOLD":
        return int(''.join(map(str, digits))) * 10
    return None

# ================= FRAME PROCESSING =================

# --- process_frame ---
# What it does:
# - Run the full pipeline on a single video frame: segment credits, warp, orient, classify, decode digits, and fake-check.
# - Returns a per-credit result tuple used for drawing and tracking.
def process_frame(frame):
    mask = remove_green_background(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), 1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    credit_boxes = [cv2.boxPoints(cv2.minAreaRect(c)) for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
    results = []
    
    for idx, box in enumerate(credit_boxes):
        cid = idx + 1
        pre_w, pre_h = estimate_w_h_from_box(box)
        print(f"Credit {cid}: Pre-warp size: {pre_w:.1f} x {pre_h:.1f}")
        
        warped = warp_credit_auto(frame, box)
        oriented = normalize_orientation_grade_c(warped, credit_id=cid)
        credit_type = classify_credit_color_from_warp(oriented)
        is_fake, fake_reasons = detect_fake_credit(credit_type, pre_w, pre_h)  # Pass both dimensions
        blocks, _ = detect_blocks_refined(oriented, credit_type, cid)
        rows = group_blocks_by_digit_row(blocks, oriented, cid)
        
        digits = []
        for did, r in enumerate(rows, 1):
            _, _, d = extract_digit_features(r, oriented, f"{cid}-{did}")
            if d != "UNKNOWN":
                digits.append(d)
        
        value = calculate_credit_value(digits, credit_type) if digits else None
        results.append((credit_type, digits, value, box, is_fake, fake_reasons))
    
    return results

# --- draw_results_on_frame ---
# What it does:
# - Draw detection boxes, labels (type/REAL/FAKE), and decoded values onto the output frame.
# - Also prints rule-violation reasons for fakes.
def draw_results_on_frame(frame, results):
    output = frame.copy()
    for credit_type, digits, value, box, is_fake, fake_reasons in results:
        box_int = np.int32(box)
        box_color = COLOR_CONFIG.get(credit_type, {}).get("display_color", (255, 255, 255))
        text_color = (0, 0, 255) if is_fake else box_color
        cv2.drawContours(output, [box_int], 0, box_color, 3)
        
        top_left = box[np.argmin(box.sum(axis=1))]
        x, y = int(top_left[0]), int(top_left[1])
        status = "FAKE" if is_fake else "REAL"
        cv2.putText(output, f"{credit_type} | {status}", (x, y - 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 3)

        if not is_fake and value is not None:
            if credit_type == "BRONZE":
                line = f"{' x '.join(map(str, digits))} = {value}"
            elif credit_type == "SILVER":
                line = str(value)
            elif credit_type == "GOLD":
                base = int("".join(map(str, digits)))
                line = f"{base} x 10 = {value}"
            else:
                line = str(value)
            cv2.putText(output, line, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        else:
            reason_text = ",".join(fake_reasons) if fake_reasons else "RULE_VIOLATION"
            cv2.putText(output, reason_text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    return output

# ================= TRACKING =================
class TrackedCredit:

    # --- __init__ ---
    # What it does:
    # - Initialize object state for this class instance (IDs, tracking timers, value history, etc.).
    def __init__(self, credit_type, digits, value, box, frame_id):
        self.id = str(uuid.uuid4())[:8]
        self.credit_type, self.box = credit_type, box
        self.first_seen = self.last_seen = frame_id
        self.counted = self.finalized = self.updated_this_frame = False
        self.value_history = [value] if value is not None else []

    # --- update ---
    # What it does:
    # - Update this tracked object with a new detection (box, frame index, and optional value).
    # - Maintains last-seen time and accumulates value history.
    def update(self, box, frame_id, value):
        self.box, self.last_seen = box, frame_id
        if value is not None:
            self.value_history.append(value)

# --- box_iou ---
# What it does:
# - Compute IoU (intersection-over-union) between two rotated boxes (as 4-point polygons).
# - Used to match detections to existing tracks.
def box_iou(boxA, boxB):
    polyA, polyB = cv2.convexHull(np.int32(boxA)), cv2.convexHull(np.int32(boxB))
    inter = cv2.intersectConvexConvex(polyA, polyB)[0]
    union = cv2.contourArea(polyA) + cv2.contourArea(polyB) - inter
    return 0 if union == 0 else inter / union

class CreditTracker:

    # --- CreditTracker.__init__ ---
    # What it does:
    # - Create a tracker that matches credits across frames using IoU and ages out stale tracks.
    # - Maintains the running TOTAL credits value.
    def __init__(self, iou_threshold=0.15, max_age=30):
        self.tracked, self.iou_threshold, self.max_age, self.total_value = [], iou_threshold, max_age, 0

    # --- update ---
    # What it does:
    # - Update this tracked object with a new detection (box, frame index, and optional value).
    # - Maintains last-seen time and accumulates value history.
    def update(self, detections, frame_id):
        for tc in self.tracked:
            tc.updated_this_frame = False

        for (ctype, digits, value, box) in detections:
            best_tc, best_iou = None, 0.0
            for tc in self.tracked:
                if not tc.updated_this_frame and tc.credit_type == ctype:
                    iou = box_iou(tc.box, box)
                    if iou > self.iou_threshold and iou > best_iou:
                        best_tc, best_iou = tc, iou

            if best_tc:
                best_tc.update(box, frame_id, value)
                best_tc.updated_this_frame = True
                if TRACKER_DEBUG:
                    print(f"[MATCH] frame={frame_id} {ctype} id={best_tc.id} IoU={best_iou:.2f} val={value}")
            else:
                tc = TrackedCredit(ctype, digits, value, box, frame_id)
                self.tracked.append(tc)
                if TRACKER_DEBUG:
                    print(f"[NEW] frame={frame_id} {ctype} id={tc.id} val={value}")

        alive = []
        for tc in self.tracked:
            if (frame_id - tc.last_seen) > self.max_age and not tc.finalized:
                final_value = None
                if tc.credit_type == "GOLD" and tc.value_history:
                    final_value = max(tc.value_history)
                elif tc.value_history:
                    non_zero = [v for v in tc.value_history if v > 0]
                    final_value = max(non_zero) if non_zero else 0
                
                if final_value is not None:
                    self.total_value += final_value
                    tc.finalized = True
                    if TRACKER_DEBUG:
                        print(f"[FINALIZE] {tc.credit_type} id={tc.id} val={final_value} TOTAL={self.total_value}")
            
            if (frame_id - tc.last_seen) <= self.max_age:
                alive.append(tc)
        
        self.tracked = alive
        if TRACKER_DEBUG:
            print(f"[STATE] frame={frame_id} tracked={len(self.tracked)} total={self.total_value}")

    # --- draw_overlay ---
    # What it does:
    # - draw_overlay: helper/function used in the credit-detection pipeline.
    # - See the code below for implementation details.
    def draw_overlay(self, frame):
        y = frame.shape[0] - 40
        cv2.rectangle(frame, (0, y - 30), (500, y + 10), (0, 0, 0), -1)
        cv2.putText(frame, f"TOTAL CREDITS VALUE: {self.total_value}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

# ================= BASLER CAMERA SETUP =================

# --- setup_basler_camera ---
# What it does:
# - Initialize a Basler camera (pypylon) and configure frame rate, exposure, and gain.
# - Returns the camera and converter objects used to grab BGR frames.
def setup_basler_camera():
    """Initialize Basler camera with proper property access"""
    if not BASLER_AVAILABLE:
        print("ERROR: pypylon library not installed.")
        print("Install with: pip install pypylon")
        return None, None
    
    try:
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        
        if not devices:
            print("ERROR: No Basler camera detected.")
            return None, None
        
        print(f"Found Basler camera: {devices[0].GetFriendlyName()}")
        
        camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
        camera.Open()
        
        print("Configuring camera...")
        
        # Frame rate - use try/except for each property
        try:
            camera.AcquisitionFrameRateEnable.SetValue(True)
            camera.AcquisitionFrameRate.SetValue(10)
            print(" Frame rate: 10 FPS")
        except Exception as e:
            print(f" Frame rate config failed: {e}")
        
        # Exposure
        try:
            camera.ExposureAuto.SetValue("Off")
            camera.ExposureTime.SetValue(8000)
            print("  Exposure: 8000 µs")
        except Exception as e:
            print(f"  Exposure config failed: {e}")
        
        # Gain
        try:
            camera.GainAuto.SetValue("Off")
            camera.Gain.SetValue(0)
            print("  Gain: 0")
        except Exception as e:
            print(f" Gain config failed: {e}")
        
        # Start grabbing
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        # Create converter
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        
        print("Basler camera ready!")
        return camera, converter
        
    except Exception as e:
        print(f"ERROR initializing Basler: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ================= MAIN =================

# --- main ---
# What it does:
# - CLI entry point: choose input source (video/webcam/Basler), run processing loop, handle UI controls, and optionally record output.
# - Manages pause/step/debug toggles and ensures resources are released cleanly.
def main():
    global SHOW_DEBUG_WINDOWS
    
    if len(sys.argv) < 2:
        print(f"Credit Detector v{__version__}")
        print("\n" + "="*60)
        print("USAGE:")
        print("="*60)
        print("\n1. VIDEO FILE:")
        print("   python script.py video <filename> [output]")
        print("   Example: python script.py video input.mp4 output.mp4")
        print("\n2. BASLER CAMERA:")
        print("   python script.py basler [output]")
        print("   Example: python script.py basler recording.mp4")
        print("\n3. WEBCAM:")
        print("   python script.py camera [output]")
        print("   Example: python script.py camera webcam_rec.mp4")
        print("\n" + "="*60)
        print("SHORTCUTS (legacy):")
        print("="*60)
        print("   python script.py <video.mp4> [output]")
        print("   python script.py basler [output]")
        print("   python script.py camera [output]")
        print("\n" + "="*60)
        print("CONTROLS:")
        print("="*60)
        print("   q / ESC  - Quit")
        print("   d        - Toggle debug windows")
        print("   p        - Pause/Resume")
        print("   SPACE    - Step (single frame when paused)")
        print("   s        - Save current frame as PNG")
        print("="*60)
        sys.exit(1)
    
    source_arg = sys.argv[1].lower()
    output_file = None
    
    # Parse command line arguments
    if source_arg == "video":
        if len(sys.argv) < 3:
            print("ERROR: Video mode requires filename")
            print("Usage: python script.py video <filename> [output]")
            sys.exit(1)
        source = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        source_name = f"Video: {source}"
        use_basler = False
        cap = cv2.VideoCapture(source)
        camera = None
        converter = None
        
    elif source_arg == "basler":
        use_basler = True
        source_name = "Basler Camera"
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        camera, converter = setup_basler_camera()
        if camera is None:
            sys.exit(1)
        cap = None
        
    elif source_arg == "camera":
        use_basler = False
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        camera_id = 0
        cap = cv2.VideoCapture(camera_id)
        source_name = f"Webcam {camera_id}"
        camera = None
        converter = None
        
    else:
        # Legacy mode: assume first argument is video file or camera
        source = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        use_basler = False
        cap = None
        camera = None
        converter = None
        
        if source.lower() == "basler":
            use_basler = True
            source_name = "Basler Camera"
            camera, converter = setup_basler_camera()
            if camera is None:
                sys.exit(1)
        elif source.isdigit():
            camera_id = int(source)
            cap = cv2.VideoCapture(camera_id)
            source_name = f"Camera {camera_id}"
        else:
            cap = cv2.VideoCapture(source)
            source_name = f"Video: {source}"
    
    if not use_basler and (cap is None or not cap.isOpened()):
        print(f"ERROR: Could not open: {source_name}")
        sys.exit(1)

    tracker = CreditTracker()
    
    # Video writer setup (will initialize for Basler after first frame)
    video_writer = None
    video_fps = 10  # Default FPS for Basler
    
    if output_file and not use_basler:
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
        if video_writer.isOpened():
            print(f"Recording to: {output_file}")
    
    print(f"\nCredit Detector v{__version__} - {source_name}")
    print("Controls: q-Quit | d-Debug | p-Pause | SPACE-Step | s-Save\n")
    
    fps_time, fps_counter, fps_display = time.time(), 0, 0
    frame_count, paused, process_single, current_output = 0, False, False, None
    
    try:
        while True:
            if not paused or process_single:
                process_single = False
                frame_count += 1
                
                if use_basler:
                    grab = camera.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
                    if not grab.GrabSucceeded():
                        grab.Release()
                        continue
                    image = converter.Convert(grab)
                    frame = image.GetArray()
                    grab.Release()
                    
                    # Initialize video writer on first frame for Basler
                    if output_file and video_writer is None and frame is not None:
                        h, w = frame.shape[:2]
                        video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (w, h))
                        if video_writer.isOpened():
                            print(f"Recording to: {output_file} ({w}x{h} @ {video_fps}fps)")
                        else:
                            print("WARNING: Could not initialize video writer")
                            video_writer = None
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break
                
                if FRAME_SKIP > 0 and frame_count % (FRAME_SKIP + 1) != 0:
                    continue
                
                start_time = time.time()
                results = process_frame(frame)
                process_time = time.time() - start_time
                
                real_results = [(ct, d, v, b) for (ct, d, v, b, fake, _) in results if not fake]
                tracker.update(real_results, frame_count)
                
                output = draw_results_on_frame(frame, results)
                tracker.draw_overlay(output)
                current_output = output.copy()
                
                if video_writer:
                    video_writer.write(output)
                
                fps_counter += 1
                if time.time() - fps_time >= 1.0:
                    fps_display, fps_counter, fps_time = fps_counter, 0, time.time()
                
                info = [f"FPS: {fps_display}", f"Process: {process_time*1000:.0f}ms", 
                       f"Credits: {len(results)}", f"Frame: {frame_count}",
                       f"Mode: {source_name}"]
                if video_writer:
                    info.append("REC")
                
                for i, text in enumerate(info):
                    cv2.putText(output, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                h, w = output.shape[:2]
                display = cv2.resize(output, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
                cv2.namedWindow("Credit Detector", cv2.WINDOW_NORMAL)
                cv2.imshow("Credit Detector", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('d'):
                SHOW_DEBUG_WINDOWS = not SHOW_DEBUG_WINDOWS
                print(f"Debug: {'ON' if SHOW_DEBUG_WINDOWS else 'OFF'}")
                if not SHOW_DEBUG_WINDOWS:
                    cv2.destroyAllWindows()
            elif key == ord('p'):
                paused = not paused
                print('PAUSED' if paused else 'RESUMED')
            elif key == ord(' ') and paused:
                process_single = True
            elif key == ord('s') and current_output is not None:
                fname = f"frame_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(fname, current_output)
                print(f"Saved: {fname}")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if use_basler and camera:
            try:
                camera.StopGrabbing()
                camera.Close()
                print("Camera closed")
            except:
                pass
        elif cap:
            cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print("Done")

if __name__ == "__main__":
    main()
