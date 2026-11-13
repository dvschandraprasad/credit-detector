# How it works (technical details)

This page explains every step of the pipeline in `credit_detector.py`, with the
exact numbers the code uses. For a short overview, see the [README](../README.md).

All HSV values use the OpenCV scale: **H 0-179, S 0-255, V 0-255**.

## Contents

1. [Find the credits](#1-find-the-credits)
2. [Measure and straighten](#2-measure-and-straighten)
3. [Turn the credit the right way up](#3-turn-the-credit-the-right-way-up)
4. [Decide the colour type](#4-decide-the-colour-type)
5. [Fake check](#5-fake-check)
6. [Find the blocks](#6-find-the-blocks)
7. [Group blocks into digit rows](#7-group-blocks-into-digit-rows)
8. [Read one digit](#8-read-one-digit)
9. [Compute the value](#9-compute-the-value)
10. [Draw the results](#10-draw-the-results)
11. [Track credits and keep the total](#11-track-credits-and-keep-the-total)
12. [Main loop details](#12-main-loop-details)

---

## 1. Find the credits

Functions: `remove_green_background`, `process_frame`

1. The frame is converted from BGR to HSV.
2. Two green masks are made and joined (OR):
   - normal green: `GREEN_LOWER = [35, 40, 40]` to `GREEN_UPPER = [90, 255, 255]`
   - bright / washed-out green: `GREEN_LOWER_BRIGHT = [35, 20, 80]` to `GREEN_UPPER_BRIGHT = [90, 255, 255]`
3. The mask is inverted: everything that is **not** green is foreground.
4. Clean-up inside `remove_green_background`: close with a 21x21 kernel (2 iterations), then open with 5x5 (1 iteration).
5. Extra clean-up in `process_frame`: close 7x7 (3 iterations), then open 5x5 (1 iteration).
6. External contours are found. Each contour with area **above** `MIN_CONTOUR_AREA = 8000` px becomes a credit, and a rotated rectangle (`cv2.minAreaRect` + `cv2.boxPoints`) is fitted to it.

Credits are numbered 1, 2, 3 ... in the order OpenCV returns the contours. For
every credit, the console prints `Credit N: Pre-warp size: W x H`.

## 2. Measure and straighten

Functions: `order_points`, `estimate_w_h_from_box`, `warp_credit_auto`

- `order_points` puts the 4 corners in the order top-left, top-right,
  bottom-right, bottom-left (smallest x+y, smallest y-x, largest x+y, largest y-x).
- `estimate_w_h_from_box` gives width and height as the average of the two
  opposite edges. These **pre-warp** sizes (in pixels of the camera image) are
  used for the fake check.
- `warp_credit_auto` warps the credit to a fixed size: **200 x 520** (portrait)
  if the box is taller than wide, otherwise **520 x 200** (landscape).

## 3. Turn the credit the right way up

Function: `normalize_orientation_grade_c` (helpers: `rotate90`, `build_insert_mask_for_type`, `detect_symbol_at_top`)

First the colour type of the warped image is found (see step 4). Then each of
the 4 rotations (0, 90, 180, 270 degrees clockwise) gets a score:

| Part | Value |
|---|---|
| Portrait bonus | +1000 if height > width, else -1000 |
| GOLD symbol bonus | 0, 300 or 500 (GOLD only, see below) |
| Top/bottom score (only if the symbol bonus is 0) | `0.004 x (top_insert - bot_insert) + 0.8 x (top_sat - bot_sat) - 0.003 x bot_insert` |

- `top_insert` / `bot_insert` = number of insert-mask pixels in the top 25% /
  bottom 25% of the image.
- `top_sat` / `bot_sat` = mean saturation in the top 25% / bottom 25%.
- The **insert mask** (`build_insert_mask_for_type`) is the credit's own colour
  range, AND saturation > 60, then open + close with a small square kernel
  (side = 0.6% of the shorter image side, at least 1 px).

**GOLD symbol** (`detect_symbol_at_top`): looks in the top 25% of the insert mask
(after close x3 and open x1 with a kernel of 2% of the shorter side, at least 3 px).
A blob counts if its area is at least 0.5% of the card and its centre x is
between 20% and 80% of the width. Score is **500** if the area is above 1.5% of
the card, otherwise **300**.

The rotation with the highest score wins. If the winner is still landscape, it
is turned 90 degrees counter-clockwise, so the result is always portrait (200 x 520).

## 4. Decide the colour type

Function: `classify_credit_color_from_warp`

Only pixels with **saturation > 80** can vote. Each type counts how many of
those pixels fall in its range from `COLOR_CONFIG`:

| Type | HSV range(s) |
|---|---|
| BRONZE | H 0-16, S 100-255, V 49-255 **or** H 170-180, S 100-255, V 49-255 |
| SILVER | H 95-128, S 105-255, V 100-255 (blue) |
| GOLD | H 22-35, S 100-255, V 100-255 (yellow) |

The type with the most pixels wins **if it has more than 250 pixels**.
Otherwise the type is `UNKNOWN`.

## 5. Fake check

Function: `detect_fake_credit`

Uses the **longer** pre-warp side, so rotation does not matter. Limits are inclusive.

| Type | Real if longer side is ... | Otherwise |
|---|---|---|
| SILVER | 1050 to 1200 px | `TOO_SMALL_SILVER` or `TOO_LARGE_SILVER` |
| GOLD | at least 700 px | `TOO_SMALL_GOLD` |
| BRONZE | at most 800 px | `TOO_LARGE_BRONZE` |
| UNKNOWN | always real | - |

The fake check does **not** stop digit reading. Digits and a value are still
computed for fakes, but fakes are not tracked and not added to the total.

## 6. Find the blocks

Function: `detect_blocks_refined`

1. Mask of the credit's own colour, AND saturation > 60.
2. Open, then close, with a square kernel of side `max(1, int(0.006 x shorter side))` (1 px for a 200 x 520 card).
3. External contours. A contour is kept as a block only if:
   - its area is between **0.2%** and **8%** of the card area (208 to 8320 px on 200 x 520),
   - area / convex-hull area is at least **0.85** (nearly rectangular / solid),
   - its bounding box is at least **10 px** from every edge.
4. Each block is stored as its bounding box `(x, y, w, h)`.

In debug mode the boxes are drawn on the card and shown in a
`Stage 4 - Blocks <n>` window. `UNKNOWN` credits return no blocks.

> Code note: this function uses `mask &= (hsv[:, :, 1] > 60)`. Because the
> right side is a boolean array, the mask ends up with values 0 and 1 instead of
> 0 and 255. OpenCV treats any non-zero pixel as foreground, so the result is
> still correct. (`build_insert_mask_for_type` uses `cv2.bitwise_and` and keeps 0/255.)

## 7. Group blocks into digit rows

Function: `group_blocks_by_digit_row`

1. Blocks are sorted by their vertical centre.
2. The threshold is `max(12, int(0.12 x card height))` = **62 px** on a 520 px card.
3. Walking from top to bottom: if a block's centre is within the threshold of
   the current row centre, it joins the row and the row centre becomes the
   average of the old centre and the new block's centre. Otherwise a new row starts.

**One row = one digit.** Rows are read top to bottom.

## 8. Read one digit

Function: `extract_digit_features`

1. **Find column 1.** Take the block with the largest area. If several blocks
   have at least 90% of that area, take the left-most of them. This is the
   *big block*.
2. **Split columns.** A block whose centre x is closer than `0.8 x big-block width`
   to the big block's centre x is in column 1. All other blocks are in column 2.
3. **C1 height class.** `ratio = tallest column-1 height / median column-2 height`
   (all column-2 blocks are used here, even tiny ones; if there are none, the ratio is 1).
   - ratio < 1.6 -> class **1**
   - 1.6 <= ratio < 3.2 -> class **2**
   - ratio >= 3.2 -> class **3**
4. **C2 pattern.** Column-2 blocks smaller than 0.1% of the card area (104 px on
   200 x 520) are dropped. With `n` blocks left, the first `min(n, 3)` positions
   are set to 1: `(0,0,0)`, `(1,0,0)`, `(1,1,0)` or `(1,1,1)`.
5. **Look up** `(class, pattern)` in `DIGIT_MAP`. Anything not in the table
   gives `UNKNOWN`, and `process_frame` drops that digit.

| | 0 C2 blocks | 1 | 2 | 3+ |
|---|:-:|:-:|:-:|:-:|
| **class 1 (short)** | 0 | 1 | 2 | 3 |
| **class 2 (medium)** | UNKNOWN | 4 | 5 | 6 |
| **class 3 (tall)** | UNKNOWN | 7 | 8 | 9 |

## 9. Compute the value

Function: `calculate_credit_value`

| Type | Rule |
|---|---|
| BRONZE | product of the digits |
| SILVER | digits joined into one number |
| GOLD | digits joined into one number, times 10 |

Returns `None` if there are no digits, if a digit is `UNKNOWN`, or if the type
is not one of the three.

## 10. Draw the results

Function: `draw_results_on_frame` (returns a copy, the input frame is not changed)

- Box outline in the type's display colour (BGR): BRONZE `(0,165,255)` orange,
  SILVER `(255,0,0)` blue, GOLD `(0,215,255)` gold. `UNKNOWN` is white.
- First line: `<TYPE> | REAL` or `<TYPE> | FAKE`. Text is red for fakes.
- Second line:
  - real **and** has a value: BRONZE `2 x 3 = 6`, SILVER `43`, GOLD `42 x 10 = 420`
  - otherwise: the fake reasons, or `RULE_VIOLATION` if there are none
    (this happens for real credits with no readable value).

## 11. Track credits and keep the total

Class: `CreditTracker(iou_threshold=0.15, max_age=30)`, helper `box_iou`

Only **real** credits (not fake) are passed to the tracker. `UNKNOWN` credits
are real, so they are tracked too, but they never have a value.

For every frame:

1. **Match.** For each detection, look at tracks of the **same type** that have
   not been matched yet in this frame. The track with the highest IoU wins, if
   that IoU is **above 0.15**. IoU is computed on the rotated boxes with
   `cv2.intersectConvexConvex` (corners rounded to whole pixels).
2. **New track.** No match -> new `TrackedCredit` with an 8-character random ID.
3. **History.** Every non-`None` value is added to the track's `value_history`.
4. **Finalize.** A track that has not been seen for **more than 30 frames**
   (`frame_id - last_seen > max_age`) is finalized:
   - GOLD: `max(value_history)`
   - BRONZE / SILVER (and UNKNOWN): the largest value above 0, or 0 if all values were 0
   - Empty history: nothing is added.

   The final value is added to `total_value`, and the track is removed.

Note: values are never negative, so the GOLD rule and the other rule give the
same number in practice. The split is kept as it is in the graded code.

`draw_overlay` draws `TOTAL CREDITS VALUE: <n>` in a black bar near the bottom
of the frame.

With `TRACKER_DEBUG = True` the console shows:

```
[NEW] frame=12 SILVER id=3fa2c1d0 val=43
[STATE] frame=12 tracked=1 total=0
[MATCH] frame=13 SILVER id=3fa2c1d0 IoU=0.91 val=43
[STATE] frame=13 tracked=1 total=0
...                                   (last seen in frame 44)
[FINALIZE] SILVER id=3fa2c1d0 val=43 TOTAL=43
[STATE] frame=75 tracked=0 total=43
```

(The IDs are random, so yours will be different.)

## 12. Main loop details

- `frame_count` goes up for every frame read. With `FRAME_SKIP = N > 0`, only
  frames where `frame_count % (N + 1) == 0` are processed.
- The tracker uses `frame_count`, so `max_age = 30` means 30 *read* frames
  (3 seconds at 10 FPS, 1 second at 30 FPS).
- The output video and the PNG saved with `s` contain the boxes, labels and the
  total, but **not** the green info text in the top-left corner (it is drawn after saving).
- The display window is resized by `DISPLAY_SCALE = 0.7`; the saved video is full size.
- Basler: frames are grabbed with `RetrieveResult(2000, TimeoutHandling_ThrowException)`
  using `GrabStrategy_LatestImageOnly`, and converted to BGR8.
