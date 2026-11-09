# Changelog

All notable changes to this project are listed here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.2.0] - 2025-01-06

### Added
- Command-line modes for three input sources: `video <file>`, `camera` (webcam) and `basler` (Basler camera via pypylon).
- Legacy shortcuts: pass a video path or a camera index directly.
- Optional recording of the annotated output to an `.mp4` file in every mode.
- Basler camera setup: 10 FPS, fixed exposure (8000 µs) and gain (0).
- Keyboard controls: quit, pause/resume, single-frame step, toggle debug windows, save frame as PNG.
- On-screen info: FPS, processing time, credit count, frame number, input mode and REC flag.

### Changed
- Fake detection uses the longer side of the credit, so it works in any orientation.

### Repository
- Packaged as a GitHub repository. The detection algorithm and all thresholds are unchanged; only the module docstring of `credit_detector.py` was rewritten.
- README with quick start, worked examples, setup tuning guide and known limitations.
- `docs/HOW_IT_WORKS.md` with every pipeline step and its exact numbers.
- Tests for the helper functions, the tracker, end-to-end runs on synthetic frames and the command line.
- GitHub Actions workflow running the tests on Python 3.9 to 3.12.
- MIT license, requirements files, `CONTRIBUTING.md`, `pyproject.toml` (pytest settings only) and this changelog.
