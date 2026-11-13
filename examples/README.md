# Examples

Put your sample videos in this folder. Video files are **not** stored in git
(`*.mp4`, `*.avi`, `*.mov` and `*.mkv` are in `.gitignore`).

A good test video:

- shows credits moving on a **green** conveyor belt,
- is filmed from above, with steady light,
- has a similar camera distance to the one used to tune the size limits
  (see "Fake detection" in the main README).

Run the detector on a sample:

```bash
python credit_detector.py video examples/sample.mp4
python credit_detector.py video examples/sample.mp4 examples/sample_out.mp4
```
