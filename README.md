<p align="center">
  <img src="skeleton_icon.png" alt="Movement Tracker" width="220">
</p>

<h1 align="center">Movement Tracker</h1>

<p align="center">
  <em>A video player built for clinicians, </em><br>
  with built-in computer vision tools for quantifying neuro exams.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: BUSL-1.1" src="https://img.shields.io/badge/license-BUSL--1.1-blue.svg"></a>
  <a href="#"><img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9+-3776ab.svg?logo=python&logoColor=white"></a>
  <a href="https://fastapi.tiangolo.com/"><img alt="FastAPI" src="https://img.shields.io/badge/web-FastAPI-009688?logo=fastapi&logoColor=white"></a>
  <a href="https://deeplabcut.github.io/DeepLabCut/"><img alt="DeepLabCut" src="https://img.shields.io/badge/pose-DeepLabCut-555.svg"></a>
  <a href="https://mediapipe.dev/"><img alt="MediaPipe" src="https://img.shields.io/badge/hands-MediaPipe-4285F4?logo=google&logoColor=white"></a>
  <a href="https://doi.org/10.5281/zenodo.19099855"><img alt="Sample data: Zenodo" src="https://img.shields.io/badge/sample%20data-Zenodo-1682d4.svg"></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick start</a> ·
  <a href="#-features">Features</a> ·
  <a href="#-typical-workflow">Workflow</a> ·
  <a href="#%EF%B8%8F-configuration">Configuration</a> ·
  <a href="#-citation">Citation</a>
</p>

---

## Quick start

```bash
git clone https://github.com/newboldd/movement-tracker
cd movement-tracker
./setup.sh
```

`setup.sh` creates a virtual environment, installs Python dependencies, downloads the ~24 MB sample video from Zenodo, and opens the app at **http://localhost:8080**.

<table>
  <tr>
    <td><strong>macOS / Linux / WSL</strong> — double-click <code>Movement Tracker.app</code> in the repo (or drag it to your Dock).</td>
  </tr>
  <tr>
    <td><strong>Windows</strong> — double-click <code>run.bat</code>. On first launch it auto-creates <code>Movement Tracker.lnk</code> with the hand icon so you can pin it like a normal app.</td>
  </tr>
</table>

---

## Features

<table>
  <tr>
    <td valign="top" width="50%">
      <h3>👤 Subject management</h3>
      Import raw stereo videos, trim trials, organise by diagnosis group. Face detection runs automatically on import so identifiable faces are blurred before storage.
    </td>
    <td valign="top" width="50%">
      <h3>DLC labeling + training</h3>
      Frame-by-frame keypoint annotation with keyboard shortcuts. Auto-detects movement events (open / peak / close) from the distance trace and renders them on a synchronised timeline.
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h3>Run pre-trained models</h3>
      Run Mediapipe, HRnet, or Apple Vision with one click. Forward, reverse, static, and bbox-cropped MediaPipe passes are fused into a single <em>Combined</em> layer that the rest of the pipeline reads. Run any source with one click.
    </td>
    <td valign="top">
      <h3>🦴 Fit Skeleton Model</h3>
      3-D hand model fit to multiple error-corrected pre-trained models with constaint bone lengths.
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h3>📊 Results</h3>
      Per-trial movement metrics (amplitude, IMI, peak velocities, sequence effect, …), group comparisons, and an Explore tool with scatter/bar plots.
    </td>
    <td valign="top">
      <h3>Remote Server Support</h3>
      Submit MediaPipe / DLC inference to a GPU host over SSH. Videos upload, jobs run remotely, results download. Jobs page monitors active jobs and tracks job history.
    </td>
  </tr>
</table>

---

## 🗂️ Sample data

A short finger-tapping example video (<code>Con01_R1.mp4</code>, ~24 MB, stereo) is on Zenodo:

> Newbold, D. (2025). *Finger tapping example* [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.19099855>

Downloaded automatically by `setup.sh`. Manual fetch:

```bash
python scripts/download_sample.py
```

---

## 🧩 Dependencies

| Library |
|---|
| [FastAPI](https://fastapi.tiangolo.com/) · [uvicorn](https://www.uvicorn.org/) |
| [DeepLabCut](https://deeplabcut.github.io/DeepLabCut/) |
| [MediaPipe](https://mediapipe.dev/) |
| [OpenCV](https://opencv.org/) |
| [NumPy](https://numpy.org/) · [SciPy](https://scipy.org/) |
| [Plotly](https://plotly.com/javascript/) |

Full list in `requirements.txt`; installed automatically by `setup.sh`.

---

## 📚 Citation

If you use this software in published research, please cite it:

```bibtex
@software{newbold_movement_tracker,
  author  = {Newbold, Dillan},
  title   = {Movement Tracker},
  url     = {https://github.com/newboldd/movement-tracker},
  license = {BUSL-1.1}
}
```

A [`CITATION.cff`](CITATION.cff) file is included so GitHub shows a **Cite this repository** button in the sidebar.

---

## 📜 License

[**Business Source License 1.1**](LICENSE) — see [`LICENSE`](LICENSE) for the full terms.

- **Additional Use Grant:** free for non-commercial academic research and educational purposes without restriction.
- **Change Date:** four years from the date each version is first published.
- **Change License:** MIT.

For commercial use before the Change Date, please get in touch.

<p align="center">
  <sub>Built for neurological research. Made open for clinicians and scientists to make video-based measurements.</sub>
</p>
