# Scene-Cut Detection — IMP 2023/24 • Exercise 1

*A lightweight baseline that finds the single hard cut in a short RGB
video using cumulative-histogram differences.*

---

## 📜 Project Summary
This repository contains:

| File | Purpose |
|------|---------|
| `ex1.py` | **Main algorithm** & CLI wrapper |
| `requirements.txt` | Exact Python dependencies (`numpy`, `mediapy`) |
| `ex1.pdf` | Four-page report describing goal, theory, results, and lessons learned |

The goal is to locate **exactly one** scene change and return  
`(last_frame_first_scene, first_frame_second_scene)` as required by the
course autograder.:contentReference[oaicite:0]{index=0}

---

## 🚀 Quick Start

```bash
# 1. Create and activate a fresh environment
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run on your own clip
python ex1.py --video path/to/clip.mp4 --type 1
