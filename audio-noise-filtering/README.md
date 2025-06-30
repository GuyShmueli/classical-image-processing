# 🎧 Targeted Audio Denoising

> Small but tidy project that demonstrates **classical DSP** techniques for removing two very different noise types from low-rate audio (4 kHz).  
> Part of my *Image & Audio Processing* course but general enough to reuse.

---

## Problem statement

| File | Symptoms | Desired outcome |
|------|----------|-----------------|
| **q1.wav** | Stationary, high-energy whistle ≈ 1.1 kHz covering entire clip | Attenuate whistle without hurting the music |
| **q2.wav** | Mid-band hum (575–630 Hz) *only* between 1.3 s → 4.2 s | Remove hum, leave rest untouched |

Automatic checkers expect two Python functions:

```python
y1 = q1("path/to/q1.wav")   # → numpy.float32, len(y1)==len(input), sr==4 000
y2 = q2("path/to/q2.wav")
