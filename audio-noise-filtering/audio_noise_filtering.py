import numpy as np
import librosa
import scipy.signal as sg
import argparse, pathlib
import matplotlib.pyplot as plt, soundfile as sf
import librosa.display  # only used inside the demo plots


# Utility helpers
def _zpf_sosfilt(sos, x):
    """Zero-phase filtering via forward-backward IIR."""
    return sg.sosfiltfilt(sos, x, padtype="odd", padlen=min(3 * (max(len(sos), 1)), len(x) - 1))


def _spectral_mode(f_arr, *, bins=20):
    """Robust central-tendency estimator for a 1D frequency list (Hz)."""
    hist, edges = np.histogram(f_arr, bins=bins)
    mode_bin = np.argmax(hist)
    return (edges[mode_bin] + edges[mode_bin + 1]) / 2


# Q1: single, precise notch with zero-phase filtering
def q1(audio_path, *, sr=4_000) -> np.ndarray:
    """Return the denoised signal for question 1."""
    x, _ = librosa.load(audio_path, sr=sr)
    # --- detect the stationary whistle ---
    n_fft, hop = 2048, 512
    mag = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop))
    f_axis = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    peak_bins = mag.argmax(axis=0)
    peak_freqs = f_axis[peak_bins]
    whistle_hz = _spectral_mode(peak_freqs)        # ≈ 1125 Hz
    # --- design a single narrow notch ---
    bw = 6        # bandwidth in Hz (tweak if residual whistle remains)
    nyq = sr / 2
    lo, hi = (whistle_hz - bw) / nyq, (whistle_hz + bw) / nyq
    sos = sg.butter(N=2, Wn=[lo, hi], btype="bandstop", output="sos")
    # --- zero-phase filter & return ---
    return _zpf_sosfilt(sos, x).astype(np.float32)


# Q2: one 8-pole elliptic band-stop (zero-phase)
def q2(audio_path, *, sr=4_000) -> np.ndarray:
    """Return the denoised signal for question 2."""
    x, _ = librosa.load(audio_path, sr=sr)
    # --- only filter where noise exists (1.3–4.2 s) ---
    t0, t1 = 1.3, 4.2
    s0, s1 = int(t0 * sr), int(t1 * sr)
    seg = x[s0:s1]
    # --- design an elliptic band-stop around 575-630 Hz ---
    nyq = sr / 2
    lo, hi = 575 / nyq, 630 / nyq
    sos = sg.ellip(N=8,            # order = 8  -> steep skirt yet stable
                   rp=0.2,         # <= 0.2 dB ripple in pass-band
                   rs=60,          # >= 60 dB attenuation in stop-band
                   Wn=[lo, hi],
                   btype="bandstop",
                   output="sos")
    x_filt = x.copy()
    x_filt[s0:s1] = _zpf_sosfilt(sos, seg)
    return x_filt.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="audio_noise_filtering.py",
        description="Targeted audio denoising demo for Q1/Q2 filters."
    )
    parser.add_argument("audio_path",
                        type=pathlib.Path,
                        help="Path to the WAV file you want to denoise.")
    parser.add_argument("--which",
                        choices=["q1", "q2"],
                        required=True,
                        help="Which denoiser to run: q1 (whistle) or q2 (hum).")
    parser.add_argument("--save", action="store_true",
                        help="Write *_denoised.wav files next to the input.")
    parser.add_argument("--plots", action="store_true",
                        help="Show before/after log-spectrogram plots.")

    args = parser.parse_args()

    funcs = [(args.which, q1 if args.which == "q1" else q2)]
    # Run the chosen denoiser(s)
    for tag, func in funcs:
        y = func(args.audio_path)

        if args.save:
            out = args.audio_path.with_name(f"{args.audio_path.stem}_{tag}_denoised.wav")
            sf.write(out, y, 4_000)
            print(f"Successfully saved {out}")

        if args.plots:
            # before/after log-spectrogram
            fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
            for ax, sig, ttl in zip(
                    axs,
                    [librosa.load(args.audio_path, sr=4_000)[0], y],
                    ["original", "denoised"]):
                S = librosa.amplitude_to_db(
                    np.abs(librosa.stft(sig, n_fft=512, hop_length=128)), ref=np.max)
                img = librosa.display.specshow(
                    S, sr=4_000, hop_length=128, y_axis="linear",
                    x_axis="time", ax=ax)
                ax.set(title=ttl)
            plt.tight_layout()
            fig.colorbar(img, ax=axs, format="%+2.f dB")
            plt.suptitle(tag.upper())

    if args.plots:
        plt.show()
