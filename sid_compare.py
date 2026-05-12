#!/usr/bin/env python3
"""Compare SID chip audio recordings/emulations against a reference.

Supports two modes:
  1. Stereo file: left channel = reference, right channel = comparison
  2. Two separate mono/stereo files: reference file + comparison file

Produces time-domain, frequency-domain, and perceptual metrics useful for
evaluating how close a SID emulation is to real hardware (8580/6581).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.gridspec import GridSpec
from scipy import signal, stats


def load_audio(path: str, mono: bool = False, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float64")
    if mono and data.ndim == 2:
        data = data.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        data = resample_audio(data, sr, target_sr)
        sr = target_sr
    return data, sr


def resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate using polyphase filtering."""
    from math import gcd
    if orig_sr == target_sr:
        return data
    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    if data.ndim == 2:
        return np.stack([signal.resample_poly(data[:, ch], up, down) for ch in range(data.shape[1])], axis=1)
    return signal.resample_poly(data, up, down)


def detect_tune_start(x: np.ndarray, sr: int, frame_ms: float = 100.0,
                      silence_frac: float = 0.10) -> int:
    """Sample index of the first frame whose RMS clearly exceeds the silence floor.

    Uses a fraction of the *typical loud* level (95th-percentile RMS over 100 ms
    frames) as the threshold. If the very first frame is already loud, returns 0
    — i.e. the tune already starts from sample zero. This avoids the pitfall of
    estimating a "noise floor" from the first 200 ms when the recording has no
    leading silence at all.
    """
    frame = int(sr * frame_ms / 1000)
    n_frames = len(x) // frame
    if n_frames < 5:
        return 0
    rms_frames = np.sqrt(np.mean(x[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1))
    loud_level = float(np.percentile(rms_frames, 95))
    threshold = silence_frac * loud_level
    above = np.flatnonzero(rms_frames > threshold)
    return int(above[0] * frame) if len(above) else 0


def align_signals(ref: np.ndarray, comp: np.ndarray, sr: int, max_lag_ms: float = 50.0) -> tuple[np.ndarray, np.ndarray, int]:
    """Align two signals using cross-correlation, compensating for recording delay."""
    max_lag = int(sr * max_lag_ms / 1000)
    chunk = min(len(ref), len(comp), sr * 5)  # use first 5s for alignment
    corr = signal.correlate(ref[:chunk], comp[:chunk], mode="full")
    mid = len(corr) // 2
    search = corr[mid - max_lag : mid + max_lag]
    lag = np.argmax(search) - max_lag

    if lag > 0:
        ref = ref[lag:]
    elif lag < 0:
        comp = comp[-lag:]

    n = min(len(ref), len(comp))
    return ref[:n], comp[:n], lag


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def crest_factor(x: np.ndarray) -> float:
    r = rms(x)
    return float(peak(x) / r) if r > 0 else 0.0


def snr_db(ref: np.ndarray, diff: np.ndarray) -> float:
    sig_power = np.mean(ref**2)
    noise_power = np.mean(diff**2)
    if noise_power == 0:
        return float("inf")
    return float(10 * np.log10(sig_power / noise_power))


def compute_spectrogram(x: np.ndarray, sr: int, nperseg: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, t, Sxx = signal.spectrogram(x, fs=sr, nperseg=nperseg, noverlap=nperseg // 2, window="hann")
    Sxx_db = 10 * np.log10(Sxx + 1e-20)
    return f, t, Sxx_db


def spectral_centroid(x: np.ndarray, sr: int, nperseg: int = 4096) -> np.ndarray:
    f, t, Sxx = signal.spectrogram(x, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    return np.sum(f[:, None] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-20)


def frequency_band_energy(x: np.ndarray, sr: int, bands: list[tuple[float, float]]) -> dict[str, float]:
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    power = np.abs(X) ** 2
    total = np.sum(power)
    result = {}
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        result[f"{lo:.0f}-{hi:.0f}Hz"] = float(np.sum(power[mask]) / total) if total > 0 else 0.0
    return result


def thd(x: np.ndarray, sr: int, fundamental_range: tuple[float, float] = (200, 5000)) -> float:
    """Estimate total harmonic distortion relative energy (simplified)."""
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    mag = np.abs(X)
    fund_mask = (freqs >= fundamental_range[0]) & (freqs < fundamental_range[1])
    fund_power = np.sum(mag[fund_mask] ** 2)
    total_power = np.sum(mag**2)
    if total_power == 0:
        return 0.0
    return float(np.sqrt((total_power - fund_power) / total_power))


def envelope_correlation(ref: np.ndarray, comp: np.ndarray, sr: int, frame_ms: float = 10.0) -> float:
    """Correlation of amplitude envelopes — measures dynamics similarity."""
    frame = int(sr * frame_ms / 1000)
    n_frames = min(len(ref), len(comp)) // frame
    ref_env = np.array([rms(ref[i * frame : (i + 1) * frame]) for i in range(n_frames)])
    comp_env = np.array([rms(comp[i * frame : (i + 1) * frame]) for i in range(n_frames)])
    if np.std(ref_env) == 0 or np.std(comp_env) == 0:
        return 0.0
    return float(np.corrcoef(ref_env, comp_env)[0, 1])


def zero_crossing_rate(x: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(np.sign(x))) > 0))


def spectral_flux(x: np.ndarray, sr: int, nperseg: int = 2048) -> np.ndarray:
    _, _, Sxx = signal.spectrogram(x, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    Sxx_norm = Sxx / (np.sum(Sxx, axis=0, keepdims=True) + 1e-20)
    flux = np.sqrt(np.sum(np.diff(Sxx_norm, axis=1) ** 2, axis=0))
    return flux


def compute_all_metrics(ref: np.ndarray, comp: np.ndarray, sr: int) -> dict:
    diff = ref - comp

    metrics = {}

    # --- Time domain ---
    metrics["ref_rms"] = rms(ref)
    metrics["comp_rms"] = rms(comp)
    metrics["diff_rms"] = rms(diff)
    metrics["ref_peak"] = peak(ref)
    metrics["comp_peak"] = peak(comp)
    metrics["diff_peak"] = peak(diff)
    metrics["ref_crest_factor"] = crest_factor(ref)
    metrics["comp_crest_factor"] = crest_factor(comp)
    metrics["snr_db"] = snr_db(ref, diff)

    # Pearson correlation
    metrics["pearson_r"] = float(np.corrcoef(ref, comp)[0, 1])

    # Sample-level error stats
    abs_diff = np.abs(diff)
    metrics["mae"] = float(np.mean(abs_diff))
    metrics["max_abs_error"] = float(np.max(abs_diff))
    metrics["rmse"] = float(np.sqrt(np.mean(diff**2)))
    metrics["median_abs_error"] = float(np.median(abs_diff))
    metrics["error_std"] = float(np.std(diff))

    # Zero crossing rate comparison
    metrics["ref_zcr"] = zero_crossing_rate(ref)
    metrics["comp_zcr"] = zero_crossing_rate(comp)

    # --- Envelope / dynamics ---
    metrics["envelope_correlation"] = envelope_correlation(ref, comp, sr)

    # --- Frequency domain ---
    sid_bands = [
        (0, 200),
        (200, 1000),
        (1000, 4000),
        (4000, 8000),
        (8000, 16000),
        (16000, sr / 2),
    ]
    ref_bands = frequency_band_energy(ref, sr, sid_bands)
    comp_bands = frequency_band_energy(comp, sr, sid_bands)
    metrics["ref_band_energy"] = ref_bands
    metrics["comp_band_energy"] = comp_bands
    metrics["band_energy_diff"] = {k: comp_bands[k] - ref_bands[k] for k in ref_bands}

    # Spectral centroid stats
    ref_sc = spectral_centroid(ref, sr)
    comp_sc = spectral_centroid(comp, sr)
    metrics["ref_spectral_centroid_mean"] = float(np.mean(ref_sc))
    metrics["comp_spectral_centroid_mean"] = float(np.mean(comp_sc))
    metrics["spectral_centroid_diff_mean"] = float(np.mean(comp_sc - ref_sc))

    # Spectral flux correlation
    ref_flux = spectral_flux(ref, sr)
    comp_flux = spectral_flux(comp, sr)
    n_flux = min(len(ref_flux), len(comp_flux))
    metrics["spectral_flux_correlation"] = float(np.corrcoef(ref_flux[:n_flux], comp_flux[:n_flux])[0, 1])

    return metrics


def print_report(metrics: dict, lag: int, sr: int, ref_label: str, comp_label: str):
    lag_ms = lag / sr * 1000

    print("=" * 70)
    print(f"  SID Audio Comparison Report")
    print(f"  Reference : {ref_label}")
    print(f"  Comparison: {comp_label}")
    print("=" * 70)

    print(f"\n--- Alignment ---")
    print(f"  Cross-correlation lag    : {lag} samples ({lag_ms:+.2f} ms)")

    print(f"\n--- Time Domain ---")
    print(f"  {'Metric':<30} {'Reference':>12} {'Comparison':>12} {'Diff/Note':>12}")
    print(f"  {'-'*66}")
    print(f"  {'RMS level':<30} {metrics['ref_rms']:>12.6f} {metrics['comp_rms']:>12.6f} {metrics['diff_rms']:>12.6f}")
    print(f"  {'Peak level':<30} {metrics['ref_peak']:>12.6f} {metrics['comp_peak']:>12.6f} {metrics['diff_peak']:>12.6f}")
    print(f"  {'Crest factor':<30} {metrics['ref_crest_factor']:>12.4f} {metrics['comp_crest_factor']:>12.4f}")
    print(f"  {'Zero crossing rate':<30} {metrics['ref_zcr']:>12.6f} {metrics['comp_zcr']:>12.6f}")

    print(f"\n--- Error Metrics ---")
    print(f"  SNR (signal vs difference) : {metrics['snr_db']:.2f} dB")
    print(f"  Pearson correlation        : {metrics['pearson_r']:.8f}")
    print(f"  RMSE                       : {metrics['rmse']:.8f}")
    print(f"  MAE                        : {metrics['mae']:.8f}")
    print(f"  Median absolute error      : {metrics['median_abs_error']:.8f}")
    print(f"  Max absolute error         : {metrics['max_abs_error']:.8f}")
    print(f"  Error std dev              : {metrics['error_std']:.8f}")

    print(f"\n--- Dynamics ---")
    print(f"  Envelope correlation       : {metrics['envelope_correlation']:.6f}")

    print(f"\n--- Frequency Domain ---")
    print(f"  Spectral centroid (mean)   : ref={metrics['ref_spectral_centroid_mean']:.1f} Hz, "
          f"comp={metrics['comp_spectral_centroid_mean']:.1f} Hz, "
          f"diff={metrics['spectral_centroid_diff_mean']:+.1f} Hz")
    print(f"  Spectral flux correlation  : {metrics['spectral_flux_correlation']:.6f}")

    print(f"\n  Band energy distribution (fraction of total):")
    print(f"  {'Band':<20} {'Reference':>12} {'Comparison':>12} {'Diff':>12}")
    print(f"  {'-'*56}")
    for band in metrics["ref_band_energy"]:
        r = metrics["ref_band_energy"][band]
        c = metrics["comp_band_energy"][band]
        d = metrics["band_energy_diff"][band]
        print(f"  {band:<20} {r:>12.6f} {c:>12.6f} {d:>+12.6f}")

    # Overall similarity score (composite)
    score = (
        0.30 * metrics["pearson_r"]
        + 0.25 * metrics["envelope_correlation"]
        + 0.20 * metrics["spectral_flux_correlation"]
        + 0.15 * max(0, min(1, metrics["snr_db"] / 60.0))
        + 0.10 * (1.0 - min(1.0, abs(metrics["spectral_centroid_diff_mean"]) / 1000.0))
    )

    print(f"\n{'=' * 70}")
    print(f"  COMPOSITE SIMILARITY SCORE : {score:.4f} / 1.0000")
    print(f"  (weighted: correlation 30%, envelope 25%, spectral flux 20%,")
    print(f"   SNR 15%, spectral centroid 10%)")
    print(f"{'=' * 70}")


def plot_comparison(ref: np.ndarray, comp: np.ndarray, sr: int, metrics: dict,
                    ref_label: str, comp_label: str, output_path: str):
    duration = len(ref) / sr
    t = np.linspace(0, duration, len(ref))

    fig = plt.figure(figsize=(18, 22))
    gs = GridSpec(5, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(f"SID Comparison: {ref_label} vs {comp_label}", fontsize=14, fontweight="bold")

    diff = ref - comp

    # 1. Waveform overlay (zoomed to a representative 50ms section)
    ax1 = fig.add_subplot(gs[0, 0])
    zoom_start = int(sr * 2)  # start at 2 seconds
    zoom_len = int(sr * 0.05)  # 50ms
    zoom_end = min(zoom_start + zoom_len, len(ref))
    zt = t[zoom_start:zoom_end] * 1000 - t[zoom_start] * 1000
    ax1.plot(zt, ref[zoom_start:zoom_end], alpha=0.8, linewidth=0.5, label=ref_label)
    ax1.plot(zt, comp[zoom_start:zoom_end], alpha=0.8, linewidth=0.5, label=comp_label)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Waveform overlay (50ms @ 2s)")
    ax1.legend(fontsize=8)

    # 2. Difference signal
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t[::100], diff[::100], linewidth=0.3, color="red", alpha=0.7)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Difference signal (ref - comp)")

    # 3. Reference spectrogram
    ax3 = fig.add_subplot(gs[1, 0])
    f, t_spec, Sxx_ref = compute_spectrogram(ref, sr)
    ax3.pcolormesh(t_spec, f, Sxx_ref, shading="gouraud", cmap="magma", vmin=-80, vmax=0)
    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_ylim(0, 16000)
    ax3.set_title(f"Spectrogram: {ref_label}")

    # 4. Comparison spectrogram
    ax4 = fig.add_subplot(gs[1, 1])
    f, t_spec, Sxx_comp = compute_spectrogram(comp, sr)
    ax4.pcolormesh(t_spec, f, Sxx_comp, shading="gouraud", cmap="magma", vmin=-80, vmax=0)
    ax4.set_ylabel("Frequency (Hz)")
    ax4.set_ylim(0, 16000)
    ax4.set_title(f"Spectrogram: {comp_label}")

    # 5. Spectral difference
    ax5 = fig.add_subplot(gs[2, 0])
    n_spec = min(Sxx_ref.shape[1], Sxx_comp.shape[1])
    spec_diff = Sxx_comp[:, :n_spec] - Sxx_ref[:, :n_spec]
    im = ax5.pcolormesh(t_spec[:n_spec], f, spec_diff, shading="gouraud", cmap="RdBu_r", vmin=-20, vmax=20)
    ax5.set_ylabel("Frequency (Hz)")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylim(0, 16000)
    ax5.set_title("Spectral difference (comp - ref, dB)")
    plt.colorbar(im, ax=ax5, label="dB")

    # 6. Average power spectrum comparison
    ax6 = fig.add_subplot(gs[2, 1])
    nperseg = 8192
    f_psd, Pref = signal.welch(ref, sr, nperseg=nperseg)
    f_psd, Pcomp = signal.welch(comp, sr, nperseg=nperseg)
    ax6.semilogy(f_psd, Pref, alpha=0.8, linewidth=0.8, label=ref_label)
    ax6.semilogy(f_psd, Pcomp, alpha=0.8, linewidth=0.8, label=comp_label)
    ax6.set_xlabel("Frequency (Hz)")
    ax6.set_ylabel("Power spectral density")
    ax6.set_xlim(0, 20000)
    ax6.set_title("Average power spectrum")
    ax6.legend(fontsize=8)

    # 7. PSD ratio (frequency response difference)
    ax7 = fig.add_subplot(gs[3, 0])
    ratio_db = 10 * np.log10((Pcomp + 1e-20) / (Pref + 1e-20))
    ax7.plot(f_psd, ratio_db, linewidth=0.8, color="purple")
    ax7.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax7.set_xlabel("Frequency (Hz)")
    ax7.set_ylabel("dB")
    ax7.set_xlim(0, 20000)
    ax7.set_ylim(-30, 30)
    ax7.set_title("Frequency response difference (comp/ref)")

    # 8. Band energy comparison bar chart
    ax8 = fig.add_subplot(gs[3, 1])
    bands = list(metrics["ref_band_energy"].keys())
    ref_vals = [metrics["ref_band_energy"][b] * 100 for b in bands]
    comp_vals = [metrics["comp_band_energy"][b] * 100 for b in bands]
    x_pos = np.arange(len(bands))
    w = 0.35
    ax8.bar(x_pos - w / 2, ref_vals, w, label=ref_label, alpha=0.8)
    ax8.bar(x_pos + w / 2, comp_vals, w, label=comp_label, alpha=0.8)
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(bands, rotation=45, ha="right", fontsize=7)
    ax8.set_ylabel("Energy (%)")
    ax8.set_title("Band energy distribution")
    ax8.legend(fontsize=8)

    # 9. Amplitude envelope comparison
    ax9 = fig.add_subplot(gs[4, 0])
    frame_ms = 10
    frame = int(sr * frame_ms / 1000)
    n_frames = min(len(ref), len(comp)) // frame
    ref_env = np.array([rms(ref[i * frame : (i + 1) * frame]) for i in range(n_frames)])
    comp_env = np.array([rms(comp[i * frame : (i + 1) * frame]) for i in range(n_frames)])
    t_env = np.arange(n_frames) * frame_ms / 1000
    ax9.plot(t_env, ref_env, alpha=0.8, linewidth=0.5, label=ref_label)
    ax9.plot(t_env, comp_env, alpha=0.8, linewidth=0.5, label=comp_label)
    ax9.set_xlabel("Time (s)")
    ax9.set_ylabel("RMS amplitude")
    ax9.set_title("Amplitude envelope (10ms frames)")
    ax9.legend(fontsize=8)

    # 10. Error histogram
    ax10 = fig.add_subplot(gs[4, 1])
    ax10.hist(diff, bins=500, density=True, alpha=0.7, color="red")
    ax10.set_xlabel("Sample difference")
    ax10.set_ylabel("Density")
    ax10.set_title(f"Error distribution (RMSE={metrics['rmse']:.6f})")
    mu, sigma = np.mean(diff), np.std(diff)
    ax10.axvline(mu, color="black", linestyle="--", linewidth=0.8, label=f"mean={mu:.6f}")
    ax10.legend(fontsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare SID audio recordings/emulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stereo file (left=reference, right=comparison):
  %(prog)s recording.wav

  # Two separate files:
  %(prog)s --ref hardware.wav --comp emulation.wav

  # Swap stereo channels (right=reference):
  %(prog)s recording.wav --swap-channels

  # Custom labels:
  %(prog)s recording.wav --ref-label "8580 Hardware" --comp-label "reSID"
        """,
    )
    parser.add_argument("stereo_file", nargs="?", help="Stereo WAV (L=ref, R=comp)")
    parser.add_argument("--ref", help="Reference audio file")
    parser.add_argument("--comp", help="Comparison audio file")
    parser.add_argument("--ref-label", default=None, help="Label for reference signal")
    parser.add_argument("--comp-label", default=None, help="Label for comparison signal")
    parser.add_argument("--swap-channels", action="store_true", help="Swap L/R channels")
    parser.add_argument("--no-align", action="store_true", help="Skip cross-correlation alignment")
    parser.add_argument("--max-lag-ms", type=float, default=50.0, help="Max alignment lag in ms (default: 50)")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plots")
    parser.add_argument("--output", "-o", default=None, help="Output plot file path")
    parser.add_argument("--target-sr", type=int, default=None,
                        help="Resample both signals to this rate. If omitted and rates differ, "
                             "resamples to the lower of the two.")
    parser.add_argument("--trim-silence", action="store_true",
                        help="Detect and trim leading silence in each signal before aligning. "
                             "Anchors comparison at the first musical event in each file.")

    args = parser.parse_args()

    if args.stereo_file and (args.ref or args.comp):
        parser.error("Provide either a stereo file OR --ref/--comp, not both")
    if not args.stereo_file and not (args.ref and args.comp):
        parser.error("Provide either a stereo file or both --ref and --comp")

    if args.stereo_file:
        print(f"Loading stereo file: {args.stereo_file}")
        data, sr = load_audio(args.stereo_file)
        if data.ndim != 2 or data.shape[1] < 2:
            print("Error: file is not stereo", file=sys.stderr)
            sys.exit(1)
        if args.swap_channels:
            ref_sig, comp_sig = data[:, 1], data[:, 0]
        else:
            ref_sig, comp_sig = data[:, 0], data[:, 1]
        ref_label = args.ref_label or "Left channel"
        comp_label = args.comp_label or "Right channel"
        output_default = str(Path(args.stereo_file).with_suffix(".png"))
    else:
        print(f"Loading reference: {args.ref}")
        print(f"Loading comparison: {args.comp}")
        ref_sig, sr_ref = load_audio(args.ref, mono=True)
        comp_sig, sr_comp = load_audio(args.comp, mono=True)
        if sr_ref != sr_comp or args.target_sr is not None:
            target_sr = args.target_sr or min(sr_ref, sr_comp)
            if sr_ref != target_sr:
                print(f"Resampling reference: {sr_ref} Hz -> {target_sr} Hz")
                ref_sig = resample_audio(ref_sig, sr_ref, target_sr)
            if sr_comp != target_sr:
                print(f"Resampling comparison: {sr_comp} Hz -> {target_sr} Hz")
                comp_sig = resample_audio(comp_sig, sr_comp, target_sr)
            sr = target_sr
        else:
            sr = sr_ref
        ref_label = args.ref_label or Path(args.ref).stem
        comp_label = args.comp_label or Path(args.comp).stem
        output_default = "comparison.png"

    print(f"Sample rate: {sr} Hz, Duration: {len(ref_sig)/sr:.2f}s")

    if args.trim_silence:
        ref_start = detect_tune_start(ref_sig, sr)
        comp_start = detect_tune_start(comp_sig, sr)
        print(f"Trimming leading silence: ref={ref_start/sr:.3f}s, comp={comp_start/sr:.3f}s")
        ref_sig = ref_sig[ref_start:]
        comp_sig = comp_sig[comp_start:]
        n = min(len(ref_sig), len(comp_sig))
        ref_sig, comp_sig = ref_sig[:n], comp_sig[:n]

    if not args.no_align:
        print(f"Aligning signals (max lag: {args.max_lag_ms:.1f} ms)...")
        ref_sig, comp_sig, lag = align_signals(ref_sig, comp_sig, sr, args.max_lag_ms)
        print(f"  Detected lag: {lag} samples ({lag/sr*1000:+.2f} ms)")
    else:
        n = min(len(ref_sig), len(comp_sig))
        ref_sig, comp_sig = ref_sig[:n], comp_sig[:n]
        lag = 0

    print("Computing metrics...")
    metrics = compute_all_metrics(ref_sig, comp_sig, sr)

    print_report(metrics, lag, sr, ref_label, comp_label)

    if not args.no_plot:
        output_path = args.output or output_default
        print(f"\nGenerating plots...")
        plot_comparison(ref_sig, comp_sig, sr, metrics, ref_label, comp_label, output_path)


if __name__ == "__main__":
    main()
