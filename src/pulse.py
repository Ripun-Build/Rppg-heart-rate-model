"""Turn a pulse waveform into BPM and a signal-quality score."""
import numpy as np
from scipy import signal as sps

MIN_HZ, MAX_HZ = 0.7, 3.0  # 42 - 180 BPM


def bandpass(wave, fs, low=MIN_HZ, high=MAX_HZ, order=2):
    b, a = sps.butter(order, [low / (fs / 2), high / (fs / 2)], btype="bandpass")
    return sps.filtfilt(b, a, np.asarray(wave, dtype=np.float64))


def to_pulse(wave, is_diff):
    """Turn a model output / label into a pulse wave before measuring HR.

    With DiffNormalized input the network predicts the frame-to-frame DIFFERENCE of the
    pulse. Differencing amplifies higher frequencies, so the 2nd harmonic (2x HR) can
    out-power the fundamental, especially at low heart rates (seen on subject37:
    55 BPM read as 110). Integrating (cumulative sum) and removing the drift restores the
    pulse shape, as rPPG-Toolbox does for diff-trained models.
    """
    wave = np.asarray(wave, dtype=np.float64)
    return sps.detrend(np.cumsum(wave)) if is_diff else wave


def _spectrum(wave, fs, n_fft=None):
    wave = np.asarray(wave, dtype=np.float64)
    wave = wave - wave.mean()
    # Zero-pad for a finer frequency grid (~0.1 BPM resolution at 30 fps).
    n_fft = n_fft or max(len(wave), int(fs * 60 * 10))
    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)
    power = np.abs(np.fft.rfft(wave * np.hanning(len(wave)), n=n_fft)) ** 2
    band = (freqs >= MIN_HZ) & (freqs <= MAX_HZ)
    return freqs[band], power[band]


def estimate_bpm(wave, fs, filtered=False):
    """Heart rate in BPM from the dominant frequency of the waveform."""
    if not filtered:
        wave = bandpass(wave, fs)
    freqs, power = _spectrum(wave, fs)
    return float(freqs[np.argmax(power)] * 60)


def snr_db(wave, fs, filtered=False, width_hz=0.1):
    """Spectral SNR: power near the HR peak (+ 1st harmonic) vs. everything else in band."""
    if not filtered:
        wave = bandpass(wave, fs)
    freqs, power = _spectrum(wave, fs)
    f0 = freqs[np.argmax(power)]
    signal_mask = (np.abs(freqs - f0) <= width_hz) | (np.abs(freqs - 2 * f0) <= width_hz)
    sig = power[signal_mask].sum()
    noise = power[~signal_mask].sum() + 1e-12
    return float(10 * np.log10(sig / noise))


def pos(rgb, fs, window_s=1.6):
    """POS (Wang et al. 2017): classical, non-learned pulse extraction from mean skin RGB.

    rgb: (T, 3) mean R, G, B of the face per frame. Used only as a sanity check / baseline.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    n, l = len(rgb), int(window_s * fs)
    projection = np.array([[0, 1, -1], [-2, 1, 1]])
    out = np.zeros(n)
    for end in range(l, n + 1):
        c = rgb[end - l : end]
        c = (c / (c.mean(axis=0) + 1e-9)).T            # temporal normalization
        s = projection @ c                              # project onto plane orthogonal to skin tone
        h = s[0] + (s[0].std() / (s[1].std() + 1e-9)) * s[1]
        out[end - l : end] += h - h.mean()              # overlap-add
    return out


def chrom(rgb, fs, window_s=1.6):
    """CHROM (de Haan & Jeanne 2013): two fixed color-difference signals, combined so that
    motion/specular changes (equal in both) cancel. Hann-windowed overlap-add."""
    rgb = np.asarray(rgb, dtype=np.float64)
    n = len(rgb)
    l = int(window_s * fs) // 2 * 2  # even window length for 50% overlap
    out = np.zeros(n)
    win = np.hanning(l)
    for start in range(0, n - l + 1, l // 2):
        c = rgb[start : start + l]
        c = c / (c.mean(axis=0) + 1e-9)
        x = 3 * c[:, 0] - 2 * c[:, 1]
        y = 1.5 * c[:, 0] + c[:, 1] - 1.5 * c[:, 2]
        xf, yf = bandpass(x, fs), bandpass(y, fs)
        s = xf - (xf.std() / (yf.std() + 1e-9)) * yf
        out[start : start + l] += win * (s - s.mean())
    return out


def green(rgb, fs):
    """GREEN (Verkruysse et al. 2008): just the mean green channel, the simplest baseline."""
    g = np.asarray(rgb, dtype=np.float64)[:, 1]
    return g - g.mean()


# SNR thresholds (dB) of the PhysNet output, chosen on the UBFC validation subjects in Phase 5:
# clips with SNR >= 0 were within 5 BPM 98% of the time; most badly wrong clips had SNR < -2.
SNR_GOOD, SNR_FAIR = 0.0, -2.0


def quality_label(snr, face_found_ratio=1.0, mean_luma=128.0):
    """Combine cues into Good / Fair / Poor."""
    if face_found_ratio < 0.9 or not (40 <= mean_luma <= 220) or snr < SNR_FAIR:
        return "Poor"
    if snr < SNR_GOOD:
        return "Fair"
    return "Good"
