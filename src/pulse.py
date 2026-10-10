"""Turn a pulse waveform into BPM and a signal-quality score."""
import numpy as np
from scipy import signal as sps

MIN_HZ, MAX_HZ = 0.7, 3.0  # 42 - 180 BPM


def bandpass(wave, fs, low=MIN_HZ, high=MAX_HZ, order=2):
    b, a = sps.butter(order, [low / (fs / 2), high / (fs / 2)], btype="bandpass")
    return sps.filtfilt(b, a, np.asarray(wave, dtype=np.float64))


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


def quality_label(snr, face_found_ratio=1.0, mean_luma=128.0):
    """Combine cues into Good / Fair / Poor. Thresholds get tuned in Phase 5."""
    if face_found_ratio < 0.9 or not (40 <= mean_luma <= 220) or snr < -3:
        return "Poor"
    if snr < 3:
        return "Fair"
    return "Good"
