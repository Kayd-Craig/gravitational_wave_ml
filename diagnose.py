import sys, numpy as np
sys.path.insert(0, '.')
from scipy.signal import welch
from scipy.stats import kurtosis
from src.data.preprocessing import bandpass_filter

syn = "data/synthetic"
fs = 4096.0

def load_sample(fname, n=50):
    return np.load(f"{syn}/{fname}")["data"][:n].astype(np.float64)

noise_segs = load_sample("noise_backgrounds.npz", 200)

# Build reference PSD
psds = []
for seg in noise_segs:
    bp = bandpass_filter(seg, fs, 20., 500.)
    f, p = welch(bp, fs=fs, nperseg=512)
    psds.append(p)
ref_freqs = f
ref_psd = np.mean(psds, axis=0)

def whiten(seg):
    seg = bandpass_filter(seg, fs, 20., 500.)
    n = len(seg)
    rfft_freqs = np.fft.rfftfreq(n, d=1.0/fs)
    psd_interp = np.interp(rfft_freqs, ref_freqs, ref_psd)
    asd = np.sqrt(np.maximum(psd_interp, 1e-100))
    asd = np.maximum(asd, np.percentile(asd[asd > 0], 5))
    Xf = np.fft.rfft(seg)
    w = np.real(np.fft.irfft(Xf / asd, n=n))
    return np.nan_to_num(w)

def stats(segs, label):
    kurts, crests = [], []
    for seg in segs:
        w = whiten(seg)
        rms = np.std(w) + 1e-30
        kurts.append(kurtosis(w))
        crests.append(np.max(np.abs(w)) / rms)
    print(f"{label:8s}: kurtosis={np.mean(kurts):7.2f}±{np.std(kurts):.1f}  "
          f"crest_factor={np.mean(crests):.2f}±{np.std(crests):.1f}")

print("=== Post-whitening feature separability ===")
stats(load_sample("noise_backgrounds.npz"), "Glitch")
stats(load_sample("bbh_injections.npz"),    "BBH")
stats(load_sample("bns_injections.npz"),    "BNS")
