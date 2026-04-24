import numpy as np
import sounddevice as sd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# -----------------------------
# PARAMETERS
# -----------------------------
fs = 48000              # MUST match transmitter phone
duration = 3.0          # seconds (record long enough)
c = 343.0               # speed of sound (m/s)

# -----------------------------
# 1. TRAINING SIGNAL (MUST MATCH TX PHONE)
# -----------------------------
# IMPORTANT: this must be EXACTLY the same signal you play on phone 1

p = np.load("p.npy")
p = p / np.max(np.abs(p))   # normalize

# -----------------------------
# 2. RECORD AUDIO
# -----------------------------
print("Recording... play sound from phone NOW")

rx = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()

rx = rx.flatten()
print("Recording finished")

# -----------------------------
# 3. ALIGN SIGNAL (CRITICAL STEP)
# -----------------------------
# Find where the transmitted signal appears in recording

correlation = np.correlate(rx, p, mode='valid')
start_idx = np.argmax(np.abs(correlation))
rx_block = rx[start_idx : start_idx + len(p)]

print("Signal detected at sample:", start_idx)

# Extract one clean block
if start_idx + len(p) > len(rx):
    print("❌ Recording too short, try increasing duration")
    exit()

# -----------------------------
# 4. CHANNEL ESTIMATION (CIR)
# -----------------------------
# Extract the segment that matches the length of your training signal p

X = np.fft.fft(p)
Y = np.fft.fft(rx_block)

H = Y / (X + 1e-8)       # avoid divide by zero
h = np.fft.ifft(H)

h = np.real(h)
mag = np.abs(h)

# -----------------------------
# 5. PEAK DETECTION
# -----------------------------
peaks, _ = find_peaks(mag, distance=20, height=np.max(mag)*0.2)

if len(peaks) < 2:
    print("❌ Not enough peaks detected. Try cleaner environment.")
    exit()

# pick top 2 peaks
top_peaks = peaks[np.argsort(mag[peaks])[-2:]]
top_peaks = np.sort(top_peaks)

print("Detected peaks at:", top_peaks)

# -----------------------------
# 6. CIRCULAR GAP (IMPORTANT)
# -----------------------------
gap = abs(top_peaks[1] - top_peaks[0])
gap = min(gap, len(h) - gap)

print("Sample gap:", gap)

# -----------------------------
# 7. DISTANCE CALCULATION
# -----------------------------
time_delay = gap / fs
extra_path = c * time_delay

# round-trip → divide by 2
distance = extra_path / 2

print(f"Extra path length: {extra_path:.4f} m")
print(f"Estimated reflector distance: {distance:.4f} m")

# -----------------------------
# 8. VISUALIZATION (FOR DEMO)
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(mag, label="CIR Magnitude")
plt.plot(top_peaks, mag[top_peaks], 'ro', label="Detected Peaks")
plt.title("Channel Impulse Response (CIR)")
plt.xlabel("Sample Index")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()
plt.show()
