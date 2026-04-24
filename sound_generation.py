import numpy as np
from scipy.io.wavfile import write

# -----------------------------
# PARAMETERS
# -----------------------------
fs = 48000          # MUST match receiver code
duration_repeat = 8 # repeat count

# -----------------------------
# 1. GENERATE TRAINING SIGNAL
# -----------------------------

p = np.load("p.npy")
p = p / np.max(np.abs(p))   # normalize to [-1,1]

# -----------------------------
# 2. REPEAT SIGNAL (IMPORTANT)
# -----------------------------
tx = np.tile(p, duration_repeat)

# -----------------------------
# 3. CONVERT TO 16-BIT AUDIO
# -----------------------------
tx_int16 = np.int16(tx * 32767)

# -----------------------------
# 4. SAVE FILE
# -----------------------------
write("echolocation_signal.wav", fs, tx_int16)

# -----------------------------
# 5. SAVE ORIGINAL SIGNAL (VERY IMPORTANT)
# -----------------------------
np.save("p.npy", p)

print("✅ Audio file saved: echolocation_signal.wav")
print("✅ Training signal saved: p.npy")