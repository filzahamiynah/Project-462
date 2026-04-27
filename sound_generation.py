import numpy as np
from scipy.io.wavfile import write

# PARAMETERS
fs = 48000         
duration_repeat = 8 #repeat count

# 1. GENERATE TRAINING SIGNAL
np.random.seed(42)
p = np.random.randn(int(fs * 0.1))
p = p / np.max(np.abs(p))

# 2. REPEAT SIGNA
tx = np.tile(p, duration_repeat)

# 3. CONVERT TO 16-BIT AUDIO
tx_int16 = np.int16(tx * 32767)

# 4. SAVE FILE
write("echolocation_signal.wav", fs, tx_int16)

# 5. SAVE ORIGINAL SIGNAL 
np.save("p.npy", p)

print(" Audio file saved: echolocation_signal.wav")
print(" Training signal saved: p.npy")