# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
Nmax = 500

# %%
OscilloscopeData_LED = pd.read_csv('LED.TXT', header=None)
OscilloscopeData_LED

# %%
OscilloscopeData_LED = OscilloscopeData_LED.transpose()
OSCData_LED = OscilloscopeData_LED[0:Nmax]
OSCData_LED.values

# %%
OscilloscopeData_EFF = pd.read_csv('EFF.TXT', header=None)
OscilloscopeData_EFF

# %%
OscilloscopeData_EFF = OscilloscopeData_EFF.transpose()
OSCData_EFF = OscilloscopeData_EFF[0:Nmax]
OSCData_EFF.values

# %%
OscilloscopeData_ELECTRO = pd.read_csv('ELECTRO.TXT', header=None)
OscilloscopeData_ELECTRO

# %%
OscilloscopeData_ELECTRO = OscilloscopeData_ELECTRO.transpose()
OSCData_ELECTRO = OscilloscopeData_ELECTRO[0:Nmax]
OSCData_ELECTRO.values

# %%
y1 = OSCData_LED.values
y2 = OSCData_EFF.values
y3 = OSCData_ELECTRO.values
x = range(len(OSCData_LED))
plt.figure(figsize=(12, 6))
plt.plot(x, y1, label='Светодиодная')
plt.plot(x, y2, label='Энергоэффективная')
plt.plot(x, y3, label='Накаливания')
plt.title('ОСЦИЛЛОГРАММА')
plt.xlabel('Номер точки измерения')
plt.ylabel('Сигнал осциллографа')
plt.grid()
plt.legend()
plt.show()

# %%
(OSCData_LED.max() - OSCData_LED.min())/2

# %%
OSCData_LED.min()

# %%
from scipy.fft import fft, fftfreq

# %%
yf = fft(y)
yf

# %%
220500 / 5980

# %% [markdown]
# --------------------------------------------------

# %%
import numpy as np
from matplotlib import pyplot as plt

SAMPLE_RATE = 44100  # Гц
DURATION = 5  # Секунды

def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate*duration, endpoint=False)
    frequencies = x * freq
    # 2pi для преобразования в радианы
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

# Генерируем волну с частотой 2 Гц, которая длится 5 секунд
x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)
plt.plot(x, y)
plt.show()

# %%
_, nice_tone = generate_sine_wave(400, SAMPLE_RATE, DURATION)
_, noise_tone = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
noise_tone = noise_tone * 0.3
mixed_tone = nice_tone + noise_tone
normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)

plt.plot(normalized_tone[:1000])
plt.show()

# %%
import scipy

# число точек в normalized_tone
N = SAMPLE_RATE * DURATION

yf = fft(normalized_tone)
xf = fftfreq(N, 1 / SAMPLE_RATE)

plt.plot(xf, np.abs(yf))
plt.show()

# %%
# обратите внимание на r в начале имён функций
yf = rfft(normalized_tone)
xf = rfftfreq(N, 1/SAMPLE_RATE)

plt.plot(xf, np.abs(yf))
plt.show()

# %%



