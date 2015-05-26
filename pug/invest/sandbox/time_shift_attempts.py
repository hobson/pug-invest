from pug.invest.util import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

np.randn(100)
np.random.randn(100)
Ts = 0.01
t = np.arange(0, 3*60, Ts)
t
3*60
t = np.arange(0, 3 * 60 + Ts, Ts)
t
e1 = .1*np.random.randn(len(t))
e2 = .1*np.random.randn(len(t2))
N1 = len(t1)
t2 = np.arange(60, 2*60+Ts, Ts)
T_cyc = 10
x1 = np.sync((ts * 5. / T_cyc) % T_cyc)
x1 = np.sinc((ts * 5. / T_cyc) % T_cyc)
x1 = np.sinc((t1 * 5. / T_cyc) % T_cyc)
t1 = t
x1 = np.sinc((t1 * 5. / T_cyc) % T_cyc)
df = pd.DataFrame(x1, columns=['signal1'], index=t1)
df.plot()
plt.show()
x2 = np.sinc((t2 * 5. / T_cyc) % T_cyc)
i1 = np.arange(0, len(t1))
delay = 42.42
i2 = np.arange(int(delay / Ts), int(2 * len(t1) / 3))
x1 = np.sinc((t1[i1] * 5. / T_cyc) % T_cyc)
x1 = x1 * np.sinc((t1[i1] - 60) * 3 / t1[-1])
df = pd.DataFrame(x1, columns=['signal 1'], index=t1)
df.plot()
plt.show()
sinc_signals()
T0=[60, 120]; TN=[240, 160]; A=[1, 1]; sigma=[.1, .1]; T_cyc=10; Ts=0.01
N1
sigma
%paste
%paste
%paste
df = sinc_signals(sigma=[0.02, 0.01])
df = sinc_signals(sigma=[0.02, 0.01])
%paste
df = sinc_signals()
%paste
df = sinc_signals()
np.correlate(df['signal 1'], df['signal 2'])
np.correlate(df['signal 1'].dropna().values, df['signal 2'].dropna().values)
xcorr = np.correlate(df['signal 1'].dropna().values, df['signal 2'].dropna().values)
np.argmax(xcorr)
t1[np.argmax(xcorr)]
plt.plot(xcorr)
plt.show(block=False)
%paste
df = double_sinc()
df = double_sinc(sigma=0)
%paste
df = double_sinc(sigma=0)
%paste
df = double_sinc(sigma=0)
df = double_sinc(sigma=0, T_0 = 120)
%paste
df1 = double_sinc(sigma=0)
df1 = double_sinc()
df2 = double_sinc()
df2 - df1
%paste
df1 = double_sinc()
df2 = df1.iloc[1234:4567]
df2 = df2 * .5 + 0.015 * np.random.randn(len(df2))
df2 = 0.5 * df2 + 0.015 * np.random.randn(len(df2))
len(df2)
np.random.randn(len(df2))
df2 = 0.5 * df2
df2 = df2 + 0.015 * np.random.randn(len(df2))
e2 = 0.015 * np.random.randn(len(df2))
e2.shape
df2.shape
df2
df2 += 0.015 * np.random.randn(len(df2), 2)
df2
df2.plot()
plt.show()
plt.show(block=Flase)
plt.show(block=False)
df1.plot()
plt.show(block=Flase)
plt.show(block=False)
df2.plot()
plt.show(block=False)
df2.index[0]
time_shift(df1['y'].values, df2['y'].values)
pd.dropna(df2['y'].values)
%paste
time_shift(df1['y'].values, df2['y'].values)
np.hanning
%paste
time_shift(df1['y'].values, df2['y'].values)
plt.plot(df1['y'].values)
plt.show(block=False)
plt.plot(df1['y'].values)
plt.show(block=False)
plt.plot(df2['y'].values)
plt.show(block=False)
delta = 2027 - 815
s1 = df1['y'].values
s2 = df2['y'].values
xcorr = smooth(np.correlate(s1, s2), window_len=min(11, int(0.15 * len(s1))))
plt.plot(xcorr)
plt.show()
xcorr = np.correlate(s1, s2)
plt.plot(xcorr)
plt.show()
xcorr = smooth(np.correlate(s1, s2), window_len=min(11, int(0.15 * len(s1))))
plt.show()
plt.plot(xcorr)
plt.show()
xcorr = np.correlate(s1, s2)
plt.plot(xcorr)
plt.show()
plt.plot(xcorr)
plt.show()
xcorr = np.corcoeff(s1, s2)
cc = np.corrcoef(s1, s2)
# s2p = np.lib.pad(s2, (len(s1)-len(s2)) / 2, 'linear_ramp', end_values=(0,0))
len(s2p)
s2p = np.append(s2, np.zeros((len(s1)-len(s2))))
cc = np.corrcoef(s1, s2p)
plt.plot(cc)
plt.show()
