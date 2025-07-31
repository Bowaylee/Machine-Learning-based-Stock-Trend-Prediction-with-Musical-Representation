# 確保你安裝了必要套件
# pip install yfinance scikit-learn MIDIUtil pandas numpy

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from midiutil import MIDIFile

# 1. 下載台積電歷史資料 (近1年每日收盤價)
tsmc = yf.download("2330.TW", period="1y", interval="1d")
tsmc = tsmc[['Close']].dropna()

# 2. 製作特徵與標籤
# 明天收盤價比今天高標為1，否則0
tsmc['Target'] = (tsmc['Close'].shift(-1) > tsmc['Close']).astype(int)
tsmc = tsmc.dropna()

# 使用前5天收盤價作為特徵
for i in range(1, 6):
    tsmc[f'lag_{i}'] = tsmc['Close'].shift(i)
tsmc = tsmc.dropna()

X = tsmc[[f'lag_{i}' for i in range(1, 6)]].values
y = tsmc['Target'].values

# 3. 分割訓練和測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# 4. 訓練隨機森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. 預測並計算準確率
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 6. 根據預測結果產生 MIDI
midi = MIDIFile(1)
track = 0
time = 0
midi.addTempo(track, time, 120)

# 漲對應C大調音符序列，跌對應A小調音符序列
c_major = [60, 62, 64, 65, 67, 69, 71, 72]  # C4開始
a_minor = [57, 59, 60, 62, 64, 65, 67, 69]  # A3開始

for i, pred in enumerate(y_pred):
    pitch_seq = c_major if pred == 1 else a_minor
    pitch = pitch_seq[i % len(pitch_seq)]
    midi.addNote(track, 0, pitch, time + i, 1, 100)

# 7. 輸出MIDI檔案
with open("tsmc_prediction.mid", "wb") as f:
    midi.writeFile(f)

print("MIDI檔案 'tsmc_prediction.mid' 已產生，可用MIDI播放器播放。")
