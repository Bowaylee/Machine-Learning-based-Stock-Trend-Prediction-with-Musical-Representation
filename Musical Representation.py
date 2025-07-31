# 先安裝必要套件（若還沒安裝）
# pip install yfinance pandas numpy scikit-learn MIDIUtil

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from midiutil import MIDIFile

# 1. 下載台積電歷史資料（近一年每日收盤價）
tsmc = yf.download("2330.TW", period="1y", interval="1d")
tsmc = tsmc[['Close']].dropna()

# 2. 製作特徵與標籤（前5天做特徵，預測第6天漲跌）
tsmc['Target'] = (tsmc['Close'].shift(-1) > tsmc['Close']).astype(int)
for i in range(1, 6):
    tsmc[f'lag_{i}'] = tsmc['Close'].shift(i)
tsmc = tsmc.dropna()

X = tsmc[[f'lag_{i}' for i in range(1, 6)]].values
y = tsmc['Target'].values

# 3. 分割訓練與測試（時間序列不打亂）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.2)

# 4. 訓練隨機森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. 預測
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 6. 設定音階與和弦
c_major_scale = [60, 62, 64, 65, 67, 69, 71]  # C D E F G A B
a_minor_scale = [57, 59, 60, 62, 64, 65, 67]  # A B C D E F G

# 和弦（大三和弦 / 小三和弦）
c_major_chord = [60, 64, 67]  # C E G
a_minor_chord = [57, 60, 64]  # A C E

# 7. 產生 MIDI：主旋律 + 和弦（兩軌）
midi = MIDIFile(2)
tempo = 100
midi.addTempo(0, 0, tempo)
midi.addTempo(1, 0, tempo)

# 主旋律用鋼琴（預設 program 0）
midi.addProgramChange(0, 0, 0, 0)
# 和弦用弦樂背景（使用 String Ensemble program 48）
midi.addProgramChange(1, 1, 0, 48)

# 8. 依漲跌生成旋律與和弦（用 step 上/下邏輯）
melody_idx = 0  # 主旋律起始在根音
prev_pred = None

for i, pred in enumerate(y_pred):
    time = i  # 每個 timestep 一個節拍

    # 決定 scale 和根音邏輯
    if pred == 1:  # 漲：C 大調
        scale = c_major_scale
        chord = c_major_chord
    else:  # 跌：A 小調
        scale = a_minor_scale
        chord = a_minor_chord

    # step logic：根據前一個預測調整位置
    if i == 0 or prev_pred is None or pred != prev_pred:
        melody_idx = 0  # 轉換時回到根音
    else:
        if pred == 1 and prev_pred == 1:
            melody_idx = min(len(scale) - 1, melody_idx + 1)  # 連續漲往上走
        elif pred == 0 and prev_pred == 0:
            melody_idx = max(0, melody_idx - 1)  # 連續跌往下走

    # 主旋律音符
    pitch = scale[melody_idx]
    midi.addNote(0, 0, pitch, time, 1, 100)

    # 加和弦（持續一拍）
    for chord_note in chord:
        midi.addNote(1, 1, chord_note, time, 1, 60)  # 和弦音量略低

    prev_pred = pred

# 9. 輸出 MIDI
output_path = "tsmc_prediction_with_chords.mid"
with open(output_path, "wb") as f:
    midi.writeFile(f)

print(f"✅ 生成完成：{output_path}")
