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

# 6. 調性與和弦設定（C大調為主）
c_major_scale = [60, 62, 64, 65, 67, 69, 71]  # C D E F G A B
# 對應漲時的常用大三和弦（包含在 C 大調裡）
major_chords = {
    'C': [60, 64, 67],  # C E G
    'F': [65, 69, 72],  # F A C (C一個八度上)
    'G': [67, 71, 74],  # G B D
}
# 漲時旋律音對應用哪個大三和弦
note_to_chord_for_rise = {
    60: 'C',  # C -> C major
    62: 'G',  # D -> G major
    64: 'C',  # E -> C major
    65: 'F',  # F -> F major
    67: 'G',  # G -> G major
    69: 'F',  # A -> F major
    71: 'G',  # B -> G major
}
# 跌時固定的小三和弦（A 小調的 root）
a_minor_chord = [57, 60, 64]  # A C E

# 7. 產生 MIDI（主旋律 + 和弦）
midi = MIDIFile(2)
tempo = 100
midi.addTempo(0, 0, tempo)
midi.addTempo(1, 0, tempo)

# 設定樂器：主旋律用鋼琴（program 0），和弦用弦樂背景（String Ensemble program 48）
midi.addProgramChange(0, 0, 0, 0)
midi.addProgramChange(1, 1, 0, 48)

# 初始化旋律位置
melody_idx = 0  # 在 C 大調裡的 index（從 C 開始）
prev_pred = None

for i, pred in enumerate(y_pred):
    time = i  # 每個 timestep 一拍

    # 決定是否要上下移動（漲=上，跌=下），方向連續才移動
    if i == 0 or prev_pred is None or pred != prev_pred:
        # 轉向或第一拍回到根音（C for 漲, C 也當作跌的起點再往下）
        melody_idx = 0
    else:
        if pred == 1 and prev_pred == 1:
            # 連續漲：往上走一階（限界在 scale 內）
            melody_idx = min(len(c_major_scale) - 1, melody_idx + 1)
        elif pred == 0 and prev_pred == 0:
            # 連續跌：往下走一階
            melody_idx = max(0, melody_idx - 1)

    pitch = c_major_scale[melody_idx]  # 主旋律音（都在 C 大調內）

    # 加主旋律
    midi.addNote(0, 0, pitch, time, 1, 100)

    # 決定和弦：漲用包含該音的大三和弦，跌用 Am 和弦
    if pred == 1:
        chord_root = note_to_chord_for_rise.get(pitch, 'C')
        chord = major_chords[chord_root]
    else:
        chord = a_minor_chord

    # 加和弦（伴奏）
    for chord_note in chord:
        midi.addNote(1, 1, chord_note, time, 1, 60)

    prev_pred = pred

# 8. 輸出 MIDI
output_path = "tsmc_prediction_chorded.mid"
with open(output_path, "wb") as f:
    midi.writeFile(f)

print(f"✅ 生成完成：{output_path}")
