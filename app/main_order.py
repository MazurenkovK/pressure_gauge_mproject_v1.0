# main_order.py
import os
import cv2
import pandas as pd
import numpy as np
import time

from core.pipeline import PressurePipeline

IMG_DIR = "app/data_model/test_dataset/test_images"
CSV_PATH = os.path.join(IMG_DIR, "test_labels_with_classes.csv")

EXTS = (".jpg", ".jpeg", ".png")

pipeline = PressurePipeline(
    gauge_model_path="app/models/best_gauge.pt",
    seg_model_path="app/models/best_seg.pt",
    pressure_max=4
)

# df = pd.read_csv(CSV_PATH, sep=';')
# df["pressure"] = df["pressure"].str.replace(",", ".").astype(float)
# gt_map = dict(zip(df["image"], df["pressure"]))

df = pd.read_csv(CSV_PATH)#, sep=';')
df["pressure"] = df["pressure"].astype(float)
gt_map = dict(zip(df["filename"], df["pressure"]))

files = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith(EXTS)
])

print(f"Found {len(files)} images\n")

y_true = []
y_pred = []
latencies = []

# CER порог
CER_THRESHOLD = 0.15  # 15%
failed_files = []  # Сюда будем собирать имена файлов, где пайплайн упал

for fname in files:

    if fname not in gt_map:
        print(f"[SKIP] {fname} — no GT in CSV")
        continue

    path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(path)

    if img is None:
        print(f"[SKIP] {fname} — cannot read image")
        continue

    # try:
    #     start = time.time()
    #     pred = pipeline.process(img, file_name=fname, visualize=False)
    #     end = time.time()

    #     latency_ms = (end - start) * 1000
    #     latencies.append(latency_ms)

    #     true = float(gt_map[fname])

    #     y_true.append(true)
    #     y_pred.append(pred)

    #     print(f"[OK] {fname} → PRED={pred:.2f} | TRUE={true:.2f} | latency={latency_ms:.1f} ms")

    # except Exception as e:
    #     print(f"[ERR] {fname} → {e}")

    try:
        start = time.time()
        pred = pipeline.process(img, file_name=fname, visualize=False)
        end = time.time()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        # --- ИСПРАВЛЕНИЕ НАЧИНАЕТСЯ ЗДЕСЬ ---
        if pred is not None:
            # Если модель вернула валидное предсказание, добавляем его в списки
            true = float(gt_map[fname])
            y_true.append(true)
            y_pred.append(pred)

            print(f"[OK] {fname} → PRED={pred:.2f} | TRUE={true:.2f} | latency={latency_ms:.1f} ms")
        else:
            # Если pred is None, просто выводим сообщение и пропускаем добавление в списки
            print(f"[SKIP] {fname} → Prediction is None (e.g., gauge not found)")
        # --- ИСПРАВЛЕНИЕ ЗАКАНЧИВАЕТСЯ ЗДЕСЬ ---

    except Exception as e:
        print(f"[ERR] {fname} → {e}")
        failed_files.append(fname) 
    


if len(y_true) == 0:
    print("\nNo valid predictions")
    exit()

y_true = np.array(y_true)
y_pred = np.array(y_pred)
latencies = np.array(latencies)

# MAE
mae = np.mean(np.abs(y_true - y_pred))

# MAPE
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# CER
cer = np.mean(np.abs(y_true - y_pred) > CER_THRESHOLD) * 100

# latency
latency_avg = np.mean(latencies)

print("\n================ METRICS ================")
print(f"{'Метрика':<10} | {'Ед. изм.':<6} | {'Значение'}")
print(f"{'-'*35}")
print(f"{'МАЕ':<10} | {'МПа':<6} | {mae:.2f}")
print(f"{'MAPE':<10} | {'%':<6} | {mape:.1f}")
print(f"{'CER':<10} | {'%':<6} | {cer:.1f}")
print(f"{'latency':<10} | {'мс':<6} | {latency_avg:.1f}")
print("=========================================")
# Выводим статистику по ошибкам
print("\n=== ОТЧЕТ ПО ОШИБКАМ ===")
print(f"Всего обработано: {len(files)}")
print(f"Успешно: {len(y_true)}")
print(f"Провал: {len(failed_files)} ({len(failed_files)/len(files):.1%})")

if failed_files:
    print("Файлы с ошибками:")
    for path in failed_files:
        print(path)
else:
    print("Все файлы обработаны успешно!")