from ultralytics import YOLO
from pathlib import Path
import os
import time
import cv2

# ==================================================
# AYARLAR
# ==================================================
MODEL_PATH = "best.pt"
IMAGE_FOLDER = "stop_sign_dataset"
OUTPUT_FOLDER = "predict_results"

CONF = 0.25
IMG_SIZE = 640

# ==================================================
# MODELİ YÜKLE
# ==================================================
model = YOLO(MODEL_PATH)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif"]

total_images = 0
total_detections = 0

confidences = []
times = []

# ==================================================
# TÜM GÖRÜNTÜLERİ İŞLE
# ==================================================
for image_path in Path(IMAGE_FOLDER).iterdir():

    if image_path.suffix.lower() not in extensions:
        continue

    total_images += 1

    start = time.perf_counter()

    results = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=CONF,
        verbose=False
    )

    elapsed = time.perf_counter() - start
    times.append(elapsed)

    result = results[0]

    print("\n" + "="*70)
    print(f"Görüntü : {image_path.name}")

    annotated = result.plot()

    cv2.imwrite(
        os.path.join(OUTPUT_FOLDER, image_path.name),
        annotated
    )

    txt_file = os.path.join(
        OUTPUT_FOLDER,
        image_path.stem + ".txt"
    )

    with open(txt_file, "w") as f:

        if len(result.boxes) == 0:
            print("Tespit bulunamadı.")
            f.write("No detection\n")

        for i, box in enumerate(result.boxes):

            total_detections += 1

            cls = int(box.cls[0])
            name = model.names[cls]

            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0]

            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)

            confidences.append(conf)

            print(f"\nDetection {i+1}")
            print("---------------------------")
            print(f"Class      : {name}")
            print(f"Confidence : {conf*100:.2f}%")
            print(f"Center     : ({cx:.2f}, {cy:.2f})")

            f.write(
                f"{name} "
                f"{conf:.4f} "
                f"{cx:.2f} "
                f"{cy:.2f}\n"
            )

# ==================================================
# RAPOR
# ==================================================

print("\n")
print("="*70)
print("MODEL PREDICTION REPORT")
print("="*70)

print(f"Toplam Görüntü        : {total_images}")
print(f"Toplam Detection      : {total_detections}")

if len(confidences):

    print(f"Ortalama Confidence   : {sum(confidences)/len(confidences)*100:.2f}%")
    print(f"En Yüksek Confidence  : {max(confidences)*100:.2f}%")
    print(f"En Düşük Confidence   : {min(confidences)*100:.2f}%")

avg_time = sum(times)/len(times)

print(f"Ortalama Süre         : {avg_time*1000:.2f} ms")
print(f"FPS                   : {1/avg_time:.2f}")

print("="*70)