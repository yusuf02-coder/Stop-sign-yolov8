from ultralytics import YOLO

def main():
    # Modeli yükle
    model = YOLO("best.pt")

    # Validation / Testing
    metrics = model.val(
        data="sontest/data.yaml",
        split="test",          # test kümesini kullan
        imgsz=640,
        batch=8,
        conf=0.25,
        iou=0.5,
        device=0,              # GPU için 0 (veya '0'), CPU için "cpu"
        workers=0,             # Windows üzerinde işlemci kilitlenmesini ve RAM taşmasını önler
        save=True,
        plots=True,
        verbose=True
    )

    print("\n========== SONUÇLAR ==========")
    print(f"Precision      : {metrics.box.mp:.4f}")
    print(f"Recall         : {metrics.box.mr:.4f}")
    print(f"mAP@50         : {metrics.box.map50:.4f}")
    print(f"mAP@50-95      : {metrics.box.map:.4f}")
    print("==============================")

if __name__ == '__main__':
    main()