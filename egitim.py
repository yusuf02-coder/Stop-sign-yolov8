import os
import torch
from ultralytics import YOLO

def main():
    # 1. Donanım Kontrolü (GPU Kullanımı)
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Eğitim Donanımı: {torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'}")

    # 2. Pre-trained Temel Modeli Yükle
    # Seçenekler: yolov8n.pt (Nano), yolov8s.pt (Small), yolov8m.pt (Medium), yolov8l.pt (Large)
    model = YOLO('yolov8s.pt') 

    # 3. Eğitimi Başlat
    results = model.train(
        data='data.yaml',      # Veri seti yapılandırma dosyası
        epochs=100,               # Toplam eğitim tur sayısı
        imgsz=640,                # Görsel girdi boyutu (piksel)
        batch=16,                 # Batch boyutu (GPU belleğinize göre 8, 16, 32 yapabilirsiniz)
        workers=8,                # Veri yükleme izlek sayısı (CPU çekirdek sayısı)
        device=device,            # CUDA cihaz ID'si veya 'cpu'
        patience=20,              # Early stopping: 20 epoch boyunca gelişme olmazsa eğitimi durdur
        save=True,                # En iyi ve son ağırlık dosyalarını kaydet (.pt)
        project='runs/detect',    # Sonuçların kaydedileceği ana klasör
        name='yolov8_custom_exp', # Deneme/Proje klasör adı
        exist_ok=True,            # Klasör varsa üzerine yaz
        
        # Hyperparameters & Augmentation (Veri Çoğaltma)
        optimizer='AdamW',        # 'SGD', 'Adam', 'AdamW' veya 'auto'
        lr0=0.01,                 # Başlangıç öğrenme oranı (Learning Rate)
        lrf=0.01,                 # Final öğrenme oranı çarpanı
        mosaic=1.0,               # Mosaic augmentation oranı
        mixup=0.1,                # Mixup augmentation oranı
        pretrained=True           # Pre-trained ağırlık transferini aktif et
    )

    print("Eğitim tamamlandı! Ağırlıklar kaydedildi.")

if __name__ == '__main__':
    # Windows sistemlerde multiprocessing çakışmasını önlemek için main bloğu şarttır
    main()