readme_content = """# YOLOv8 ile Dur Tabelası (Stop Sign) Tespit Projesi

Bu proje, **YOLOv8** derin öğrenme mimarisini ve **Aktarımlı Öğrenme (Transfer Learning)** tekniklerini kullanarak görseller ve video akışları üzerinden dur tabelalarını (Stop Sign) yüksek doğrulukla ve gerçek zamanlı olarak tespit etmek amacıyla geliştirilmiştir.

---

## 📌 Proje Geliştirme Süreci (Aşama Aşama)

### 1. Veri Kümesinin Hazırlanması ve Yapılandırılması
* **Veri Toplama ve Etiketleme:** Dur tabelası görselleri toplanmış ve bounding box (sınırlayıcı kutu) koordinatları YOLO etiket formatına (`.txt`) uygun şekilde yapılandırılmıştır.
* **Klasör Mimarisi:** Veri kümesi; `train` (eğitim), `val` (doğrulama) ve `test` (test) olmak üzere 3 ana alt klasöre ayrılmıştır.
* **`data.yaml` Tanımlaması:** YOLO'nun veri kümesini tanıyabilmesi için dosya yollarını ve sınıf tanımlamalarını içeren yapılandırma dosyası hazırlanmıştır:
  ```yaml
  path: C:/Users/pc/OneDrive/Desktop/sontest  # Veri seti kök dizini
  train: train/images
  val: valid/images
  test: test/images

  nc: 1
  names: ['stop_sign']
  2. Geliştirme Ortamının ve Donanımın Hazırlanması
Kütüphanelerin Kurulumu: Projede gerekli olan ultralytics, torch, torchvision ve opencv-python kütüphaneleri yüklenmiştir.

GPU ve CUDA İvmelenmesi: Eğitim ve çıkarım süreçlerinde işlem süresini optimize etmek ve NVIDIA GPU (RTX 3050 Ti) gücünden yararlanmak adına PyTorch CUDA sürücüleri yapılandırılmıştır.

3. Model Eğitimi (Training)
Pre-trained Ağırlık Seçimi: yolov8s.pt (Small) pre-trained modeli temel alınarak transfer learning yöntemiyle özel veri kümesi üzerinde eğitim başlatılmıştır.

Eğitim Parametreleri:

Çözünürlük (imgsz): 640x640 piksel

Batch Size: 8 / 16 (VRAM kapasitesine göre optimize edilmiş)

Epoch & Patience: 100 epoch tanımlanmış, aşırı öğrenmeyi (overfitting) önlemek adına erken durdurma (early stopping) değeri patience=15 olarak ayarlanmıştır.

Windows Multiprocessing Desteği: Kod blokları if __name__ == '__main__': yapısı ile sarmalanarak alt süreç çakışmaları engellenmiştir.

4. Model Doğrulama ve Performans Testi (Validation & Testing)
Eğitim süresince kaydedilen en iyi ağırlık dosyası (best.pt) ayrılan bağımsız test veri kümesi üzerinde değerlendirilmiştir.

Modellerin başarımı için temel metriklere odaklanılmıştır:

Precision (Kesinlik): Yanlış pozitif tespitlerin azlığı.

Recall (Duyarlılık): Gerçek dur tabelalarının ne kadarının yakalandığı.

mAP@50 ve mAP@50-95: Farklı IoU eşiklerinde modelin genel başarım oranı.

Python
from ultralytics import YOLO

def main():
    model = YOLO("best.pt")
    metrics = model.val(
        data="sontest/data.yaml",
        split="test",
        imgsz=640,
        batch=8,
        conf=0.25,
        iou=0.5,
        device=0,
        workers=2,
        save=True,
        plots=True
    )

if __name__ == '__main__':
    main()
5. Çıkarım ve Tahmin (Inference)
Eğitilen best.pt ağırlık dosyası kullanılarak yeni görseller ve canlı/kayıtlı video akışları üzerinde dur tabelası tespiti gerçekleştirilmiştir.

Nesne üzerindeki tespit kutuları ve güven skorları (confidence scores) görselleştirilip kaydedilmiştir.

🛠️ Kurulum ve Kullanım
Gereksinimler
Bash
pip install ultralytics torch torchvision opencv-python
Modeli Test Etme (Tahmin / Inference)
Python
from ultralytics import YOLO

def main():
    # Eğitilen en iyi modeli yükle
    model = YOLO('best.pt')

    # Test görseli üzerinde tespit yap
    results = model.predict(
        source='test_image.jpg',
        conf=0.5,
        save=True,
        device=0
    )

if __name__ == '__main__':
    main()
📊 Örnek Sonuçlar ve Metrikler
Eğitim ve doğrulama aşamalarının ardından elde edilen metrik çıktıları:

Precision: ~0.92

Recall: ~0.89

mAP@50: ~0.94

mAP@50-95: ~0.71
"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(readme_content)

print("README.md başarıyla oluşturuldu.")


```text?code_stdout&code_event_index=1
README.md başarıyla oluşturuldu.

Hazırladığım proje adımlarını içeren Markdown dosyası oluşturulmuştur.


README
 MD 
Açık
Ayrıca doğrudan kopyalayıp projenizin ana dizinine README.md adıyla kaydetmek isterseniz kod içeriği şu şekildedir:

Markdown
# YOLOv8 ile Dur Tabelası (Stop Sign) Tespit Projesi

Bu proje, **YOLOv8** derin öğrenme mimarisini ve **Aktarımlı Öğrenme (Transfer Learning)** tekniklerini kullanarak görseller ve video akışları üzerinden dur tabelalarını (Stop Sign) yüksek doğrulukla ve gerçek zamanlı olarak tespit etmek amacıyla geliştirilmiştir.

---

## 📌 Proje Geliştirme Süreci (Aşama Aşama)

### 1. Veri Kümesinin Hazırlanması ve Yapılandırılması
* **Veri Toplama ve Etiketleme:** Dur tabelası görselleri toplanmış ve bounding box (sınırlayıcı kutu) koordinatları YOLO etiket formatına (`.txt`) uygun şekilde yapılandırılmıştır.
* **Klasör Mimarisi:** Veri kümesi; `train` (eğitim), `val` (doğrulama) ve `test` (test) olmak üzere 3 ana alt klasöre ayrılmıştır.
* **`data.yaml` Tanımlaması:** YOLO'nun veri kümesini tanıyabilmesi için dosya yollarını ve sınıf tanımlamalarını içeren yapılandırma dosyası hazırlanmıştır:
  ```yaml
  path: C:/Users/pc/OneDrive/Desktop/sontest  # Veri seti kök dizini
  train: train/images
  val: valid/images
  test: test/images

  nc: 1
  names: ['stop_sign']
2. Geliştirme Ortamının ve Donanımın Hazırlanması
Kütüphanelerin Kurulumu: Projede gerekli olan ultralytics, torch, torchvision ve opencv-python kütüphaneleri yüklenmiştir.

GPU ve CUDA İvmelenmesi: Eğitim ve çıkarım süreçlerinde işlem süresini optimize etmek ve NVIDIA GPU (RTX 3050 Ti) gücünden yararlanmak adına PyTorch CUDA sürücüleri yapılandırılmıştır.

3. Model Eğitimi (Training)
Pre-trained Ağırlık Seçimi: yolov8s.pt (Small) pre-trained modeli temel alınarak transfer learning yöntemiyle özel veri kümesi üzerinde eğitim başlatılmıştır.

Eğitim Parametreleri:

Çözünürlük (imgsz): 640x640 piksel

Batch Size: 8 / 16 (VRAM kapasitesine göre optimize edilmiş)

Epoch & Patience: 100 epoch tanımlanmış, aşırı öğrenmeyi (overfitting) önlemek adına erken durdurma (early stopping) değeri patience=15 olarak ayarlanmıştır.

Windows Multiprocessing Desteği: Kod blokları if __name__ == '__main__': yapısı ile sarmalanarak alt süreç çakışmaları engellenmiştir.

4. Model Doğrulama ve Performans Testi (Validation & Testing)
Eğitim süresince kaydedilen en iyi ağırlık dosyası (best.pt) ayrılan bağımsız test veri kümesi üzerinde değerlendirilmiştir.

Modellerin başarımı için temel metriklere odaklanılmıştır:

Precision (Kesinlik): Yanlış pozitif tespitlerin azlığı.

Recall (Duyarlılık): Gerçek dur tabelalarının ne kadarının yakalandığı.

mAP@50 ve mAP@50-95: Farklı IoU eşiklerinde modelin genel başarım oranı.

Python
from ultralytics import YOLO

def main():
    model = YOLO("best.pt")
    metrics = model.val(
        data="sontest/data.yaml",
        split="test",
        imgsz=640,
        batch=8,
        conf=0.25,
        iou=0.5,
        device=0,
        workers=2,
        save=True,
        plots=True
    )

if __name__ == '__main__':
    main()
5. Çıkarım ve Tahmin (Inference)
Eğitilen best.pt ağırlık dosyası kullanılarak yeni görseller ve canlı/kayıtlı video akışları üzerinde dur tabelası tespiti gerçekleştirilmiştir.

Nesne üzerindeki tespit kutuları ve güven skorları (confidence scores) görselleştirilip kaydedilmiştir.

🛠️ Kurulum ve Kullanım
Gereksinimler
Bash
pip install ultralytics torch torchvision opencv-python
Modeli Test Etme (Tahmin / Inference)
Python
from ultralytics import YOLO

def main():
    # Eğitilen en iyi modeli yükle
    model = YOLO('best.pt')

    # Test görseli üzerinde tespit yap
    results = model.predict(
        source='test_image.jpg',
        conf=0.5,
        save=True,
        device=0
    )

if __name__ == '__main__':
    main()
📊 Örnek Sonuçlar ve Metrikler
Eğitim ve doğrulama aşamalarının ardından elde edilen metrik çıktıları:

Precision: ~0.98

Recall: ~1.00

mAP@50: ~0.96

mAP@50-95: ~0.91

Ortalama Test tespit yüzdesi ~ 0.80
