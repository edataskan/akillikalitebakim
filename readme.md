# Akıllı Kalite & Bakım (Simülasyon + Sensör Anomali + YOLO)

Bu repo, **sensör verisi ile anomali tespiti** (RandomForest) ve **görüntü analizi** (Keras model yükleme + YOLO eğitim scripti) etrafında bir örnek akış içerir.

- **Sensör tarafı**: `train_sensor_model.py` ile CSV üzerinden RandomForest eğitilir ve `sensor_anomaly_model.pkl` üretilir.
- **Simülasyon/çalıştırma**: `main_simulation.py` sensör verisini simüle eder, anomali varsa klasörden rastgele görsel seçip `VISION_MODEL_PATH` ile yüklenen Keras model ile tahmin dener.
- **YOLO eğitimi**: `yolo_tra.py` Ultralytics YOLO ile eğitim/validation çalıştırır.

> Not: Repoda bir `venv/` klasörü bulunuyor. Genelde bu klasör **kaynak kontrolüne eklenmez** ve farklı makinelerde yeniden oluşturulur. Kurulum adımlarında kendi sanal ortamınızı oluşturmanız önerilir.

## Gereksinimler

- **Windows 10/11**
- **Python 3.10+** (önerilen)
- (Opsiyonel) CUDA/GPU kurulumları: TensorFlow/Ultralytics için sisteminize göre değişir.

Bağımlılıklar `requirements.txt` içindedir.

## Kurulum

Proje klasöründe aşağıdaki adımları çalıştırın.

### 1) Sanal ortam oluşturma

PowerShell:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Paketleri yükleme

```bash
pip install -r requirements.txt
```

## Hızlı Başlangıç

### A) Sensör modelini eğitme (RandomForest)

Script:
- `train_sensor_model.py`

Beklenen veri:
- `sensor/ai4i2020.csv`

Çalıştırma:

```bash
python train_sensor_model.py
```

Çıktı:
- `sensor_anomaly_model.pkl`

### B) Simülasyonu çalıştırma

Script:
- `main_simulation.py`

Bu dosya iki modeli yükler:
- `sensor_anomaly_model.pkl` (repo kökünde mevcut/üretilmiş olmalı)
- `vision_defect_model.h5` (**repoda yoksa siz sağlamalısınız**)

Ayrıca test görselleri için bir klasör bekler:
- `TEST_IMAGE_FOLDER` değişkeni (varsayılan: `dataset_klasorunun_yolu/defect`)

Çalıştırmadan önce `main_simulation.py` içinde şunları düzenleyin:
- `VISION_MODEL_PATH`: Keras `.h5` modelinizin yolu
- `TEST_IMAGE_FOLDER`: test görselleri klasör yolu

Çalıştırma:

```bash
python main_simulation.py
```

Akış:
- Sensör verisi simüle edilir.
- Model “anomali” dönerse, klasörden rastgele görsel seçilir ve `vision_model.predict(...)` ile sınıflandırma yapılır.

### C) YOLO eğitimi/raporlama (Ultralytics)

Script:
- `yolo_tra.py`

Önemli: Script içinde veri yaml yolu hardcoded:

- `data='E:/akıllı_kalitebakim/FDM.v2i.yolov11/data.yaml'`

Kendi makinenize göre bu yolu güncelleyin ve sonra çalıştırın:

```bash
python yolo_tra.py
```

Çıktılar:
- Ultralytics çıktıları genelde `runs/` altında oluşur.
- Script ayrıca `final_model.pt` kaydetmeye çalışır.

## Proje İçeriği (kısa)

- `main_simulation.py`: sensör simülasyonu + anomali tetikleyince görsel analiz
- `train_sensor_model.py`: RandomForest eğitim ve `sensor_anomaly_model.pkl` üretimi
- `yolo_tra.py`: Ultralytics YOLO eğitim/validation + metrik raporu
- `sensor_anomaly_model.pkl`: eğitilmiş sensör anomali modeli (hazır dosya)
- `yolov8n.pt`, `final_model_colab.pt`: YOLO ağırlıkları (hazır dosyalar)
- `FDM.v2i.yolov11.zip`, `sensor.zip`: veri/örnek içerikler (zip)

## Sık Karşılaşılan Problemler

- **`FileNotFoundError: vision_defect_model.h5`**
  - `VISION_MODEL_PATH` doğru değil veya `.h5` model dosyası projede yok.
  - Çözüm: Kendi eğitilmiş Keras modelinizi ekleyin veya yolu düzeltin.

- **`TEST_IMAGE_FOLDER` boş / yanlış**
  - Klasör yolunu tam (absolute) verin ve içinde görsel olduğundan emin olun.

- **TensorFlow kurulumu/uyumsuzluğu (Windows)**
  - CPU/GPU desteği ve Python sürümü uyumu önemlidir.
  - Çözüm: Python 3.10/3.11 kullanın; gerekirse `tensorflow` sürümünü uyumlu bir aralığa çekin.

## (Opsiyonel) Dashboard

Mevcut repoda `dash.py` dosyası **boş**. Eğer bir Streamlit arayüzü hedefleniyorsa:

```bash
pip install streamlit
streamlit run dash.py
```

> Not: Eski README’de geçen `dashboard.py` dosyası bu repoda bulunmuyor.