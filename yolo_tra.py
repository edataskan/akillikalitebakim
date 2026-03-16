from ultralytics import YOLO
import pandas as pd

def train_validate_and_report():
    # 1. Modeli Yükle
    model = YOLO('yolov8n.pt') 

    # 2. Eğitimi Başlat
    # NOT: 'plots=True' eğitim sonunda sonuç grafiklerini runs klasörüne kaydeder.
    results = model.train(
        data='E:/akıllı_kalitebakim/FDM.v2i.yolov11/data.yaml', 
        epochs=50, 
        imgsz=640, 
        name='havelsan_model'
    )

    # 3. Validation (Doğrulama) Yap - Metrikleri Hesapla
    print("\n--- Model Performans Değerlendiriliyor ---")
    metrics = model.val() 

    # 4. Değerleri Ekrana Yazdır
    print("\n" + "="*30)
    print(f"MODEL PERFORMANS ÖZETİ:")
    print(f"mAP50 (Genel Doğruluk): {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision (Keskinlik): {metrics.box.mp:.4f}")
    print(f"Recall (Duyarlılık): {metrics.box.mr:.4f}")
    print("="*30)

    # 5. Sınıf Bazlı Doğruluk Değerlerini Yazdır
    print("\nSINIF BAZLI mAP50 DEĞERLERİ:")
    names = model.names
    for i, map50 in enumerate(metrics.box.maps50):
        print(f" - {names[i]}: {map50:.4f}")

    # 6. Modeli Kaydet
    model.save('final_model.pt')
    print("\nModel 'final_model.pt' adıyla kaydedildi.")

if __name__ == "__main__":
    train_validate_and_report()