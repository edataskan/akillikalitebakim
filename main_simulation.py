import time
import random
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image


SENSOR_MODEL_PATH = 'sensor_anomaly_model.pkl'
VISION_MODEL_PATH = 'vision_defect_model.h5'

TEST_IMAGE_FOLDER = 'dataset_klasorunun_yolu/defect' 

print("Sistem başlatılıyor...")
print("1. Sensör Modeli Yükleniyor...")
sensor_model = joblib.load(SENSOR_MODEL_PATH)

print("2. Görsel İşleme Modeli Yükleniyor...")
vision_model = tf.keras.models.load_model(VISION_MODEL_PATH)
print("Sistem Hazır! Simülasyon Başlıyor...\n")


LABELS = ['Defect', 'Normal'] 

def get_simulated_sensor_data():
    
    # Normal değerler üretelim
    air_temp = random.uniform(295, 305)
    proc_temp = random.uniform(305, 315)
    speed = random.uniform(1300, 1600)
    torque = random.uniform(30, 50) # Düşük tork = Normal
    wear = random.uniform(0, 100)
    
    # %20 ihtimalle ANOMALİ (Hata) verisi üretelim
    if random.random() < 0.2:
        torque = random.uniform(60, 90) # Yüksek tork = Titreşim/Hata
        proc_temp = random.uniform(315, 325) # Yüksek Isı
        
    # Modelin beklediği formatta DataFrame oluştur
    data = pd.DataFrame([[air_temp, proc_temp, speed, torque, wear]], 
                        columns=['Air temperature [K]', 'Process temperature [K]', 
                                 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
    return data

def analyze_image():
    
    try:
        # Klasörden rastgele bir resim seç
        files = os.listdir(TEST_IMAGE_FOLDER)
        if not files:
            print("HATA: Test klasöründe resim yok!")
            return "Bilinmiyor"
            
        random_file = random.choice(files)
        img_path = os.path.join(TEST_IMAGE_FOLDER, random_file)
        
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0 

        prediction = vision_model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        
        result_label = LABELS[predicted_class_index]
        return f"{result_label} (Güven: %{confidence*100:.2f}) - Dosya: {random_file}"
        
    except Exception as e:
        return f"Görüntü İşleme Hatası: {e}"

try:
    while True:
        
        sensor_data = get_simulated_sensor_data()
        
        anomaly_check = sensor_model.predict(sensor_data)[0]
        
        print("-" * 30)
        print(f"Sensör Verisi: Tork={sensor_data['Torque [Nm]'].values[0]:.1f} Nm, Isı={sensor_data['Process temperature [K]'].values[0]:.1f} K")
        
        if anomaly_check == 1:
            print("ANOMALİ TESPİT EDİLDİ! (Tetikleyici Devrede)")
            print("Fotoğraf çekiliyor ve analiz ediliyor...")
            
    
            vision_result = analyze_image()
            print(f"SONUÇ: {vision_result}")
            
            
        else:
            print("Sistem Normal.")
            
        time.sleep(2) 

except KeyboardInterrupt:
    print("\nSimülasyon durduruldu.")