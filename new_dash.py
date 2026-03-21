import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import cv2
import random
import os
from ultralytics import YOLO
import plotly.express as px  

st.set_page_config(
    page_title="IoT Akıllı Kalite Kontrol Merkezi",
    page_icon="🏭",
    layout="wide"
)

st.title("Akıllı Kalite Kontrol Sistemi (YOLOv8)")
st.markdown("---")

# --- PATH CONFIGURATION ---
SENSOR_MODEL_PATH = 'sensor_anomaly_model.pkl' 
# Point this to your trained YOLO model (use the .pt file or NCNN folder)
VISION_MODEL_PATH = 'final_model_colab.pt' 
TEST_IMAGE_FOLDER = "test_image"

@st.cache_resource
def load_all_models():
    try:
        s_model = joblib.load(SENSOR_MODEL_PATH)
        # Load the YOLOv8 model for object detection
        v_model = YOLO(VISION_MODEL_PATH)
        return s_model, v_model
    except Exception as e:
        st.error(f"Modeller yüklenirken teknik hata oluştu: {e}")
        return None, None

sensor_model, vision_model = load_all_models()

if sensor_model is None or vision_model is None:
    st.warning("⚠️ Sistem başlatılamadı. Model dosyalarını kontrol edin.")
    st.stop()
else:
    st.success("✅ YOLOv8 ve Sensör Modelleri Entegre Edildi.")

def get_sensor_data():
    """Modelin 'HATA' olarak algılayacağı daha keskin değerler üretir"""
    
    ambient_c = random.uniform(24, 26)
    
    # Anomali ihtimalini test için %20'ye çıkaralım
    if random.random() < 0.20: 
        # KRİTİK DURUM: Yüksek Sıcaklık + Çok Yüksek Tork
        object_c = random.uniform(95, 105) # Ciddi ısınma
        x_g = random.uniform(7.0, 9.0)    # Şiddetli sarsıntı
        y_g = random.uniform(7.0, 9.0)
        z_g = random.uniform(10.0, 12.0)
        status = "CRITICAL"
        
        # Modelin "Machine Failure" demesi için Torku 70+ üzerine çıkaralım
        simulated_torque = random.uniform(75.0, 90.0) 
        simulated_rpm = random.uniform(1100, 1200) # Düşük devir, yüksek yük
    else:
        # NORMAL DURUM
        object_c = random.uniform(35, 45)
        x_g = random.uniform(0.1, 0.5)
        y_g = random.uniform(0.1, 0.5)
        z_g = random.uniform(0.9, 1.1)
        status = "NORMAL"
        
        simulated_torque = random.uniform(35.0, 45.0) # İdeal tork aralığı
        simulated_rpm = 1500.0

    # Model girişi (Kelvin dönüşümü)
    data = pd.DataFrame([[
        ambient_c + 273.15, 
        object_c + 273.15, 
        simulated_rpm, 
        simulated_torque, 
        180.0 # Tool wear: Yüksek aşınma da hatayı tetikler
    ]], columns=[
        'Air temperature [K]', 'Process temperature [K]', 
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ])
    
    return data, status, (ambient_c, object_c, x_g, y_g, z_g)

def analyze_image_with_yolo():
    """YOLOv8 ile hata tespiti ve lokalizasyonu yapar"""
    if not os.path.exists(TEST_IMAGE_FOLDER):
        return None, "Klasör Yok", 0
        
    files = [f for f in os.listdir(TEST_IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not files: return None, "Resim Yok", 0
    
    img_path = os.path.join(TEST_IMAGE_FOLDER, random.choice(files))
    
    # YOLO Inference
    results = vision_model(img_path, conf=0.4, verbose=False)
    result_obj = results[0]
    
    # Localization Logic: Check if any bounding boxes (defects) were found
    if len(result_obj.boxes) > 0:
        label = "Defect"
        confidence = float(result_obj.boxes[0].conf)
    else:
        label = "Normal"
        confidence = 0.95 

    # Plot results on the image (Draws the bounding boxes)
    annotated_frame = result_obj.plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    return annotated_frame, label, confidence

# --- DASHBOARD LAYOUT ---
col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    st.subheader("⚙️ Kontrol Paneli")
    run_sim = st.toggle('Simülasyonu Başlat', value=False)
    st.markdown("---")
    kpi_amb = st.empty()
    kpi_obj = st.empty()
    kpi_gforce = st.empty() # X-Y-Z'yi birleşik veya ayrı verebilirsin
    kpi_status = st.empty()

with col2:
    st.subheader("📈 Canlı Sensör Verileri")
    chart_placeholder = st.empty()

with col3:
    st.subheader("👁️ YOLOv8 Hata Lokalizasyonu")
    camera_placeholder = st.empty()
    alert_box = st.empty()

if run_sim:
    # 1. Geçmiş verileri tutmak için genişletilmiş DataFrame
    history_df = pd.DataFrame(columns=['Time', 'Ambient', 'Object', 'X-G', 'Y-G', 'Z-G'])
    
    for i in range(100):
        # --- DÜZELTME BURADA: 3 değer bekliyoruz ---
        data, _, physical_vals = get_sensor_data()
        
        # Fiziksel değerleri (Ambient, Object, X-Y-Z) ayrı değişkenlere açalım
        amb, obj, xg, yg, zg = physical_vals
        
        # 3. Model Tahmini (Artık hata vermeyecek çünkü 'data' içinde modelin beklediği isimler var)
        sensor_pred = sensor_model.predict(data)[0]
        
        # 4. Sol Paneldeki KPI Metriklerini Güncelle
        # Not: kpi_temp ve kpi_vib değişkenlerinin yukarıda tanımlandığından emin ol
        kpi_amb.metric(label="Ortam Sıcaklığı", value=f"{amb:.1f} °C")
        kpi_obj.metric(label="Nesne Sıcaklığı", value=f"{obj:.1f} °C", delta=f"{obj-40:.1f} °C")
        kpi_status.empty() # Durum kutusunu temizle
        
        # 5. Grafik Verisini Hazırla ve Plotly ile Çiz
        new_row = pd.DataFrame({
            'Time': [i], 
            'Ambient': [amb], 
            'Object': [obj], 
            'X-G': [xg], 
            'Y-G': [yg], 
            'Z-G': [zg]
        })
        history_df = pd.concat([history_df, new_row], ignore_index=True)
        
        # Son 30 veriyi gösteren interaktif grafik
        fig = px.line(
            history_df.tail(30), 
            x='Time', 
            y=['Ambient', 'Object', 'X-G', 'Y-G', 'Z-G'],
            labels={'value': 'Değer', 'variable': 'Sensör Tipi'},
            title="Gerçek Zamanlı Telemetri Analizi"
        )
        fig.update_layout(template="plotly_dark", legend_orientation="h")
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        # 6. Karar Mekanizması ve YOLO Entegrasyonu
        if sensor_pred == 1: 
            kpi_status.error("⚠️ ANOMALİ TESPİT EDİLDİ")
            proc_img, label, conf = analyze_image_with_yolo()
            
            if proc_img is not None:
                camera_placeholder.image(proc_img, caption="YOLOv8 Hata Analizi", use_container_width=True)
                if label == 'Defect':
                    alert_box.warning(f"🚨 KRİTİK HATA: {label} (Güven: %{conf*100:.1f})")
                else:
                    alert_box.success(f"✅ Görsel Kontrol Temiz (Güven: %{conf*100:.1f})")
        else:
            kpi_status.success("✅ SİSTEM NORMAL")
            camera_placeholder.info("Kamera Beklemede... (Anomali bekleniyor)")
            alert_box.empty()
            
        time.sleep(0.8)