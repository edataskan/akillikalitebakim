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
    """Simüle edilmiş sensör verisi üretir"""
    air_temp = random.uniform(295, 305)
    if random.random() < 0.15:
        proc_temp = random.uniform(315, 330) 
        torque = random.uniform(60, 90)      
        status = "CRITICAL"
    else:
        proc_temp = random.uniform(300, 310)
        torque = random.uniform(30, 45)
        status = "NORMAL"
    data = pd.DataFrame([[air_temp, proc_temp, 1400.0, torque, 50.0]], 
                        columns=['Air temperature [K]', 'Process temperature [K]', 
                                 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
    return data, status

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
    kpi_temp = st.empty()
    kpi_vib = st.empty()
    kpi_status = st.empty()

with col2:
    st.subheader("📈 Canlı Sensör Verileri")
    chart_placeholder = st.empty()

with col3:
    st.subheader("👁️ YOLOv8 Hata Lokalizasyonu")
    camera_placeholder = st.empty()
    alert_box = st.empty()

if run_sim:
    history_df = pd.DataFrame(columns=['Time', 'Temperature', 'Torque'])
    for i in range(100):
        data, _ = get_sensor_data()
        current_temp = data['Process temperature [K]'].values[0]
        current_torque = data['Torque [Nm]'].values[0]
        
        sensor_pred = sensor_model.predict(data)[0]
        
        kpi_temp.metric(label="Sıcaklık (K)", value=f"{current_temp:.1f} K")
        kpi_vib.metric(label="Titreşim (Nm)", value=f"{current_torque:.1f} Nm")
        
        # Plotting
        new_row = pd.DataFrame({'Time': [i], 'Temperature': [current_temp], 'Torque': [current_torque]})
        history_df = pd.concat([history_df, new_row], ignore_index=True)
        fig = px.line(history_df.tail(20), x='Time', y=['Temperature', 'Torque'])
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        if sensor_pred == 1: 
            kpi_status.error("⚠️ ANOMALİ TESPİT EDİLDİ")
            proc_img, label, conf = analyze_image_with_yolo()
            
            if proc_img is not None:
                camera_placeholder.image(proc_img, caption="AI Hata Lokalizasyonu", use_container_width=True)
                if label == 'Defect':
                    alert_box.warning(f"🚨 HATA TESPİTİ: {label} (%{conf*100:.1f})")
                else:
                    alert_box.success(f"✅ Görüntü Temiz (%{conf*100:.1f})")
        else:
            kpi_status.success("✅ SİSTEM NORMAL")
            camera_placeholder.info("Kamera Beklemede...")
            alert_box.empty()
            
        time.sleep(1)