import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


df = pd.read_csv('sensor/ai4i2020.csv')

features = ['Air temperature [K]', 'Process temperature [K]', 
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

target = 'Machine failure'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MODEL EĞİTİMİ (Random Forest)
# Raspberry Pi üzerinde hafif ve hızlı çalıştığı için Random Forest seçtik.

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu: {accuracy:.2f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))


joblib.dump(model, 'sensor_anomaly_model.pkl')
print("Model 'sensor_anomaly_model.pkl' olarak kaydedildi.")

# --- SİMÜLASYON ÖRNEĞİ ---
print("\n--- SİMÜLASYON: ARDUINO'DAN VERİ GELİYORMUŞ GİBİ ---")
# Örnek: Sıcaklık çok yüksek, Tork (Titreşim) ani artmış
ornek_veri = [[300, 315, 1400, 60, 200]]  # Manuel girilen tehlikeli değerler
sonuc = model.predict(ornek_veri)

if sonuc[0] == 1:
    print(f"DURUM: ANOMALİ TESPİT EDİLDİ! (Simülasyon Değeri: {ornek_veri})")
    print("AKSİYON: Kamera tetikleniyor... [FOTOĞRAF ÇEKİLİYOR]")
else:
    print("DURUM: Sistem Normal.")