import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statsmodels.api as sm

# === KONFIGURASI FIREBASE ===
API_KEY = "AIzaSyDsM-j-nbNPMdTz1irbXOXD1b8bS_mjrPk"
PROJECT_ID = "sparm-b9de0"
USER_EMAIL = "esp32@test.com"
USER_PASSWORD = "12345678"
COLLECTION_SOURCE = "power_history"
COLLECTION_FORECAST = "monthly_forecast"
PRICE_PER_KWH = 1500  # Rp per kWh

# === LOGIN MENGGUNAKAN EMAIL & PASSWORD ===
def login_firebase(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    res = requests.post(url, json=payload)
    if res.status_code != 200:
        raise Exception(f"Login gagal: {res.text}")
    data = res.json()
    return data["idToken"]

print("🔐 Login ke Firebase...")
id_token = login_firebase(USER_EMAIL, USER_PASSWORD)
print("✅ Login berhasil!")

# === FUNGSI UNTUK BACA DATA FIRESTORE ===
def fetch_firestore_data():
    url = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/{COLLECTION_SOURCE}?pageSize=10000"
    headers = {"Authorization": f"Bearer {id_token}"}
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        raise Exception(f"Gagal ambil data: {res.text}")

    docs = res.json().get("documents", [])
    data = []
    for doc in docs:
        fields = doc["fields"]
        power = float(fields["power"]["doubleValue"])
        waktu = fields["simulated_time"]["stringValue"]
        data.append({"time": waktu, "power": power})

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df

print("📥 Mengambil data 1 tahun dari Firestore...")
df = fetch_firestore_data()
print(f"✅ Ditemukan {len(df)} data.")

# === AMBIL 1 TAHUN TERAKHIR ===
df = df.tail(8760)

# === LATIH MODEL SARIMA ===
print("🤖 Melatih model SARIMA...")
ts = df.set_index("time")["power"].asfreq("H")
ts = ts.fillna(method="ffill")

model = sm.tsa.statespace.SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,24))
result = model.fit(disp=False)
print("✅ Model dilatih!")

# === PREDIKSI 30 HARI KE DEPAN ===
future_steps = 24 * 30  # 720 jam
forecast = result.forecast(steps=future_steps)

total_kwh = forecast.sum()
total_cost = total_kwh * PRICE_PER_KWH
print(f"🔮 Prediksi 30 hari ke depan:")
print(f"   Total energi: {total_kwh:.2f} kWh")
print(f"   Estimasi biaya: Rp{total_cost:,.0f}")

# === SIMPAN KE FIRESTORE ===
url = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/{COLLECTION_FORECAST}"
headers = {"Authorization": f"Bearer {id_token}", "Content-Type": "application/json"}

payload = {
    "fields": {
        "forecast_start": {"timestampValue": datetime.utcnow().isoformat() + "Z"},
        "forecast_end": {"timestampValue": (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z"},
        "predicted_total_kwh": {"doubleValue": float(total_kwh)},
        "predicted_total_cost": {"doubleValue": float(total_cost)},
        "price_per_kwh": {"doubleValue": PRICE_PER_KWH},
        "created_at": {"timestampValue": datetime.utcnow().isoformat() + "Z"},
    }
}

res = requests.post(url, headers=headers, data=json.dumps(payload))
if res.status_code in [200, 201]:
    print("✅ Prediksi berhasil disimpan ke Firestore!")
else:
    print("❌ Gagal simpan:", res.text)
