import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import statsmodels.api as sm
from tqdm import tqdm

# === 1. KONFIGURASI FIREBASE ===
API_KEY = "AIzaSyDsM-j-nbNPMdTz1irbXOXD1b8bS_mjrPk"
PROJECT_ID = "sparm-b9de0"
USER_EMAIL = "esp32@test.com"
USER_PASSWORD = "12345678"
COLLECTION_SOURCE = "power_history"
COLLECTION_FORECAST_DATA = "monthly_forecast_data"
COLLECTION_FORECAST_SUMMARY = "monthly_forecast_summary"
PRICE_PER_KWH = 1500  # Rp per kWh

# === 2. LOGIN FIREBASE ===
def login_firebase(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    res = requests.post(url, json=payload)
    if res.status_code != 200:
        raise Exception(f"Login gagal: {res.text}")
    return res.json()["idToken"]

print("🔐 Login ke Firebase...")
id_token = login_firebase(USER_EMAIL, USER_PASSWORD)
print("✅ Login berhasil!")

# === 3. AMBIL DATA DARI FIRESTORE ===
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
        # Ambil energy_per_hour (bukan power)
        if "energy_per_hour" not in fields:
            continue
        energy = float(fields["energy_per_hour"]["doubleValue"])
        timestamp = fields["timestamp"]["timestampValue"]
        data.append({"time": timestamp, "energy_per_hour": energy})

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # 🔧 FIX: hapus timestamp duplikat
    df = df.drop_duplicates(subset="time", keep="first")
    
    # 🔧 pastikan data berfrekuensi 1 jam penuh
    df = df.set_index("time").resample("h").mean().interpolate().reset_index()
    return df


print("📥 Mengambil data 1 tahun terakhir...")
df = fetch_firestore_data()
df = df.tail(8760)
print(f"✅ Ditemukan {len(df)} data energi per jam.")

# === 4. LATIH MODEL SARIMA UNTUK ENERGI ===
print("🤖 Melatih model SARIMA...")
ts = df.set_index("time")["energy_per_hour"].asfreq("H")
ts = ts.fillna(method="ffill")

model = sm.tsa.statespace.SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,24))
result = model.fit(disp=False)
print("✅ Model berhasil dilatih!")

# === 5. PREDIKSI 30 HARI KE DEPAN (720 JAM) ===
future_steps = 24 * 30
forecast = result.forecast(steps=future_steps)

future_dates = [df["time"].iloc[-1] + timedelta(hours=i+1) for i in range(future_steps)]
forecast_df = pd.DataFrame({
    "time": future_dates,
    "predicted_energy_kwh": forecast
})

# === 6. HITUNG BIAYA LISTRIK ===
total_kwh = forecast_df["predicted_energy_kwh"].sum()
total_cost = total_kwh * PRICE_PER_KWH

print("🔮 HASIL PREDIKSI:")
print(f"   Total Energi: {total_kwh:.2f} kWh")
print(f"   Estimasi Biaya: Rp{total_cost:,.0f}")

# === 4. HAPUS DATA LEBIH DARI 8760 JAM TERAKHIR ===
print("🧹 Mengecek dan menghapus data lama di 'power_history'...")

def cleanup_old_power_history():
    url = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/{collection_name}?pageSize=10000"
    headers = {"Authorization": f"Bearer {id_token}"}
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        print("⚠️ Gagal mengambil data:", res.text)
        return

    docs = res.json().get("documents", [])
    print(f"📊 Total dokumen saat ini: {len(docs)}")

    # Urutkan berdasarkan timestamp
    data = []
    for doc in docs:
        fields = doc.get("fields", {})
        timestamp = fields.get("timestamp", {}).get("timestampValue", None)
        if timestamp:
            data.append((doc["name"], timestamp))

    data.sort(key=lambda x: x[1])  # urut dari paling lama ke terbaru

    # Jika lebih dari 8760 data, hapus sisanya dari depan
    if len(data) > 8760:
        to_delete = data[:len(data) - 8760]
        print(f"🗑️ Menghapus {len(to_delete)} data lama...")
        for doc_name, _ in to_delete:
            requests.delete(f"https://firestore.googleapis.com/v1/{doc_name}", headers=headers)
        print("✅ Data lama berhasil dihapus.")
    else:
        print("✅ Tidak ada data lama yang perlu dihapus (masih di bawah 8760).")

cleanup_old_power_history()

# === HAPUS DATA PREDIKSI LAMA ===
print("🧹 Menghapus data prediksi lama...")

delete_url = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/{COLLECTION_FORECAST_DATA}"
headers = {"Authorization": f"Bearer {id_token}"}
res = requests.get(delete_url, headers=headers)

if res.status_code == 200:
    docs = res.json().get("documents", [])
    for doc in docs:
        name = doc["name"]
        requests.delete(f"https://firestore.googleapis.com/v1/{name}", headers=headers)
    print(f"✅ {len(docs)} data prediksi lama berhasil dihapus.")
else:
    print("⚠️ Gagal mengambil data lama untuk dihapus:", res.text)

# === 7. UNGGAH DATA PREDIKSI PER JAM KE FIRESTORE ===
print("📤 Mengunggah data prediksi per jam...")
batch_size = 500
headers = {"Authorization": f"Bearer {id_token}", "Content-Type": "application/json"}

month_key = datetime.now().strftime("%Y_%m")

for batch_start in tqdm(range(0, len(forecast_df), batch_size)):
    batch_data = forecast_df.iloc[batch_start:batch_start + batch_size]
    writes = []

    for _, row in batch_data.iterrows():
        doc_name = f"projects/{PROJECT_ID}/databases/(default)/documents/{COLLECTION_FORECAST_DATA}/{month_key}_{row['time'].isoformat()}"
        writes.append({
            "update": {
                "name": doc_name,
                "fields": {
                    "time": {"timestampValue": row["time"].isoformat() + "Z"},
                    "predicted_energy_kwh": {"doubleValue": float(row["predicted_energy_kwh"])},
                    "predicted_cost": {"doubleValue": float(row["predicted_energy_kwh"] * PRICE_PER_KWH)}
                }
            },
            "updateMask": {"fieldPaths": ["time", "predicted_energy_kwh", "predicted_cost"]},
            "currentDocument": {"exists": False}
        })

    requests.post(
        f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents:commit",
        headers=headers,
        json={"writes": writes}
    )

# === 8. SIMPAN DATA RINGKASAN KE FIRESTORE ===
print("📤 Menyimpan ringkasan biaya bulanan...")

summary_payload = {
    "fields": {
        "forecast_month": {"stringValue": datetime.now().strftime("%B %Y")},
        "predicted_total_kwh": {"doubleValue": float(total_kwh)},
        "predicted_total_cost": {"doubleValue": float(total_cost)},
        "price_per_kwh": {"doubleValue": PRICE_PER_KWH},
        "created_at": {"timestampValue": datetime.utcnow().isoformat() + "Z"}
    }
}

summary_url = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/{COLLECTION_FORECAST_SUMMARY}"
res = requests.post(summary_url, headers=headers, data=json.dumps(summary_payload))

if res.status_code in [200, 201]:
    print("✅ Ringkasan biaya berhasil disimpan!")
else:
    print("❌ Gagal simpan ringkasan:", res.text)

print("\n🎉 Semua prediksi & ringkasan berhasil dikirim ke Firestore!")
