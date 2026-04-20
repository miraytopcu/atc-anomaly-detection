from opensky_api import OpenSkyApi, TokenManager
import pandas as pd
import os
import time
from datetime import datetime

# Dosya yollarını ayarla
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CRED_PATH = os.path.join(BASE_DIR, "credentials.json")
CSV_PATH = os.path.join(BASE_DIR, "data/raw_flight_data.csv")

def fetch_and_save():
    try:
        # OAuth2 Flow: TokenManager credentials.json'ı otomatik okur
        tm = TokenManager.from_json_file(CRED_PATH)
        api = OpenSkyApi(token_manager=tm)
        
        # Marmara Bölgesi
        states = api.get_states(bbox=(40.0, -75.0, 42.0, -72.0))
        
        if states and states.states:
            data_list = []
            for s in states.states:
                data_list.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "icao24": s.icao24,
                    "callsign": s.callsign.strip() if s.callsign else "Unknown",
                    "longitude": s.longitude,
                    "latitude": s.latitude,
                    "baro_altitude": s.baro_altitude,
                    "velocity": s.velocity,
                    "true_track": s.true_track
                })
            
            df = pd.DataFrame(data_list)
            file_exists = os.path.isfile(CSV_PATH)
            df.to_csv(CSV_PATH, mode='a', index=False, header=not file_exists)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {len(df)} uçak kaydedildi.")
        else:
            print("Uçuş bulunamadı.")

    except Exception as e:
        print(f"Sistem Hatası: {e}")

if __name__ == "__main__":
    while True:
        fetch_and_save()
        time.sleep(30)