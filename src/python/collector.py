import pandas as pd # dataframe and parquet
import os
import time
import logging # instead of print
import json
import signal # ctrl+c for interrupting
import sys # safe close
from datetime import datetime
from collections import defaultdict
from opensky_api import OpenSkyApi, TokenManager
from utils import calculate_heading_delta, calculate_rate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("collector.log"), logging.StreamHandler()]
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CRED_PATH = os.path.join(BASE_DIR, "credentials.json")
DATA_DIR = os.path.join(BASE_DIR, "data/raw_parquet/")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "collector_checkpoint.json")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {"total_rows_saved": 0}
    return {"total_rows_saved": 0}

checkpoint = load_checkpoint()
total_rows_saved = checkpoint["total_rows_saved"]
prev_states = {}   
seen_records = {}  

def shutdown_handler(sig, frame):
    logging.info("Signal received, saving current status...")
    try:
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump({"total_rows_saved": total_rows_saved}, f)
        logging.info("Checkpoint updated successfully. Shutting down.")
    except Exception as e:
        logging.error(f"Error occurred during shutdown: {e}")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)

try:
    tm = TokenManager.from_json_file(CRED_PATH)
    api = OpenSkyApi(token_manager=tm)
    logging.info("Connected to API.")
except Exception as e:
    logging.error(f"API Error: {e}"); sys.exit(1)

def fetch_and_save():
    global total_rows_saved, seen_records, prev_states
    
    try:
        states = api.get_states()
        
        if not states or not states.states: # internet koptuysa ya da rate limit (çok istek atıldığı için engelleme) geldiyse program çökmez, 10s bekleyip tekrar dener
            logging.warning("API returned an empty response; waiting 10 seconds (May be rate limited).")
            time.sleep(10)
            return

        now_real = time.time()
        api_time = states.time # Bilgisayarınla OpenSky sunucusu arasında internet hızı yüzünden birkaç saniye latency olabilir. Bizim için uçağın konumu, bilgisayara ulaştığı anda değil, sunucudan çıktığı anda doğrudur.
        data_list = []
        
        for s in states.states:
            if s.longitude is None or s.latitude is None or s.icao24 is None:
                continue # konumu ya da kimliği belli olmayan uçak trainingde işimize yaramaz, onları skipliyoruz.
            
            if s.velocity is not None and (s.velocity < 0 or s.velocity > 400):
                continue # outlier filtresi

            if s.baro_altitude is not None and (s.baro_altitude < -500 or s.baro_altitude > 18000):
                continue

            icao = s.icao24
            msg_time = s.time_position if s.time_position else api_time # eğer uçağın kendi girili bir zamanı varsa onu kullan, yoksa api sunucusunun vaktini kullan
            
            record_key = (icao, msg_time) # duplicate filter için key oluşturuyor
            if record_key in seen_records:
                continue
            seen_records[record_key] = now_real

            accel, turn_rate, v_rate = 0, 0, 0
            if icao in prev_states:
                prev = prev_states[icao]
                dt = msg_time - prev["time"]
                if dt > 0:
                    accel = calculate_rate(s.velocity, prev["vel"], dt)
                    v_rate = calculate_rate(s.baro_altitude, prev["alt"], dt)
                    turn_rate = calculate_heading_delta(s.true_track, prev["track"]) / dt

            prev_states[icao] = {"vel": s.velocity, "track": s.true_track, "alt": s.baro_altitude, "time": msg_time}

            data_list.append({
                "timestamp": pd.to_datetime(msg_time, unit='s'),
                "icao24": icao,
                "callsign": s.callsign.strip() if s.callsign else "Unknown",
                "lat": s.latitude, "lon": s.longitude,
                "alt": s.baro_altitude, "velocity": s.velocity, "true_track": s.true_track,
                "acceleration": accel, "turn_rate": turn_rate, "vertical_rate": v_rate,
                "on_ground": s.on_ground, "category": s.category
            })

        if data_list:
            dt_now = datetime.now()
            path = os.path.join(DATA_DIR, f"y={dt_now.year}/m={dt_now.month:02}/d={dt_now.day:02}/h={dt_now.hour:02}") # Partitioning, verileri saat saat klasörler
            os.makedirs(path, exist_ok=True)
            
            file_path = os.path.join(path, f"batch_{dt_now.strftime('%M%S_%f')}.parquet") # CSV 100 MB ise Parquet 10 MB'tır ve okuması 10 kat daha hızlıdır. snappy ise veriyi çok hızlı sıkıştıran bir algoritmadır.
            df = pd.DataFrame(data_list)
            df.to_parquet(file_path, engine='pyarrow', compression='snappy', index=False)
            
            total_rows_saved += len(df)
            
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump({"total_rows_saved": total_rows_saved}, f)

            logging.info(f"Saved: {len(df)} | Total: {total_rows_saved}")

        seen_records = {k: v for k, v in seen_records.items() if now_real - v < 600}
        expired_aircraft = [i for i, st in prev_states.items() if api_time - st["time"] > 3600]
        for i in expired_aircraft: prev_states.pop(i) # memory cleanup: Eğer bir uçak radar alanından çıktıysa ve 1 saattir görünmüyorsa, onun bilgisini prev_states içinde tutmaya devam etmeyelim. Yoksa bilgisayarın RAM'i yavaş yavaş dolar ve sonunda bilgisayar donar (Memory Leak).

    except Exception as e:
        logging.error(f"Loop Error: {e}"); time.sleep(10)

if __name__ == "__main__":
    logging.info("Collector enabled.")
    while True:
        fetch_and_save()
        time.sleep(30)