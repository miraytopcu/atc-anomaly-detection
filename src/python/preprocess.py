import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

class FlightPreprocessor:
    def __init__(self, raw_path, output_path, min_seq_len=8, max_gap=300):
        self.raw_path = raw_path
        self.output_path = output_path
        self.min_seq_len = min_seq_len
        self.max_gap = max_gap

        self.stages = ["cleaned", "segmented", "resampled", "windowed"]
        for stage in self.stages:
            os.makedirs(os.path.join(output_path, stage), exist_ok=True)

    def clean_outliers(self, df):
        """Hard filtering and outlier flagging."""
        df = df.sort_values(['icao24', 'timestamp']).copy()

        # 1. Hard Filtering
        df = df[(df['velocity'] >= 0) & (df['velocity'] <= 400)]
        df = df[(df['alt'] >= -500) & (df['alt'] <= 15000)]

        # 2. Soft Outlier Flagging (Preserving anomaly signal)
        df['vertical_rate'] = df['vertical_rate'].clip(-50, 50)

        # 3. Handle Missing Values
        # df = df.groupby('icao24', group_keys=False).apply(lambda x: x.ffill().bfill())
        return df

    def create_segments(self, df):
        """Splits a single aircraft's data into continuous segments based on time gaps, ground status, and icao changes."""
        df = df.sort_values(['icao24', 'timestamp'])
        df['time_diff'] = df.groupby('icao24')['timestamp'].diff().dt.total_seconds().fillna(0)
        
        on_ground_change = df.groupby('icao24')['on_ground'].diff().fillna(0) != 0
        icao_change = df['icao24'] != df['icao24'].shift(1)

        df['new_segment'] = (
            (df['time_diff'] > self.max_gap) |
            on_ground_change |
            icao_change
        )

        df['segment_id'] = df.groupby('icao24')['new_segment'].cumsum()
        return df
    
    def add_features(self, df):
        df = df.sort_values('timestamp')
        df['delta_velocity'] = df['velocity'].diff().fillna(0)
        df['delta_alt'] = df['alt'].diff().fillna(0)

        def heading_diff(x):
            return (x.diff() + 180) % 360 - 180

        df['delta_heading'] = heading_diff(df['true_track']).fillna(0)

        return df

        
    def resample_segment(self, segment_df):
        """Converts irregular time series to fixed 30s intervals."""
        numeric_cols = segment_df.select_dtypes(include=[np.number]).columns.drop('segment_id', errors='ignore')

        segment_df = segment_df.drop_duplicates(subset='timestamp')
        resampled = segment_df.set_index('timestamp')[numeric_cols].resample('30S').mean()
        resampled = resampled.interpolate(method='linear', limit=2)

        resampled['icao24'] = segment_df['icao24'].iloc[0]
        resampled['segment_id'] = segment_df['segment_id'].iloc[0]

        return resampled
    
    def filter_segments(self, df):
        return df.groupby(['icao24', 'segment_id']) \
                 .filter(lambda x: len(x) >= self.min_seq_len)
    
    def create_windows(self, segment, window_size=30):
        """Creates sliding windows for LSTM input."""
        features = segment.select_dtypes(include=[np.number]).drop(columns=['segment_id'], errors='ignore')

        X = []
        for i in range(len(features) - window_size + 1):
            X.append(features.iloc[i:i+window_size].values)
        return np.array(X)

    def run(self):
        files = glob.glob(os.path.join(self.raw_path, "**/*.parquet"), recursive=True)
        print(f"--- DEBUG: {len(files)} files found ---")

        if not files:
            print("Error: No parquet files found!")
            return

        out_dir = os.path.join(self.output_path, "windowed")
        all_windows_list = []

        batch_size = 100
        for i in range(0, len(files), batch_size):
            batch_files = files[i : i + batch_size]
            print(f"\n--- Processing Batch: {i//batch_size + 1} ({len(batch_files)} files) ---")

            df_list = [pd.read_parquet(f) for f in batch_files]
            df = pd.concat(df_list, ignore_index=True)

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"DEBUG: Total rows loaded: {len(df)}")

            df = self.clean_outliers(df)
            df = self.create_segments(df)

            segments = df.groupby(['icao24', 'segment_id'])

            for(_, _), seg in segments:
                if len(seg) < self.min_seq_len:
                    continue

                resampled = seg.sort_values('timestamp').copy()
                resampled = resampled.ffill().bfill()
                resampled = self.add_features(resampled)
                
                if len(resampled) >= self.min_seq_len:
                    windows = self.create_windows(resampled, window_size=10)

                    if windows is not None and len(windows) > 0:
                        all_windows_list.append(windows)

        if all_windows_list:
            print("\nConcatenating all windows. Please wait.")
            final_data = np.concatenate(all_windows_list, axis=0)

            save_path = os.path.join(out_dir, "all_training_data_final.npy")
            np.save(save_path, final_data)

        else:
            print("No valid windows found to save.")
        
if __name__ == "__main__":
    RAW_DATA_PATH = r"C:\Users\LENOVO\Documents\GitHub\atc-anomaly-detection\data\raw_parquet\y=2026\m=04"
    PROCESSED_PATH = r"C:\Users\LENOVO\Documents\GitHub\atc-anomaly-detection\data\processed"

    processor = FlightPreprocessor(RAW_DATA_PATH, PROCESSED_PATH)
    if os.path.exists(RAW_DATA_PATH):
        processor.run()
    else:
        print(f"Error: Path does not exist! Please check: {RAW_DATA_PATH}")
