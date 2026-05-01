import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

def prepare_model_data(input_path, output_path):

    np.random.seed(42)
    print(f"Loading data from {input_path}...")
    
    # 1. Memory Optimization
    data = np.load(input_path, mmap_mode='r')
    
    num_windows, seq_len, total_features = data.shape
    print(f"Original shape (mmap): {data.shape}")

    # 2. Feature Selection
    FEATURE_NAMES = [
        "lat", "lon", "alt", "velocity", "true_track", "acceleration",
        "turn_rate", "vertical_rate", "category", "delta_velocity", "delta_alt", "delta_heading"
    ]

    if len(FEATURE_NAMES) != total_features:
        print(f"WARNING: Feature name list ({len(FEATURE_NAMES)}) does not match data ({total_features})!")

    features_to_drop = ["lat", "lon"]
    keep_indices = [i for i, f in enumerate(FEATURE_NAMES) if f not in features_to_drop]

    data_filtered = data[:, :, keep_indices] 
    num_features = data_filtered.shape[2]
    print(f"Features kept: {[FEATURE_NAMES[i] for i in keep_indices]}")

    # 3. Train / Test Split (Time-Aware)
    split_idx = int(0.8 * num_windows)
    train_raw = data_filtered[:split_idx]
    test_raw = data_filtered[split_idx:]
    
    print(f"Train set size: {len(train_raw)}")
    print(f"Test set size: {len(test_raw)}")

    # 4. Scaling
    scaler = StandardScaler()
    train_reshaped = train_raw.reshape(-1, num_features).astype(np.float32)
    
    scaler.fit(train_reshaped)
    train_scaled = scaler.transform(train_reshaped).reshape(train_raw.shape).astype(np.float32)

    del train_reshaped
    
    test_reshaped = test_raw.reshape(-1, num_features).astype(np.float32)
    test_scaled = scaler.transform(test_reshaped).reshape(test_raw.shape).astype(np.float32)

    del test_reshaped

    # 5. Outlier Clipping
    train_scaled = np.clip(train_scaled, -5, 5)
    test_scaled = np.clip(test_scaled, -5, 5)

    # 6. Quality Check
    print(f"NaN count in Train: {np.isnan(train_scaled).sum()}")
    print(f"NaN count in Test: {np.isnan(test_scaled).sum()}")
    print(f"INF count in Train: {np.isinf(train_scaled).sum()}")
    print(f"INF count in Test: {np.isinf(test_scaled).sum()}")

    # 7. Shuffle Training Data
    indices = np.arange(len(train_scaled))
    np.random.shuffle(indices)
    train_scaled = train_scaled[indices]

    # 8. Saving the Scaler
    scaler_path = os.path.join(output_path, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # 9. Saving Final Datasets
    np.save(os.path.join(output_path, "X_train.npy"), train_scaled)
    np.save(os.path.join(output_path, "X_test.npy"), test_scaled)
    
    print("--- Data Preparation Complete ---")
    print(f"Final Train Shape: {train_scaled.shape}")
    print(f"Final Test Shape: {test_scaled.shape}")

if __name__ == "__main__":

    INPUT_FILE = r"C:\Users\LENOVO\Documents\GitHub\atc-anomaly-detection\data\processed\windowed\all_training_data_final.npy"
    OUTPUT_DIR = r"C:\Users\LENOVO\Documents\GitHub\atc-anomaly-detection\data\processed"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    prepare_model_data(INPUT_FILE, OUTPUT_DIR)