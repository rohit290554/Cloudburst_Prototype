import pandas as pd
import numpy as np
import os

# Ensure the folder exists
os.makedirs("data/data_processed", exist_ok=True)

# Number of dummy rows
n_rows = 100

# Generate dummy data
data = {
    "time": pd.date_range(start="2024-07-01", periods=n_rows, freq="H"),
    "latitude": np.random.uniform(28.2, 29.0, n_rows),
    "longitude": np.random.uniform(76.9, 77.8, n_rows),
    "t2m": np.random.uniform(295, 310, n_rows),       # K
    "d2m": np.random.uniform(290, 305, n_rows),       # K
    "tp": np.random.uniform(0, 20, n_rows),           # mm
    "cape": np.random.uniform(0, 2000, n_rows),       # J/kg
    "sp": np.random.uniform(95000, 105000, n_rows),   # Pa
    "number": np.zeros(n_rows, dtype=int),
    "expver": np.ones(n_rows, dtype=int),
    "lon": np.random.uniform(76.9, 77.8, n_rows),
    "lat": np.random.uniform(28.2, 29.0, n_rows),
    "precipitation": np.random.uniform(0, 100, n_rows),
    "cloudburst": np.random.choice([0, 1], size=n_rows, p=[0.8, 0.2]),  # 20% chance
    "elevation": np.random.uniform(200, 300, n_rows),  # m
    "slope": np.random.uniform(0, 90, n_rows)          # degrees
}

# Create DataFrame
df = pd.DataFrame(data)

# Save CSV
csv_path = "data/data_processed/dummy_noida_data.csv"
df.to_csv(csv_path, index=False)

print(f"✅ Dummy dataset saved: {csv_path}")
