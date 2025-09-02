import streamlit as st
import pandas as pd
import xarray as xr
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Cloudburst Prediction", layout="wide")
st.title("🌩 Cloudburst Prediction Web App")

# Sidebar
st.sidebar.header("Upload Data Files")
imerg_file = st.sidebar.file_uploader("IMERG Rainfall (.nc4)", type=["nc4"])
era5_file = st.sidebar.file_uploader("ERA5 Data (.nc)", type=["nc"])
dem_file = st.sidebar.file_uploader("DEM File (.tif)", type=["tif"])
model_file = st.sidebar.file_uploader("Trained Model (.pkl)", type=["pkl"])

# Step 1: Load files
if imerg_file and era5_file and dem_file:
    st.success("✅ Files uploaded successfully!")

    # ---------- IMERG ----------
    ds_imerg = xr.open_dataset(imerg_file)
    if "precipitationCal" in ds_imerg.variables:
        rain = ds_imerg["precipitationCal"][0,:,:]
    else:
        rain = ds_imerg["precipitation"][0,:,:]

    st.subheader("📊 IMERG Rainfall Preview")
    fig, ax = plt.subplots()
    rain.plot(ax=ax, cmap="Blues")
    st.pyplot(fig)

    # ---------- ERA5 ----------
    ds_era5 = xr.open_dataset(era5_file)
    st.subheader("🌡 ERA5 Variables Preview")
    st.write(ds_era5.to_dataframe().head())

    # ---------- DEM ----------
    with rasterio.open(dem_file) as dem_src:
        dem = dem_src.read(1)
        dem_extent = (dem_src.bounds.left, dem_src.bounds.right,
                      dem_src.bounds.bottom, dem_src.bounds.top)

    st.subheader("🌍 DEM + Rainfall Overlay")
    fig, ax = plt.subplots()
    ax.imshow(dem, extent=dem_extent, cmap="terrain", alpha=0.6)
    ax.imshow(rain, extent=dem_extent, cmap="Blues", alpha=0.5)
    st.pyplot(fig)

    # ---------- Prediction ----------
    if model_file:
        model = joblib.load(model_file)
        st.subheader("⚡ Cloudburst Prediction")
        features = {
            "rain_mean": float(rain.mean().values),
            "rain_max": float(rain.max().values),
            "rain_std": float(rain.std().values),
            "dem_mean": float(np.mean(dem)),
            "dem_std": float(np.std(dem))
        }
        X = np.array([list(features.values())])
        pred_proba = model.predict_proba(X)[0][1]
        pred_class = model.predict(X)[0]
        if pred_class == 1:
            st.metric("Cloudburst Risk", f"⚠ High ({pred_proba:.2%})")
        else:
            st.metric("Cloudburst Risk", f"✅ Low ({pred_proba:.2%})")
        st.json(features)