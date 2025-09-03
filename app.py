import streamlit as st
import pandas as pd
import xarray as xr
import rasterio
import richdem as rd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Cloudburst Prediction", layout="wide")
st.title("🌩 Cloudburst Prediction Web App")

# Sidebar
st.sidebar.header("Upload Data Files")
imerg_file = st.sidebar.file_uploader("IMERG Rainfall (.nc4)", type=["nc4"])
era5_instant = st.sidebar.file_uploader("ERA5 Instantaneous (.nc)", type=["nc"])
era5_accum = st.sidebar.file_uploader("ERA5 Accumulated (.nc)", type=["nc"])
dem_file = st.sidebar.file_uploader("DEM File (.tif)", type=["tif"])
model_file = st.sidebar.file_uploader("Trained Model (.pkl)", type=["pkl"])

# Sidebar threshold adjustment
threshold = st.sidebar.slider(
    "🌧 Cloudburst Threshold (mm/hr)", min_value=10, max_value=100, value=50, step=5
)

if imerg_file and era5_instant and era5_accum and dem_file:
    st.success("✅ Files uploaded successfully!")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📜 Logs", "📊 Preview Plots", "📑 Merged Dataset", "⚡ Prediction"]
    )

    # ---------- LOGS ----------
    with tab1:
        logs = []

        # --- IMERG ---
        ds_imerg = xr.open_dataset(io.BytesIO(imerg_file.read()))
        imerg_file.seek(0)
        logs.append("=== IMERG Dataset Info ===")
        logs.append(str(ds_imerg))

        if "precipitationCal" in ds_imerg.variables:
            rain = ds_imerg["precipitationCal"].resample(time="1h").mean()
        else:
            rain = ds_imerg["precipitation"].resample(time="1h").mean()

        imerg_df = rain.to_dataframe().reset_index()
        imerg_df["cloudburst"] = (imerg_df.iloc[:, -1] >= threshold).astype(int)
        logs.append(f"IMERG DataFrame shape: {imerg_df.shape}")
        logs.append(f"Cloudburst threshold used: {threshold} mm/hr")

        # --- ERA5 ---
        ds_instant = xr.open_dataset(io.BytesIO(era5_instant.read()))
        era5_instant.seek(0)
        ds_accum = xr.open_dataset(io.BytesIO(era5_accum.read()))
        era5_accum.seek(0)
        era5 = xr.merge([ds_instant, ds_accum])
        logs.append("=== ERA5 Dataset Variables ===")
        logs.append(str(list(era5.data_vars)))

        era5_df = era5[["t2m", "d2m", "tp", "cape", "sp"]].to_dataframe().reset_index()
        if "valid_time" in era5_df.columns:
            era5_df = era5_df.rename(columns={"valid_time": "time"})
        era5_df["time"] = pd.to_datetime(era5_df["time"].astype(str))
        logs.append(f"ERA5 DataFrame shape: {era5_df.shape}")

        # --- DEM ---
        with rasterio.open(io.BytesIO(dem_file.read())) as dem_src:
            dem = dem_src.read(1)
            dem_extent = (dem_src.bounds.left, dem_src.bounds.right,
                          dem_src.bounds.bottom, dem_src.bounds.top)
            logs.append(f"DEM shape: {dem.shape}, Resolution: {dem_src.res}")

        dem_file.seek(0)
        dem_rd = rd.LoadGDAL(dem_file)
        slope = rd.TerrainAttribute(dem_rd, attrib="slope_degrees")

        # --- Merge ---
        merged = pd.merge(era5_df, imerg_df, on="time", how="inner")
        merged["elevation"] = np.nanmean(dem)
        merged["slope"] = np.nanmean(slope)
        logs.append(f"Merged dataset shape: {merged.shape}")

        # Show all logs
        st.subheader("📝 Calculation Logs")
        st.code("\n".join(logs))

    # ---------- PREVIEWS ----------
    with tab2:
        st.subheader("📊 IMERG Rainfall Snapshot")
        fig, ax = plt.subplots()
        rain.isel(time=0).plot(ax=ax, cmap="Blues")
        st.pyplot(fig)

        st.subheader("🌡 ERA5 Variables Preview")
        st.write(era5_df.head())

        st.subheader("🌍 DEM + Rainfall Overlay")
        fig, ax = plt.subplots()
        ax.imshow(dem, cmap="terrain", alpha=0.6)
        ax.imshow(rain.isel(time=0), cmap="Blues", alpha=0.5)
        st.pyplot(fig)

    # ---------- MERGED DATASET ----------
    with tab3:
        st.subheader("📑 Merged Dataset")
        st.dataframe(merged.head())

        csv_buffer = BytesIO()
        merged.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇ Download Merged Dataset (CSV)",
            data=csv_buffer.getvalue(),
            file_name="cloudburst_merged_dataset.csv",
            mime="text/csv"
        )

    # ---------- PREDICTION ----------
    with tab4:
        if model_file:
            model = joblib.load(model_file)
            st.subheader("⚡ Cloudburst Prediction")

            # Features for single snapshot prediction
            features = {
                "rain_mean": float(rain.mean().values),
                "rain_max": float(rain.max().values),
                "rain_std": float(rain.std().values),
                "dem_mean": float(np.mean(dem)),
                "dem_std": float(np.std(dem)),
                "slope_mean": float(np.nanmean(slope)),
                "slope_std": float(np.nanstd(slope)),
            }
            X_single = np.array([list(features.values())])
            pred_proba = model.predict_proba(X_single)[0][1]
            pred_class = model.predict(X_single)[0]

            if pred_class == 1:
                st.metric("Cloudburst Risk", f"⚠ High ({pred_proba:.2%})")
            else:
                st.metric("Cloudburst Risk", f"✅ Low ({pred_proba:.2%})")

            st.json(features)

            # Predict for full dataset
            X_all = merged[["t2m", "d2m", "tp", "cape", "sp", "elevation", "slope"]].fillna(0)
            merged["cloudburst_pred"] = model.predict(X_all)

            # Add combined status
            def combined_status(row):
                if row["cloudburst"] == 1 and row["cloudburst_pred"] == 1:
                    return "True Positive"
                elif row["cloudburst"] == 0 and row["cloudburst_pred"] == 0:
                    return "True Negative"
                elif row["cloudburst"] == 0 and row["cloudburst_pred"] == 1:
                    return "False Positive"
                else:
                    return "False Negative"

            merged["status"] = merged.apply(combined_status, axis=1)

            # Interactive scatter plot
            st.subheader("📈 Actual vs Predicted Cloudburst Events")
            fig = px.scatter(
                merged,
                x="time",
                y="tp",
                color="status",
                labels={"tp": "Rainfall (tp mm/hr)", "time": "Time"},
                hover_data=["t2m", "d2m", "cape", "sp", "elevation", "slope", "cloudburst", "cloudburst_pred"]
            )
            fig.add_trace(
                go.Scatter(
                    x=merged["time"],
                    y=[threshold] * len(merged),
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    name=f"Threshold ({threshold} mm/hr)"
                )
            )
            fig.update_traces(marker=dict(size=9, line=dict(width=1, color="DarkSlateGrey")))
            fig.update_layout(
                title=f"Actual vs Predicted Cloudburst Events (Threshold = {threshold} mm/hr)",
                xaxis_title="Time",
                yaxis_title="Rainfall (tp mm/hr)",
                hovermode="closest"
            )
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("👨‍💻 Developed by **git@rohit290554**")