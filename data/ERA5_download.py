import cdsapi
import os

def download_era5(year, month, bbox):
    """
    Download ERA5 single-level data for given year, month, and bounding box.
    Saves file under \\data\\data_raw\\era5\\era5_<year>_<month>.nc

    Parameters
    ----------
    year : int or str
        Year of data (e.g. 2024)
    month : int or str
        Month of data (1–12)
    bbox : list or tuple
        [North, West, South, East] bounding box
    """
    # Define base directory
    base_dir = os.path.join("data", "data_raw", "era5")
    os.makedirs(base_dir, exist_ok=True)

    # Build output filename
    output_file = os.path.join(base_dir, f"era5_{year}_{int(month):02d}.nc")

    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '2m_temperature',
                '2m_dewpoint_temperature',
                'surface_pressure',
                'convective_available_potential_energy',
                'total_precipitation',
            ],
            'year': str(year),
            'month': f"{int(month):02d}",
            'day': [f"{d:02d}" for d in range(1, 32)],   # all days
            'time': [f"{h:02d}:00" for h in range(24)],  # all hours
            'area': bbox,  # [North, West, South, East]
            'format': 'netcdf',
        },
        output_file
    )

    print(f"✅ ERA5 data saved to {output_file}")
