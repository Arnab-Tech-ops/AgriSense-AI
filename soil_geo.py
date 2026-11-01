"""Optional point query for modelled SoilGrids WCS estimates.

This module is isolated from the crop model. WCS returns spatial estimates,
never direct soil measurements, and is not used as a substitute for lab NPK.
"""
from datetime import datetime, timezone
from io import BytesIO
from math import cos, radians

import requests

BASE = "https://maps.isric.org/mapserv"
CRS = "urn:ogc:def:crs:EPSG::4326"
PROPERTIES = {
    "pH in water (0–5 cm)": ("phh2o", "phh2o_0-5cm_Q0.5", "ph", 10.0),
    "pH lower 90% bound": ("phh2o", "phh2o_0-5cm_Q0.05", "ph_low", 10.0),
    "pH upper 90% bound": ("phh2o", "phh2o_0-5cm_Q0.95", "ph_high", 10.0),
    "Organic carbon (0–5 cm)": ("soc", "soc_0-5cm_Q0.5", "organic_carbon", 10.0),
    "Organic carbon lower 90% bound": ("soc", "soc_0-5cm_Q0.05", "organic_carbon_low", 10.0),
    "Organic carbon upper 90% bound": ("soc", "soc_0-5cm_Q0.95", "organic_carbon_high", 10.0),
}


def fetch_soilgrids_estimate(latitude, longitude, timeout=20):
    """Fetch SoilGrids WCS median and 90% interval for pH and organic carbon.

    Requires optional `rasterio`. Estimates represent modeled 250 m gridded
    products at 0–5 cm depth, not point measurements or the crop-model features.
    """
    lat, lon = float(latitude), float(longitude)
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Latitude/longitude are outside valid WGS84 ranges.")
    try:
        import numpy as np
        import rasterio
        from rasterio.io import MemoryFile
    except ImportError as exc:
        raise RuntimeError("Optional geospatial packages are missing. Install requirements-geo.txt to enable SoilGrids WCS.") from exc
    half_lat = 125 / 111320
    half_lon = min(1.0, half_lat / max(cos(radians(lat)), .01))
    bounds = {
        "Lat": (max(-90, lat-half_lat), min(90, lat+half_lat)),
        "Long": (max(-180, lon-half_lon), min(180, lon+half_lon)),
    }
    raw = {}
    session = requests.Session()
    for label, (service, coverage, key, scale) in PROPERTIES.items():
        params = [
            ("map", f"/map/{service}.map"), ("SERVICE", "WCS"),
            ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"),
            ("COVERAGEID", coverage), ("FORMAT", "GEOTIFF_INT16"),
            ("SUBSETTINGCRS", CRS), ("OUTPUTCRS", CRS),
            ("SUBSET", f"Lat({bounds['Lat'][0]},{bounds['Lat'][1]})"),
            ("SUBSET", f"Long({bounds['Long'][0]},{bounds['Long'][1]})"),
        ]
        try:
            response = session.get(BASE, params=params, timeout=timeout)
            response.raise_for_status()
            if not response.content.startswith((b"II*\x00", b"MM\x00*")):
                raise RuntimeError("SoilGrids WCS returned an error document instead of a GeoTIFF.")
            with MemoryFile(response.content) as memory:
                with memory.open() as raster:
                    sample = next(raster.sample([(lon, lat)], indexes=1, masked=True))
                    raw_value = sample[0]
                    if np.ma.is_masked(raw_value):
                        raise RuntimeError("SoilGrids returned no estimate at these coordinates.")
                    value = float(raw_value)
                    if value == raster.nodata or value == -32768:
                        raise RuntimeError("SoilGrids returned no estimate at these coordinates.")
            raw[key] = value / scale
        except requests.RequestException as exc:
            raise RuntimeError(f"SoilGrids WCS request failed for {label}: {exc}") from exc
        except rasterio.errors.RasterioError as exc:
            raise RuntimeError(f"Could not read the SoilGrids estimate for {label}: {exc}") from exc
        except StopIteration as exc:
            raise RuntimeError(f"SoilGrids returned no raster cells for {label} at these coordinates.") from exc
    raw["source"] = "ISRIC SoilGrids WCS"
    raw["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    raw["latitude"] = lat
    raw["longitude"] = lon
    raw["depth"] = "0–5 cm"
    raw["product_resolution"] = "250 m"
    raw["uncertainty"] = "Q0.05–Q0.95 (90% prediction interval)"
    raw["limitations"] = "Modelled map-cell estimates; not a lab sample. Do not use as crop-model NPK inputs."
    return raw
