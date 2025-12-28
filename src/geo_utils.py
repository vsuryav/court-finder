"""Geospatial utilities for coordinate conversion and bounding box generation."""

from dataclasses import dataclass
from typing import Tuple
import math

import pgeocode
from pyproj import CRS, Transformer


@dataclass
class BBox:
    """Bounding box in EPSG:4326 (lat/lon)."""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    
    @property
    def center(self) -> Tuple[float, float]:
        """Return center as (lat, lon)."""
        return (
            (self.min_lat + self.max_lat) / 2,
            (self.min_lon + self.max_lon) / 2
        )
    
    @property
    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return as (min_lon, min_lat, max_lon, max_lat) for rasterio."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)
    
    def to_wkt(self) -> str:
        """Convert to WKT polygon string."""
        return (
            f"POLYGON(({self.min_lon} {self.min_lat}, {self.max_lon} {self.min_lat}, "
            f"{self.max_lon} {self.max_lat}, {self.min_lon} {self.max_lat}, "
            f"{self.min_lon} {self.min_lat}))"
        )


# Earth's radius in miles
EARTH_RADIUS_MILES = 3958.8

# NAIP nominal resolution
NAIP_METERS_PER_PIXEL = 0.6


def zipcode_to_center(zipcode: str, country: str = "US") -> Tuple[float, float]:
    """
    Convert a US zipcode to lat/lon center point.
    
    Args:
        zipcode: US zipcode (e.g., "30306")
        country: Country code (default "US")
        
    Returns:
        Tuple of (latitude, longitude)
        
    Raises:
        ValueError: If zipcode is not found
    """
    nomi = pgeocode.Nominatim(country)
    result = nomi.query_postal_code(zipcode)
    
    # pgeocode returns a pandas Series, math.isnan expects a scalar float
    lat = float(result.latitude)
    lon = float(result.longitude)
    
    if math.isnan(lat) or math.isnan(lon):
        raise ValueError(f"Zipcode '{zipcode}' not found")
    
    return (lat, lon)


def create_search_bbox(center: Tuple[float, float], radius_miles: float) -> BBox:
    """
    Create a square bounding box around a center point.
    
    Args:
        center: (latitude, longitude) center point
        radius_miles: Radius in miles from center to edge of box
        
    Returns:
        BBox in EPSG:4326
    """
    lat, lon = center
    
    # Convert miles to degrees
    # Latitude: 1 degree ≈ 69 miles
    lat_delta = radius_miles / 69.0
    
    # Longitude: varies with latitude
    # 1 degree ≈ 69 * cos(lat) miles
    lon_delta = radius_miles / (69.0 * math.cos(math.radians(lat)))
    
    return BBox(
        min_lon=lon - lon_delta,
        min_lat=lat - lat_delta,
        max_lon=lon + lon_delta,
        max_lat=lat + lat_delta
    )


def meters_per_pixel_at_lat(latitude: float) -> float:
    """
    Calculate NAIP ground resolution at a given latitude.
    
    NAIP imagery is nominally 60cm (0.6m) per pixel.
    
    Args:
        latitude: Latitude in degrees
        
    Returns:
        Meters per pixel (approximately 0.6 for NAIP)
    """
    # NAIP is orthorectified, so resolution is fairly consistent
    return NAIP_METERS_PER_PIXEL


def pixel_area_to_square_meters(pixel_count: int, latitude: float) -> float:
    """
    Convert pixel area to square meters.
    
    Args:
        pixel_count: Number of pixels
        latitude: Latitude for resolution calculation
        
    Returns:
        Area in square meters
    """
    mpp = meters_per_pixel_at_lat(latitude)
    return pixel_count * (mpp ** 2)


def create_utm_transformer(bbox: BBox) -> Tuple[Transformer, CRS]:
    """
    Create a transformer from EPSG:4326 to appropriate UTM zone.
    
    Args:
        bbox: Bounding box to get UTM zone for
        
    Returns:
        Tuple of (transformer, utm_crs)
    """
    center_lat, center_lon = bbox.center
    
    # Calculate UTM zone
    zone = int((center_lon + 180) / 6) + 1
    hemisphere = "north" if center_lat >= 0 else "south"
    
    utm_crs = CRS.from_proj4(
        f"+proj=utm +zone={zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
    )
    
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),
        utm_crs,
        always_xy=True
    )
    
    return transformer, utm_crs
