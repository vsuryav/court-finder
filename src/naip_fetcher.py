"""NAIP imagery fetcher using Microsoft Planetary Computer."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple
import tempfile

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import rasterio.crs
import rasterio.windows
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from pyproj import Transformer
from rasterio.crs import CRS

from .geo_utils import BBox


@dataclass
class NAIPImage:
    """Container for NAIP imagery data."""
    data: np.ndarray  # Shape: (bands, height, width)
    transform: rasterio.Affine
    crs: rasterio.CRS
    bounds: BBox
    imagery_date: date
    
    @property
    def height(self) -> int:
        return self.data.shape[1]
    
    @property
    def width(self) -> int:
        return self.data.shape[2]
    
    @property
    def rgb(self) -> np.ndarray:
        """Return RGB channels as (height, width, 3) array."""
        # NAIP bands: R, G, B, NIR
        return np.transpose(self.data[:3], (1, 2, 0))


class NAIPFetcher:
    """Fetch NAIP imagery from Microsoft Planetary Computer."""
    
    STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    COLLECTION = "naip"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the NAIP fetcher.
        
        Args:
            cache_dir: Directory to cache downloaded imagery
        """
        self.catalog = pystac_client.Client.open(
            self.STAC_URL,
            modifier=planetary_computer.sign_inplace
        )
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "naip_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch(self, bbox: BBox) -> NAIPImage:
        """
        Fetch NAIP imagery for a bounding box.
        
        Args:
            bbox: Area to fetch imagery for
            
        Returns:
            NAIPImage with merged/mosaicked data
            
        Raises:
            ValueError: If no NAIP imagery found for area
        """
        # Search for NAIP items
        items = self._search_items(bbox)
        
        if not items:
            raise ValueError(f"No NAIP imagery found for bbox: {bbox}")
        
        # Get the most recent imagery date
        imagery_date = self._get_imagery_date(items)
        
        # Read and merge imagery
        data, transform, crs = self._read_and_merge(items, bbox)
        
        return NAIPImage(
            data=data,
            transform=transform,
            crs=crs,
            bounds=bbox,
            imagery_date=imagery_date
        )
    
    def get_imagery_date(self, bbox: BBox) -> Optional[date]:
        """
        Get the most recent NAIP imagery date for a bbox without downloading.
        
        Args:
            bbox: Area to check
            
        Returns:
            Most recent imagery date, or None if no imagery found
        """
        items = self._search_items(bbox)
        if not items:
            return None
        return self._get_imagery_date(items)
    
    def _search_items(self, bbox: BBox) -> List:
        """Search for NAIP items covering the bbox."""
        search = self.catalog.search(
            collections=[self.COLLECTION],
            bbox=bbox.as_tuple,
            sortby=[{"field": "datetime", "direction": "desc"}]
        )
        return list(search.items())
    
    def _get_imagery_date(self, items: List) -> date:
        """Extract the most recent imagery date from items."""
        # Items are sorted by datetime desc, so first is most recent
        datetime_str = items[0].properties.get("datetime", "")
        if datetime_str:
            return date.fromisoformat(datetime_str[:10])
        return date.today()
    
    def _read_and_merge(
        self, 
        items: List, 
        bbox: BBox
    ) -> Tuple[np.ndarray, rasterio.Affine, rasterio.CRS]:
        """Read imagery from items and merge into single array."""
        datasets = []
        
        try:
            # Only use the most recent item for now (sorted by date desc)
            item = items[0]
            asset = item.assets.get("image")
            if not asset:
                raise ValueError("No image asset found")
            
            href = asset.href
            src = rasterio.open(href)
            datasets.append(src)
            
            # Transform bbox from EPSG:4326 to source CRS
            src_crs = src.crs
            transformer = Transformer.from_crs(
                "EPSG:4326", 
                src_crs.to_string(),
                always_xy=True
            )
            
            # Transform bbox corners
            min_x, min_y = transformer.transform(bbox.min_lon, bbox.min_lat)
            max_x, max_y = transformer.transform(bbox.max_lon, bbox.max_lat)
            
            # Create window from transformed bounds
            window = from_bounds(min_x, min_y, max_x, max_y, src.transform)
            
            # Read data
            data = src.read(window=window)
            transform = rasterio.windows.transform(window, src.transform)
            
            # Handle case where window is outside bounds or empty
            if data.size == 0 or data.shape[1] == 0 or data.shape[2] == 0:
                # Try reading the full intersection
                # Get the intersection of our bbox with the tile bounds
                tile_bounds = src.bounds
                int_min_x = max(min_x, tile_bounds.left)
                int_max_x = min(max_x, tile_bounds.right)
                int_min_y = max(min_y, tile_bounds.bottom)
                int_max_y = min(max_y, tile_bounds.top)
                
                if int_min_x < int_max_x and int_min_y < int_max_y:
                    window = from_bounds(int_min_x, int_min_y, int_max_x, int_max_y, src.transform)
                    data = src.read(window=window)
                    transform = rasterio.windows.transform(window, src.transform)
            
            return data, transform, src_crs
            
        finally:
            # Close all datasets
            for ds in datasets:
                ds.close()
