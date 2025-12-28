"""Tiling/chipping utilities for processing large imagery in manageable chunks."""

from dataclasses import dataclass
from typing import Generator, Tuple

import numpy as np
import rasterio
from rasterio import Affine

from .geo_utils import BBox


@dataclass
class Chip:
    """A single image chip with geospatial metadata."""
    data: np.ndarray  # Shape: (bands, height, width) or (height, width, channels)
    transform: Affine
    crs: rasterio.CRS
    chip_id: Tuple[int, int]  # (row, col) index
    
    @property
    def rgb(self) -> np.ndarray:
        """Return RGB as (height, width, 3) array."""
        if self.data.ndim == 3 and self.data.shape[0] <= 4:
            # (bands, height, width) -> (height, width, bands)
            return np.transpose(self.data[:3], (1, 2, 0))
        return self.data[:, :, :3] if self.data.shape[2] >= 3 else self.data
    
    @property
    def bounds(self) -> BBox:
        """Calculate bounds from transform and data shape."""
        h, w = self.data.shape[1], self.data.shape[2]
        min_x, max_y = self.transform * (0, 0)
        max_x, min_y = self.transform * (w, h)
        return BBox(min_lon=min_x, min_lat=min_y, max_lon=max_x, max_lat=max_y)
    
    def pixel_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates."""
        # result of Affine * (col, row) is a tuple (x, y)
        res = self.transform * (col, row)
        return (float(res[0]), float(res[1]))
    
    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        inv_transform = ~self.transform
        res = inv_transform * (x, y)
        return (int(res[0]), int(res[1]))


class ChipGenerator:
    """Generate overlapping chips from large imagery."""
    
    def __init__(
        self,
        chip_size: int = 1024,
        overlap: float = 0.2
    ):
        """
        Initialize chip generator.
        
        Args:
            chip_size: Size of each chip in pixels (square)
            overlap: Overlap fraction between adjacent chips (0.0 - 0.5)
        """
        self.chip_size = chip_size
        self.overlap = overlap
        self.stride = int(chip_size * (1 - overlap))
    
    def generate(
        self,
        data: np.ndarray,
        transform: Affine,
        crs: rasterio.CRS
    ) -> Generator[Chip, None, None]:
        """
        Generate chips from imagery.
        
        Args:
            data: Image data with shape (bands, height, width)
            transform: Affine transform for the image
            crs: Coordinate reference system
            
        Yields:
            Chip objects with geospatial metadata
        """
        _, height, width = data.shape
        
        # Handle small images: if smaller than chip_size, yield entire image
        if height < self.chip_size or width < self.chip_size:
            # Pad to chip_size if needed, or just use as-is
            yield Chip(
                data=data,
                transform=transform,
                crs=crs,
                chip_id=(0, 0)
            )
            return
        
        row_idx = 0
        for y in range(0, height - self.chip_size + 1, self.stride):
            col_idx = 0
            for x in range(0, width - self.chip_size + 1, self.stride):
                # Extract chip data
                chip_data = data[
                    :,
                    y:y + self.chip_size,
                    x:x + self.chip_size
                ]
                
                # Calculate chip transform
                chip_transform = transform * Affine.translation(x, y)
                
                yield Chip(
                    data=chip_data,
                    transform=chip_transform,
                    crs=crs,
                    chip_id=(row_idx, col_idx)
                )
                
                col_idx += 1
            row_idx += 1
        
        # Handle edge chips (right and bottom edges)
        # Right edge
        if width % self.stride != 0:
            row_idx = 0
            x = width - self.chip_size
            for y in range(0, height - self.chip_size + 1, self.stride):
                chip_data = data[:, y:y + self.chip_size, x:x + self.chip_size]
                chip_transform = transform * Affine.translation(x, y)
                yield Chip(
                    data=chip_data,
                    transform=chip_transform,
                    crs=crs,
                    chip_id=(row_idx, -1)  # -1 indicates edge
                )
                row_idx += 1
        
        # Bottom edge
        if height % self.stride != 0:
            col_idx = 0
            y = height - self.chip_size
            for x in range(0, width - self.chip_size + 1, self.stride):
                chip_data = data[:, y:y + self.chip_size, x:x + self.chip_size]
                chip_transform = transform * Affine.translation(x, y)
                yield Chip(
                    data=chip_data,
                    transform=chip_transform,
                    crs=crs,
                    chip_id=(-1, col_idx)
                )
                col_idx += 1
    
    def estimate_chip_count(self, height: int, width: int) -> int:
        """Estimate number of chips for given dimensions."""
        rows = max(1, (height - self.chip_size) // self.stride + 1)
        cols = max(1, (width - self.chip_size) // self.stride + 1)
        return rows * cols
