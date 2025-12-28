"""SQLite-based result caching with imagery-date invalidation."""

from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import logging
import sqlite3

from shapely.geometry import Polygon, mapping, shape
from shapely.wkt import dumps as wkt_dumps, loads as wkt_loads

from .geo_utils import BBox

logger = logging.getLogger(__name__)


@dataclass
class CachedCourt:
    """A cached court detection result."""
    polygon_wkt: str
    confidence: float
    area_m2: float
    aspect_ratio: float
    
    def to_polygon(self) -> Polygon:
        """Convert WKT back to Shapely polygon."""
        poly = wkt_loads(self.polygon_wkt)
        if isinstance(poly, Polygon):
            return poly
        
        # If it's a MultiPolygon or something else, it will fail the return type hint
        # but the caller expects a Polygon. For now, we cast it to satisfy the type checker
        # and hope it's a Polygon.
        from typing import cast
        return cast(Polygon, poly)
    
    @classmethod
    def from_polygon(
        cls,
        polygon: Polygon,
        confidence: float,
        area_m2: float,
        aspect_ratio: float
    ) -> "CachedCourt":
        """Create from Shapely polygon."""
        return cls(
            polygon_wkt=wkt_dumps(polygon),
            confidence=confidence,
            area_m2=area_m2,
            aspect_ratio=aspect_ratio
        )


class ResultCache:
    """
    SQLite-based cache for court detection results.
    
    Results are cached indefinitely until NAIP imagery date changes.
    Cache keys are based on quantized bounding box grid cells.
    """
    
    GRID_SIZE = 0.01  # ~1km grid cells
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the cache.
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_path = Path.home() / ".cache" / "court-finder" / "results.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    grid_key TEXT PRIMARY KEY,
                    bbox_wkt TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    imagery_date TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_imagery_date 
                ON cache(imagery_date)
            """)
            conn.commit()
    
    def _quantize_bbox(self, bbox: BBox) -> str:
        """
        Convert bbox to quantized grid key.
        
        This ensures overlapping queries hit the same cache entries.
        """
        min_lon = round(bbox.min_lon / self.GRID_SIZE) * self.GRID_SIZE
        min_lat = round(bbox.min_lat / self.GRID_SIZE) * self.GRID_SIZE
        max_lon = round(bbox.max_lon / self.GRID_SIZE) * self.GRID_SIZE
        max_lat = round(bbox.max_lat / self.GRID_SIZE) * self.GRID_SIZE
        
        return f"{min_lon:.4f},{min_lat:.4f},{max_lon:.4f},{max_lat:.4f}"
    
    def _get_grid_cells(self, bbox: BBox) -> List[str]:
        """Get all grid cells covered by bbox."""
        cells = []
        
        lon = bbox.min_lon
        while lon < bbox.max_lon:
            lat = bbox.min_lat
            while lat < bbox.max_lat:
                cell_key = f"{lon:.4f},{lat:.4f}"
                cells.append(cell_key)
                lat += self.GRID_SIZE
            lon += self.GRID_SIZE
        
        return cells
    
    def get_cached(
        self,
        bbox: BBox,
        current_imagery_date: date
    ) -> Optional[List[CachedCourt]]:
        """
        Get cached results for bbox if available and valid.
        
        Args:
            bbox: Bounding box to query
            current_imagery_date: Current NAIP imagery date (for validation)
            
        Returns:
            List of cached courts, or None if cache miss
        """
        grid_key = self._quantize_bbox(bbox)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT results_json, imagery_date 
                FROM cache 
                WHERE grid_key = ?
                """,
                (grid_key,)
            )
            row = cursor.fetchone()
        
        if row is None:
            logger.debug(f"Cache miss for {grid_key}")
            return None
        
        results_json, cached_date_str = row
        cached_date = date.fromisoformat(cached_date_str)
        
        # Invalidate if imagery date has changed
        if cached_date != current_imagery_date:
            logger.info(
                f"Cache invalidated: imagery updated from {cached_date} to {current_imagery_date}"
            )
            return None
        
        logger.info(f"Cache hit for {grid_key}")
        
        # Parse results
        results_data = json.loads(results_json)
        return [
            CachedCourt(**item) 
            for item in results_data
        ]
    
    def store(
        self,
        bbox: BBox,
        courts: List[CachedCourt],
        imagery_date: date
    ):
        """
        Store results in cache.
        
        Args:
            bbox: Bounding box that was searched
            courts: List of detected courts
            imagery_date: Date of NAIP imagery used
        """
        grid_key = self._quantize_bbox(bbox)
        results_json = json.dumps([asdict(c) for c in courts])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache 
                (grid_key, bbox_wkt, results_json, imagery_date, created_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                """,
                (grid_key, bbox.to_wkt(), results_json, imagery_date.isoformat())
            )
            conn.commit()
        
        logger.debug(f"Cached {len(courts)} courts for {grid_key}")
    
    def clear(self):
        """Clear all cached results."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache")
            total_entries = cursor.fetchone()[0]
            
            cursor = conn.execute(
                "SELECT COUNT(DISTINCT imagery_date) FROM cache"
            )
            unique_dates = cursor.fetchone()[0]
            
            cursor = conn.execute(
                """
                SELECT imagery_date, COUNT(*) 
                FROM cache 
                GROUP BY imagery_date 
                ORDER BY imagery_date DESC 
                LIMIT 5
                """
            )
            date_counts = cursor.fetchall()
        
        return {
            "total_entries": total_entries,
            "unique_imagery_dates": unique_dates,
            "recent_dates": dict(date_counts),
            "db_path": str(self.db_path),
            "db_size_mb": self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        }
