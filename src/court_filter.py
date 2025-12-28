"""Geometric filtering to verify tennis court detections."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

import cv2
import numpy as np
from shapely.geometry import Polygon, box
from shapely.validation import make_valid

logger = logging.getLogger(__name__)


@dataclass
class CourtCandidate:
    """A candidate tennis court detection."""
    polygon: Polygon  # Shapely polygon in pixel coordinates
    mask: np.ndarray
    confidence: float
    area_m2: float
    aspect_ratio: float
    rectangularity: float
    line_score: float
    
    @property
    def passes_all_filters(self) -> bool:
        """Check if candidate passes all geometric filters."""
        return (
            self.area_m2 >= CourtFilter.MIN_AREA_M2 and
            self.area_m2 <= CourtFilter.MAX_AREA_M2 and
            abs(self.aspect_ratio - CourtFilter.TARGET_ASPECT) <= CourtFilter.ASPECT_TOLERANCE and
            self.rectangularity >= CourtFilter.MIN_RECTANGULARITY and
            self.line_score >= CourtFilter.MIN_LINE_SCORE
        )


class CourtFilter:
    """
    Filter segmentation results to identify likely tennis courts.
    
    Uses structural and geometric invariants rather than color:
    - Approximate area range (handles practice courts, non-ITF sizes)
    - Aspect ratio ~2:1 with generous tolerance
    - Moderate rectangularity (courts can have irregular boundaries)
    - Optional line detection (some courts have worn/minimal markings)
    
    Note: These are intentionally LOOSE filters. It's better to have 
    false positives that can be manually reviewed than to miss courts.
    Pickleball lines, worn markings, and non-standard dimensions are common.
    """
    
    # Area range: very flexible
    # - Singles court: ~195 m², Doubles: ~260 m²
    # - With runout space: can be 600-800 m²
    # - Practice/mini courts: can be smaller
    MIN_AREA_M2 = 150.0   # Catch small/practice courts
    MAX_AREA_M2 = 1500.0  # Allow courts with surrounding area
    
    # Aspect ratio: generous tolerance
    # - Standard is ~2:1, but practice courts and viewing angles vary
    TARGET_ASPECT = 2.0
    ASPECT_TOLERANCE = 0.8  # Allow 1.2:1 to 2.8:1
    
    # Rectangularity: lowered to handle irregular boundaries
    # - Courts often have curved corners, overlapping lines, shadows
    MIN_RECTANGULARITY = 0.60
    
    # Line detection: DISABLED
    # - At 0.6m/pixel NAIP resolution, tennis court lines (5cm wide) are 
    #   sub-pixel and not detectable with Canny edge detection
    # - The geometry filters (area, aspect, rectangularity) are sufficient
    MIN_LINE_SCORE = 0.0
    
    def __init__(self, meters_per_pixel: float = 0.6):
        """
        Initialize court filter.
        
        Args:
            meters_per_pixel: Ground resolution for area calculation
        """
        self.meters_per_pixel = meters_per_pixel
    
    def filter_candidates(
        self,
        polygons: List[List[Tuple[float, float]]],
        masks: List[np.ndarray],
        scores: List[float],
        image: np.ndarray
    ) -> List[CourtCandidate]:
        """
        Filter segmentation results to find tennis courts.
        
        Args:
            polygons: List of polygon vertices (pixel coordinates)
            masks: List of binary masks
            scores: List of confidence scores
            image: Original RGB image for line detection
            
        Returns:
            List of CourtCandidate objects that pass filtering
        """
        candidates = []
        
        for polygon_pts, mask, score in zip(polygons, masks, scores):
            if polygon_pts is None or len(polygon_pts) < 3:
                continue
            
            try:
                # Create shapely polygon
                geom = Polygon(polygon_pts)
                if not geom.is_valid:
                    geom = make_valid(geom)
                
                if geom.is_empty:
                    continue
                
                # If make_valid returned a MultiPolygon, take the largest part
                if geom.geom_type == 'MultiPolygon':
                    polygon = max(geom.geoms, key=lambda p: p.area)
                elif geom.geom_type == 'Polygon':
                    polygon = geom
                else:
                    logger.debug(f"Skipping non-polygon geometry: {geom.geom_type}")
                    continue
                
                # Calculate metrics
                area_m2 = self._calculate_area_m2(polygon)
                aspect_ratio = self._calculate_aspect_ratio(polygon)
                rectangularity = self._calculate_rectangularity(polygon)
                line_score = self._calculate_line_score(image, mask)
                
                candidate = CourtCandidate(
                    polygon=polygon,
                    mask=mask,
                    confidence=score,
                    area_m2=area_m2,
                    aspect_ratio=aspect_ratio,
                    rectangularity=rectangularity,
                    line_score=line_score
                )
                
                # Only include if passes all filters
                if candidate.passes_all_filters:
                    candidates.append(candidate)
                    logger.debug(
                        f"Court candidate: area={area_m2:.0f}m², "
                        f"aspect={aspect_ratio:.2f}, rect={rectangularity:.2f}, "
                        f"lines={line_score:.2f}"
                    )
                    
            except Exception as e:
                logger.warning(f"Error processing polygon: {e}")
                continue
        
        return candidates
    
    def _calculate_area_m2(self, polygon: Polygon) -> float:
        """Calculate polygon area in square meters."""
        pixel_area = polygon.area
        return pixel_area * (self.meters_per_pixel ** 2)
    
    def _calculate_aspect_ratio(self, polygon: Polygon) -> float:
        """Calculate aspect ratio from minimum bounding rectangle."""
        # Get minimum rotated rectangle
        minx, miny, maxx, maxy = polygon.bounds
        
        # Use convex hull for better rectangle fit
        hull = polygon.convex_hull
        
        # Get oriented bounding box using OpenCV
        if hasattr(hull, 'exterior'):
            coords = np.array(hull.exterior.coords, dtype=np.float32)
        else:
            coords = np.array(polygon.exterior.coords, dtype=np.float32)
        
        if len(coords) < 3:
            return 0.0
        
        rect = cv2.minAreaRect(coords)
        width, height = rect[1]
        
        if width == 0 or height == 0:
            return 0.0
        
        # Return ratio of longer to shorter side
        return max(width, height) / min(width, height)
    
    def _calculate_rectangularity(self, polygon: Polygon) -> float:
        """
        Calculate how rectangular the polygon is.
        
        Rectangularity = polygon_area / minimum_bounding_rectangle_area
        A perfect rectangle has rectangularity = 1.0
        """
        if hasattr(polygon, 'exterior'):
            coords = np.array(polygon.exterior.coords, dtype=np.float32)
        else:
            return 0.0
        
        if len(coords) < 3:
            return 0.0
        
        rect = cv2.minAreaRect(coords)
        rect_area = rect[1][0] * rect[1][1]
        
        if rect_area == 0:
            return 0.0
        
        return polygon.area / rect_area
    
    def _calculate_line_score(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Calculate white-line grid score using edge detection.
        
        Tennis courts have distinctive internal line markings that create
        a regular grid pattern. This function detects these using Canny
        edge detection and Hough line transform.
        """
        # Extract masked region
        if image.shape[:2] != mask.shape:
            # Resize mask if needed
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        
        # Create masked image
        masked = image.copy()
        masked[~mask] = 0
        
        # Convert to grayscale
        if len(masked.shape) == 3:
            gray = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
        else:
            gray = masked
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply mask to edges
        edges[~mask] = 0
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=20,
            maxLineGap=10
        )
        
        if lines is None:
            return 0.0
        
        # Calculate score based on:
        # 1. Number of lines found
        # 2. Line orientations (should be mostly horizontal/vertical)
        
        num_lines = len(lines)
        
        # Count orthogonal lines (horizontal or vertical ±15°)
        orthogonal_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Check if horizontal (0° ± 15°) or vertical (90° ± 15°)
            if angle < 15 or angle > 165 or (75 < angle < 105):
                orthogonal_count += 1
        
        # Score based on line density and orthogonality
        mask_area = np.sum(mask)
        if mask_area == 0:
            return 0.0
        
        line_density = num_lines / (mask_area / 1000)  # Lines per 1000 pixels
        orthogonal_ratio = orthogonal_count / max(1, num_lines)
        
        # Combined score (0-1)
        score = min(1.0, line_density * 0.1) * orthogonal_ratio
        
        return score
