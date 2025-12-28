"""Non-Maximum Suppression for merging overlapping detections."""

from typing import List, Tuple, Callable, Union, Optional
import logging

from shapely.geometry import Polygon
from shapely.ops import unary_union

from .court_filter import CourtCandidate

logger = logging.getLogger(__name__)


def calculate_iou(poly1: Polygon, poly2: Polygon) -> float:
    """
    Calculate Intersection over Union between two polygons.
    
    Args:
        poly1: First polygon
        poly2: Second polygon
        
    Returns:
        IoU value between 0 and 1
    """
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    
    try:
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        if union == 0:
            return 0.0
        
        return intersection / union
    except Exception:
        return 0.0


def nms_merge(
    candidates: List[CourtCandidate],
    iou_threshold: float = 0.5
) -> List[CourtCandidate]:
    """
    Apply Non-Maximum Suppression to merge overlapping detections.
    
    When the same court appears in overlapping image chips, this
    merges them into a single detection, keeping the one with
    higher confidence.
    
    Args:
        candidates: List of court candidates (from multiple chips)
        iou_threshold: IoU threshold for considering overlap
        
    Returns:
        Merged list of candidates
    """
    if not candidates:
        return []
    
    # Sort by confidence (descending)
    sorted_candidates = sorted(
        candidates, 
        key=lambda c: c.confidence, 
        reverse=True
    )
    
    merged = []
    used = set()
    
    for i, candidate in enumerate(sorted_candidates):
        if i in used:
            continue
        
        # Find all overlapping candidates
        overlapping_indices = [i]
        
        for j in range(i + 1, len(sorted_candidates)):
            if j in used:
                continue
            
            iou = calculate_iou(candidate.polygon, sorted_candidates[j].polygon)
            
            if iou >= iou_threshold:
                overlapping_indices.append(j)
                used.add(j)
        
        if len(overlapping_indices) == 1:
            # No overlaps, keep as-is
            merged.append(candidate)
        else:
            # Merge overlapping polygons
            merged_candidate = _merge_candidates(
                [sorted_candidates[idx] for idx in overlapping_indices]
            )
            merged.append(merged_candidate)
        
        used.add(i)
    
    logger.info(f"NMS: {len(candidates)} candidates -> {len(merged)} merged")
    
    return merged


def _merge_candidates(candidates: List[CourtCandidate]) -> CourtCandidate:
    """
    Merge multiple overlapping candidates into one.
    
    Uses the highest-confidence candidate's metrics but
    optionally unions the polygons for a cleaner boundary.
    """
    # Use the highest confidence candidate as base
    best = candidates[0]  # Already sorted by confidence
    
    # Optionally: merge polygon boundaries
    # This creates a smoother boundary when court spans chip edges
    if len(candidates) > 1:
        try:
            polygons = [c.polygon for c in candidates]
            merged_polygon = unary_union(polygons)
            
            # If union results in MultiPolygon, take largest
            if merged_polygon.geom_type == 'MultiPolygon':
                from shapely.geometry import MultiPolygon
                if isinstance(merged_polygon, MultiPolygon):
                    merged_polygon = max(
                        merged_polygon.geoms, 
                        key=lambda p: p.area
                    )
                else:
                    # Defensive: if it says it's MultiPolygon but isn't instance
                    # (sometimes happens with different shapely versions/wrappers)
                    pass
            
            # Create new candidate with merged polygon
            return CourtCandidate(
                polygon=merged_polygon,
                mask=best.mask,  # Keep best mask
                confidence=best.confidence,
                area_m2=best.area_m2,
                aspect_ratio=best.aspect_ratio,
                rectangularity=best.rectangularity,
                line_score=best.line_score
            )
        except Exception as e:
            logger.warning(f"Polygon merge failed: {e}")
    
    return best


def world_nms_merge(
    candidates_with_transforms: List[Tuple[CourtCandidate, Callable[[float, float], Tuple[float, float]]]],
    iou_threshold: float = 0.5
) -> List[CourtCandidate]:
    """
    NMS merge in world coordinates.
    
    Converts all polygons to world coordinates before comparing,
    which handles cases where chips from different areas might
    have similar pixel-space polygons.
    
    Args:
        candidates_with_transforms: List of (candidate, transform_func) tuples
            where transform_func converts pixel coords to world coords
        iou_threshold: IoU threshold for overlap
        
    Returns:
        Merged candidates with polygons in world coordinates
    """
    # Transform all polygons to world coordinates
    world_candidates = []
    
    for candidate, transform in candidates_with_transforms:
        try:
            # Transform polygon vertices
            world_coords = [
                transform(x, y) 
                for x, y in candidate.polygon.exterior.coords
            ]
            world_polygon = Polygon(world_coords)
            
            # Create new candidate with world polygon
            world_candidate = CourtCandidate(
                polygon=world_polygon,
                mask=candidate.mask,
                confidence=candidate.confidence,
                area_m2=candidate.area_m2,
                aspect_ratio=candidate.aspect_ratio,
                rectangularity=candidate.rectangularity,
                line_score=candidate.line_score
            )
            world_candidates.append(world_candidate)
            
        except Exception as e:
            logger.warning(f"Transform failed: {e}")
    
    return nms_merge(world_candidates, iou_threshold)
