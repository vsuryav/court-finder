"""Main pipeline orchestrating court detection from NAIP imagery."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Callable
import json
import logging

from shapely.geometry import mapping

from .geo_utils import BBox, zipcode_to_center, create_search_bbox
from .naip_fetcher import NAIPFetcher, NAIPImage
from .tiling import ChipGenerator, Chip
from .segmentation import SAM3Segmenter, MockSAM3Segmenter, SegmentationResult
from .court_filter import CourtFilter, CourtCandidate
from .cache import ResultCache, CachedCourt
from .nms import nms_merge, world_nms_merge

logger = logging.getLogger(__name__)


@dataclass
class CourtDetection:
    """A detected tennis court with world coordinates."""
    polygon_geojson: Dict[str, Any]  # GeoJSON geometry
    confidence: float
    area_m2: float
    aspect_ratio: float
    
    def to_feature(self, properties: Optional[Dict] = None) -> Dict:
        """Convert to GeoJSON Feature."""
        props = {
            "confidence": self.confidence,
            "area_m2": round(self.area_m2, 1),
            "aspect_ratio": round(self.aspect_ratio, 2),
            "type": "tennis_court"
        }
        if properties:
            props.update(properties)
        
        return {
            "type": "Feature",
            "geometry": self.polygon_geojson,
            "properties": props
        }


class CourtFinderPipeline:
    """
    Main pipeline for detecting tennis courts from NAIP imagery.
    
    Pipeline flow:
    1. Check cache for existing results
    2. Convert zipcode → bounding box
    3. Fetch NAIP imagery
    4. Generate image chips
    5. Run SAM 3 segmentation
    6. Apply geometric filters
    7. NMS merge overlapping detections
    8. Cache results
    9. Return GeoJSON
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_mock_segmenter: bool = False
    ):
        """
        Initialize the pipeline.
        
        Args:
            cache_dir: Directory for caching imagery and results
            use_mock_segmenter: Use mock segmenter for testing
        """
        self.cache = ResultCache()
        self.fetcher = NAIPFetcher(cache_dir)
        self.chip_generator = ChipGenerator(chip_size=1024, overlap=0.2)
        self.court_filter = CourtFilter()
        
        self.segmenter: Union[SAM3Segmenter, MockSAM3Segmenter]
        # Use mock segmenter for testing without SAM 3 installed
        if use_mock_segmenter:
            self.segmenter = MockSAM3Segmenter()
        else:
            self.segmenter = SAM3Segmenter()
    
    def search(
        self,
        zipcode: str,
        radius_miles: float,
        skip_cache: bool = False,
        text_prompt: str = "tennis court",
        progress_callback: Optional[Callable[[str, float, Optional[int], Optional[int]], None]] = None
    ) -> Dict[str, Any]:
        """
        Search for tennis courts in an area.
        
        Args:
            zipcode: US zipcode to center search on
            radius_miles: Search radius in miles
            skip_cache: If True, bypass cache and reprocess
            text_prompt: Text prompt for SAM 3 segmentation
            progress_callback: Callable for progress updates.
                Signature: (description, progress_fraction, current_step, total_steps)
            
        Returns:
            GeoJSON FeatureCollection of detected courts
        """
        def _update_progress(desc: str, frac: float = 0.0, current: int = 0, total: int = 0):
            if progress_callback:
                progress_callback(desc, frac, current, total)
        
        logger.info(f"Searching for tennis courts: {zipcode}, {radius_miles} miles")
        _update_progress("Finding coordinates...", 0.05)
        
        # Step 1: Convert zipcode to bbox
        center = zipcode_to_center(zipcode)
        bbox = create_search_bbox(center, radius_miles)
        logger.info(f"Search bbox: {bbox}")
        
        _update_progress("Checking imagery availability...", 0.1)
        
        # Step 2: Check imagery date for cache validation
        imagery_date = self.fetcher.get_imagery_date(bbox)
        if imagery_date is None:
            raise ValueError(f"No NAIP imagery available for {zipcode}")
        
        logger.info(f"NAIP imagery date: {imagery_date}")
        _update_progress("Checking cache...", 0.15)
        
        # Step 3: Check cache
        if not skip_cache:
            cached = self.cache.get_cached(bbox, imagery_date)
            if cached is not None:
                logger.info(f"Cache hit: {len(cached)} courts")
                _update_progress("Cache hit, finalizing results...", 1.0)
                return self._cached_to_geojson(cached, zipcode, radius_miles)
        
        # Step 4: Fetch imagery
        logger.info("Fetching NAIP imagery...")
        _update_progress("Fetching NAIP imagery from Planetary Computer...", 0.3)
        naip_image = self.fetcher.fetch(bbox)
        
        # Step 5: Generate chips
        _update_progress("Generating image chips...", 0.4)
        chips = list(self.chip_generator.generate(
            naip_image.data,
            naip_image.transform,
            naip_image.crs
        ))
        logger.info(f"Generated {len(chips)} image chips")
        
        # Step 6: Process chips with SAM 3
        all_candidates = []
        total_chips = len(chips)
        
        for i, chip in enumerate(chips):
            step_desc = f"Processing chip {i+1}/{total_chips}..."
            logger.debug(step_desc)
            _update_progress(step_desc, 0.4 + (i / total_chips) * 0.5, i + 1, total_chips)
            
            # Run segmentation
            results = self.segmenter.segment_with_text(
                chip.rgb,
                text_prompt=text_prompt
            )
            
            if not results:
                continue
            
            # Apply geometric filtering
            # Ensure we only pass valid polygons (filter out None)
            valid_results = [r for r in results if r.polygon is not None]
            
            candidates = self.court_filter.filter_candidates(
                polygons=[r.polygon for r in valid_results if r.polygon is not None],
                masks=[r.mask for r in valid_results],
                scores=[r.score for r in valid_results],
                image=chip.rgb
            )
            
            # Store with chip transform for world coordinate conversion
            for candidate in candidates:
                all_candidates.append((candidate, chip.pixel_to_world))
        
        _update_progress("Merging overlapping detections...", 0.95)
        logger.info(f"Found {len(all_candidates)} candidates before NMS")
        
        # Step 7: NMS merge in world coordinates
        merged = world_nms_merge(all_candidates, iou_threshold=0.5)
        logger.info(f"After NMS: {len(merged)} courts")
        
        # Step 8: Convert to detections
        _update_progress("Finalizing results...", 0.98)
        detections = []
        cached_courts = []
        
        for candidate in merged:
            # Convert polygon to GeoJSON
            geojson_geom = mapping(candidate.polygon)
            
            detection = CourtDetection(
                polygon_geojson=geojson_geom,
                confidence=candidate.confidence,
                area_m2=candidate.area_m2,
                aspect_ratio=candidate.aspect_ratio
            )
            detections.append(detection)
            
            # Prepare for caching
            cached_courts.append(CachedCourt.from_polygon(
                candidate.polygon,
                candidate.confidence,
                candidate.area_m2,
                candidate.aspect_ratio
            ))
        
        # Step 9: Cache results
        self.cache.store(bbox, cached_courts, imagery_date)
        
        # Step 10: Return GeoJSON
        return self._detections_to_geojson(
            detections, zipcode, radius_miles, imagery_date
        )
    
    def _detections_to_geojson(
        self,
        detections: List[CourtDetection],
        zipcode: str,
        radius_miles: float,
        imagery_date: date
    ) -> Dict[str, Any]:
        """Convert detections to GeoJSON FeatureCollection."""
        features = [d.to_feature() for d in detections]
        
        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "zipcode": zipcode,
                "radius_miles": radius_miles,
                "imagery_date": imagery_date.isoformat(),
                "court_count": len(features),
                "generated_by": "court-finder"
            }
        }
    
    def _cached_to_geojson(
        self,
        cached: List[CachedCourt],
        zipcode: str,
        radius_miles: float
    ) -> Dict[str, Any]:
        """Convert cached results to GeoJSON."""
        features = []
        
        for court in cached:
            polygon = court.to_polygon()
            feature = {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {
                    "confidence": court.confidence,
                    "area_m2": round(court.area_m2, 1),
                    "aspect_ratio": round(court.aspect_ratio, 2),
                    "type": "tennis_court",
                    "cached": True
                }
            }
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "zipcode": zipcode,
                "radius_miles": radius_miles,
                "court_count": len(features),
                "from_cache": True,
                "generated_by": "court-finder"
            }
        }


def save_geojson(geojson: Dict[str, Any], output_path: Path):
    """Save GeoJSON to file."""
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    logger.info(f"Saved GeoJSON to {output_path}")
