#!/usr/bin/env python3
"""Debug script to test tennis court detection at known locations."""

import logging
import numpy as np
from PIL import Image
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from src.geo_utils import BBox, create_search_bbox
from src.naip_fetcher import NAIPFetcher
from src.tiling import ChipGenerator
from src.segmentation import SAM3Segmenter
from src.court_filter import CourtFilter


def debug_at_location(lat: float, lon: float, name: str, radius_miles: float = 0.3):
    """Debug detection at a specific location."""
    print(f"\n{'='*60}")
    print(f"Debugging: {name}")
    print(f"Coordinates: {lat}, {lon}")
    print(f"{'='*60}")
    
    # Create bbox around the location
    bbox = create_search_bbox((lat, lon), radius_miles)
    print(f"BBox: {bbox}")
    
    # Fetch imagery
    print("\n1. Fetching NAIP imagery...")
    fetcher = NAIPFetcher()
    try:
        naip = fetcher.fetch(bbox)
        print(f"   ✓ Image shape: {naip.data.shape}")
        print(f"   ✓ Imagery date: {naip.imagery_date}")
        
        # Save the raw image for inspection
        rgb = naip.rgb
        print(f"   ✓ RGB shape: {rgb.shape}, dtype: {rgb.dtype}, range: [{rgb.min()}, {rgb.max()}]")
        
        # Save image
        output_dir = Path("debug_output")
        output_dir.mkdir(exist_ok=True)
        
        if rgb.max() > 1:
            rgb_normalized = (rgb / 255.0 * 255).astype(np.uint8)
        else:
            rgb_normalized = (rgb * 255).astype(np.uint8)
        
        Image.fromarray(rgb_normalized).save(output_dir / f"{name.replace(' ', '_')}_raw.png")
        print(f"   ✓ Saved raw image to debug_output/{name.replace(' ', '_')}_raw.png")
        
    except Exception as e:
        print(f"   ✗ Failed to fetch imagery: {e}")
        return
    
    # Generate chips
    print("\n2. Generating chips...")
    chip_gen = ChipGenerator(chip_size=1024, overlap=0.2)
    chips = list(chip_gen.generate(naip.data, naip.transform, naip.crs))
    print(f"   ✓ Generated {len(chips)} chips")
    
    # Run segmentation on each chip
    print("\n3. Running SAM segmentation...")
    segmenter = SAM3Segmenter()
    court_filter = CourtFilter()
    
    all_results = []
    all_candidates = []
    
    for i, chip in enumerate(chips):
        print(f"\n   Chip {i+1}/{len(chips)}:")
        chip_rgb = chip.rgb
        print(f"   - RGB shape: {chip_rgb.shape}, dtype: {chip_rgb.dtype}")
        
        # Save chip image
        if chip_rgb.max() > 1:
            chip_normalized = (chip_rgb / 255.0 * 255).astype(np.uint8)
        else:
            chip_normalized = (chip_rgb * 255).astype(np.uint8)
        Image.fromarray(chip_normalized).save(output_dir / f"{name.replace(' ', '_')}_chip_{i}.png")
        
        # Run segmentation
        results = segmenter.segment_with_text(chip_rgb, text_prompt="tennis court", min_score=0.1)
        print(f"   - SAM found {len(results)} segments")
        
        for j, r in enumerate(results):
            print(f"     * Segment {j}: score={r.score:.3f}, polygon_pts={len(r.polygon) if r.polygon else 0}")
            all_results.append(r)
        
        # Run court filter with verbose output
        if results:
            print(f"\n   4. Running court filter on {len(results)} segments...")
            
            polygons = [r.polygon for r in results if r.polygon]
            masks = [r.mask for r in results if r.polygon]
            scores = [r.score for r in results if r.polygon]
            
            print(f"   - Valid polygons: {len(polygons)}")
            
            # Check each polygon manually
            from shapely.geometry import Polygon
            from shapely.validation import make_valid
            
            for k, (pts, mask, score) in enumerate(zip(polygons, masks, scores)):
                if pts is None or len(pts) < 3:
                    print(f"     Polygon {k}: Invalid (no points or < 3 pts)")
                    continue
                
                try:
                    poly = Polygon(pts)
                    if not poly.is_valid:
                        poly = make_valid(poly)
                    
                    if poly.geom_type == 'MultiPolygon':
                        poly = max(poly.geoms, key=lambda p: p.area)
                    elif poly.geom_type != 'Polygon':
                        print(f"     Polygon {k}: Skipped ({poly.geom_type})")
                        continue
                    
                    area_m2 = court_filter._calculate_area_m2(poly)
                    aspect = court_filter._calculate_aspect_ratio(poly)
                    rect = court_filter._calculate_rectangularity(poly)
                    lines = court_filter._calculate_line_score(chip_rgb, mask)
                    
                    print(f"     Polygon {k}:")
                    print(f"       Area: {area_m2:.1f} m² (min={court_filter.MIN_AREA_M2}, max={court_filter.MAX_AREA_M2})")
                    print(f"       Aspect: {aspect:.2f} (target={court_filter.TARGET_ASPECT} ± {court_filter.ASPECT_TOLERANCE})")
                    print(f"       Rect: {rect:.3f} (min={court_filter.MIN_RECTANGULARITY})")
                    print(f"       Lines: {lines:.3f} (min={court_filter.MIN_LINE_SCORE})")
                    
                    passes_area = court_filter.MIN_AREA_M2 <= area_m2 <= court_filter.MAX_AREA_M2
                    passes_aspect = abs(aspect - court_filter.TARGET_ASPECT) <= court_filter.ASPECT_TOLERANCE
                    passes_rect = rect >= court_filter.MIN_RECTANGULARITY
                    passes_lines = lines >= court_filter.MIN_LINE_SCORE
                    
                    print(f"       Passes: area={passes_area}, aspect={passes_aspect}, rect={passes_rect}, lines={passes_lines}")
                    
                except Exception as e:
                    print(f"     Polygon {k}: Error - {e}")
            
            candidates = court_filter.filter_candidates(polygons, masks, scores, chip_rgb)
            print(f"   - Court filter passed: {len(candidates)} candidates")
            all_candidates.extend(candidates)
    
    print(f"\n{'='*60}")
    print(f"Summary for {name}:")
    print(f"  Total SAM segments: {len(all_results)}")
    print(f"  Valid candidates: {len(all_candidates)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Deep Run High School tennis courts
    # 37°40′29″N 77°35′55″W -> 37.6747, -77.5986
    debug_at_location(37.6747, -77.5986, "Deep Run HS")
    
    # Twin Hickory Community Center (approximately)
    # 5011 Twin Hickory Road, Glen Allen, VA 23059
    # Approximate coords: 37.671, -77.599
    debug_at_location(37.671, -77.599, "Twin Hickory CC")
