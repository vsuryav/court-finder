#!/usr/bin/env python3
"""Find tennis courts near a specific address and sort by distance."""

import json
import math
from pathlib import Path

from src.geo_utils import create_search_bbox
from src.pipeline import CourtFinderPipeline


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in miles."""
    R = 3958.8  # Earth's radius in miles
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def get_polygon_center(geojson_geometry):
    """Get center of a polygon from GeoJSON geometry."""
    coords = geojson_geometry["coordinates"][0]  # First ring of polygon
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (sum(lats) / len(lats), sum(lons) / len(lons))


def find_nearest_courts(origin_lat: float, origin_lon: float, radius_miles: float = 5, top_n: int = 10):
    """Find the N closest tennis courts to a given location."""
    print(f"🎾 Finding tennis courts near ({origin_lat}, {origin_lon})")
    print(f"   Search radius: {radius_miles} miles")
    print(f"   Looking for top {top_n} closest courts")
    print()
    
    # Create pipeline
    pipeline = CourtFinderPipeline(use_mock_segmenter=False)
    
    # Create bbox and search
    bbox = create_search_bbox((origin_lat, origin_lon), radius_miles)
    
    # Run search using the pipeline directly with coordinates
    from src.naip_fetcher import NAIPFetcher
    from src.tiling import ChipGenerator
    from src.court_filter import CourtFilter
    from src.nms import world_nms_merge
    from shapely.geometry import mapping
    
    print("Fetching NAIP imagery...")
    fetcher = NAIPFetcher()
    imagery_date = fetcher.get_imagery_date(bbox)
    if not imagery_date:
        print("No imagery available for this area!")
        return
    
    print(f"Imagery date: {imagery_date}")
    naip = fetcher.fetch(bbox)
    
    print("Generating chips...")
    chip_gen = ChipGenerator(chip_size=1024, overlap=0.2)
    chips = list(chip_gen.generate(naip.data, naip.transform, naip.crs))
    print(f"Processing {len(chips)} chips...")
    
    # Process
    all_candidates = []
    court_filter = CourtFilter()
    
    for i, chip in enumerate(chips):
        print(f"  Chip {i+1}/{len(chips)}...", end="\r")
        results = pipeline.segmenter.segment_with_text(chip.rgb, "tennis court", min_score=0.1)
        
        if results:
            valid_results = [r for r in results if r.polygon is not None]
            candidates = court_filter.filter_candidates(
                [r.polygon for r in valid_results],
                [r.mask for r in valid_results],
                [r.score for r in valid_results],
                chip.rgb
            )
            for c in candidates:
                all_candidates.append((c, chip.pixel_to_world))
    
    print(f"\nFound {len(all_candidates)} candidates before NMS")
    
    # Merge
    merged = world_nms_merge(all_candidates, iou_threshold=0.5)
    print(f"After NMS: {len(merged)} courts")
    
    # Calculate distances and sort
    courts_with_distance = []
    for court in merged:
        geojson_geom = mapping(court.polygon)
        center_lat, center_lon = get_polygon_center(geojson_geom)
        distance = haversine_distance(origin_lat, origin_lon, center_lat, center_lon)
        courts_with_distance.append({
            "distance_miles": distance,
            "center": (center_lat, center_lon),
            "area_m2": court.area_m2,
            "confidence": court.confidence,
            "geometry": geojson_geom
        })
    
    # Sort by distance
    courts_with_distance.sort(key=lambda x: x["distance_miles"])
    
    # Print top N
    print(f"\n{'='*60}")
    print(f"Top {min(top_n, len(courts_with_distance))} Closest Tennis Courts")
    print(f"{'='*60}")
    
    for i, court in enumerate(courts_with_distance[:top_n]):
        print(f"\n{i+1}. Distance: {court['distance_miles']:.2f} miles")
        print(f"   Center: ({court['center'][0]:.6f}, {court['center'][1]:.6f})")
        print(f"   Area: {court['area_m2']:.0f} m²")
        print(f"   Confidence: {court['confidence']:.2f}")
        print(f"   Google Maps: https://www.google.com/maps?q={court['center'][0]},{court['center'][1]}")
    
    # Save full results
    output = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": c["geometry"],
                "properties": {
                    "rank": i + 1,
                    "distance_miles": round(c["distance_miles"], 2),
                    "area_m2": round(c["area_m2"], 1),
                    "confidence": round(c["confidence"], 2)
                }
            }
            for i, c in enumerate(courts_with_distance[:top_n])
        ],
        "properties": {
            "origin": {"lat": origin_lat, "lon": origin_lon},
            "search_radius_miles": radius_miles,
            "imagery_date": str(imagery_date)
        }
    }
    
    with open("nearest_courts.geojson", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to nearest_courts.geojson")
    print(f"  View at: https://geojson.io")
    
    return courts_with_distance[:top_n]


if __name__ == "__main__":
    # 755 North Ave Apartments, Atlanta, GA
    # Coordinates: 33.772590, -84.364440
    find_nearest_courts(33.772590, -84.364440, radius_miles=5, top_n=10)
