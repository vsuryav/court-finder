# Tennis Court Finder

Detect tennis courts from NAIP aerial imagery using SAM 3 segmentation.

## Features

- **Zipcode + Radius Search**: Specify a US zipcode and mile radius to search
- **SAM 3 Text Prompts**: Uses "tennis court" concept segmentation
- **Geometric Filtering**: ITF standard dimensions, aspect ratio, line detection
- **Smart Caching**: Results cached indefinitely until NAIP imagery updates
- **GeoJSON Output**: Compatible with QGIS, Leaflet, geojson.io

## Requirements

- Python 3.12+
- PyTorch 2.7+
- CUDA (recommended) or MPS (Apple Silicon)

## Installation

```bash
pip install -e .
```

For SAM 3, follow installation from [facebookresearch/sam3](https://github.com/facebookresearch/sam3).

## Usage

```bash
# Search for tennis courts
python cli.py search --zipcode 30306 --radius 2 --output courts.geojson

# Skip cache
python cli.py search --zipcode 30306 --radius 1 --no-cache

# View cache stats
python cli.py cache stats

# Clear cache
python cli.py cache clear
```

## Output

Results are saved as GeoJSON with court polygons:
- Open in [geojson.io](https://geojson.io) for quick visualization
- Import into QGIS for overlay with basemaps
