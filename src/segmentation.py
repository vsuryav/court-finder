"""SAM 3 segmentation wrapper with device auto-detection."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

import numpy as np

# Torch is optional - only required when actually running segmentation
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Result from SAM 3 segmentation."""
    mask: np.ndarray  # Binary mask (height, width)
    score: float  # Confidence score
    polygon: Optional[List[Tuple[float, float]]] = None  # Polygon vertices (pixel coords)


def get_device():
    """
    Auto-detect the best available device.
    
    Returns:
        torch.device for CUDA, MPS, or CPU, or None if torch unavailable
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not installed - segmentation unavailable")
        return None
        
    if torch.cuda.is_available():
        logger.info("Using CUDA device")
        return torch.device("cuda")
    elif TORCH_AVAILABLE and hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using MPS device (Apple Silicon)")
        return torch.device("mps")
    else:
        logger.warning("No GPU available, using CPU (this will be slow)")
        return torch.device("cpu")


class SAM3Segmenter:
    """
    SAM segmenter using HuggingFace transformers.
    
    Uses SAM 2 from transformers which works on Apple Silicon MPS.
    For tennis court detection, we use automatic mask generation
    and filter results by geometry.
    """
    
    def __init__(
        self,
        device = None,
        model_name: str = "facebook/sam2-hiera-large"
    ):
        """
        Initialize SAM segmenter.
        
        Args:
            device: Torch device to use (auto-detected if None)
            model_name: HuggingFace model name
        """
        self.device = device or get_device()
        self.model_name = model_name
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load SAM model. Called lazily on first use."""
        if self.model is not None:
            return
        
        try:
            from transformers import Sam2Model, Sam2Processor
            
            logger.info(f"Loading SAM 2 model: {self.model_name}")
            
            self.processor = Sam2Processor.from_pretrained(self.model_name)
            self.model = Sam2Model.from_pretrained(self.model_name)
            
            if self.device:
                self.model.to(self.device)
            self.model.eval()
            
            logger.info("SAM 2 model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load SAM 2: {e}")
            raise
    
    def segment_with_text(
        self,
        image: np.ndarray,
        text_prompt: str = "tennis court",
        min_score: float = 0.3
    ) -> List[SegmentationResult]:
        """
        Segment objects in the image using SAM 2.
        
        Note: SAM 2 doesn't support text prompts directly. We use
        grid-based point prompts to generate masks across the image.
        
        Args:
            image: RGB image array (height, width, 3) with values 0-255
            text_prompt: Ignored for SAM 2 (no text prompt support)
            min_score: Minimum confidence score to include
            
        Returns:
            List of SegmentationResult for each detected object
        """
        self.load_model()
        
        from PIL import Image
        import torch
        import cv2
        
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        
        # Convert to PIL
        pil_image = Image.fromarray(image)
        height, width = image.shape[:2]
        
        # Generate grid of point prompts
        if self.processor is None or self.model is None:
            raise RuntimeError("SAM 2 model or processor not loaded")

        # Sample points across the image to find potential objects
        grid_size = 4  # 4x4 grid = 16 points
        points = []
        for y in range(grid_size):
            for x in range(grid_size):
                px = int((x + 0.5) * width / grid_size)
                py = int((y + 0.5) * height / grid_size)
                points.append([px, py])
        
        results = []
        
        # Process points in batches
        # SAM 2 input_points format: [image_level][object_level][point_level][coords]
        # For single image, single object per point: [[[[x, y]]]]
        for point in points:
            try:
                # Format: [image][object][points][coords]
                input_points = [[[[point[0], point[1]]]]]
                
                inputs = self.processor(
                    pil_image, 
                    input_points=input_points,
                    return_tensors="pt"
                )
                
                if self.device:
                    inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Get best mask (highest iou score)
                # pred_masks shape: [batch, num_objects, num_masks, H, W]
                # iou_scores shape: [batch, num_objects, num_masks]
                scores = outputs.iou_scores[0][0].cpu().numpy()  # [3]
                masks_raw = outputs.pred_masks[0][0]  # [3, 256, 256]
                
                best_idx = scores.argmax()
                best_score = scores[best_idx]
                
                if best_score >= min_score:
                    # Upscale mask from 256x256 to original size
                    mask_256 = masks_raw[best_idx].cpu().numpy()
                    mask_upscaled = cv2.resize(
                        mask_256, 
                        (width, height), 
                        interpolation=cv2.INTER_LINEAR
                    )
                    mask_bool = mask_upscaled > 0  # Threshold to boolean
                    
                    polygon = self._mask_to_polygon(mask_bool)
                    
                    if polygon and len(polygon) >= 4:
                        results.append(SegmentationResult(
                            mask=mask_bool,
                            score=float(best_score),
                            polygon=polygon
                        ))
            except Exception as e:
                logger.debug(f"Point prompt failed: {e}")
                continue
        
        return results
    
    def _mask_to_polygon(
        self,
        mask: np.ndarray
    ) -> Optional[List[Tuple[float, float]]]:
        """Convert binary mask to polygon vertices."""
        import cv2
        
        # Find contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Simplify polygon
        epsilon = 0.01 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        
        # Convert to list of tuples
        return [(float(pt[0][0]), float(pt[0][1])) for pt in approx]


class MockSAM3Segmenter(SAM3Segmenter):
    """
    Mock SAM 3 segmenter for testing without the actual model.
    
    Generates random rectangular regions as placeholder detections.
    """
    
    def load_model(self):
        """No model to load in mock mode."""
        logger.info("Using MockSAM3Segmenter (no actual model)")
    
    def segment_with_text(
        self,
        image: np.ndarray,
        text_prompt: str = "tennis court",
        min_score: float = 0.5
    ) -> List[SegmentationResult]:
        """Generate mock segmentation results."""
        import cv2
        
        # Guard against empty or invalid images
        if image is None or image.size == 0:
            logger.warning("Empty image received, returning no results")
            return []
        
        # Use simple color/edge detection as placeholder
        # This is NOT how real SAM 3 works - just for testing
        height, width = image.shape[:2]
        
        if height == 0 or width == 0:
            return []
        
        # Convert to grayscale and detect rectangles
        if len(image.shape) == 3 and image.shape[2] >= 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 2:
            gray = image
        else:
            logger.warning(f"Unexpected image shape: {image.shape}")
            return []
        
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size (tennis court should be significant portion of 1024x1024 chip)
            if area < 5000 or area > 100000:
                continue
            
            # Get polygon approximation
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Need at least 4 points for a valid polygon
            if len(approx) < 4:
                continue
            
            polygon = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
            
            # Create mask
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, -1)
            mask = mask.astype(bool)
            
            results.append(SegmentationResult(
                mask=mask,
                score=0.7,  # Mock confidence
                polygon=polygon
            ))
        
        return results[:10]  # Limit results
