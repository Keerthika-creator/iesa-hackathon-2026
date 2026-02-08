"""
Stage-Aware and Tile-Based Inference Engine
IESA DeepTech Hackathon 2026
"""

import numpy as np
import cv2
from typing import Tuple, Union

CONFIDENCE_THRESHOLD = 0.6
TILE_SIZE = 224
TILE_STRIDE = 112


def stage_aware_inference(model, image: np.ndarray, stage: str = 'wafer') -> Tuple[Union[int, str], float]:
    """
    Route inference to correct classification head based on stage.
    
    Args:
        model: Trained dual-head Keras model
        image: Preprocessed input image (224x224x1, normalized 0-1)
        stage: 'wafer' or 'die'
    
    Returns:
        tuple: (predicted_class, confidence)
    """
    
    img_array = np.expand_dims(image, axis=0)
    predictions = model.predict(img_array, verbose=0)
    
    if stage == 'wafer':
        pred = predictions[0][0]
    elif stage == 'die':
        pred = predictions[1][0]
    else:
        raise ValueError(f"Invalid stage: {stage}")
    
    confidence = float(np.max(pred))
    class_idx = int(np.argmax(pred))
    
    if confidence < CONFIDENCE_THRESHOLD:
        return 'UNKNOWN', confidence
    
    return class_idx, confidence


def tile_based_inference(model, large_image: np.ndarray, stage: str = 'wafer') -> Tuple[Union[int, str], float]:
    """
    Sliding-window tile-based inference for high-resolution images.
    
    Args:
        model: Trained dual-head Keras model
        large_image: High-resolution grayscale image (HxW)
        stage: 'wafer' or 'die'
    
    Returns:
        tuple: (final_class, final_confidence)
    """
    
    h, w = large_image.shape[:2]
    tile_predictions = []
    
    for y in range(0, h - TILE_SIZE + 1, TILE_STRIDE):
        for x in range(0, w - TILE_SIZE + 1, TILE_STRIDE):
            tile = large_image[y:y+TILE_SIZE, x:x+TILE_SIZE]
            
            if tile.shape == (TILE_SIZE, TILE_SIZE):
                tile_norm = tile.astype(np.float32) / 255.0
                tile_input = np.expand_dims(tile_norm, axis=-1)
                
                class_idx, conf = stage_aware_inference(model, tile_input, stage)
                tile_predictions.append((class_idx, conf))
    
    if tile_predictions:
        votes = {}
        for cls, conf in tile_predictions:
            if cls not in votes:
                votes[cls] = 0.0
            votes[cls] += conf
        
        final_class = max(votes, key=votes.get)
        final_conf = votes[final_class] / len(tile_predictions)
        
        return final_class, final_conf
    
    return 'UNKNOWN', 0.0


def detect_stage(image_path: str) -> str:
    """Detect whether image is wafer or die based on size heuristics."""
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    h, w = img.shape
    
    if h > 512 or w > 512:
        return 'wafer'
    else:
        return 'die'


if __name__ == "__main__":
    print("Inference module - IESA DeepTech Hackathon 2026")
