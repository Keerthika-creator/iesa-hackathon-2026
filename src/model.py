"""
Dual-Head MobileNetV2 Architecture for Semiconductor Defect Classification
IESA DeepTech Hackathon 2026
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def build_dual_head_model(num_wafer_classes=9, num_die_classes=3):
    """
    Build dual-head MobileNetV2 model for wafer and die classification.
    
    Architecture:
    - Shared backbone: MobileNetV2 (ImageNet pretrained)
    - Wafer head: Dense(256) → Dropout(0.3) → Dense(num_wafer_classes)
    - Die head: Dense(128) → Dropout(0.3) → Dense(num_die_classes)
    
    Args:
        num_wafer_classes (int): Number of wafer defect classes (default: 9)
        num_die_classes (int): Number of die classes (default: 3)
    
    Returns:
        tuple: (model, base_model)
    """
    
    # Input layer for grayscale images
    base_input = layers.Input(shape=(224, 224, 1), name='input_image')
    
    # Convert grayscale to RGB
    x = layers.Concatenate(axis=-1)([base_input, base_input, base_input])
    
    # Shared MobileNetV2 backbone
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False
    
    features = base_model(x, training=False)
    
    # Wafer classification head
    wafer_dense = layers.Dense(256, activation='relu', name='wafer_dense')(features)
    wafer_dropout = layers.Dropout(0.3, name='wafer_dropout')(wafer_dense)
    wafer_output = layers.Dense(
        num_wafer_classes, 
        activation='softmax', 
        name='wafer_output'
    )(wafer_dropout)
    
    # Die classification head
    die_dense = layers.Dense(128, activation='relu', name='die_dense')(features)
    die_dropout = layers.Dropout(0.3, name='die_dropout')(die_dense)
    die_output = layers.Dense(
        num_die_classes, 
        activation='softmax', 
        name='die_output'
    )(die_dropout)
    
    # Build final model
    model = models.Model(
        inputs=base_input,
        outputs=[wafer_output, die_output],
        name='DualHead_MobileNetV2'
    )
    
    return model, base_model


if __name__ == "__main__":
    print("Model architecture module - IESA DeepTech Hackathon 2026")
