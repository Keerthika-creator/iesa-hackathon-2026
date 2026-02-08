"""
ONNX Model Conversion for NXP eIQ Deployment
IESA DeepTech Hackathon 2026
"""

import tensorflow as tf
from tensorflow import keras
import tf2onnx
import onnx
import os


def convert_to_onnx(keras_model_path: str, onnx_output_path: str, 
                    opset_version: int = 13) -> dict:
    """
    Convert Keras model to ONNX format for NXP eIQ deployment.
    
    Args:
        keras_model_path: Path to .h5 Keras model file
        onnx_output_path: Output path for .onnx file
        opset_version: ONNX opset version (default: 13)
    
    Returns:
        dict: Conversion statistics
    """
    
    print("="*60)
    print("KERAS → ONNX CONVERSION")
    print("="*60)
    
    # Load Keras model
    print(f"\nLoading Keras model from {keras_model_path}...")
    model = keras.models.load_model(keras_model_path)
    print(f"✓ Model loaded: {model.count_params():,} parameters")
    
    # Define input signature
    spec = (tf.TensorSpec((None, 224, 224, 1), tf.float32, name="input"),)
    
    # Convert to ONNX
    print(f"\nConverting to ONNX (opset {opset_version})...")
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=opset_version,
        output_path=onnx_output_path
    )
    print(f"✓ ONNX model saved to {onnx_output_path}")
    
    # Verify ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified successfully")
    
    # Statistics
    keras_size_mb = os.path.getsize(keras_model_path) / (1024 * 1024)
    onnx_size_mb = os.path.getsize(onnx_output_path) / (1024 * 1024)
    
    stats = {
        'keras_size_mb': keras_size_mb,
        'onnx_size_mb': onnx_size_mb,
        'compression_ratio': keras_size_mb / onnx_size_mb,
        'opset_version': opset_version
    }
    
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"Keras Model Size:  {stats['keras_size_mb']:.2f} MB")
    print(f"ONNX Model Size:   {stats['onnx_size_mb']:.2f} MB")
    print(f"Opset Version:     {stats['opset_version']}")
    print("="*60)
    
    return stats


if __name__ == "__main__":
    print("ONNX export module - IESA DeepTech Hackathon 2026")
