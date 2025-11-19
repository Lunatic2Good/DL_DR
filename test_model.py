"""
Quick test script to check if your model loads and predicts correctly
Run this to diagnose model issues: python test_model.py
"""

import os
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow import keras
    print("✓ TensorFlow imported successfully")
except ImportError:
    print("✗ TensorFlow not available. Please install TensorFlow.")
    exit(1)

def test_model_loading():
    """Test loading the 5 layer CNN model"""
    model_path = "models/5layer_cnn.h5"
    
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        return None
    
    print(f"\n{'='*60}")
    print("Testing Model Loading")
    print(f"{'='*60}\n")
    
    # Try different loading methods
    loading_methods = [
        ("Standard Keras", lambda: keras.models.load_model(model_path)),
        ("Keras (compile=False)", lambda: keras.models.load_model(model_path, compile=False)),
        ("TF Keras", lambda: tf.keras.models.load_model(model_path)),
        ("TF Keras (compile=False)", lambda: tf.keras.models.load_model(model_path, compile=False)),
    ]
    
    model = None
    for method_name, load_func in loading_methods:
        try:
            print(f"Trying: {method_name}...")
            model = load_func()
            print(f"✓ Success with {method_name}!\n")
            break
        except Exception as e:
            print(f"✗ Failed: {str(e)}\n")
            continue
    
    if model is None:
        print("✗ Could not load model with any method")
        return None
    
    # Inspect model
    print(f"{'='*60}")
    print("Model Information")
    print(f"{'='*60}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    if len(model.output_shape) > 1:
        num_classes = model.output_shape[1]
        print(f"Number of classes: {num_classes}")
    
    print(f"\nModel layers:")
    for i, layer in enumerate(model.layers[-5:]):  # Show last 5 layers
        print(f"  {i}: {layer.name} ({type(layer).__name__})")
    
    return model

def test_prediction(model):
    """Test making a prediction with a dummy image"""
    if model is None:
        return
    
    print(f"\n{'='*60}")
    print("Testing Prediction")
    print(f"{'='*60}\n")
    
    # Create a dummy image (224x224 RGB)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_array = dummy_image.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    print(f"Input shape: {img_array.shape}")
    print(f"Input range: [{img_array.min():.3f}, {img_array.max():.3f}]")
    
    try:
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        print(f"\n✓ Prediction successful!")
        print(f"Prediction shape: {predictions.shape}")
        print(f"Prediction dtype: {predictions.dtype}")
        print(f"Prediction values: {predictions[0]}")
        print(f"Min: {np.min(predictions[0]):.6f}, Max: {np.max(predictions[0]):.6f}")
        print(f"Sum: {np.sum(predictions[0]):.6f}")
        print(f"Argmax (predicted class): {np.argmax(predictions[0])}")
        
        # Check if values are probabilities or logits
        if np.max(predictions[0]) > 1.0 or np.min(predictions[0]) < 0.0:
            print("\n⚠ Values are outside [0,1] range - these are likely logits")
            print("Applying softmax...")
            exp_pred = np.exp(predictions[0] - np.max(predictions[0]))
            probs = exp_pred / np.sum(exp_pred)
            print(f"After softmax - Max: {np.max(probs):.6f} at index {np.argmax(probs)}")
        else:
            print("\n✓ Values are in [0,1] range - these are probabilities")
            print(f"Top class: {np.argmax(predictions[0])} with confidence {np.max(predictions[0]):.6f}")
        
        # Show top 5
        top_indices = np.argsort(predictions[0])[::-1][:5]
        print(f"\nTop 5 predictions:")
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. Class {idx}: {predictions[0][idx]:.6f} ({predictions[0][idx]*100:.2f}%)")
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Model Testing Script")
    print("=" * 60)
    
    model = test_model_loading()
    if model:
        test_prediction(model)
    
    print(f"\n{'='*60}")
    print("Test Complete")
    print(f"{'='*60}")

