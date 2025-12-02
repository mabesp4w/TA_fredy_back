#!/usr/bin/env python3
"""
Test script untuk memverifikasi loading model CNN
"""

import os
import sys
import tensorflow as tf
import numpy as np

# Force CPU usage
tf.config.set_visible_devices([], 'GPU')

def test_model_loading():
    """Test loading model dengan berbagai metode"""
    model_path = 'api/utils/model_sound.h5'
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return False
    
    print(f"Testing model loading from: {model_path}")
    
    # Method 1: Standard loading
    try:
        print("\n=== Method 1: Standard Loading ===")
        model = tf.keras.models.load_model(model_path)
        print("✓ Standard loading successful")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"✗ Standard loading failed: {e}")
    
    # Method 2: Loading with compile=False
    try:
        print("\n=== Method 2: Loading with compile=False ===")
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("✓ Loading with compile=False successful")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"✗ Loading with compile=False failed: {e}")
    
    # Method 3: Custom InputLayer
    try:
        print("\n=== Method 3: Custom InputLayer ===")
        class CustomInputLayer(tf.keras.layers.InputLayer):
            def __init__(self, **kwargs):
                if 'batch_shape' in kwargs:
                    batch_shape = kwargs.pop('batch_shape')
                    if batch_shape[0] is None:
                        kwargs['input_shape'] = batch_shape[1:]
                super().__init__(**kwargs)
        
        custom_objects = {'InputLayer': CustomInputLayer}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("✓ Custom InputLayer loading successful")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"✗ Custom InputLayer loading failed: {e}")
    
    # Method 4: Load weights only
    try:
        print("\n=== Method 4: Load weights only ===")
        # Create a simple model structure first
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(128, 216, 1)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(13, activation='softmax')
        ])
        
        # Try to load weights
        model.load_weights(model_path)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("✓ Load weights only successful")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"✗ Load weights only failed: {e}")
    
    print("\n✗ All loading methods failed")
    return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
