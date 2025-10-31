"""
SIMPLE EXAMPLE - Change Detection CNN
Quick start guide for deforestation detection using deep learning

Framework: TensorFlow/Keras
Output: Binary mask of deforestation probability (0-1 map)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


def create_simple_change_detection_model(img_height=256, img_width=256):
    """
    Simple CNN model for change detection
    
    Args:
        img_height: image height
        img_width: image width
    
    Returns:
        compiled Keras model
    """
    
    # Input: two images (T0 and T1)
    input_t0 = layers.Input(shape=(img_height, img_width, 3), name='image_t0')
    input_t1 = layers.Input(shape=(img_height, img_width, 3), name='image_t1')
    
    # Concatenate both images
    concat = layers.Concatenate()([input_t0, input_t1])
    
    # Encoder (feature extraction)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(concat)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    
    # Decoder (mask reconstruction)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    
    # Output: probability mask (0-1)
    output = layers.Conv2D(1, 1, activation='sigmoid', name='change_mask')(x)
    
    # Create model
    model = keras.Model(inputs=[input_t0, input_t1], outputs=output)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


def generate_synthetic_data(n_samples=100, img_size=256):
    """
    Generate synthetic data for testing
    (replace with real data in production)
    """
    images_t0 = np.random.rand(n_samples, img_size, img_size, 3).astype(np.float32)
    images_t1 = np.random.rand(n_samples, img_size, img_size, 3).astype(np.float32)
    
    # Generate masks (1 = deforestation, 0 = no change)
    masks = np.random.rand(n_samples, img_size, img_size, 1).astype(np.float32)
    masks = (masks > 0.8).astype(np.float32)  # Only 20% is deforested
    
    return images_t0, images_t1, masks


def visualize_prediction(img_t0, img_t1, mask_true, mask_pred, save_path='result.png'):
    """
    Visualize prediction results
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(img_t0)
    axes[0].set_title('T₀ - Before')
    axes[0].axis('off')
    
    axes[1].imshow(img_t1)
    axes[1].set_title('T₁ - After')
    axes[1].axis('off')
    
    axes[2].imshow(mask_true.squeeze(), cmap='Reds', vmin=0, vmax=1)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    axes[3].imshow(mask_pred.squeeze(), cmap='Reds', vmin=0, vmax=1)
    axes[3].set_title('Prediction')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Result saved: {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("CHANGE DETECTION CNN - Simple Example")
    print("="*60 + "\n")
    
    # 1. CREATE MODEL
    print("1️⃣  Creating model...")
    model = create_simple_change_detection_model(img_height=256, img_width=256)
    print(f"   ✓ Model created with {model.count_params():,} parameters\n")
    
    # Show architecture
    model.summary()
    
    # 2. GENERATE DATA (in production: load your data)
    print("\n2️⃣  Generating example data...")
    X_t0, X_t1, y_masks = generate_synthetic_data(n_samples=100, img_size=256)
    print(f"   ✓ {len(X_t0)} image pairs generated")
    print(f"   ✓ T₀ images shape: {X_t0.shape}")
    print(f"   ✓ T₁ images shape: {X_t1.shape}")
    print(f"   ✓ Masks shape: {y_masks.shape}\n")
    
    # Train/validation split
    split_idx = int(0.8 * len(X_t0))
    X_t0_train, X_t0_val = X_t0[:split_idx], X_t0[split_idx:]
    X_t1_train, X_t1_val = X_t1[:split_idx], X_t1[split_idx:]
    y_train, y_val = y_masks[:split_idx], y_masks[split_idx:]
    
    # 3. TRAINING
    print("3️⃣  Training model...")
    print("   (this may take a few minutes...)\n")
    
    history = model.fit(
        [X_t0_train, X_t1_train],
        y_train,
        validation_data=([X_t0_val, X_t1_val], y_val),
        epochs=10,  # Increase for better results
        batch_size=8,
        verbose=1
    )
    
    print("\n   ✓ Training completed!\n")
    
    # 4. EVALUATION
    print("4️⃣  Evaluating on validation set...")
    results = model.evaluate([X_t0_val, X_t1_val], y_val, verbose=0)
    print(f"   ✓ Loss: {results[0]:.4f}")
    print(f"   ✓ Accuracy: {results[1]:.4f}")
    print(f"   ✓ Precision: {results[2]:.4f}")
    print(f"   ✓ Recall: {results[3]:.4f}\n")
    
    # 5. EXAMPLE PREDICTION
    print("5️⃣  Example prediction...")
    
    # Take an example from validation set
    idx = 0
    test_t0 = X_t0_val[idx:idx+1]
    test_t1 = X_t1_val[idx:idx+1]
    true_mask = y_val[idx:idx+1]
    
    # Predict
    predicted_mask = model.predict([test_t0, test_t1], verbose=0)
    
    # Calculate statistics
    binary_pred = (predicted_mask > 0.5).astype(np.float32)
    deforestation_area = np.sum(binary_pred) / binary_pred.size * 100
    
    print(f"   ✓ Detected deforestation area: {deforestation_area:.2f}%")
    
    # Visualize
    visualize_prediction(
        test_t0[0], 
        test_t1[0], 
        true_mask[0], 
        predicted_mask[0],
        save_path='prediction_example.png'
    )
    
    # 6. SAVE MODEL
    print("\n6️⃣  Saving model...")
    model.save('change_detection_model.keras')
    print("   ✓ Model saved as: change_detection_model.keras\n")
    
    print("="*60)
    print("✅ PROCESS COMPLETED!")
    print("="*60)
    print("\nTo load the model later:")
    print(">>> model = keras.models.load_model('change_detection_model.keras')")
    print("\nTo make predictions:")
    print(">>> prediction = model.predict([img_t0, img_t1])")
    print()


# ============================================================================
# UTILITY FUNCTIONS FOR REAL IMAGES
# ============================================================================

def load_image_pair(path_t0, path_t1, target_size=(256, 256)):
    """
    Load a pair of real images from files
    
    Example usage:
    >>> img_t0, img_t1 = load_image_pair('drone_2020.jpg', 'drone_2024.jpg')
    """
    import cv2
    
    # Load images
    img_t0 = cv2.imread(path_t0)
    img_t1 = cv2.imread(path_t1)
    
    # Convert BGR -> RGB
    img_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2RGB)
    img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)
    
    # Resize
    img_t0 = cv2.resize(img_t0, target_size)
    img_t1 = cv2.resize(img_t1, target_size)
    
    # Normalize (0-1)
    img_t0 = img_t0.astype(np.float32) / 255.0
    img_t1 = img_t1.astype(np.float32) / 255.0
    
    return img_t0, img_t1


def predict_on_real_images(model, path_t0, path_t1, threshold=0.5):
    """
    Predict changes on real images
    
    Example usage:
    >>> model = keras.models.load_model('change_detection_model.keras')
    >>> mask = predict_on_real_images(model, 'before.jpg', 'after.jpg')
    """
    # Load images
    img_t0, img_t1 = load_image_pair(path_t0, path_t1)
    
    # Add batch dimension
    img_t0 = np.expand_dims(img_t0, axis=0)
    img_t1 = np.expand_dims(img_t1, axis=0)
    
    # Predict
    mask_prob = model.predict([img_t0, img_t1], verbose=0)
    
    # Binarize
    mask_binary = (mask_prob > threshold).astype(np.uint8) * 255
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_t0[0])
    axes[0].set_title('Before (T₀)')
    axes[0].axis('off')
    
    axes[1].imshow(img_t1[0])
    axes[1].set_title('After (T₁)')
    axes[1].axis('off')
    
    axes[2].imshow(mask_binary[0, :, :, 0], cmap='Reds')
    axes[2].set_title(f'Deforestation (threshold={threshold})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('real_prediction.png', dpi=150)
    plt.close()
    
    # Statistics
    defor_pixels = np.sum(mask_binary > 0)
    total_pixels = mask_binary.size
    percentage = (defor_pixels / total_pixels) * 100
    
    print(f"\n📊 Results:")
    print(f"   Deforested area: {percentage:.2f}%")
    print(f"   Deforested pixels: {defor_pixels:,}/{total_pixels:,}")
    print(f"   ✓ Image saved: real_prediction.png")
    
    return mask_prob[0]
