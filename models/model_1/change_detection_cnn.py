"""
Change Detection CNN for Deforestation Monitoring
Complete implementation with Siamese architecture

Framework: TensorFlow/Keras
Input: Pairs of drone/satellite images (T₀/T₁)
Output: Binary mask of deforestation probability (0-1 map)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import os


class ChangeDetectionCNN:
    """
    Siamese CNN model for detecting canopy loss and forest disturbance
    Input: pair of images (T₀, T₁)
    Output: binary probability mask for deforestation (0-1)
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 3)):
        """
        Args:
            input_shape: input image dimensions (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = None
        
    def build_encoder(self, name: str = "encoder") -> keras.Model:
        """
        Build CNN encoder for feature extraction
        """
        inputs = layers.Input(shape=self.input_shape, name=f"{name}_input")
        
        # Block 1
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        # Block 2
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        # Block 3
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        # Block 4
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        
        return keras.Model(inputs=inputs, outputs=x, name=name)
    
    def build_model(self) -> keras.Model:
        """
        Build complete model with Siamese architecture and decoder
        """
        # Inputs for both temporal images
        input_t0 = layers.Input(shape=self.input_shape, name='image_t0')
        input_t1 = layers.Input(shape=self.input_shape, name='image_t1')
        
        # Shared encoder (Siamese)
        encoder = self.build_encoder()
        
        # Extract features
        features_t0 = encoder(input_t0)
        features_t1 = encoder(input_t1)
        
        # Concatenate features
        merged = layers.Concatenate()([features_t0, features_t1])
        
        # Decoder with skip connections
        # Upsampling 1
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(merged)
        x = layers.UpSampling2D((2, 2))(x)
        
        # Upsampling 2
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        # Upsampling 3
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        # Final upsampling
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        
        # Output: probability mask
        output = layers.Conv2D(1, (1, 1), activation='sigmoid', name='change_mask')(x)
        
        self.model = keras.Model(
            inputs=[input_t0, input_t1], 
            outputs=output,
            name='ChangeDetectionCNN'
        )
        
        return self.model
    
    def compile_model(self, learning_rate: float = 0.0001):
        """
        Compile model with appropriate loss and metrics
        """
        if self.model is None:
            self.build_model()
        
        # Binary Cross-Entropy + Dice Loss (better for segmentation)
        def combined_loss(y_true, y_pred):
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            
            # Dice Loss
            smooth = 1e-6
            y_true_f = tf.reshape(y_true, [-1])
            y_pred_f = tf.reshape(y_pred, [-1])
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
            dice_loss = 1 - dice
            
            return bce + dice_loss
        
        # Custom metrics
        def dice_coefficient(y_true, y_pred):
            smooth = 1e-6
            y_true_f = tf.reshape(y_true, [-1])
            y_pred_f = tf.reshape(y_pred, [-1])
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        
        def iou_metric(y_true, y_pred):
            smooth = 1e-6
            y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
            intersection = tf.reduce_sum(y_true * y_pred_binary)
            union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_binary) - intersection
            return (intersection + smooth) / (union + smooth)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=combined_loss,
            metrics=[
                'accuracy',
                dice_coefficient,
                iou_metric,
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        print("✓ Model compiled successfully")
        return self.model
    
    def summary(self):
        """Show model summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()


class DataGenerator(keras.utils.Sequence):
    """
    Data generator for training/validation
    """
    
    def __init__(self, image_pairs: List[Tuple[str, str]], masks: List[str], 
                 batch_size: int = 8, img_size: Tuple[int, int] = (256, 256),
                 augment: bool = True):
        """
        Args:
            image_pairs: list of tuples (path_t0, path_t1)
            masks: list of paths to ground truth masks
            batch_size: batch size
            img_size: target image size
            augment: apply data augmentation
        """
        self.image_pairs = image_pairs
        self.masks = masks
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.indexes = np.arange(len(self.image_pairs))
        
    def __len__(self):
        return int(np.ceil(len(self.image_pairs) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_t0 = []
        batch_t1 = []
        batch_masks = []
        
        for idx in batch_indexes:
            # Load images (simulated with random data here)
            # In production: use cv2.imread() or tf.io.read_file()
            img_t0 = self._load_and_preprocess_image(self.image_pairs[idx][0])
            img_t1 = self._load_and_preprocess_image(self.image_pairs[idx][1])
            mask = self._load_and_preprocess_mask(self.masks[idx])
            
            if self.augment:
                img_t0, img_t1, mask = self._augment(img_t0, img_t1, mask)
            
            batch_t0.append(img_t0)
            batch_t1.append(img_t1)
            batch_masks.append(mask)
        
        return [np.array(batch_t0), np.array(batch_t1)], np.array(batch_masks)
    
    def _load_and_preprocess_image(self, path: str) -> np.ndarray:
        """Load and preprocess an image"""
        # Simulation - replace with actual loading
        img = np.random.rand(*self.img_size, 3).astype(np.float32)
        return img
    
    def _load_and_preprocess_mask(self, path: str) -> np.ndarray:
        """Load and preprocess a mask"""
        # Simulation - replace with actual loading
        mask = np.random.rand(*self.img_size, 1).astype(np.float32)
        mask = (mask > 0.7).astype(np.float32)  # Binarize
        return mask
    
    def _augment(self, img_t0, img_t1, mask):
        """Apply data augmentation"""
        # Horizontal flip
        if np.random.rand() > 0.5:
            img_t0 = np.fliplr(img_t0)
            img_t1 = np.fliplr(img_t1)
            mask = np.fliplr(mask)
        
        # Vertical flip
        if np.random.rand() > 0.5:
            img_t0 = np.flipud(img_t0)
            img_t1 = np.flipud(img_t1)
            mask = np.flipud(mask)
        
        # 90° rotation
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            img_t0 = np.rot90(img_t0, k)
            img_t1 = np.rot90(img_t1, k)
            mask = np.rot90(mask, k)
        
        return img_t0, img_t1, mask
    
    def on_epoch_end(self):
        """Shuffle data at end of each epoch"""
        np.random.shuffle(self.indexes)


def train_model_example():
    """
    Complete training example
    """
    print("=" * 60)
    print("CHANGE DETECTION CNN - Training Example")
    print("=" * 60)
    
    # 1. Initialize model
    print("\n1. Building model...")
    change_detector = ChangeDetectionCNN(input_shape=(256, 256, 3))
    model = change_detector.build_model()
    change_detector.compile_model(learning_rate=0.0001)
    
    print(f"\nTotal parameters: {model.count_params():,}")
    
    # 2. Prepare data (simulated example)
    print("\n2. Preparing dataset...")
    
    # Simulate paths (use real paths in production)
    n_samples = 100
    image_pairs_train = [(f"t0_{i}.jpg", f"t1_{i}.jpg") for i in range(n_samples)]
    masks_train = [f"mask_{i}.png" for i in range(n_samples)]
    
    image_pairs_val = [(f"t0_val_{i}.jpg", f"t1_val_{i}.jpg") for i in range(20)]
    masks_val = [f"mask_val_{i}.png" for i in range(20)]
    
    # Generators
    train_gen = DataGenerator(image_pairs_train, masks_train, batch_size=8, augment=True)
    val_gen = DataGenerator(image_pairs_val, masks_val, batch_size=8, augment=False)
    
    print(f"Training samples: {len(image_pairs_train)}")
    print(f"Validation samples: {len(image_pairs_val)}")
    
    # 3. Callbacks
    print("\n3. Configuring callbacks...")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'best_change_detection_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    
    # 4. Training
    print("\n4. Starting training...")
    print("-" * 60)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✓ Training completed!")
    
    # 5. Visualize results
    plot_training_history(history)
    
    return model, history


def plot_training_history(history):
    """Visualize training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Dice Coefficient
    axes[1, 0].plot(history.history['dice_coefficient'], label='Train Dice')
    axes[1, 0].plot(history.history['val_dice_coefficient'], label='Val Dice')
    axes[1, 0].set_title('Dice Coefficient')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # IoU
    axes[1, 1].plot(history.history['iou_metric'], label='Train IoU')
    axes[1, 1].plot(history.history['val_iou_metric'], label='Val IoU')
    axes[1, 1].set_title('IoU Metric')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('IoU')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'training_history.png'")


def predict_example(model, img_t0_path: str, img_t1_path: str):
    """
    Example prediction on new image pair
    """
    # Load images (simulated)
    img_t0 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    img_t1 = np.random.rand(1, 256, 256, 3).astype(np.float32)
    
    # Prediction
    change_mask = model.predict([img_t0, img_t1])
    
    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img_t0[0])
    axes[0].set_title('Image T₀ (Before)')
    axes[0].axis('off')
    
    axes[1].imshow(img_t1[0])
    axes[1].set_title('Image T₁ (After)')
    axes[1].axis('off')
    
    axes[2].imshow(change_mask[0, :, :, 0], cmap='jet', vmin=0, vmax=1)
    axes[2].set_title('Probability Map')
    axes[2].axis('off')
    
    # Binary mask with threshold
    binary_mask = (change_mask[0, :, :, 0] > 0.5).astype(np.float32)
    axes[3].imshow(binary_mask, cmap='gray')
    axes[3].set_title('Binary Change Mask (threshold=0.5)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=300, bbox_inches='tight')
    print("\n✓ Prediction saved as 'prediction_result.png'")
    
    # Calculate statistics
    deforestation_percentage = np.sum(binary_mask) / binary_mask.size * 100
    print(f"\n📊 Detected deforestation area: {deforestation_percentage:.2f}%")
    
    return change_mask, binary_mask


if __name__ == "__main__":
    # Example usage
    print("\n🌲 DEFORESTATION DETECTION SYSTEM 🌲\n")
    
    # Training
    model, history = train_model_example()
    
    # Example prediction
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION")
    print("=" * 60)
    change_mask, binary_mask = predict_example(model, "test_t0.jpg", "test_t1.jpg")
    
    print("\n✓ Process completed!")
    print("\nGenerated files:")
    print("  - best_change_detection_model.keras (saved model)")
    print("  - training_history.png (training plots)")
    print("  - prediction_result.png (example prediction)")
