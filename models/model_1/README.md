# 🌲 Change Detection CNN - Deforestation Monitoring

Convolutional Neural Network for detecting canopy loss and forest disturbance from paired drone/satellite images.

**Framework:** TensorFlow/Keras  
**Input:** Pairs of images (T₀/T₁)  
**Output:** Binary mask of deforestation probability (0-1 map)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Model Architecture](#model-architecture)
6. [Usage Examples](#usage-examples)
7. [Dataset Preparation](#dataset-preparation)
8. [Performance Metrics](#performance-metrics)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This system uses deep learning to detect changes between two temporal images.

**Key Features:**
- Siamese CNN architecture with shared weights
- Combined loss function (BCE + Dice)
- Data augmentation for robust training
- Multiple evaluation metrics (Accuracy, Dice, IoU, Precision, Recall)

**Applications:**
- Canopy loss detection
- Forest disturbance monitoring
- Illegal logging identification
- Change impact assessment

---

## 📦 Requirements

```
Python >= 3.8
TensorFlow >= 2.10.0
NumPy >= 1.21.0
Matplotlib >= 3.5.0
OpenCV >= 4.5.0
```

---

## 🔧 Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install tensorflow numpy matplotlib opencv-python
```

---

## 🚀 Quick Start

### Simple Example (5 minutes)

```bash
python simple_example.py
```

This will train a basic model and show example predictions.

### Advanced Example

```bash
python change_detection_cnn.py
```

Full Siamese architecture with advanced features.

---

## 🏗️ Model Architecture

### Simple Model

```
Input T₀ + Input T₁ → Concatenate → Encoder → Decoder → Output (Probability Map)
```

### Advanced Model (Siamese)

```
Input T₀ → Shared Encoder ┐
                           ├→ Concatenate → Decoder → Output
Input T₁ → Shared Encoder ┘
```

**Why Siamese?**
- Consistent feature extraction
- Better generalization
- More parameter-efficient
- Captures temporal relationships

---

## 📖 Usage Examples

### Example 1: Basic Training

```python
from simple_example import create_simple_change_detection_model

# Create model
model = create_simple_change_detection_model(256, 256)

# Train (with your data)
history = model.fit(
    [X_t0_train, X_t1_train],
    y_train,
    epochs=20,
    batch_size=8,
    validation_split=0.2
)

# Save
model.save('my_model.keras')
```

### Example 2: Prediction

```python
from tensorflow import keras
import numpy as np

# Load model
model = keras.models.load_model('my_model.keras')

# Load images (normalized 0-1)
img_t0 = np.load('before.npy')  # Shape: (1, 256, 256, 3)
img_t1 = np.load('after.npy')

# Predict
probability_mask = model.predict([img_t0, img_t1])

# Binarize
binary_mask = (probability_mask > 0.5).astype(np.uint8)

# Calculate area
deforestation_pct = binary_mask.sum() / binary_mask.size * 100
print(f"Deforested: {deforestation_pct:.2f}%")
```

### Example 3: Load Real Images

```python
from simple_example import load_image_pair

# Load and preprocess
img_t0, img_t1 = load_image_pair('before.jpg', 'after.jpg', target_size=(256, 256))

# Add batch dimension
img_t0 = np.expand_dims(img_t0, axis=0)
img_t1 = np.expand_dims(img_t1, axis=0)

# Predict
mask = model.predict([img_t0, img_t1])
```

---

## 📊 Dataset Preparation

### Directory Structure

```
dataset/
├── images_t0/          # Before images
│   ├── area001.jpg
│   └── ...
├── images_t1/          # After images
│   ├── area001.jpg
│   └── ...
└── masks/              # Ground truth (0=no change, 255=change)
    ├── area001.png
    └── ...
```

### Image Requirements

- **Format:** JPG, PNG, TIFF
- **Size:** Any (will be resized)
- **Channels:** RGB (3)
- **Range:** 0-255 (will be normalized)
- **Alignment:** T₀ and T₁ must be aligned

### Mask Requirements

- **Format:** PNG grayscale
- **Values:** 0 (no change) or 255 (deforestation)
- **Size:** Same as images

---

## 📈 Performance Metrics

### Key Metrics

| Metric | Formula | Good Value | Meaning |
|--------|---------|------------|---------|
| **Accuracy** | (TP+TN)/Total | > 0.90 | Overall correctness |
| **Dice** | 2×TP/(2×TP+FP+FN) | > 0.75 | Overlap measure |
| **IoU** | TP/(TP+FP+FN) | > 0.65 | Intersection/Union |
| **Precision** | TP/(TP+FP) | > 0.80 | Few false alarms |
| **Recall** | TP/(TP+FN) | > 0.80 | Few missed detections |

### Expected Performance

**Synthetic Data:**
- Accuracy: 85-92%
- Dice: 0.70-0.85
- IoU: 0.60-0.75

**Real Data (well-prepared):**
- Accuracy: 90-96%
- Dice: 0.80-0.92
- IoU: 0.70-0.88

---

## 🐛 Troubleshooting

### Out of Memory

```python
# Reduce batch size
batch_size = 2

# Reduce image size
img_size = 128

# Enable memory growth (GPU)
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### Model Not Learning

**Check:**
- Data range (images 0-1, masks 0-1)
- Learning rate (try 0.0001)
- Data quality (visualize samples)

```python
# Verify data
print(f"Images: {X_t0.min():.3f} to {X_t0.max():.3f}")
print(f"Masks: {y_masks.min():.3f} to {y_masks.max():.3f}")
```

### Overfitting

```python
# Add dropout
x = layers.Dropout(0.3)(x)

# Add regularization
x = layers.Conv2D(64, 3, kernel_regularizer=keras.regularizers.l2(0.01))(x)

# Early stopping
keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
```

### Slow Training

```python
# Use GPU (verify)
print(tf.config.list_physical_devices('GPU'))

# Reduce image size
img_size = 128

# Use mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

---

## 🌍 Working with Satellite Data

### Sentinel-2 Example

```python
import rasterio
import numpy as np

with rasterio.open('sentinel2.tif') as src:
    # RGB bands
    r = src.read(4)
    g = src.read(3)
    b = src.read(2)
    
    img = np.dstack([r, g, b])
    img = img.astype(np.float32) / 10000.0  # Normalize
    img = np.clip(img, 0, 1)
```

### Landsat Example

```python
with rasterio.open('landsat8.tif') as src:
    r = src.read(4)
    g = src.read(3)
    b = src.read(2)
    
    img = np.dstack([r, g, b])
    img = img.astype(np.float32) / 65535.0
```

---

## 🎓 Training Tips

### Hyperparameters

**Quick Test:**
- Epochs: 10
- Batch size: 16
- Learning rate: 0.001
- Image size: 128

**Good Results:**
- Epochs: 50
- Batch size: 8
- Learning rate: 0.0001
- Image size: 256

**Best Accuracy:**
- Epochs: 100
- Batch size: 4
- Learning rate: 0.00005
- Image size: 512

### Callbacks

```python
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        save_best_only=True
    )
]
```

---

## 📚 Additional Resources

### Scientific Papers

- Daudt et al. (2018): "Fully Convolutional Siamese Networks for Change Detection"
- Zhang et al. (2020): "Deep Learning for Remote Sensing Change Detection"

### Datasets

- **Sentinel-2**: Free, 10m resolution (ESA Copernicus)
- **Landsat 8/9**: Free, 30m resolution (USGS)
- **Planet**: Commercial, 3m daily imagery

### Tools

- **QGIS**: Open-source GIS
- **Google Earth Engine**: Cloud geospatial analysis
- **Rasterio**: Python raster I/O
- **TensorFlow**: Deep learning framework

---

## 📝 Citation

```bibtex
@software{change_detection_cnn,
  title = {Change Detection CNN for Deforestation Monitoring},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/change-detection-cnn}
}
```

---

## ⚖️ License

This code is provided for educational and research purposes.

---

## 💡 Tips for Success

1. **Start small** - Test on 128×128 images first
2. **Visualize** - Always check data and predictions
3. **Use callbacks** - Early stopping saves time
4. **Monitor metrics** - Track Dice and IoU, not just accuracy
5. **Augment data** - Helps with limited training samples
6. **GPU recommended** - Much faster training

---

## 🎉 Summary

This package provides:
- ✅ Simple and advanced implementations
- ✅ Complete training pipeline
- ✅ Data preprocessing utilities
- ✅ Evaluation metrics
- ✅ Visualization tools
- ✅ Production-ready code

**Start with `simple_example.py` and progress to `change_detection_cnn.py`!**

---

**Questions? Check the code comments or open an issue.**

**Happy monitoring! 🌲🛰️📊**
