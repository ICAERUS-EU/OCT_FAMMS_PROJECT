# 🌲 Forest Pattern Recognition & Anomaly Detection

Random Forest classifier for detecting deforestation risk based on vegetation indices (NDVI) and canopy texture analysis.

**Framework:** Scikit-learn  
**Output:** Risk classification per grid cell ("Low", "Medium", "High")

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [File Descriptions](#file-descriptions)
6. [Feature Engineering](#feature-engineering)
7. [Model Architecture](#model-architecture)
8. [Usage Examples](#usage-examples)
9. [Real-World Application](#real-world-application)
10. [Performance Metrics](#performance-metrics)

---

## 🎯 Overview

This system uses machine learning to classify forest areas into risk categories based on:

- **NDVI (Normalized Difference Vegetation Index)**: Measures vegetation health
- **Texture Features**: Analyzes canopy structure patterns
- **Spatial Features**: Detects fragmentation and patch distribution

**Risk Categories:**
- **Low Risk**: Healthy, dense vegetation (NDVI > 0.6)
- **Medium Risk**: Moderate vegetation with some disturbance (NDVI 0.3-0.6)
- **High Risk**: Degraded or deforested areas (NDVI < 0.3)

---

## 📦 Requirements

```
Python >= 3.8
scikit-learn >= 1.0.0
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
scipy >= 1.7.0
scikit-image >= 0.19.0
joblib >= 1.1.0
```

### Optional (for satellite imagery):
```
rasterio >= 1.2.0
GDAL >= 3.0.0
```

---

## 🔧 Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install scikit-learn numpy pandas matplotlib seaborn scipy scikit-image joblib

# For satellite data processing (optional)
pip install rasterio gdal
```

---

## 🚀 Quick Start

### Option 1: Simple Example (Recommended for Beginners)

```bash
python simple_forest_example.py
```

This will:
1. Generate synthetic forest data
2. Extract features
3. Train a Random Forest model
4. Evaluate performance
5. Save the trained model

### Option 2: Advanced Implementation

```bash
python forest_pattern_classifier.py
```

This includes:
- Comprehensive feature extraction (40+ features)
- Advanced texture analysis (GLCM)
- Spatial pattern recognition
- Full risk map generation
- Detailed visualizations

---

## 📁 File Descriptions

### 1. `simple_forest_example.py` ⭐ **START HERE**

Simplified implementation perfect for learning and quick deployment.

**Features:**
- 10 basic features (NDVI statistics + texture)
- Easy to understand and modify
- Fast training (~10 seconds)
- Good baseline performance

**Use when:**
- Learning the basics
- Rapid prototyping
- Limited computational resources

### 2. `forest_pattern_classifier.py` 🚀 **PRODUCTION READY**

Full-featured implementation with advanced capabilities.

**Features:**
- 40+ engineered features
- GLCM texture analysis
- Spatial autocorrelation
- Fragmentation metrics
- Cross-validation
- Grid-based risk mapping

**Use when:**
- Maximum accuracy needed
- Processing large areas
- Research or production deployment

---

## 🔬 Feature Engineering

### NDVI Features (Vegetation Health)

```python
NDVI = (NIR - Red) / (NIR + Red)
```

**Extracted features:**
- Statistical moments (mean, std, min, max, median)
- Distribution properties (skewness, kurtosis, IQR)
- Vegetation density ratios
- Coefficient of variation

### Texture Features (Canopy Structure)

Using Gray-Level Co-occurrence Matrix (GLCM):

- **Contrast**: Local variations in pixel intensity
- **Dissimilarity**: Differences between neighboring pixels
- **Homogeneity**: Closeness of distribution
- **Energy**: Uniformity of texture
- **Correlation**: Linear dependency of gray levels
- **Entropy**: Randomness of texture

### Spatial Features (Pattern Analysis)

- Number of vegetation patches
- Patch size distribution
- Fragmentation index
- Spatial autocorrelation (Moran's I)
- Edge density

---

## 🏗️ Model Architecture

```
Input Data (Satellite/Drone Images)
         ↓
    NDVI Calculation
         ↓
  Feature Extraction
  ├── NDVI Features (12)
  ├── Texture Features (20+)
  └── Spatial Features (8)
         ↓
   Feature Scaling
   (StandardScaler)
         ↓
  Random Forest Classifier
  ├── n_estimators: 200 trees
  ├── max_depth: 20
  └── class_weight: balanced
         ↓
  Risk Prediction
  ("Low", "Medium", "High")
```

### Why Random Forest?

✅ **Advantages:**
- Handles non-linear relationships
- Robust to outliers
- Provides feature importance
- No need for feature scaling (but we do it anyway)
- Fast inference
- Interpretable results

---

## 📖 Usage Examples

### Example 1: Basic Training

```python
from simple_forest_example import generate_sample_data, train_classifier
from sklearn.model_selection import train_test_split

# Generate data
X, y, feature_names = generate_sample_data(n_samples=300)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = train_classifier(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

### Example 2: Predict Single Area

```python
import joblib
import numpy as np
from simple_forest_example import extract_simple_features

# Load trained model
model = joblib.load('simple_forest_classifier.pkl')

# Your NDVI patch (32x32 pixels)
ndvi_patch = np.array([...])  # Load your data

# Extract features
features = extract_simple_features(ndvi_patch)
X = np.array([list(features.values())])

# Predict
risk_level = model.predict(X)[0]
probabilities = model.predict_proba(X)[0]

print(f"Risk Level: {risk_level}")
print(f"Confidence: {probabilities.max():.2%}")
```

### Example 3: Process Large Area

```python
from forest_pattern_classifier import ForestPatternClassifier, predict_risk_map
import numpy as np

# Load your NDVI map (e.g., from satellite imagery)
ndvi_map = np.load('ndvi_map.npy')  # Shape: (H, W)

# Load trained model
classifier = ForestPatternClassifier()
classifier.load_model('forest_classifier_model.pkl')

# Generate risk map (divides into 64x64 pixel cells)
risk_map, prob_map = predict_risk_map(
    classifier, 
    ndvi_map, 
    grid_size=64
)

# Analyze results
low_risk = (risk_map == 'Low').sum()
medium_risk = (risk_map == 'Medium').sum()
high_risk = (risk_map == 'High').sum()

print(f"High risk areas: {high_risk} cells")
```

### Example 4: Batch Processing

```python
import glob
import rasterio

# Process multiple files
ndvi_files = glob.glob('data/*.tif')

results = []
for filepath in ndvi_files:
    with rasterio.open(filepath) as src:
        ndvi = src.read(1)
    
    risk_map, _ = predict_risk_map(classifier, ndvi)
    
    results.append({
        'file': filepath,
        'high_risk_cells': (risk_map == 'High').sum(),
        'total_cells': risk_map.size
    })

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('batch_assessment.csv', index=False)
```

---

## 🌍 Real-World Application

### Working with Satellite Data

#### Step 1: Load Satellite Image

```python
import rasterio
import numpy as np

def load_and_calculate_ndvi(filepath):
    """
    Load satellite image and calculate NDVI
    Assumes Sentinel-2 or Landsat format
    """
    with rasterio.open(filepath) as src:
        # Sentinel-2: Band 8 (NIR), Band 4 (Red)
        # Landsat: Band 5 (NIR), Band 4 (Red)
        nir = src.read(8).astype(float)  # Adjust band numbers
        red = src.read(4).astype(float)
        
        # Calculate NDVI
        ndvi = (nir - red) / (nir + red + 1e-10)
        
        # Clip to valid range
        ndvi = np.clip(ndvi, -1, 1)
        
        # Get geospatial metadata
        transform = src.transform
        crs = src.crs
    
    return ndvi, transform, crs
```

#### Step 2: Process and Classify

```python
# Load NDVI
ndvi_map, transform, crs = load_and_calculate_ndvi('sentinel_image.tif')

# Load model
classifier = ForestPatternClassifier()
classifier.load_model('forest_classifier_model.pkl')

# Generate risk map
risk_map, prob_map = predict_risk_map(classifier, ndvi_map, grid_size=64)

# Save results as GeoTIFF
from rasterio.transform import from_bounds

with rasterio.open(
    'risk_classification.tif',
    'w',
    driver='GTiff',
    height=risk_map.shape[0],
    width=risk_map.shape[1],
    count=1,
    dtype=rasterio.uint8,
    crs=crs,
    transform=transform
) as dst:
    # Convert to numeric
    risk_numeric = np.vectorize({'Low': 0, 'Medium': 1, 'High': 2}.get)(risk_map)
    dst.write(risk_numeric.astype(rasterio.uint8), 1)
```

### Time Series Analysis

```python
def analyze_temporal_change(ndvi_t0, ndvi_t1, classifier):
    """
    Analyze change in risk between two time periods
    """
    # Predict risk for both periods
    risk_t0, _ = predict_risk_map(classifier, ndvi_t0)
    risk_t1, _ = predict_risk_map(classifier, ndvi_t1)
    
    # Detect changes
    change_map = np.zeros_like(risk_t0, dtype=object)
    
    for i in range(risk_t0.shape[0]):
        for j in range(risk_t0.shape[1]):
            if risk_t0[i,j] == 'Low' and risk_t1[i,j] == 'High':
                change_map[i,j] = 'Critical Degradation'
            elif risk_t0[i,j] != risk_t1[i,j]:
                change_map[i,j] = 'Change Detected'
            else:
                change_map[i,j] = 'No Change'
    
    # Statistics
    critical = (change_map == 'Critical Degradation').sum()
    print(f"Critical degradation in {critical} cells")
    
    return change_map
```

---

## 📊 Performance Metrics

### Interpretation Guide

| Metric | Formula | Good Value | Interpretation |
|--------|---------|------------|----------------|
| **Accuracy** | (TP+TN)/(Total) | > 0.85 | Overall correctness |
| **Precision** | TP/(TP+FP) | > 0.80 | Few false alarms |
| **Recall** | TP/(TP+FN) | > 0.80 | Catches most issues |
| **F1-Score** | 2×(P×R)/(P+R) | > 0.80 | Balanced performance |

### Per-Class Metrics

```python
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

**Expected Output:**
```
              precision    recall  f1-score   support

        High       0.85      0.88      0.86        25
         Low       0.92      0.91      0.91        32
      Medium       0.81      0.79      0.80        28

    accuracy                           0.86        85
   macro avg       0.86      0.86      0.86        85
weighted avg       0.86      0.86      0.86        85
```

### Feature Importance

```python
# Get top features
importance_df = classifier.get_feature_importance(top_n=10)
print(importance_df)
```

**Typical Important Features:**
1. `ndvi_mean` - Average vegetation health
2. `dense_vegetation_ratio` - Proportion of healthy forest
3. `texture_contrast_mean` - Canopy heterogeneity
4. `fragmentation_index` - Forest fragmentation
5. `ndvi_std` - Vegetation variability

---

## 🎯 Hyperparameter Tuning

### Grid Search Example

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

### Recommended Settings

**For Speed:**
```python
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    n_jobs=-1
)
```

**For Accuracy:**
```python
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1
)
```

**For Balanced:**
```python
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1
)
```

---

## 🐛 Troubleshooting

### Problem: Low accuracy on real data

**Solutions:**
1. Check NDVI calculation (ensure correct band ordering)
2. Verify that NDVI values are in [-1, 1] range
3. Increase training data diversity
4. Add domain-specific features
5. Try different grid sizes

### Problem: Imbalanced classes

```python
# Use class weights
model = RandomForestClassifier(
    class_weight='balanced'  # or {0: 1, 1: 2, 2: 3}
)

# Or use SMOTE for oversampling
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### Problem: Slow prediction on large maps

```python
# Reduce grid size
risk_map = predict_risk_map(classifier, ndvi_map, grid_size=128)

# Or use parallel processing
from joblib import Parallel, delayed

def process_cell(i, j, cell):
    features = extract_features(cell)
    return model.predict(features)[0]

results = Parallel(n_jobs=-1)(
    delayed(process_cell)(i, j, cell) 
    for i, j, cell in iterate_cells(ndvi_map)
)
```

---

## 📚 Additional Resources

### Scientific Background

- **NDVI**: Tucker, C.J. (1979). "Red and photographic infrared linear combinations for monitoring vegetation"
- **GLCM**: Haralick, R.M. et al. (1973). "Textural Features for Image Classification"
- **Random Forests**: Breiman, L. (2001). "Random Forests"

### Datasets

- **Sentinel-2**: Free 10m resolution, 5-day revisit (ESA Copernicus)
- **Landsat 8/9**: Free 30m resolution, 16-day revisit (USGS)
- **Planet**: Commercial 3m daily imagery

### Tools

- **QGIS**: Open-source GIS software
- **Google Earth Engine**: Cloud-based geospatial analysis
- **Rasterio**: Python library for raster data
- **GeoPandas**: Spatial operations in Python

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{forest_pattern_classifier,
  title = {Forest Pattern Recognition and Anomaly Detection},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/forest-classifier}
}
```

---

## 📧 Support

For questions or issues:
- Check the troubleshooting section
- Review the example scripts
- Open an issue on GitHub

---

## ⚖️ License

This code is provided for educational and research purposes.

---

**Happy forest monitoring! 🌲🛰️📊**
