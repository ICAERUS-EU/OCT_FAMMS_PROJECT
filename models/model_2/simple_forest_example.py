"""
SIMPLE EXAMPLE - Forest Pattern Recognition
Quick start guide for anomaly detection using Random Forest
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib


def calculate_ndvi(nir, red):
    """
    Calculate NDVI (Normalized Difference Vegetation Index)
    
    Formula: NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        nir: Near-Infrared band values
        red: Red band values
    
    Returns:
        NDVI values (range: -1 to 1)
    """
    return (nir - red) / (nir + red + 1e-10)


def extract_simple_features(ndvi_patch):
    """
    Extract basic features from NDVI patch
    
    Args:
        ndvi_patch: 2D array of NDVI values
    
    Returns:
        dict with features
    """
    features = {
        'mean_ndvi': np.mean(ndvi_patch),
        'std_ndvi': np.std(ndvi_patch),
        'min_ndvi': np.min(ndvi_patch),
        'max_ndvi': np.max(ndvi_patch),
        'median_ndvi': np.median(ndvi_patch),
        'range_ndvi': np.max(ndvi_patch) - np.min(ndvi_patch),
        
        # Vegetation density
        'high_veg_ratio': (ndvi_patch > 0.6).sum() / ndvi_patch.size,
        'low_veg_ratio': (ndvi_patch < 0.3).sum() / ndvi_patch.size,
        
        # Texture approximation (standard deviation of differences)
        'texture_h': np.std(np.diff(ndvi_patch, axis=0)),
        'texture_v': np.std(np.diff(ndvi_patch, axis=1)),
    }
    
    return features


def generate_sample_data(n_samples=200):
    """
    Generate synthetic forest data with risk labels
    
    Returns:
        features: array of extracted features
        labels: risk classifications
    """
    print(f"Generating {n_samples} samples...")
    
    features_list = []
    labels = []
    
    patch_size = 32
    
    # Low Risk (healthy forest) - 40%
    for _ in range(int(n_samples * 0.4)):
        ndvi = np.random.normal(0.7, 0.08, (patch_size, patch_size))
        ndvi = np.clip(ndvi, 0, 1)
        
        feats = extract_simple_features(ndvi)
        features_list.append(list(feats.values()))
        labels.append('Low')
    
    # Medium Risk - 35%
    for _ in range(int(n_samples * 0.35)):
        ndvi = np.random.normal(0.45, 0.15, (patch_size, patch_size))
        ndvi = np.clip(ndvi, 0, 1)
        
        feats = extract_simple_features(ndvi)
        features_list.append(list(feats.values()))
        labels.append('Medium')
    
    # High Risk (degraded) - 25%
    for _ in range(n_samples - int(n_samples * 0.4) - int(n_samples * 0.35)):
        ndvi = np.random.normal(0.25, 0.18, (patch_size, patch_size))
        ndvi = np.clip(ndvi, 0, 1)
        
        feats = extract_simple_features(ndvi)
        features_list.append(list(feats.values()))
        labels.append('High')
    
    feature_names = list(extract_simple_features(np.zeros((2, 2))).keys())
    
    print(f"✓ Generated {len(labels)} samples")
    print(f"  Features: {len(feature_names)}")
    
    return np.array(features_list), np.array(labels), feature_names


def train_classifier(X_train, y_train):
    """
    Train a Random Forest classifier
    
    Args:
        X_train: training features
        y_train: training labels
    
    Returns:
        trained model
    """
    print("\nTraining Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("✓ Training complete")
    
    return model


def visualize_results(model, X_test, y_test, feature_names):
    """Visualize model performance"""
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    axes[0].barh(range(len(indices)), importances[indices])
    axes[0].set_yticks(range(len(indices)))
    axes[0].set_yticklabels([feature_names[i] for i in indices])
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Top 10 Feature Importances')
    axes[0].invert_yaxis()
    
    # Class distribution
    unique, counts = np.unique(y_test, return_counts=True)
    pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
    
    x = np.arange(len(unique))
    width = 0.35
    
    axes[1].bar(x - width/2, counts, width, label='True', alpha=0.8)
    axes[1].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
    axes[1].set_xlabel('Risk Class')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Risk Distribution (Accuracy: {accuracy:.2%})')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(unique)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Results saved: simple_results.png")


def predict_single_patch(model, ndvi_patch, feature_names):
    """
    Predict risk for a single NDVI patch
    
    Args:
        model: trained classifier
        ndvi_patch: 2D NDVI array
        feature_names: list of feature names
    
    Returns:
        prediction and probability
    """
    # Extract features
    features = extract_simple_features(ndvi_patch)
    X = np.array([list(features.values())])
    
    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    # Map to class names
    class_names = model.classes_
    prob_dict = {cls: prob for cls, prob in zip(class_names, probabilities)}
    
    return prediction, prob_dict


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("="*60)
    print("FOREST PATTERN RECOGNITION - Simple Example")
    print("="*60)
    
    # 1. Generate data
    print("\n1️⃣  Generating synthetic data...")
    X, y, feature_names = generate_sample_data(n_samples=200)
    print(f"   Dataset shape: {X.shape}")
    
    # 2. Split data
    print("\n2️⃣  Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # 3. Train model
    print("\n3️⃣  Training model...")
    model = train_classifier(X_train, y_train)
    
    # 4. Evaluate
    print("\n4️⃣  Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n   Accuracy: {accuracy:.2%}")
    
    print("\n   Detailed Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Visualize
    print("5️⃣  Creating visualizations...")
    visualize_results(model, X_test, y_test, feature_names)
    
    # 6. Save model
    print("\n6️⃣  Saving model...")
    joblib.dump(model, 'simple_forest_classifier.pkl')
    print("   ✓ Model saved: simple_forest_classifier.pkl")
    
    # 7. Example prediction
    print("\n7️⃣  Example prediction...")
    
    # Create test patches
    print("\n   Testing on synthetic patches:")
    
    # Healthy forest
    healthy_patch = np.random.normal(0.75, 0.05, (32, 32))
    healthy_patch = np.clip(healthy_patch, 0, 1)
    pred, probs = predict_single_patch(model, healthy_patch, feature_names)
    print(f"\n   Healthy forest (NDVI={healthy_patch.mean():.2f}):")
    print(f"   Prediction: {pred}")
    print(f"   Probabilities: {probs}")
    
    # Degraded forest
    degraded_patch = np.random.normal(0.25, 0.15, (32, 32))
    degraded_patch = np.clip(degraded_patch, 0, 1)
    pred, probs = predict_single_patch(model, degraded_patch, feature_names)
    print(f"\n   Degraded forest (NDVI={degraded_patch.mean():.2f}):")
    print(f"   Prediction: {pred}")
    print(f"   Probabilities: {probs}")
    
    print("\n" + "="*60)
    print("✅ PROCESS COMPLETE")
    print("="*60)
    print("\nTo use the model later:")
    print(">>> import joblib")
    print(">>> model = joblib.load('simple_forest_classifier.pkl')")
    print(">>> prediction = model.predict(features)")


# ============================================================================
# UTILITY FUNCTIONS FOR REAL DATA
# ============================================================================

def load_satellite_image(filepath):
    """
    Load satellite image and extract NDVI
    
    Example for GeoTIFF files (requires rasterio):
    >>> import rasterio
    >>> with rasterio.open('satellite.tif') as src:
    >>>     nir = src.read(4)  # NIR band
    >>>     red = src.read(3)  # Red band
    >>>     ndvi = calculate_ndvi(nir, red)
    """
    pass


def process_large_area(model, ndvi_map, patch_size=32):
    """
    Process a large NDVI map by dividing into patches
    
    Args:
        model: trained classifier
        ndvi_map: 2D NDVI array
        patch_size: size of each patch
    
    Returns:
        risk_map: 2D array with risk classifications
    """
    h, w = ndvi_map.shape
    n_rows = h // patch_size
    n_cols = w // patch_size
    
    risk_map = np.empty((n_rows, n_cols), dtype=object)
    
    print(f"Processing {n_rows}x{n_cols} = {n_rows*n_cols} patches...")
    
    for i in range(n_rows):
        for j in range(n_cols):
            patch = ndvi_map[
                i*patch_size:(i+1)*patch_size,
                j*patch_size:(j+1)*patch_size
            ]
            
            features = extract_simple_features(patch)
            X = np.array([list(features.values())])
            
            risk_map[i, j] = model.predict(X)[0]
    
    print("✓ Processing complete")
    
    return risk_map


def visualize_risk_map_simple(risk_map):
    """Visualize risk classification map"""
    
    risk_colors = {'Low': 0, 'Medium': 1, 'High': 2}
    risk_numeric = np.vectorize(risk_colors.get)(risk_map)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(risk_numeric, cmap='RdYlGn_r', vmin=0, vmax=2)
    plt.colorbar(ticks=[0, 1, 2], label='Risk Level')
    plt.gca().set_yticks([0, 1, 2])
    plt.gca().set_yticklabels(['Low', 'Medium', 'High'])
    plt.title('Forest Risk Classification Map')
    plt.savefig('risk_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Risk map saved: risk_map.png")


def export_results_csv(risk_map, output_file='risk_assessment.csv'):
    """
    Export risk assessment results to CSV
    
    Args:
        risk_map: 2D array with risk classifications
        output_file: output CSV filename
    """
    import pandas as pd
    
    # Flatten and create DataFrame
    results = []
    for i in range(risk_map.shape[0]):
        for j in range(risk_map.shape[1]):
            results.append({
                'row': i,
                'col': j,
                'grid_cell': f"cell_{i}_{j}",
                'risk_level': risk_map[i, j]
            })
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    print(f"✓ Results exported: {output_file}")
    
    # Summary statistics
    print("\nRisk Summary:")
    print(df['risk_level'].value_counts())
