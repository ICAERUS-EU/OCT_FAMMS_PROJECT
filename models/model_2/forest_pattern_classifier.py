"""
Pattern Recognition Forest Classifier
Random Forest model for anomaly detection based on NDVI and canopy texture
Framework: Scikit-learn
Output: Risk classification (Low, Medium, High) per grid cell
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
import joblib
import warnings
warnings.filterwarnings('ignore')


class ForestPatternClassifier:
    """
    Random Forest classifier for detecting forest anomalies and deforestation risk
    based on vegetation indices (NDVI) and canopy texture features.
    """
    
    def __init__(self, n_estimators=200, max_depth=20, random_state=42):
        """
        Initialize the Forest Pattern Classifier
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
    def extract_ndvi_features(self, ndvi_image):
        """
        Extract features from NDVI (Normalized Difference Vegetation Index) image
        
        NDVI formula: (NIR - Red) / (NIR + Red)
        Values range from -1 to 1, where:
        - High values (0.6 to 1.0) = dense vegetation
        - Medium values (0.2 to 0.6) = sparse vegetation
        - Low values (-1.0 to 0.2) = no vegetation
        
        Args:
            ndvi_image: 2D numpy array with NDVI values
            
        Returns:
            dict with NDVI-based features
        """
        features = {}
        
        # Basic statistics
        features['ndvi_mean'] = np.mean(ndvi_image)
        features['ndvi_std'] = np.std(ndvi_image)
        features['ndvi_min'] = np.min(ndvi_image)
        features['ndvi_max'] = np.max(ndvi_image)
        features['ndvi_median'] = np.median(ndvi_image)
        features['ndvi_range'] = features['ndvi_max'] - features['ndvi_min']
        
        # Percentiles
        features['ndvi_p25'] = np.percentile(ndvi_image, 25)
        features['ndvi_p75'] = np.percentile(ndvi_image, 75)
        features['ndvi_iqr'] = features['ndvi_p75'] - features['ndvi_p25']
        
        # Vegetation density categories
        dense_veg = (ndvi_image >= 0.6).sum()
        sparse_veg = ((ndvi_image >= 0.2) & (ndvi_image < 0.6)).sum()
        no_veg = (ndvi_image < 0.2).sum()
        total_pixels = ndvi_image.size
        
        features['dense_vegetation_ratio'] = dense_veg / total_pixels
        features['sparse_vegetation_ratio'] = sparse_veg / total_pixels
        features['no_vegetation_ratio'] = no_veg / total_pixels
        
        # Variability measures
        features['ndvi_cv'] = features['ndvi_std'] / (features['ndvi_mean'] + 1e-7)
        features['ndvi_skewness'] = self._calculate_skewness(ndvi_image)
        features['ndvi_kurtosis'] = self._calculate_kurtosis(ndvi_image)
        
        return features
    
    def extract_texture_features(self, image, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """
        Extract texture features using Gray-Level Co-occurrence Matrix (GLCM)
        
        GLCM measures how often pairs of pixel values occur in a specified direction
        and distance. Useful for detecting patterns in canopy structure.
        
        Args:
            image: 2D numpy array (grayscale image)
            distances: list of pixel distances to consider
            angles: list of angles to consider (in radians)
            
        Returns:
            dict with texture features
        """
        features = {}
        
        # Normalize image to 0-255 and convert to uint8
        if image.max() > 0:
            image_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image_norm = np.zeros_like(image, dtype=np.uint8)
        
        # Calculate GLCM
        glcm = graycomatrix(
            image_norm,
            distances=distances,
            angles=angles,
            levels=256,
            symmetric=True,
            normed=True
        )
        
        # Extract properties
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        
        for prop in properties:
            values = graycoprops(glcm, prop).flatten()
            features[f'texture_{prop}_mean'] = np.mean(values)
            features[f'texture_{prop}_std'] = np.std(values)
            features[f'texture_{prop}_max'] = np.max(values)
        
        # Additional texture measures
        features['texture_entropy'] = self._calculate_entropy(image_norm)
        features['texture_smoothness'] = 1 - (1 / (1 + features['texture_contrast_mean']))
        
        # Edge detection features
        sobel_h = ndimage.sobel(image, axis=0)
        sobel_v = ndimage.sobel(image, axis=1)
        edges = np.hypot(sobel_h, sobel_v)
        
        features['edge_density'] = (edges > edges.mean()).sum() / edges.size
        features['edge_mean'] = np.mean(edges)
        features['edge_std'] = np.std(edges)
        
        return features
    
    def extract_spatial_features(self, ndvi_image):
        """
        Extract spatial pattern features
        
        Args:
            ndvi_image: 2D numpy array with NDVI values
            
        Returns:
            dict with spatial features
        """
        features = {}
        
        # Create binary mask for vegetation (NDVI > 0.3)
        veg_mask = (ndvi_image > 0.3).astype(int)
        
        # Connected components analysis
        labeled, num_features = ndimage.label(veg_mask)
        
        features['num_vegetation_patches'] = num_features
        
        if num_features > 0:
            # Patch sizes
            patch_sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            features['mean_patch_size'] = np.mean(patch_sizes)
            features['std_patch_size'] = np.std(patch_sizes)
            features['max_patch_size'] = np.max(patch_sizes)
            features['min_patch_size'] = np.min(patch_sizes)
            
            # Fragmentation index
            features['fragmentation_index'] = num_features / (veg_mask.sum() + 1)
        else:
            features['mean_patch_size'] = 0
            features['std_patch_size'] = 0
            features['max_patch_size'] = 0
            features['min_patch_size'] = 0
            features['fragmentation_index'] = 0
        
        # Spatial autocorrelation
        features['spatial_autocorr'] = self._calculate_morans_i(ndvi_image)
        
        return features
    
    def extract_all_features(self, ndvi_image, nir_image=None, red_image=None):
        """
        Extract all features from an image
        
        Args:
            ndvi_image: 2D numpy array with NDVI values
            nir_image: Optional NIR band
            red_image: Optional Red band
            
        Returns:
            dict with all features
        """
        features = {}
        
        # NDVI features
        ndvi_feats = self.extract_ndvi_features(ndvi_image)
        features.update(ndvi_feats)
        
        # Texture features from NDVI
        texture_feats = self.extract_texture_features(ndvi_image)
        features.update(texture_feats)
        
        # Spatial features
        spatial_feats = self.extract_spatial_features(ndvi_image)
        features.update(spatial_feats)
        
        # If NIR and Red bands available, extract additional features
        if nir_image is not None and red_image is not None:
            features['nir_mean'] = np.mean(nir_image)
            features['red_mean'] = np.mean(red_image)
            features['nir_red_ratio'] = features['nir_mean'] / (features['red_mean'] + 1e-7)
        
        return features
    
    def prepare_training_data(self, images_list, labels_list):
        """
        Prepare training data from a list of images and labels
        
        Args:
            images_list: list of NDVI images (2D numpy arrays)
            labels_list: list of labels ('Low', 'Medium', 'High')
            
        Returns:
            X: feature matrix
            y: label array
        """
        features_list = []
        
        print(f"Extracting features from {len(images_list)} images...")
        
        for i, (image, label) in enumerate(zip(images_list, labels_list)):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(images_list)} images...")
            
            feats = self.extract_all_features(image)
            features_list.append(feats)
        
        # Convert to DataFrame
        X = pd.DataFrame(features_list)
        y = np.array(labels_list)
        
        self.feature_names = list(X.columns)
        
        print(f"✓ Extracted {len(self.feature_names)} features")
        
        return X.values, y
    
    def fit(self, X, y):
        """
        Train the Random Forest model
        
        Args:
            X: feature matrix
            y: labels
        """
        print("\nTraining Random Forest Classifier...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        
        print("✓ Model trained successfully!")
        
        return self
    
    def predict(self, X):
        """
        Predict risk class
        
        Args:
            X: feature matrix
            
        Returns:
            predictions: array of class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: feature matrix
            
        Returns:
            probabilities: array of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance rankings
        
        Args:
            top_n: number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importances = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath):
        """Save the trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"✓ Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        print(f"✓ Model loaded from: {filepath}")
    
    # Helper methods
    def _calculate_skewness(self, data):
        """Calculate skewness of distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_entropy(self, image):
        """Calculate Shannon entropy"""
        histogram, _ = np.histogram(image, bins=256, range=(0, 256))
        histogram = histogram / histogram.sum()
        histogram = histogram[histogram > 0]
        return -np.sum(histogram * np.log2(histogram))
    
    def _calculate_morans_i(self, image):
        """Calculate Moran's I for spatial autocorrelation"""
        n = image.size
        mean = np.mean(image)
        
        # Simplified version for 2D grid
        diff = image - mean
        numerator = np.sum(diff[:-1, :] * diff[1:, :]) + np.sum(diff[:, :-1] * diff[:, 1:])
        denominator = np.sum(diff ** 2)
        
        if denominator == 0:
            return 0
        
        return (n / (2 * (image.shape[0] - 1) * image.shape[1])) * (numerator / denominator)


def generate_synthetic_data(n_samples=300, img_size=64):
    """
    Generate synthetic NDVI data with different risk patterns
    
    Args:
        n_samples: number of samples to generate
        img_size: size of each image patch
        
    Returns:
        images: list of NDVI images
        labels: list of risk labels
    """
    print(f"\nGenerating {n_samples} synthetic forest samples...")
    
    images = []
    labels = []
    
    # Distribution: 40% Low, 35% Medium, 25% High
    n_low = int(n_samples * 0.4)
    n_medium = int(n_samples * 0.35)
    n_high = n_samples - n_low - n_medium
    
    # Generate LOW risk samples (healthy forest)
    for _ in range(n_low):
        # High NDVI, low variability
        base_ndvi = np.random.uniform(0.6, 0.85)
        noise = np.random.normal(0, 0.05, (img_size, img_size))
        ndvi = np.clip(base_ndvi + noise, 0, 1)
        
        images.append(ndvi)
        labels.append('Low')
    
    # Generate MEDIUM risk samples (some disturbance)
    for _ in range(n_medium):
        # Medium NDVI with some patches of degradation
        base_ndvi = np.random.uniform(0.35, 0.6)
        noise = np.random.normal(0, 0.1, (img_size, img_size))
        ndvi = np.clip(base_ndvi + noise, 0, 1)
        
        # Add random degraded patches
        n_patches = np.random.randint(1, 4)
        for _ in range(n_patches):
            x, y = np.random.randint(0, img_size-10, 2)
            size = np.random.randint(5, 15)
            ndvi[y:y+size, x:x+size] *= np.random.uniform(0.3, 0.6)
        
        images.append(ndvi)
        labels.append('Medium')
    
    # Generate HIGH risk samples (severe degradation)
    for _ in range(n_high):
        # Low NDVI with high variability and fragmentation
        base_ndvi = np.random.uniform(0.1, 0.4)
        noise = np.random.normal(0, 0.15, (img_size, img_size))
        ndvi = np.clip(base_ndvi + noise, 0, 1)
        
        # Add large degraded areas
        n_patches = np.random.randint(3, 7)
        for _ in range(n_patches):
            x, y = np.random.randint(0, img_size-15, 2)
            size = np.random.randint(10, 25)
            ndvi[y:y+size, x:x+size] *= np.random.uniform(0.1, 0.4)
        
        images.append(ndvi)
        labels.append('High')
    
    print(f"✓ Generated samples:")
    print(f"  - Low risk: {n_low}")
    print(f"  - Medium risk: {n_medium}")
    print(f"  - High risk: {n_high}")
    
    return images, labels


def visualize_samples(images, labels, n_samples=9):
    """Visualize sample images with their labels"""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(n_samples, len(images))):
        idx = np.random.randint(0, len(images))
        
        im = axes[i].imshow(images[idx], cmap='RdYlGn', vmin=0, vmax=1)
        axes[i].set_title(f'Risk: {labels[idx]}\nMean NDVI: {images[idx].mean():.3f}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('sample_ndvi_images.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Sample visualization saved: sample_ndvi_images.png")


def evaluate_model(model, X_test, y_test, class_names):
    """
    Comprehensive model evaluation
    
    Args:
        model: trained classifier
        X_test: test features
        y_test: test labels
        class_names: list of class names
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=class_names, average=None
    )
    
    print("\nPer-Class Metrics:")
    print("-" * 60)
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    print(metrics_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Feature Importance
    importance_df = model.get_feature_importance(top_n=15)
    axes[1].barh(range(len(importance_df)), importance_df['importance'])
    axes[1].set_yticks(range(len(importance_df)))
    axes[1].set_yticklabels(importance_df['feature'])
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Evaluation plots saved: model_evaluation.png")
    
    return accuracy, cm


def predict_risk_map(model, ndvi_map, grid_size=64):
    """
    Predict risk classification for an entire NDVI map by dividing into grid cells
    
    Args:
        model: trained ForestPatternClassifier
        ndvi_map: 2D numpy array with NDVI values
        grid_size: size of each grid cell
        
    Returns:
        risk_map: 2D array with risk classifications
        prob_map: 3D array with class probabilities
    """
    h, w = ndvi_map.shape
    
    # Calculate grid dimensions
    n_rows = h // grid_size
    n_cols = w // grid_size
    
    print(f"\nProcessing NDVI map: {h}x{w} pixels")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Number of cells: {n_rows}x{n_cols} = {n_rows * n_cols}")
    
    # Initialize output maps
    risk_map = np.empty((n_rows, n_cols), dtype=object)
    prob_map = np.zeros((n_rows, n_cols, 3))
    
    # Process each grid cell
    for i in range(n_rows):
        for j in range(n_cols):
            # Extract grid cell
            y_start, y_end = i * grid_size, (i + 1) * grid_size
            x_start, x_end = j * grid_size, (j + 1) * grid_size
            
            cell = ndvi_map[y_start:y_end, x_start:x_end]
            
            # Extract features
            features = model.extract_all_features(cell)
            X = pd.DataFrame([features])[model.feature_names].values
            
            # Predict
            risk_map[i, j] = model.predict(X)[0]
            prob_map[i, j, :] = model.predict_proba(X)[0]
    
    print("✓ Risk map generated")
    
    return risk_map, prob_map


def visualize_risk_map(ndvi_map, risk_map, prob_map):
    """Visualize the risk classification map"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Original NDVI
    im0 = axes[0, 0].imshow(ndvi_map, cmap='RdYlGn', vmin=0, vmax=1)
    axes[0, 0].set_title('Original NDVI Map', fontsize=14, fontweight='bold')
    plt.colorbar(im0, ax=axes[0, 0], label='NDVI')
    
    # Risk Classification
    risk_colors = {'Low': 0, 'Medium': 1, 'High': 2}
    risk_numeric = np.vectorize(risk_colors.get)(risk_map)
    
    im1 = axes[0, 1].imshow(risk_numeric, cmap='RdYlGn_r', vmin=0, vmax=2)
    axes[0, 1].set_title('Risk Classification Map', fontsize=14, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=axes[0, 1], ticks=[0, 1, 2])
    cbar1.set_ticklabels(['Low', 'Medium', 'High'])
    
    # Probability maps
    im2 = axes[1, 0].imshow(prob_map[:, :, 2], cmap='Reds', vmin=0, vmax=1)
    axes[1, 0].set_title('High Risk Probability', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=axes[1, 0], label='Probability')
    
    # Risk statistics
    axes[1, 1].axis('off')
    
    total_cells = risk_map.size
    low_count = (risk_map == 'Low').sum()
    medium_count = (risk_map == 'Medium').sum()
    high_count = (risk_map == 'High').sum()
    
    stats_text = f"""
    RISK STATISTICS
    ═══════════════════════════
    
    Total Grid Cells: {total_cells}
    
    Low Risk:     {low_count:4d} ({low_count/total_cells*100:5.1f}%)
    Medium Risk:  {medium_count:4d} ({medium_count/total_cells*100:5.1f}%)
    High Risk:    {high_count:4d} ({high_count/total_cells*100:5.1f}%)
    
    ═══════════════════════════
    
    Mean NDVI: {ndvi_map.mean():.3f}
    Std NDVI:  {ndvi_map.std():.3f}
    
    Critical Areas (High Risk):
    {high_count} cells require attention
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('risk_classification_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Risk map visualization saved: risk_classification_map.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("="*60)
    print("FOREST PATTERN RECOGNITION & ANOMALY DETECTION")
    print("Random Forest Classifier for Deforestation Risk Assessment")
    print("="*60)
    
    # 1. Generate synthetic data
    images, labels = generate_synthetic_data(n_samples=300, img_size=64)
    
    # Visualize samples
    visualize_samples(images, labels)
    
    # 2. Initialize classifier
    print("\n" + "="*60)
    print("INITIALIZING CLASSIFIER")
    print("="*60)
    
    classifier = ForestPatternClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42
    )
    
    # 3. Prepare training data
    print("\n" + "="*60)
    print("FEATURE EXTRACTION")
    print("="*60)
    
    X, y = classifier.prepare_training_data(images, labels)
    
    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # 5. Train model
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    classifier.fit(X_train, y_train)
    
    # 6. Evaluate model
    class_names = ['High', 'Low', 'Medium']
    accuracy, cm = evaluate_model(classifier, X_test, y_test, class_names)
    
    # 7. Feature importance analysis
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    top_features = classifier.get_feature_importance(top_n=20)
    print("\nTop 20 Most Important Features:")
    print(top_features.to_string(index=False))
    
    # 8. Cross-validation
    print("\n" + "="*60)
    print("CROSS-VALIDATION")
    print("="*60)
    
    cv_scores = cross_val_score(
        classifier.model,
        classifier.scaler.transform(X),
        y,
        cv=5,
        scoring='accuracy'
    )
    
    print(f"\n5-Fold Cross-Validation Scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    print(f"\nMean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 9. Save model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    classifier.save_model('forest_classifier_model.pkl')
    
    # 10. Example: Predict on full NDVI map
    print("\n" + "="*60)
    print("RISK MAP PREDICTION EXAMPLE")
    print("="*60)
    
    # Generate a large synthetic NDVI map
    large_map_size = 512
    ndvi_large = np.random.rand(large_map_size, large_map_size)
    
    # Add patterns (healthy areas, degraded areas)
    ndvi_large[:256, :256] = np.clip(np.random.normal(0.7, 0.1, (256, 256)), 0, 1)  # Healthy
    ndvi_large[256:, 256:] = np.clip(np.random.normal(0.3, 0.15, (256, 256)), 0, 1)  # Degraded
    
    # Predict risk map
    risk_map, prob_map = predict_risk_map(classifier, ndvi_large, grid_size=64)
    
    # Visualize
    visualize_risk_map(ndvi_large, risk_map, prob_map)
    
    print("\n" + "="*60)
    print("✓ PROCESS COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nGenerated files:")
    print("  1. sample_ndvi_images.png - Sample NDVI visualizations")
    print("  2. model_evaluation.png - Model performance metrics")
    print("  3. risk_classification_map.png - Risk prediction map")
    print("  4. forest_classifier_model.pkl - Trained model file")
    print("\nTo use the model later:")
    print(">>> classifier = ForestPatternClassifier()")
    print(">>> classifier.load_model('forest_classifier_model.pkl')")
    print(">>> predictions = classifier.predict(X_new)")
