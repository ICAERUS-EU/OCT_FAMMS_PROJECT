"""
Forest Pattern Classifier - Executable Demo with Real Images
=============================================================
Questo script permette di classificare il rischio di deforestazione da immagini reali.

Uso:
    python run_forest_classifier.py --image <immagine>
    python run_forest_classifier.py --demo  (usa immagini di esempio del progetto)
    python run_forest_classifier.py --folder <cartella>  (analizza tutte le immagini)

Requisiti:
    pip install scikit-learn numpy matplotlib opencv-python Pillow scipy scikit-image
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non interattivo
import matplotlib.pyplot as plt
from pathlib import Path

# Verifica dipendenze
def check_dependencies():
    missing = []
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        missing.append('scikit-learn')
    try:
        import cv2
    except ImportError:
        missing.append('opencv-python')
    try:
        from PIL import Image
    except ImportError:
        missing.append('Pillow')
    try:
        from scipy import ndimage
    except ImportError:
        missing.append('scipy')

    if missing:
        print(f"Dipendenze mancanti: {', '.join(missing)}")
        print(f"Installa con: pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
from scipy import ndimage
import joblib


def load_image(image_path):
    """
    Carica un'immagine da file

    Args:
        image_path: percorso dell'immagine

    Returns:
        numpy array RGB
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")

    # Carica con OpenCV
    img = cv2.imread(image_path)
    if img is None:
        # Prova con PIL
        img = np.array(Image.open(image_path).convert('RGB'))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def calculate_pseudo_ndvi(rgb_image):
    """
    Calcola uno pseudo-NDVI da immagine RGB.

    Per immagini RGB senza banda NIR, usiamo:
    pseudo_NDVI = (G - R) / (G + R + epsilon)

    dove G = Green channel, R = Red channel.
    Questo approssima la differenza di riflettanza vegetazione/suolo.

    Args:
        rgb_image: immagine RGB (numpy array)

    Returns:
        pseudo_ndvi: array 2D con valori pseudo-NDVI
    """
    # Estrai canali e normalizza
    r = rgb_image[:, :, 0].astype(np.float32)
    g = rgb_image[:, :, 1].astype(np.float32)

    # Calcola pseudo-NDVI (Green-Red Vegetation Index)
    epsilon = 1e-10
    pseudo_ndvi = (g - r) / (g + r + epsilon)

    # Normalizza a range 0-1
    pseudo_ndvi = (pseudo_ndvi - pseudo_ndvi.min()) / (pseudo_ndvi.max() - pseudo_ndvi.min() + epsilon)

    return pseudo_ndvi


def calculate_excess_green(rgb_image):
    """
    Calcola Excess Green Index (ExG) - un altro indicatore di vegetazione.

    ExG = 2*G - R - B

    Args:
        rgb_image: immagine RGB

    Returns:
        exg: array 2D con valori normalizzati
    """
    r = rgb_image[:, :, 0].astype(np.float32) / 255.0
    g = rgb_image[:, :, 1].astype(np.float32) / 255.0
    b = rgb_image[:, :, 2].astype(np.float32) / 255.0

    exg = 2 * g - r - b

    # Normalizza a 0-1
    exg = (exg - exg.min()) / (exg.max() - exg.min() + 1e-10)

    return exg


def extract_features(image_patch):
    """
    Estrae features da un patch di immagine per la classificazione.

    Args:
        image_patch: patch RGB

    Returns:
        dict con features
    """
    # Calcola indici vegetazione
    pseudo_ndvi = calculate_pseudo_ndvi(image_patch)
    exg = calculate_excess_green(image_patch)

    features = {}

    # === Features da pseudo-NDVI ===
    features['ndvi_mean'] = np.mean(pseudo_ndvi)
    features['ndvi_std'] = np.std(pseudo_ndvi)
    features['ndvi_min'] = np.min(pseudo_ndvi)
    features['ndvi_max'] = np.max(pseudo_ndvi)
    features['ndvi_median'] = np.median(pseudo_ndvi)
    features['ndvi_range'] = features['ndvi_max'] - features['ndvi_min']

    # Percentili
    features['ndvi_p25'] = np.percentile(pseudo_ndvi, 25)
    features['ndvi_p75'] = np.percentile(pseudo_ndvi, 75)
    features['ndvi_iqr'] = features['ndvi_p75'] - features['ndvi_p25']

    # Rapporti vegetazione
    features['high_veg_ratio'] = (pseudo_ndvi > 0.6).sum() / pseudo_ndvi.size
    features['medium_veg_ratio'] = ((pseudo_ndvi >= 0.3) & (pseudo_ndvi <= 0.6)).sum() / pseudo_ndvi.size
    features['low_veg_ratio'] = (pseudo_ndvi < 0.3).sum() / pseudo_ndvi.size

    # === Features da Excess Green ===
    features['exg_mean'] = np.mean(exg)
    features['exg_std'] = np.std(exg)

    # === Features di colore RGB ===
    features['red_mean'] = np.mean(image_patch[:, :, 0]) / 255.0
    features['green_mean'] = np.mean(image_patch[:, :, 1]) / 255.0
    features['blue_mean'] = np.mean(image_patch[:, :, 2]) / 255.0

    features['green_red_ratio'] = features['green_mean'] / (features['red_mean'] + 1e-10)
    features['green_blue_ratio'] = features['green_mean'] / (features['blue_mean'] + 1e-10)

    # === Features di texture ===
    gray = cv2.cvtColor(image_patch, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Gradiente (edge detection)
    sobel_x = ndimage.sobel(gray, axis=1)
    sobel_y = ndimage.sobel(gray, axis=0)
    gradient_magnitude = np.hypot(sobel_x, sobel_y)

    features['texture_gradient_mean'] = np.mean(gradient_magnitude)
    features['texture_gradient_std'] = np.std(gradient_magnitude)
    features['edge_density'] = (gradient_magnitude > gradient_magnitude.mean()).sum() / gradient_magnitude.size

    # Varianza locale (misura di eterogeneita)
    local_variance = ndimage.generic_filter(gray, np.var, size=5)
    features['local_variance_mean'] = np.mean(local_variance)
    features['local_variance_std'] = np.std(local_variance)

    # === Features spaziali ===
    # Frammentazione basata su soglia vegetazione
    veg_mask = (pseudo_ndvi > 0.4).astype(int)
    labeled, num_patches = ndimage.label(veg_mask)
    features['num_vegetation_patches'] = num_patches

    if num_patches > 0 and veg_mask.sum() > 0:
        features['fragmentation_index'] = num_patches / (veg_mask.sum() + 1)
    else:
        features['fragmentation_index'] = 0

    return features


def create_and_train_classifier():
    """
    Crea e addestra un classificatore Random Forest con dati sintetici.
    In produzione, usare un dataset reale etichettato.

    Returns:
        model: classificatore addestrato
        feature_names: lista nomi features
    """
    print("Creazione dataset di training sintetico...")

    # Genera patches sintetici
    n_samples = 300
    patch_size = 64
    features_list = []
    labels = []

    # LOW RISK - foresta sana (verde uniforme)
    for _ in range(int(n_samples * 0.35)):
        # Verde intenso con poca variazione
        g = np.random.randint(100, 180, (patch_size, patch_size))
        r = g - np.random.randint(30, 60, (patch_size, patch_size))
        b = g - np.random.randint(20, 50, (patch_size, patch_size))
        patch = np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)], axis=-1).astype(np.uint8)

        features = extract_features(patch)
        features_list.append(list(features.values()))
        labels.append('Low')

    # MEDIUM RISK - vegetazione mista
    for _ in range(int(n_samples * 0.35)):
        g = np.random.randint(80, 150, (patch_size, patch_size))
        r = np.random.randint(80, 150, (patch_size, patch_size))
        b = np.random.randint(60, 120, (patch_size, patch_size))
        patch = np.stack([r, g, b], axis=-1).astype(np.uint8)

        # Aggiungi qualche zona degradata
        n_degraded = np.random.randint(1, 3)
        for _ in range(n_degraded):
            x, y = np.random.randint(0, patch_size - 15, 2)
            size = np.random.randint(8, 20)
            patch[y:y+size, x:x+size, 1] = np.random.randint(40, 80)  # Meno verde

        features = extract_features(patch)
        features_list.append(list(features.values()))
        labels.append('Medium')

    # HIGH RISK - suolo esposto / deforestazione
    for _ in range(n_samples - int(n_samples * 0.35) - int(n_samples * 0.35)):
        # Colori marroni/grigi (suolo)
        base = np.random.randint(100, 180, (patch_size, patch_size))
        r = base + np.random.randint(10, 40, (patch_size, patch_size))
        g = base - np.random.randint(0, 30, (patch_size, patch_size))
        b = base - np.random.randint(20, 50, (patch_size, patch_size))
        patch = np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)], axis=-1).astype(np.uint8)

        features = extract_features(patch)
        features_list.append(list(features.values()))
        labels.append('High')

    # Ottieni nomi features
    dummy_patch = np.zeros((64, 64, 3), dtype=np.uint8)
    feature_names = list(extract_features(dummy_patch).keys())

    X = np.array(features_list)
    y = np.array(labels)

    print(f"  Samples: {len(X)} (Low: {(y=='Low').sum()}, Medium: {(y=='Medium').sum()}, High: {(y=='High').sum()})")
    print(f"  Features: {len(feature_names)}")

    # Addestramento
    print("Addestramento Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"  Accuracy su test set: {accuracy:.2%}")

    return model, feature_names


def classify_image(model, feature_names, image, grid_size=64):
    """
    Classifica un'immagine dividendola in una griglia.

    Args:
        model: classificatore addestrato
        feature_names: nomi delle features
        image: immagine RGB
        grid_size: dimensione di ogni cella della griglia

    Returns:
        risk_map: mappa delle classificazioni
        prob_map: mappa delle probabilita
        stats: statistiche
    """
    h, w = image.shape[:2]
    n_rows = h // grid_size
    n_cols = w // grid_size

    print(f"Analisi immagine: {w}x{h} pixel")
    print(f"Griglia: {n_cols}x{n_rows} celle ({grid_size}x{grid_size} pixel ciascuna)")

    risk_map = np.empty((n_rows, n_cols), dtype=object)
    prob_map = np.zeros((n_rows, n_cols, 3))

    for i in range(n_rows):
        for j in range(n_cols):
            y_start, y_end = i * grid_size, (i + 1) * grid_size
            x_start, x_end = j * grid_size, (j + 1) * grid_size

            patch = image[y_start:y_end, x_start:x_end]

            # Estrai features
            features = extract_features(patch)
            X = np.array([[features[fn] for fn in feature_names]])

            # Predici
            risk_map[i, j] = model.predict(X)[0]
            prob_map[i, j, :] = model.predict_proba(X)[0]

    # Calcola statistiche
    total = risk_map.size
    low_count = (risk_map == 'Low').sum()
    medium_count = (risk_map == 'Medium').sum()
    high_count = (risk_map == 'High').sum()

    stats = {
        'total_cells': total,
        'low_risk': low_count,
        'medium_risk': medium_count,
        'high_risk': high_count,
        'low_pct': (low_count / total) * 100,
        'medium_pct': (medium_count / total) * 100,
        'high_pct': (high_count / total) * 100
    }

    return risk_map, prob_map, stats


def visualize_results(image, risk_map, prob_map, stats, output_path='forest_risk_assessment.png'):
    """Visualizza i risultati della classificazione"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Immagine originale
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Immagine Originale', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Mappa di rischio
    risk_colors = {'Low': 0, 'Medium': 1, 'High': 2}
    risk_numeric = np.vectorize(risk_colors.get)(risk_map)

    cmap = plt.cm.RdYlGn_r
    im = axes[0, 1].imshow(risk_numeric, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
    axes[0, 1].set_title('Mappa di Rischio Deforestazione', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    cbar = plt.colorbar(im, ax=axes[0, 1], ticks=[0, 1, 2], fraction=0.046)
    cbar.set_ticklabels(['Low', 'Medium', 'High'])

    # Probabilita rischio alto
    im2 = axes[1, 0].imshow(prob_map[:, :, 0], cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    axes[1, 0].set_title('Probabilita Rischio ALTO', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, label='Probability')

    # Statistiche
    axes[1, 1].axis('off')

    stats_text = f"""
    REPORT ANALISI FORESTA
    ══════════════════════════════════════

    Celle analizzate: {stats['total_cells']}

    CLASSIFICAZIONE RISCHIO:

      BASSO (Low):    {stats['low_risk']:4d} celle ({stats['low_pct']:5.1f}%)
      MEDIO (Medium): {stats['medium_risk']:4d} celle ({stats['medium_pct']:5.1f}%)
      ALTO (High):    {stats['high_risk']:4d} celle ({stats['high_pct']:5.1f}%)

    ══════════════════════════════════════

    ALERT: {stats['high_risk']} celle richiedono attenzione

    Legenda:
      - Low:    Vegetazione sana
      - Medium: Vegetazione degradata
      - High:   Possibile deforestazione
    """

    axes[1, 1].text(0.05, 0.5, stats_text, fontsize=12, family='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nRisultati salvati: {output_path}")


def run_demo():
    """Esegue demo con immagini del progetto"""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # Cerca immagini DJI
    dji_images = []
    for f in os.listdir(project_root):
        if f.startswith('DJI_') and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            dji_images.append(os.path.join(project_root, f))

    if dji_images:
        dji_images.sort()
        print(f"Trovate {len(dji_images)} immagini DJI")
        image_path = dji_images[0]
        print(f"Analisi di: {os.path.basename(image_path)}")
        return load_image(image_path)
    else:
        print("Nessuna immagine DJI trovata. Creo immagine sintetica...")
        # Crea immagine sintetica
        size = 512
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Quadrante verde (foresta sana)
        img[:size//2, :size//2, 1] = np.random.randint(100, 180, (size//2, size//2))
        img[:size//2, :size//2, 0] = img[:size//2, :size//2, 1] - 40
        img[:size//2, :size//2, 2] = img[:size//2, :size//2, 1] - 30

        # Quadrante misto
        img[:size//2, size//2:, :] = np.random.randint(80, 140, (size//2, size//2, 3))

        # Quadrante marrone (deforestato)
        img[size//2:, :size//2, 0] = np.random.randint(140, 200, (size//2, size//2))
        img[size//2:, :size//2, 1] = np.random.randint(100, 140, (size//2, size//2))
        img[size//2:, :size//2, 2] = np.random.randint(80, 120, (size//2, size//2))

        # Quadrante grigio (urbano/suolo)
        img[size//2:, size//2:, :] = np.random.randint(120, 160, (size//2, size//2, 3))

        return img


def main():
    parser = argparse.ArgumentParser(
        description='Forest Pattern Classifier - Analisi rischio deforestazione',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python run_forest_classifier.py --demo
  python run_forest_classifier.py --image satellite.jpg
  python run_forest_classifier.py --image drone.jpg --grid 32
        """
    )

    parser.add_argument('--image', '-i', type=str, help='Immagine da analizzare')
    parser.add_argument('--demo', action='store_true', help='Esegui demo')
    parser.add_argument('--grid', '-g', type=int, default=64, help='Dimensione griglia (default: 64)')
    parser.add_argument('--output', '-o', type=str, default='forest_risk_assessment.png', help='File output')

    args = parser.parse_args()

    print("="*60)
    print("FOREST PATTERN CLASSIFIER")
    print("Random Forest - Analisi Rischio Deforestazione")
    print("="*60)

    # Carica immagine
    if args.demo:
        print("\nModalita DEMO")
        image = run_demo()
    elif args.image:
        print(f"\nCaricamento: {args.image}")
        image = load_image(args.image)
    else:
        parser.print_help()
        print("\nErrore: specifica --demo oppure --image")
        sys.exit(1)

    print(f"Dimensioni: {image.shape[1]}x{image.shape[0]} pixel")

    # Crea e addestra classificatore
    print("\n" + "-"*40)
    model, feature_names = create_and_train_classifier()
    print("-"*40)

    # Classifica immagine
    print("\nClassificazione in corso...")
    risk_map, prob_map, stats = classify_image(model, feature_names, image, args.grid)

    # Mostra risultati
    print("\n" + "="*40)
    print("RISULTATI:")
    print("="*40)
    print(f"  Celle totali:  {stats['total_cells']}")
    print(f"  Rischio BASSO: {stats['low_risk']} ({stats['low_pct']:.1f}%)")
    print(f"  Rischio MEDIO: {stats['medium_risk']} ({stats['medium_pct']:.1f}%)")
    print(f"  Rischio ALTO:  {stats['high_risk']} ({stats['high_pct']:.1f}%)")
    print("="*40)

    # Visualizza
    visualize_results(image, risk_map, prob_map, stats, args.output)

    print("\n" + "="*60)
    print("COMPLETATO")
    print("="*60)

    return stats


if __name__ == "__main__":
    main()
