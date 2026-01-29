"""
Change Detection - Executable Demo with Real Images
====================================================
Questo script rileva cambiamenti tra due immagini usando tecniche di image differencing.

Uso:
    python run_change_detection.py --before <img1> --after <img2>
    python run_change_detection.py --demo  (usa immagini di esempio del progetto)

Requisiti:
    pip install numpy matplotlib opencv-python Pillow scipy scikit-learn
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non interattivo
import matplotlib.pyplot as plt

# Verifica dipendenze
def check_dependencies():
    missing = []
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

import cv2
from PIL import Image
from scipy import ndimage


class ChangeDetector:
    """
    Rileva cambiamenti tra due immagini usando tecniche di image differencing
    e analisi multispettrale.
    """

    def __init__(self, threshold=0.15):
        self.threshold = threshold

    def detect(self, img_before, img_after):
        """
        Rileva cambiamenti tra due immagini.

        Args:
            img_before: immagine T0 (numpy array normalizzato 0-1)
            img_after: immagine T1 (numpy array normalizzato 0-1)

        Returns:
            change_prob: mappa di probabilita dei cambiamenti
        """
        # Converti a float se necessario
        if img_before.max() > 1:
            img_before = img_before.astype(np.float32) / 255.0
        if img_after.max() > 1:
            img_after = img_after.astype(np.float32) / 255.0

        # 1. Differenza assoluta per canale
        diff = np.abs(img_after.astype(np.float32) - img_before.astype(np.float32))

        # 2. Calcola indice di vegetazione per entrambe
        vi_before = self._vegetation_index(img_before)
        vi_after = self._vegetation_index(img_after)

        # 3. Differenza nell'indice di vegetazione
        vi_diff = np.abs(vi_after - vi_before)

        # 4. Combina le metriche
        # Media pesata: differenza colore + differenza vegetazione
        color_diff = np.mean(diff, axis=2)  # Media sui canali RGB

        # Combinazione: 40% colore, 60% vegetazione
        combined = 0.4 * color_diff + 0.6 * vi_diff

        # 5. Applica smoothing per ridurre rumore
        combined = ndimage.gaussian_filter(combined, sigma=2)

        # 6. Normalizza a 0-1
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-10)

        # 7. Enfatizza i cambiamenti significativi
        # Applica una curva sigmoidale per aumentare il contrasto
        change_prob = 1 / (1 + np.exp(-10 * (combined - 0.3)))

        return change_prob

    def _vegetation_index(self, img):
        """
        Calcola un indice di vegetazione approssimato da RGB.
        Usa Excess Green Index: ExG = 2*G - R - B
        """
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        exg = 2 * g - r - b
        # Normalizza
        exg = (exg - exg.min()) / (exg.max() - exg.min() + 1e-10)
        return exg


def load_and_preprocess_image(image_path, target_size=(512, 512)):
    """
    Carica e preprocessa un'immagine reale

    Args:
        image_path: percorso dell'immagine
        target_size: dimensioni target (height, width)

    Returns:
        numpy array normalizzato (0-1)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")

    # Carica con OpenCV
    img = cv2.imread(image_path)
    if img is None:
        # Prova con PIL per formati non supportati da OpenCV
        img = np.array(Image.open(image_path).convert('RGB'))
    else:
        # Converti BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Ridimensiona
    img = cv2.resize(img, target_size)

    # Normalizza (0-1)
    img = img.astype(np.float32) / 255.0

    return img


def detect_changes(detector, img_before, img_after, threshold=0.5):
    """
    Rileva i cambiamenti tra due immagini

    Args:
        detector: istanza di ChangeDetector
        img_before: immagine T0 (numpy array normalizzato)
        img_after: immagine T1 (numpy array normalizzato)
        threshold: soglia per binarizzazione

    Returns:
        change_prob: mappa di probabilita
        change_binary: mappa binaria dei cambiamenti
        stats: statistiche sui cambiamenti
    """
    # Predizione
    change_prob = detector.detect(img_before, img_after)

    # Binarizzazione
    change_binary = (change_prob > threshold).astype(np.float32)

    # Calcola statistiche
    total_pixels = change_binary.size
    changed_pixels = np.sum(change_binary)
    change_percentage = (changed_pixels / total_pixels) * 100

    stats = {
        'total_pixels': total_pixels,
        'changed_pixels': int(changed_pixels),
        'change_percentage': change_percentage,
        'mean_probability': float(np.mean(change_prob)),
        'max_probability': float(np.max(change_prob))
    }

    return change_prob, change_binary, stats


def visualize_results(img_before, img_after, change_prob, change_binary, stats, output_path='change_detection_result.png'):
    """Visualizza i risultati"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Immagine Before
    axes[0, 0].imshow(img_before)
    axes[0, 0].set_title('BEFORE (T0)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Immagine After
    axes[0, 1].imshow(img_after)
    axes[0, 1].set_title('AFTER (T1)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Mappa di probabilita
    im = axes[1, 0].imshow(change_prob, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Change Probability Map', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, label='Probability')

    # Mappa binaria con overlay
    # Crea overlay rosso sulle aree cambiate
    overlay = img_after.copy()
    mask_rgb = np.stack([change_binary, np.zeros_like(change_binary), np.zeros_like(change_binary)], axis=-1)
    overlay = overlay * 0.6 + mask_rgb * 0.4
    overlay = np.clip(overlay, 0, 1)

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f'Detected Changes: {stats["change_percentage"]:.2f}%', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Aggiungi statistiche
    stats_text = f"""
    Change Detection Results
    ========================
    Total pixels: {stats['total_pixels']:,}
    Changed pixels: {stats['changed_pixels']:,}
    Change area: {stats['change_percentage']:.2f}%
    Mean probability: {stats['mean_probability']:.3f}
    Max probability: {stats['max_probability']:.3f}
    """

    plt.figtext(0.02, 0.02, stats_text, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nRisultati salvati: {output_path}")


def run_demo():
    """Esegue una demo con le immagini DJI del progetto"""

    # Trova la root del progetto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # Cerca immagini DJI
    dji_images = []
    for f in os.listdir(project_root):
        if f.startswith('DJI_') and f.lower().endswith(('.jpg', '.jpeg', '.png')):
            dji_images.append(os.path.join(project_root, f))

    if len(dji_images) < 2:
        print("Demo: non ci sono abbastanza immagini DJI nel progetto.")
        print("Creo immagini sintetiche per la demo...")

        # Crea immagini sintetiche
        img_before = np.random.rand(256, 256, 3).astype(np.float32) * 0.3 + 0.4  # Verde
        img_after = img_before.copy()
        # Aggiungi "deforestazione" simulata
        img_after[80:180, 80:180, :] = np.random.rand(100, 100, 3).astype(np.float32) * 0.3 + 0.2

    else:
        dji_images.sort()
        print(f"Trovate {len(dji_images)} immagini DJI")
        print(f"  - Before: {os.path.basename(dji_images[0])}")
        print(f"  - After: {os.path.basename(dji_images[1])}")

        img_before = load_and_preprocess_image(dji_images[0])
        img_after = load_and_preprocess_image(dji_images[1])

    return img_before, img_after


def main():
    parser = argparse.ArgumentParser(
        description='Change Detection CNN - Rileva cambiamenti tra due immagini',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python run_change_detection.py --demo
  python run_change_detection.py --before forest_2020.jpg --after forest_2024.jpg
  python run_change_detection.py --before img1.jpg --after img2.jpg --threshold 0.3
        """
    )

    parser.add_argument('--before', '-b', type=str, help='Immagine BEFORE (T0)')
    parser.add_argument('--after', '-a', type=str, help='Immagine AFTER (T1)')
    parser.add_argument('--demo', action='store_true', help='Esegui demo con immagini del progetto')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Soglia di binarizzazione (default: 0.5)')
    parser.add_argument('--output', '-o', type=str, default='change_detection_result.png', help='File output')
    parser.add_argument('--size', '-s', type=int, default=512, help='Dimensione immagine (default: 512)')

    args = parser.parse_args()

    print("="*60)
    print("CHANGE DETECTION CNN - Deforestation Detector")
    print("="*60)

    # Carica immagini
    if args.demo:
        print("\nModalita DEMO")
        img_before, img_after = run_demo()
    elif args.before and args.after:
        print(f"\nCaricamento immagini...")
        print(f"  Before: {args.before}")
        print(f"  After: {args.after}")
        img_before = load_and_preprocess_image(args.before, (args.size, args.size))
        img_after = load_and_preprocess_image(args.after, (args.size, args.size))
    else:
        parser.print_help()
        print("\nErrore: specifica --demo oppure --before e --after")
        sys.exit(1)

    print(f"\nDimensioni immagine: {img_before.shape}")

    # Crea detector
    print("\nCreazione Change Detector...")
    detector = ChangeDetector(threshold=args.threshold)
    print("  Metodo: Image Differencing + Vegetation Index Analysis")

    # Rileva cambiamenti
    print(f"\nRilevamento cambiamenti (threshold={args.threshold})...")
    change_prob, change_binary, stats = detect_changes(detector, img_before, img_after, args.threshold)

    # Mostra statistiche
    print("\n" + "-"*40)
    print("RISULTATI:")
    print("-"*40)
    print(f"  Pixel totali:    {stats['total_pixels']:,}")
    print(f"  Pixel cambiati:  {stats['changed_pixels']:,}")
    print(f"  Area cambiata:   {stats['change_percentage']:.2f}%")
    print(f"  Prob. media:     {stats['mean_probability']:.3f}")
    print(f"  Prob. massima:   {stats['max_probability']:.3f}")
    print("-"*40)

    # Visualizza
    visualize_results(img_before, img_after, change_prob, change_binary, stats, args.output)

    print("\n" + "="*60)
    print("COMPLETATO")
    print("="*60)

    return stats


if __name__ == "__main__":
    main()
