"""Genera un dataset sintetico de figuras (circulo, cuadrado, triangulo) en formato YOLO.
Todo se dibuja con OpenCV/NumPy: no depende de ningun dataset externo.
Uso: python data/generate_dataset.py --n 1800 --out data/shapes_yolo
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from shapes_utils import CLASSES, PALETTE  # noqa: E402

SIZE = 640  # lado de la imagen (coincide con imgsz=640 del entrenamiento)


def jitter(color, j=8):
    """Variacion pequena del color base (para no tener siempre el mismo tono)."""
    c = np.array(color) + np.random.randint(-j, j + 1, 3)
    return tuple(int(x) for x in np.clip(c, 0, 255))


def make_background():
    """Fondo aleatorio: liso, degradado, 'hoja', madera o nubes."""
    kind = random.choice(["solid", "gradient", "paper", "wood", "clouds"])
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    base = np.random.randint(40, 230, 3).astype(np.float32)
    if kind == "solid":
        img = np.ones((SIZE, SIZE, 3), np.float32) * base
    elif kind == "gradient":
        ang = random.uniform(0, np.pi)
        t = (xx * np.cos(ang) + yy * np.sin(ang)) / SIZE
        t = (t - t.min()) / (t.max() - t.min() + 1e-6)
        base2 = np.random.randint(40, 230, 3).astype(np.float32)
        img = base * (1 - t[..., None]) + base2 * t[..., None]
    elif kind == "paper":
        base = np.random.randint(190, 250, 3).astype(np.float32)
        noise = cv2.GaussianBlur(np.random.randn(SIZE, SIZE).astype(np.float32), (0, 0), 1.2) * 6
        img = np.ones((SIZE, SIZE, 3), np.float32) * base + noise[..., None]
    elif kind == "wood":
        base = np.array([random.randint(40, 90), random.randint(80, 140), random.randint(130, 200)], np.float32)
        n = cv2.GaussianBlur(np.random.randn(SIZE, SIZE).astype(np.float32), (0, 0), 25)
        n = n / (n.std() + 1e-6)
        wave = np.sin(yy * random.uniform(0.02, 0.06) + 1.5 * n + random.uniform(0, 6.28))
        img = base * (1 + 0.12 * wave[..., None])
    else:  # clouds
        low = cv2.resize(np.random.rand(8, 8, 3).astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
        img = base * 0.6 + 90 * low
    return np.clip(img, 0, 255).astype(np.uint8)


def rotate(pts, ang):
    c, s = np.cos(ang), np.sin(ang)
    return pts @ np.array([[c, -s], [s, c]]).T


def shape_polygon(cls, s):
    """Poligono centrado en el origen. s ~ tamano de la figura en pixeles."""
    if cls == 0:  # circulo (a veces ligeramente eliptico, como por perspectiva)
        t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        pts = np.stack([s * np.cos(t), s * random.uniform(0.85, 1.0) * np.sin(t)], 1)
        return rotate(pts, random.uniform(0, np.pi))
    if cls == 1:  # cuadrado
        a = s * 0.85
        b = a * random.uniform(0.92, 1.08)
        pts = np.array([[-a, -b], [a, -b], [a, b], [-a, b]], np.float32)
        return rotate(pts, random.uniform(0, np.pi / 2))
    # triangulo (equilatero deformado un poco, con rotacion libre)
    angs = np.deg2rad(np.array([0, 120, 240]) + np.random.uniform(-20, 20, 3) + random.uniform(0, 360))
    r = s * 1.1 * np.random.uniform(0.92, 1.08, 3)
    return np.stack([r * np.cos(angs), r * np.sin(angs)], 1)


def overlaps(box, boxes, pad=6):
    for b in boxes:
        if not (box[2] + pad < b[0] or b[2] + pad < box[0] or box[3] + pad < b[1] or b[3] + pad < box[1]):
            return True
    return False


def postprocess(img):
    """Luz, ruido y desenfoque para parecerse mas a una camara real."""
    img = img.astype(np.float32) * random.uniform(0.75, 1.2) + random.uniform(-20, 20)
    if random.random() < 0.5:  # gradiente de iluminacion
        ang = random.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
        t = (xx * np.cos(ang) + yy * np.sin(ang)) / SIZE
        t = (t - t.min()) / (t.max() - t.min() + 1e-6)
        img *= (random.uniform(0.8, 1.0) + t * random.uniform(0.0, 0.25))[..., None]
    img += np.random.randn(*img.shape) * random.uniform(0, 8)
    img = np.clip(img, 0, 255).astype(np.uint8)
    r = random.random()
    if r < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    elif r < 0.45:  # desenfoque de movimiento
        k = random.choice([5, 7, 9])
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), random.uniform(0, 180), 1)
        kernel = cv2.warpAffine(kernel, M, (k, k))
        img = cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))
    return img


def make_sample():
    """Una imagen con 1 a 5 figuras. Devuelve (imagen, [(clase, color, (x1,y1,x2,y2)), ...])."""
    while True:
        img = make_background()
        bg_mean = img.reshape(-1, 3).mean(0)
        objs, boxes = [], []
        for _ in range(random.randint(1, 5)):
            cls = random.randrange(len(CLASSES))
            for _try in range(30):
                pts = shape_polygon(cls, random.uniform(28, 115))
                outline = random.random() < 0.3
                th = random.randint(2, 4) if outline else 0
                pad = th / 2 + 1
                xmin, ymin = pts.min(0) - pad
                xmax, ymax = pts.max(0) + pad
                m = 4
                if SIZE - xmax - m <= m - xmin or SIZE - ymax - m <= m - ymin:
                    continue
                cx = random.uniform(m - xmin, SIZE - xmax - m)
                cy = random.uniform(m - ymin, SIZE - ymax - m)
                box = (cx + xmin, cy + ymin, cx + xmax, cy + ymax)
                if overlaps(box, boxes):
                    continue
                name = random.choice(list(PALETTE))
                color = jitter(PALETTE[name])
                if np.linalg.norm(np.array(color) - bg_mean) < 70:  # que contraste con el fondo
                    continue
                p = np.round((pts + [cx, cy]) * 16).astype(np.int32)  # subpixel (shift=4)
                cv2.fillPoly(img, [p], color, lineType=cv2.LINE_AA, shift=4)
                if outline:
                    dark = tuple(int(c * 0.45) for c in color)
                    cv2.polylines(img, [p], True, dark, th, cv2.LINE_AA, shift=4)
                boxes.append(box)
                objs.append((cls, name, box))
                break
        if objs:
            return postprocess(img), objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1800)
    ap.add_argument("--out", default="data/shapes_yolo")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    n_train, n_val = int(a.n * 0.7), int(a.n * 0.2)
    splits = [("train", n_train), ("val", n_val), ("test", a.n - n_train - n_val)]  # 70/20/10

    meta = ["split,image,cls,color,x1,y1,x2,y2"]
    idx = 0
    for split, count in splits:
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for _ in range(count):
            img, objs = make_sample()
            name = f"{split}_{idx:05d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, random.randint(75, 95)])
            lines = []
            for cls, color, (x1, y1, x2, y2) in objs:
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, SIZE), min(y2, SIZE)
                # formato YOLO: clase x_centro y_centro ancho alto (todo relativo a 0-1)
                lines.append(f"{cls} {(x1 + x2) / 2 / SIZE:.6f} {(y1 + y2) / 2 / SIZE:.6f} "
                             f"{(x2 - x1) / SIZE:.6f} {(y2 - y1) / SIZE:.6f}")
                meta.append(f"{split},{name}.jpg,{cls},{color},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}")
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            idx += 1

    (out / "meta.csv").write_text("\n".join(meta))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print(f"Dataset generado en {out.resolve()} -> " + ", ".join(f"{s}: {c}" for s, c in splits))


if __name__ == "__main__":
    main()
