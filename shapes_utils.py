"""Utilidades compartidas: clases, paleta, color por HSV, IoU y dibujo en pantalla."""
import cv2
import numpy as np

CLASSES = ["circulo", "cuadrado", "triangulo"]
BOX_COLORS = [(0, 200, 255), (255, 140, 0), (0, 220, 0)]  # BGR del recuadro por clase

# Paleta que se usa al generar el dataset (BGR). Sirve tambien para probar el detector de color.
PALETTE = {
    "rojo": (40, 40, 210),
    "naranja": (30, 130, 240),
    "amarillo": (30, 220, 230),
    "verde": (60, 170, 60),
    "azul": (200, 80, 30),
    "morado": (150, 50, 130),
    "negro": (25, 25, 25),
    "blanco": (240, 240, 240),
}


def color_name(pixels_bgr):
    """Nombre del color a partir de pixeles BGR: mediana en BGR y luego umbrales en HSV."""
    px = np.asarray(pixels_bgr).reshape(-1, 3)
    med = np.median(px, axis=0).astype(np.uint8).reshape(1, 1, 3)
    h, s, v = [int(x) for x in cv2.cvtColor(med, cv2.COLOR_BGR2HSV)[0, 0]]
    if v < 70:
        return "negro"
    if s < 45:
        return "blanco" if v > 150 else "gris"
    if h < 8 or h >= 165:
        return "rojo"
    if h < 20:
        return "naranja"
    if h < 36:
        return "amarillo"
    if h < 85:
        return "verde"
    if h < 135:
        return "azul"
    return "morado"


def dominant_color(roi):
    """Color de la figura dentro de su bounding box.
    Estima el fondo con las esquinas del recuadro y se queda con los pixeles distintos al fondo
    de la zona central; si casi no hay (p. ej. cuadrado alineado), usa el centro del recuadro."""
    h, w = roi.shape[:2]
    if h < 4 or w < 4:
        return "desconocido"
    k = max(2, int(0.1 * min(h, w)))
    corners = np.concatenate([
        roi[:k, :k].reshape(-1, 3), roi[:k, -k:].reshape(-1, 3),
        roi[-k:, :k].reshape(-1, 3), roi[-k:, -k:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(roi.astype(np.float32) - bg, axis=2)
    inner = np.zeros((h, w), bool)
    inner[int(.2 * h):int(.8 * h), int(.2 * w):int(.8 * w)] = True
    mask = (dist > 45) & inner
    if mask.sum() >= 0.1 * max(inner.sum(), 1):
        return color_name(roi[mask])
    return color_name(roi[int(.35 * h):int(.65 * h), int(.35 * w):int(.65 * w)])


def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def annotate(frame, result, acc_val=None):
    """Dibuja bbox + clase + color + confianza y el overlay fijo Acc_val sobre el frame."""
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        conf = result.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            x2, y2 = min(int(x2), frame.shape[1]), min(int(y2), frame.shape[0])
            col = dominant_color(frame[y1:y2, x1:x2])
            text = f"{CLASSES[c]} | {col} | {p:.2f}"
            box_color = BOX_COLORS[c % len(BOX_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ty = max(y1, th + 8)
            cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 6, ty), box_color, -1)
            cv2.putText(frame, text, (x1 + 3, ty - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 1, cv2.LINE_AA)
    # Overlay fijo (esquina superior izquierda)
    txt = "Acc_val=N/A" if acc_val is None else f"Acc_val={acc_val:.1f}%"
    cv2.rectangle(frame, (8, 8), (230, 44), (0, 0, 0), -1)
    cv2.putText(frame, txt, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return frame
