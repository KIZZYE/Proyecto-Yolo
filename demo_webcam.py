"""Demo en tiempo real: bbox + clase + color + confianza + overlay fijo Acc_val.
Uso:
  python demo_webcam.py                       # webcam 0
  python demo_webcam.py --cam 1               # otra camara
  python demo_webcam.py --save demo.mp4       # ademas graba el video
  python demo_webcam.py --source video.mp4 --no-show --save salida.mp4   # sin ventana (p. ej. Colab)
Teclas: q / ESC salir, s captura de pantalla.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from shapes_utils import annotate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--metrics", default="models/metrics.json")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--source", default=None, help="archivo de video en vez de webcam")
    ap.add_argument("--conf", type=float, default=0.9)
    ap.add_argument("--save", default=None, help="ruta del mp4 de salida")
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()

    acc = None
    if Path(a.metrics).exists():
        acc = json.loads(Path(a.metrics).read_text()).get("acc_val")
    else:
        print("Aviso: no hay models/metrics.json, corre primero: python evaluate.py")

    model = YOLO(a.weights)
    cap = cv2.VideoCapture(a.source if a.source else a.cam)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la camara/video.")
    writer, prev, fps = None, time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, conf=a.conf, imgsz=640, verbose=False)[0]
        frame = annotate(frame, res, acc)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.0f}", (frame.shape[1] - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if a.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(frame)
        if not a.no_show:
            cv2.imshow("YOLO - formas geometricas", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                cv2.imwrite(f"captura_{int(time.time())}.png", frame)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
