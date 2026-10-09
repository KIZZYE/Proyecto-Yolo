"""Captura cuadros de la webcam para reentrenar con datos reales.
Version lista para FIGURAS: basta con correr  python capture_figuras.py
Uso (en tu PC):
  python capture_frames.py --mode vacio     # SIN figuras: techo, lamparas, paredes, tu cara, la mesa...
  python capture_frames.py --mode figuras   # CON figuras (dibujadas, impresas o recortadas)
Teclas: ESPACIO guarda un cuadro | a activa/desactiva captura automatica | q o ESC salir
Las fotos quedan en mis_fotos/<modo>/ (maximo 640 px de lado).
"""
import argparse
import time
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vacio", "figuras"], default="figuras")
    ap.add_argument("--out", default="mis_fotos")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--auto", type=float, default=0.8, help="segundos entre fotos en modo automatico")
    a = ap.parse_args()

    out = Path(a.out) / a.mode
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.cam)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la camara.")

    n = len(list(out.glob("*.jpg")))
    auto, last = False, 0.0
    print(f"Guardando en {out}/  (ya hay {n} fotos)")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        s = 640 / max(h, w)
        small = cv2.resize(frame, (int(w * s), int(h * s))) if s < 1 else frame
        now = time.time()

        guardar = False
        if auto and now - last >= a.auto:
            guardar, last = True, now
        if guardar:
            cv2.imwrite(str(out / f"{a.mode}_{int(now * 1000)}.jpg"), small)
            n += 1

        view = frame.copy()
        msg = f"{a.mode.upper()} | fotos: {n} | auto: {'ON' if auto else 'OFF'}"
        cv2.rectangle(view, (8, 8), (8 + 12 * len(msg), 40), (0, 0, 0), -1)
        cv2.putText(view, msg, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Captura de cuadros", view)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            cv2.imwrite(str(out / f"{a.mode}_{int(now * 1000)}.jpg"), small)
            n += 1
        if k == ord("a"):
            auto = not auto

    cap.release()
    cv2.destroyAllWindows()
    print(f"Listo: {n} fotos en {out}/")


if __name__ == "__main__":
    main()
