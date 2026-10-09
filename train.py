"""Entrena YOLO (>= v8) sobre el dataset de figuras y guarda los mejores pesos en models/best.pt.
Uso: python train.py --epochs 50 --batch 16
"""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/shapes_yolo/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="modelo base (>= v8): yolov8n/s/m, yolo11n...")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="bajar a 8 si la GPU se queda sin memoria")
    ap.add_argument("--patience", type=int, default=10, help="early stopping (epocas sin mejorar)")
    a = ap.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    model = YOLO(a.model)  # parte de pesos preentrenados (transfer learning)
    model.train(
        data=str(Path(a.data).resolve()), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        patience=a.patience,            # early stopping activado
        device=device, seed=42, plots=True,
        project=str(Path("runs").resolve()), name="shapes", exist_ok=True,
    )

    save_dir = Path(model.trainer.save_dir)
    Path("models").mkdir(exist_ok=True)
    shutil.copy(save_dir / "weights" / "best.pt", "models/best.pt")
    print(f"Pesos finales guardados en models/best.pt (corrida: {save_dir})")


if __name__ == "__main__":
    main()
