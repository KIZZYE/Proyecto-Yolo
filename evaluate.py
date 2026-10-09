"""Reproduce las metricas: mAP@0.5, precision, recall, matriz de confusion y Accuracy_val.
Escribe models/metrics.json (la demo en tiempo real lee de ahi el Acc_val).
Uso: python evaluate.py
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from shapes_utils import CLASSES, iou_xyxy


def read_gt(label_path, w, h):
    """Lee un TXT YOLO y devuelve [(clase, (x1,y1,x2,y2) en pixeles)]."""
    gts = []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            c, x, y, bw, bh = line.split()
            x, y, bw, bh = float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
            gts.append((int(c), (x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2)))
    return gts


def acc_val(model, img_dir, lbl_dir, conf=0.25, iou_thr=0.5):
    """Accuracy_val segun el enunciado: una deteccion es correcta si empareja un Ground Truth
    con IoU >= 0.5 y clase correcta.
        TP = detecciones correctas
        FP = detecciones que no empatan con ningun GT (mal ubicadas, clase incorrecta o duplicadas)
        FN = GT sin deteccion correcta
        Acc_val = TP / (TP + FP + FN)
    """
    tp = fp = fn = 0
    paths = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for p in paths:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gts = read_gt(Path(lbl_dir) / f"{p.stem}.txt", w, h)
        res = model.predict(img, conf=conf, verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            dets = list(zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int),
                            res.boxes.conf.cpu().numpy()))
        dets.sort(key=lambda d: -d[2])  # primero las de mayor confianza
        matched = set()
        for box, c, _ in dets:
            best, best_j = 0.0, -1
            for j, (gc, gb) in enumerate(gts):
                if j in matched or gc != c:
                    continue
                v = iou_xyxy(box, gb)
                if v > best:
                    best, best_j = v, j
            if best >= iou_thr:
                tp += 1
                matched.add(best_j)
            else:
                fp += 1
        fn += len(gts) - len(matched)
    return {"tp": tp, "fp": fp, "fn": fn, "acc_val": tp / max(tp + fp + fn, 1),
            "n_images": len(paths)}


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--data", default="data/shapes_yolo/data.yaml")
    ap.add_argument("--conf", type=float, default=0.25, help="umbral de confianza para Acc_val")
    a = ap.parse_args()

    model = YOLO(a.weights)
    root = Path(a.data).parent
    out = {}
    for split in ["val", "test"]:
        m = model.val(data=str(Path(a.data).resolve()), split=split, imgsz=640, conf=0.001,
                      plots=True, verbose=False, project=str(Path("runs").resolve()),
                      name=f"eval_{split}", exist_ok=True)
        per_class = {}
        for k, ci in enumerate(m.box.ap_class_index):
            per_class[CLASSES[int(ci)]] = {"precision": float(m.box.p[k]), "recall": float(m.box.r[k]),
                                           "ap50": float(m.box.ap50[k])}
        acc = acc_val(model, root / "images" / split, root / "labels" / split, conf=a.conf)
        out[split] = {"map50": float(m.box.map50), "precision": float(m.box.mp), "recall": float(m.box.mr),
                      "per_class": per_class, "confusion_matrix_dir": str(m.save_dir), **acc}
        print(f"\n== {split.upper()} ==  mAP@0.5={m.box.map50:.3f}  P={m.box.mp:.3f}  R={m.box.mr:.3f}  "
              f"Acc_val={acc['acc_val'] * 100:.1f}%  (TP={acc['tp']} FP={acc['fp']} FN={acc['fn']})")
        for c, d in per_class.items():
            print(f"   {c:10s} P={d['precision']:.3f}  R={d['recall']:.3f}  AP50={d['ap50']:.3f}")

    out["acc_val"] = out["val"]["acc_val"] * 100  # lo que muestra la demo
    Path("models").mkdir(exist_ok=True)
    Path("models/metrics.json").write_text(json.dumps(out, indent=2))
    print("\nGuardado en models/metrics.json")


if __name__ == "__main__":
    main()
