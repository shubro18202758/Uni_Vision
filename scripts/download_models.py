"""Download pre-trained YOLOv8n weights and export to ONNX format.

Usage::

    python -m scripts.download_models          # download + export
    python -m scripts.download_models --skip-export  # download only

Requires ``ultralytics`` to be installed::

    pip install ultralytics

The script downloads two YOLOv8n variants:
  1. ``yolov8n.pt``  — COCO-pretrained vehicle detector (cars, trucks, buses, motorcycles)
  2. ``yolov8n.pt``  — same base model, to be fine-tuned for plate detection

Both are exported to ONNX with dynamic batch and opset 17 for
ONNX Runtime CUDA EP compatibility.

For TensorRT INT8 engines, use ``trtexec`` on the target GPU::

    trtexec --onnx=models/vehicle_detector.onnx \
            --saveEngine=models/vehicle_detector.engine \
            --int8 --fp16 --workspace=4096
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _ensure_ultralytics():
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        print(
            "ERROR: 'ultralytics' package not found.\n"
            "Install it with:  pip install ultralytics",
            file=sys.stderr,
        )
        sys.exit(1)


def download_and_export(skip_export: bool = False) -> None:
    """Download YOLOv8n and optionally export to ONNX."""
    _ensure_ultralytics()
    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Vehicle detector (COCO classes: car=2, truck=7, bus=5, motorcycle=3)
    print("[1/2] Downloading YOLOv8n for vehicle detection ...")
    vehicle_model = YOLO("yolov8n.pt")

    if not skip_export:
        print("       Exporting to ONNX ...")
        vehicle_model.export(
            format="onnx",
            imgsz=640,
            opset=17,
            dynamic=True,
            simplify=True,
        )
        # ultralytics saves the .onnx next to the .pt; move it
        exported = Path("yolov8n.onnx")
        target = MODELS_DIR / "vehicle_detector.onnx"
        if exported.exists():
            shutil.move(str(exported), str(target))
            print(f"       Saved → {target}")

    pt_target = MODELS_DIR / "vehicle_detector.pt"
    shutil.copy2("yolov8n.pt", str(pt_target))
    print(f"       Weights → {pt_target}")

    # ── Plate detector (same architecture, needs fine-tuning)
    print("[2/2] Copying YOLOv8n for plate detection (fine-tune required) ...")
    plate_pt = MODELS_DIR / "plate_detector.pt"
    shutil.copy2("yolov8n.pt", str(plate_pt))

    if not skip_export:
        # Export plate detector ONNX from the same base model
        plate_model = YOLO(str(plate_pt))
        plate_model.export(
            format="onnx",
            imgsz=640,
            opset=17,
            dynamic=True,
            simplify=True,
        )
        plate_onnx = MODELS_DIR / "plate_detector.onnx"
        exported_plate = plate_pt.with_suffix(".onnx")
        if exported_plate.exists() and exported_plate != plate_onnx:
            shutil.move(str(exported_plate), str(plate_onnx))
        print(f"       Saved → {plate_onnx}")

    # ── Summary
    print("\n✓ Models ready in:", MODELS_DIR)
    print("  vehicle_detector.onnx  —  COCO-pretrained (filter classes 2,3,5,7)")
    print("  plate_detector.onnx    —  base model (fine-tune on plate dataset)")
    print(
        "\nTo build TensorRT INT8 engines on your target GPU:\n"
        "  trtexec --onnx=models/vehicle_detector.onnx "
        "--saveEngine=models/vehicle_detector.engine --int8 --fp16\n"
        "  trtexec --onnx=models/plate_detector.onnx "
        "--saveEngine=models/plate_detector.engine --int8 --fp16"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download & export YOLOv8n models")
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Download .pt weights only — skip ONNX export",
    )
    args = parser.parse_args()
    download_and_export(skip_export=args.skip_export)


if __name__ == "__main__":
    main()
