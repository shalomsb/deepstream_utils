#!/usr/bin/env python3
"""Export all RF-DETR model sizes to ONNX.

Usage (inside Docker container):
    pip install rfdetr
    python3 /models/export_rfdetr_all.py

For 2XLarge:
    pip install rfdetr[plus]
"""

from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium, RFDETRLarge

MODELS = {
    "rfdetr_nano":   RFDETRNano,
    "rfdetr_small":  RFDETRSmall,
    "rfdetr_base":   RFDETRBase,
    "rfdetr_medium": RFDETRMedium,
    "rfdetr":        RFDETRLarge,
}

try:
    from rfdetr import RFDETRXLarge, RFDETR2XLarge
    MODELS["rfdetr_xl"] = RFDETRXLarge
    MODELS["rfdetr_2xl"] = RFDETR2XLarge
except ImportError:
    print("Skipping XLarge/2XLarge (install rfdetr[plus] to enable)")

for name, cls in MODELS.items():
    output_dir = f"/models/{name}"
    print(f"\n{'='*60}")
    print(f"Exporting {cls.__name__} -> {output_dir}")
    print(f"{'='*60}")
    m = cls()
    m.export(output_dir=output_dir)

print("\nDone! All models exported.")
