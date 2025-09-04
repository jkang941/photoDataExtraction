#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from io import BytesIO
from pathlib import Path

from PIL.TiffImagePlugin import IFDRational
import datetime
import numpy as np
from PIL import Image, ExifTags
from pillow_heif import register_heif_opener

def _json_default(o):
    # Pillow EXIF ratios -> float
    if isinstance(o, IFDRational):
        try:
            return float(o)
        except Exception:
            return str(o)
    # bytes -> utf-8 (lossy ok)
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", errors="replace")
    # numpy scalars -> python scalars
    if isinstance(o, np.generic):
        return o.item()
    # sets/tuples -> lists
    if isinstance(o, (set, tuple)):
        return list(o)
    # datetimes -> ISO
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    # fallback: string
    return str(o)



register_heif_opener()  # Enables Image.open for .heic/.heif

def check_exiftool() -> bool:
    try:
        out = subprocess.run(["exiftool", "-ver"], capture_output=True, text=True)
        return out.returncode == 0
    except FileNotFoundError:
        return False


def run_exiftool_json(path: Path) -> dict:
    """Return rich metadata (JSON) from exiftool."""
    res = subprocess.run(
        ["exiftool", "-j", "-g1", str(path)],
        capture_output=True, text=True
    )
    if res.returncode != 0 or not res.stdout.strip():
        return {}
    try:
        data = json.loads(res.stdout)
        return data[0] if data else {}
    except json.JSONDecodeError:
        return {}


def run_exiftool_binary(path: Path, tag: str) -> bytes | None:
    """
    Extract a binary tag (e.g., DepthMapImage, PortraitEffectsMatte).
    Returns bytes or None if tag not found.
    """
    res = subprocess.run(
        ["exiftool", "-b", f"-{tag}", str(path)],
        capture_output=True
    )
    if res.returncode != 0 or not res.stdout:
        return None
    return res.stdout


def save_image_bytes(img_bytes: bytes, out_path: Path) -> Path | None:
    """Try to load bytes via PIL and save as PNG."""
    try:
        im = Image.open(BytesIO(img_bytes))
        im.save(out_path)
        return out_path
    except Exception:
        return None


def read_base_image(path: Path, out_dir: Path) -> Path:
    """Open HEIC/HEIF/JPEG with Pillow and save a PNG copy for consistency."""
    im = Image.open(path)
    # Convert to 8-bit RGB (handles HEIC/HEIF with different modes)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    out_path = out_dir / f"{path.stem}_base.png"
    im.save(out_path)
    return out_path


def read_basic_exif(path: Path) -> dict:
    """Basic EXIF via Pillow (not as complete as exiftool)."""
    try:
        im = Image.open(path)
        exif = im.getexif()
        if not exif:
            return {}
        decoded = {}
        for k, v in exif.items():
            tag = ExifTags.TAGS.get(k, k)
            decoded[tag] = v
        return decoded
    except Exception:
        return {}


def normalize_depth_png(depth_png: Path) -> Path:
    """
    Optional: normalize a 16-bit depth image to 8-bit for quick viewing.
    Saves alongside original as *_depth_vis.png.
    """
    try:
        im = Image.open(depth_png)
        arr = np.array(im)
        # Handle common 16-bit or floating depth
        arr = arr.astype("float32")
        finite = np.isfinite(arr)
        if not np.any(finite):
            return depth_png  # nothing to normalize
        vmin = float(np.nanmin(arr[finite]))
        vmax = float(np.nanmax(arr[finite]))
        if vmax <= vmin:
            return depth_png
        norm = (arr - vmin) / (vmax - vmin)
        vis = (norm * 255.0).clip(0, 255).astype("uint8")
        out_path = depth_png.with_name(depth_png.stem + "_vis.png")
        Image.fromarray(vis).save(out_path)
        return out_path
    except Exception:
        return depth_png


def main():
    parser = argparse.ArgumentParser(
        description="Extract photo data & depth/mattes from iPhone HEIC/HEIF Portrait photos."
    )
    parser.add_argument("photo", type=str, help="Path to .HEIC/.HEIF/.JPG")
    parser.add_argument("--out", type=str, default=".", help="Output folder")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip making an 8-bit visualization of depth")
    args = parser.parse_args()

    photo_path = Path(args.photo).expanduser()
    out_dir = Path(args.out).expanduser()
    if not out_dir.is_absolute():
        out_dir = Path.home() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Base image copy (PNG)
    base_png = read_base_image(photo_path, out_dir)

    # 2) Metadata: Pillow + exiftool
    pillow_exif = read_basic_exif(photo_path)
    exiftool_ok = check_exiftool()
    exiftool_meta = run_exiftool_json(photo_path) if exiftool_ok else {}

    # 3) Depth & mattes via exiftool
    depth_png = None
    pem_png = None
    segm_outputs = []

    if exiftool_ok:
        # Depth map
        depth_bytes = run_exiftool_binary(photo_path, "DepthMapImage")
        if depth_bytes:
            out_depth = out_dir / f"{photo_path.stem}_depth.png"
            if save_image_bytes(depth_bytes, out_depth):
                depth_png = out_depth
                if not args.no_normalize:
                    normalize_depth_png(depth_png)

        # Portrait Effects Matte (subject cutout)
        pem_bytes = run_exiftool_binary(photo_path, "PortraitEffectsMatte")
        if pem_bytes:
            out_pem = out_dir / f"{photo_path.stem}_portrait_effects_matte.png"
            if save_image_bytes(pem_bytes, out_pem):
                pem_png = out_pem

        # Common semantic segmentation mattes on newer iOS
        segm_tags = [
            ("SemanticSegmentationSkinMatte", "skin"),
            ("SemanticSegmentationHairMatte", "hair"),
            ("SemanticSegmentationTeethMatte", "teeth"),
            ("SemanticSegmentationGlassesMatte", "glasses"),
            ("SemanticSegmentationSkyMatte", "sky"),
            ("SemanticSegmentationBackgroundMatte", "background"),
            ("SemanticSegmentationHeadMatte", "head"),
            ("SemanticSegmentationTorsoMatte", "torso"),
        ]
        for tag, label in segm_tags:
            b = run_exiftool_binary(photo_path, tag)
            if b:
                out_p = out_dir / f"{photo_path.stem}_segm_{label}_matte.png"
                if save_image_bytes(b, out_p):
                    segm_outputs.append(str(out_p))

    # 4) Write a compact metadata JSON
    summary = {
        "input": str(photo_path),
        "outputs": {
            "base_png": str(base_png),
            "depth_png": str(depth_png) if depth_png else None,
            "portrait_effects_matte": str(pem_png) if pem_png else None,
            "segmentation_mattes": segm_outputs,
        },
        "metadata": {
            "pillow_exif": pillow_exif,     # basic EXIF
            "exiftool_full": exiftool_meta, # very detailed
        },
        "notes": {
            "exiftool_available": exiftool_ok,
            "tip": "If depth_png is None, this photo may not be a Portrait (no depth). Try another image."
        }
    }

    meta_path = out_dir / f"{photo_path.stem}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
