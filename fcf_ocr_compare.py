#!/usr/bin/env python3
"""
FCF OCR Comparison Pipeline
============================
Processes a folder of FCF crop images through Surya OCR and generates
a self-contained HTML file showing each image side-by-side with its
detected text for easy visual comparison.

Optimizations for small GD&T / FCF crops:
  - Upscales images 3x so the text detector can actually find lines
  - Disables math mode to prevent LaTeX-style output artifacts
  - Optionally skips detection and treats the whole image as one text box

Usage:
    python3 fcf_ocr_compare.py --input_folder /path/to/images --output_dir ./ocr_output
    python3 fcf_ocr_compare.py --input_folder /path/to/images --output_dir ./ocr_output --scale 4
    python3 fcf_ocr_compare.py --input_folder /path/to/images --output_dir ./ocr_output --no-detect
"""

import argparse
import base64
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Surya OCR imports
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.common.surya.schema import TaskNames


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# HTML tags Surya sometimes injects (formatting markup, not actual text)
HTML_TAGS_TO_FILTER = [
    "u", "b", "i", "s", "em", "strong", "sub", "sup", "mark", "br",
    "p", "li", "ul", "ol", "table", "td", "tr", "th", "tbody", "pre",
]

# Regex to strip any residual HTML tags from final text
HTML_TAG_RE = re.compile(r'</?(?:u|b|i|s|em|strong|sub|sup|mark)\s*/?>', re.IGNORECASE)


def clean_ocr_text(text: str) -> str:
    """Remove spurious HTML tags and clean up OCR output."""
    text = HTML_TAG_RE.sub('', text)
    text = text.strip()
    return text


def encode_image_base64(image_path: str) -> str:
    """Read an image file and return its Base64-encoded data URI."""
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".webp": "image/webp",
    }
    mime = mime_map.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def upscale_image(img: Image.Image, scale: int) -> Image.Image:
    """Upscale an image by integer factor using LANCZOS resampling."""
    if scale <= 1:
        return img
    new_size = (img.width * scale, img.height * scale)
    return img.resize(new_size, Image.LANCZOS)


def remove_modifier_circles(pil_img: Image.Image) -> tuple[Image.Image, Image.Image | None]:
    """
    Detect and erase GD&T modifier circles from an image.

    Returns:
        (cleaned_image, annotated_image_or_None)
        - cleaned_image: circles erased, letter preserved
        - annotated_image: original with green highlights on detected circles
          (None if no circles found)
    """
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return pil_img, None

    cleaned = img_cv.copy()
    annotated = img_cv.copy()
    circles_found = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        diameter = max(bw, bh)
        has_child = hierarchy[0][i][2] != -1

        if (circularity > 0.85
                and 0.85 < aspect < 1.15
                and (0.25 * h) < diameter < (0.6 * h)
                and has_child):
            # Erase circle on cleaned image
            cv2.drawContours(cleaned, contours, i, (255, 255, 255), thickness=3)
            circles_found.append((i, x, y, bw, bh, circularity))

    if not circles_found:
        return pil_img, None

    # Build annotated image: draw green highlight over detected circles
    for idx, x, y, bw, bh, circ in circles_found:
        # Green circle outline
        cv2.drawContours(annotated, contours, idx, (0, 200, 0), thickness=2)
        # Green bounding box
        cv2.rectangle(annotated, (x - 2, y - 2), (x + bw + 2, y + bh + 2), (0, 200, 0), 1)
        # Label with circularity score
        label = f"circ={circ:.2f}"
        font_scale = max(0.3, min(h / 120, 0.6))
        cv2.putText(annotated, label, (x, max(y - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 200, 0), 1, cv2.LINE_AA)

    cleaned_pil = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    return cleaned_pil, annotated_pil


def build_html(results: list[dict], show_debug: bool = False) -> str:
    """
    Build a self-contained HTML string.

    Each entry in `results` is:
        {
            "filename": str,
            "image_data_uri": str,
            "annotated_data_uri": str | None,  (circle detection debug view)
            "text_lines": list[{"text": str, "confidence": float}],
        }
    """
    cards_html = []
    for i, r in enumerate(results):
        lines_html = ""
        if r["text_lines"]:
            for line in r["text_lines"]:
                conf = line["confidence"]
                conf_class = "high" if conf >= 0.8 else ("med" if conf >= 0.5 else "low")
                text = line["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                lines_html += (
                    f'<div class="line">'
                    f'<span class="text">{text}</span>'
                    f'<span class="conf {conf_class}">{conf:.0%}</span>'
                    f'</div>\n'
                )
        else:
            lines_html = '<div class="line"><span class="text no-text">No text detected</span></div>'

        # Middle column: annotated debug image (only when --remove-circles)
        debug_col = ""
        if show_debug:
            ann_uri = r.get("annotated_data_uri")
            if ann_uri:
                debug_col = f"""
                <div class="image-side debug-side">
                    <div class="step-label">Circle Detection</div>
                    <img src="{ann_uri}" alt="circle detection" />
                </div>"""
            else:
                debug_col = f"""
                <div class="image-side debug-side">
                    <div class="step-label">Circle Detection</div>
                    <div class="no-circles">No modifier circles found</div>
                </div>"""

        cards_html.append(f"""
        <div class="card">
            <div class="card-header">
                <span class="index">#{i + 1}</span>
                <span class="filename">{r["filename"]}</span>
            </div>
            <div class="card-body">
                <div class="image-side">
                    <div class="step-label">Original Input</div>
                    <img src="{r["image_data_uri"]}" alt="{r["filename"]}" />
                </div>
                {debug_col}
                <div class="text-side">
                    <div class="text-header">Detected Text</div>
                    {lines_html}
                </div>
            </div>
        </div>
        """)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>FCF OCR Comparison</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242836;
    --border: #2e3345;
    --text: #e4e6ed;
    --text-dim: #8b8fa3;
    --accent: #6c7aff;
    --accent2: #a78bfa;
    --high: #34d399;
    --med: #fbbf24;
    --low: #f87171;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 2rem;
    line-height: 1.5;
  }}
  h1 {{
    text-align: center;
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: .4rem;
  }}
  .subtitle {{
    text-align: center;
    color: var(--text-dim);
    font-size: .9rem;
    margin-bottom: 2rem;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 1.5rem;
    overflow: hidden;
    transition: border-color .2s;
  }}
  .card:hover {{ border-color: var(--accent); }}
  .card-header {{
    padding: .75rem 1.25rem;
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: .75rem;
  }}
  .index {{
    background: var(--accent);
    color: #fff;
    font-size: .75rem;
    font-weight: 700;
    padding: .15rem .55rem;
    border-radius: 6px;
  }}
  .filename {{
    font-size: .85rem;
    font-weight: 600;
    color: var(--text-dim);
    word-break: break-all;
  }}
  .card-body {{
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    padding: 1.25rem;
  }}
  .image-side {{
    flex: 1 1 250px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: var(--surface2);
    border-radius: 8px;
    padding: .75rem;
    min-height: 100px;
    gap: .5rem;
  }}
  .image-side img {{
    max-width: 100%;
    max-height: 400px;
    object-fit: contain;
    border-radius: 4px;
  }}
  .step-label {{
    font-size: .65rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--accent2);
    font-weight: 700;
  }}
  .debug-side {{
    border: 1px dashed var(--accent2);
  }}
  .no-circles {{
    color: var(--text-dim);
    font-style: italic;
    font-size: .8rem;
    padding: 1rem;
  }}
  .text-side {{
    flex: 1 1 300px;
    min-width: 0;
  }}
  .text-header {{
    font-size: .75rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--accent);
    font-weight: 700;
    margin-bottom: .6rem;
  }}
  .line {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: .75rem;
    padding: .35rem .6rem;
    border-radius: 6px;
    margin-bottom: .25rem;
    background: var(--surface2);
  }}
  .line .text {{
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: .85rem;
    word-break: break-all;
  }}
  .no-text {{ color: var(--text-dim); font-style: italic; }}
  .conf {{
    font-size: .7rem;
    font-weight: 700;
    flex-shrink: 0;
    padding: .1rem .4rem;
    border-radius: 4px;
  }}
  .conf.high {{ color: var(--high); background: rgba(52,211,153,.12); }}
  .conf.med  {{ color: var(--med);  background: rgba(251,191,36,.12); }}
  .conf.low  {{ color: var(--low);  background: rgba(248,113,113,.12); }}

  @media (max-width: 700px) {{
    body {{ padding: 1rem; }}
    .card-body {{ flex-direction: column; }}
  }}
</style>
</head>
<body>
<h1>FCF OCR Comparison</h1>
<p class="subtitle">{len(results)} images processed with Surya OCR</p>
{"".join(cards_html)}
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(
        description="Run Surya OCR on a folder of FCF crop images and generate an HTML comparison."
    )
    parser.add_argument(
        "--input_folder", required=True, help="Path to folder containing input images"
    )
    parser.add_argument(
        "--output_dir",
        default="./ocr_output",
        help="Directory to write comparison.html (default: ./ocr_output)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=3,
        help="Upscale factor for images before OCR (default: 3). Higher = better detection on small crops.",
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Skip text detection — treat the entire image as one text region. "
             "Use this for already-cropped FCF elements.",
    )
    parser.add_argument(
        "--task",
        choices=["ocr_with_boxes", "ocr_without_boxes", "block_without_boxes"],
        default="ocr_with_boxes",
        help="Surya OCR task name (default: ocr_with_boxes). "
             "Try 'ocr_without_boxes' or 'block_without_boxes' if results are poor.",
    )
    parser.add_argument(
        "--remove-circles",
        action="store_true",
        help="Remove GD&T modifier circles (Ⓜ, Ⓟ, Ⓛ, etc.) before OCR. "
             "Improves extraction of letters enclosed in circles.",
    )
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_dir = Path(args.output_dir)

    if not input_folder.is_dir():
        print(f"Error: {input_folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Collect image paths
    image_paths = sorted(
        [p for p in input_folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )
    if not image_paths:
        print(f"Error: No supported images found in {input_folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {input_folder}")
    # Cap scale for --no-detect mode (full-image bboxes + high scale = tensor overflow)
    if args.no_detect and args.scale > 2:
        print(f"⚠  --no-detect mode: capping scale from {args.scale}x to 2x (large images crash the encoder)")
        args.scale = 2

    print(f"Settings: scale={args.scale}x, skip_detect={args.no_detect}, task={args.task}, math_mode=off")

    # Load Surya OCR models
    print("Loading Surya OCR models...")
    foundation_predictor = FoundationPredictor()
    recognition_predictor = RecognitionPredictor(foundation_predictor)
    if not args.no_detect:
        detection_predictor = DetectionPredictor()
    print("Models loaded.")

    # Load, preprocess, and upscale images
    circle_label = ", circle-removal=on" if args.remove_circles else ""
    print(f"Loading & preprocessing images ({args.scale}x{circle_label})...")
    pil_images = []
    annotated_images = []  # debug visualizations (None if no circles)
    valid_paths = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            img = upscale_image(img, args.scale)
            ann = None
            if args.remove_circles:
                img, ann = remove_modifier_circles(img)
            pil_images.append(img)
            annotated_images.append(ann)
            valid_paths.append(p)
        except Exception as e:
            print(f"  ⚠ Skipping {p.name}: {e}")

    # Map task name string to TaskNames constant
    task_name_map = {
        "ocr_with_boxes": TaskNames.ocr_with_boxes,
        "ocr_without_boxes": TaskNames.ocr_without_boxes,
        "block_without_boxes": TaskNames.block_without_boxes,
    }
    task = task_name_map[args.task]

    # Run OCR
    print("Running OCR on all images...")
    if args.no_detect:
        # Process one-by-one in no-detect mode to avoid hangs on large images
        # max_tokens caps generation length, drop_repeated_text stops loops
        predictions = []
        for idx, img in enumerate(pil_images):
            print(f"  [{idx+1}/{len(pil_images)}] {valid_paths[idx].name} ...", end="", flush=True)
            try:
                pred = recognition_predictor(
                    [img],
                    bboxes=[[[0, 0, img.width, img.height]]],
                    task_names=[task],
                    math_mode=False,
                    max_tokens=128,
                    drop_repeated_text=True,
                    filter_tag_list=HTML_TAGS_TO_FILTER,
                )
                predictions.extend(pred)
                n_lines = len(pred[0].text_lines) if pred else 0
                print(f" {n_lines} line(s)")
            except Exception as e:
                print(f" ⚠ ERROR: {e}")
                from surya.recognition.schema import OCRResult
                predictions.append(OCRResult(text_lines=[], image_bbox=[0, 0, img.width, img.height]))
    else:
        predictions = recognition_predictor(
            pil_images,
            det_predictor=detection_predictor,
            task_names=[task] * len(pil_images),
            math_mode=False,
            max_tokens=128,
            drop_repeated_text=True,
            filter_tag_list=HTML_TAGS_TO_FILTER,
        )

    # Build results
    results = []
    for idx_r, (path, pred) in enumerate(zip(valid_paths, predictions)):
        text_lines = []
        for line in pred.text_lines:
            cleaned = clean_ocr_text(line.text)
            if not cleaned:
                continue
            text_lines.append({
                "text": cleaned,
                "confidence": line.confidence,
            })

        # Build annotated data URI if available
        ann_uri = None
        if args.remove_circles and annotated_images[idx_r] is not None:
            import io
            buf = io.BytesIO()
            annotated_images[idx_r].save(buf, format="PNG")
            ann_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            ann_uri = f"data:image/png;base64,{ann_b64}"

        results.append({
            "filename": path.name,
            "image_data_uri": encode_image_base64(str(path)),
            "annotated_data_uri": ann_uri,
            "text_lines": text_lines,
        })
        line_count = len(text_lines)
        preview = text_lines[0]["text"][:50] if text_lines else ""
        print(f"  ✓ {path.name} — {line_count} line(s)" + (f'  "{preview}"' if preview else ""))

    # Generate HTML
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "comparison.html"
    html_content = build_html(results, show_debug=args.remove_circles)
    html_path.write_text(html_content, encoding="utf-8")

    print(f"\n✅ Done! HTML written to: {html_path.resolve()}")
    print(f"   Open it in your browser to compare results.")


if __name__ == "__main__":
    main()
