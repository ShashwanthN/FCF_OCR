# FCF OCR Comparison Pipeline

A specialized OCR pipeline designed for processing **Feature Control Frame (FCF)** and **Geometric Dimensioning and Tolerancing (GD&T)** crops from engineering drawings. 

This tool uses [Surya OCR](https://github.com/vikashelina/surya) with custom preprocessing to handle the unique challenges of small, high-density engineering symbols.

## Key Features

- **Adaptive Upscaling**: Automatically upscales small crops (default 3x) to improve text detection performance.
- **Modifier Circle Removal**: Specialized logic to detect and erase GD&T modifier circles (like Ⓜ, Ⓟ, Ⓛ) to isolate the enclosed letters for better recognition.
- **Optimized for Small Text**: Disables LaTeX/math mode and offers a "no-detect" mode for specific FCF segments.
- **Interactive Reports**: Generates a self-contained, premium dark-mode HTML report (`comparison.html`) for side-by-side visual verification.

## Setup

### Prerequisites
- Python 3.9+
- [Surya OCR](https://github.com/vikashelina/surya)
- OpenCV, NumPy, and Pillow

### Installation
```bash
pip install surya-ocr opencv-python numpy pillow
```

## Usage

Basic execution:
```bash
python3 fcf_ocr_compare.py --input_folder /path/to/crops --output_dir ./ocr_results
```

### Advanced Flags

| Flag | Purpose | Recommended For |
| :--- | :--- | :--- |
| `--scale [N]` | Sets upscaling factor (default: 3). | Very small or blurry symbol crops. |
| `--no-detect` | Skips the detection step; treats the whole image as one box. | Already isolated single-character or single-line FCF elements. |
| `--remove-circles` | Erases modifier circles before OCR. | FCFs containing Ⓜ (MMC), Ⓟ (Projected Tolerance Zone), etc. |
| `--task [task]` | Choose `ocr_with_boxes`, `ocr_without_boxes`, or `block_without_boxes`. | Fine-tuning extraction based on layout density. |

### Example Command
```bash
python3 fcf_ocr_compare.py \
    --input_folder ./data/fcf_crops \
    --output_dir ./output \
    --scale 4 \
    --remove-circles \
    --no-detect
```

## Output
The script generates an `ocr_output/comparison.html` file.
- **Left Column**: Original input image.
- **Middle Column** (Optional): Debug view showing detected/removed modifier circles.
- **Right Column**: Extracted text with per-line confidence scores.

---
*Part of the Adeos SuryaOCR Toolset.*
