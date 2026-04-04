"""Translate text within images using OCR + Claude + Pillow rendering."""
from __future__ import annotations

import io
import base64
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


MIN_IMAGE_DIMENSION = 150  # pixels — skip images smaller than this
UPSCALE_FACTOR = 3  # scale images up before OCR/rendering for precision


def is_image_too_small(image_bytes: bytes) -> bool:
    """Check if an image is too small to contain meaningful text."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        return w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION
    except Exception:
        return True


@dataclass
class TextRegion:
    """A detected text region in an image."""
    bbox: List[List[int]]  # 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    text: str
    confidence: float


def _get_easyocr_reader():
    """Lazy-load EasyOCR reader (downloads model on first use)."""
    import warnings
    import logging
    warnings.filterwarnings("ignore", message=".*pin_memory.*")
    logging.getLogger("easyocr").setLevel(logging.ERROR)
    import easyocr
    if not hasattr(_get_easyocr_reader, "_reader"):
        _get_easyocr_reader._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _get_easyocr_reader._reader


def detect_text(image_bytes: bytes, confidence_threshold: float = 0.3) -> List[TextRegion]:
    """Detect text regions in an image using EasyOCR."""
    reader = _get_easyocr_reader()
    nparr = np.frombuffer(image_bytes, np.uint8)

    import cv2
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    results = reader.readtext(img)

    regions = []
    for bbox, text, conf in results:
        if conf >= confidence_threshold and text.strip():
            # Convert float coords to int
            int_bbox = [[int(p[0]), int(p[1])] for p in bbox]
            regions.append(TextRegion(bbox=int_bbox, text=text.strip(), confidence=conf))

    return regions


def _bbox_to_rect(bbox: List[List[int]]) -> Tuple[int, int, int, int]:
    """Convert 4-corner bbox to (left, top, right, bottom) rectangle."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def sample_background_color(
    image: Image.Image, bbox: List[List[int]], margin: int = 3
) -> Tuple[int, int, int]:
    """Sample the background color around a bounding box by taking the median
    of pixels just outside the box edges."""
    left, top, right, bottom = _bbox_to_rect(bbox)
    w, h = image.size
    pixels = []

    # Sample pixels along each edge, just outside the bbox
    for x in range(max(0, left - margin), min(w, right + margin)):
        for dy in [-margin, bottom - top + margin]:
            y = top + dy
            if 0 <= y < h:
                pixels.append(image.getpixel((x, y)))

    for y in range(max(0, top - margin), min(h, bottom + margin)):
        for dx in [-margin, right - left + margin]:
            x = left + dx
            if 0 <= x < w:
                pixels.append(image.getpixel((x, y)))

    if not pixels:
        return (255, 255, 255)  # Default to white

    # Take median of each channel
    pixels_arr = np.array(pixels)
    if pixels_arr.ndim == 1:
        return (255, 255, 255)

    # Handle RGBA by only using RGB
    median = np.median(pixels_arr[:, :3], axis=0).astype(int)
    return tuple(median)


def _detect_text_color(
    image: Image.Image, bbox: List[List[int]], bg_color: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    """Estimate the text color by finding the most common non-background color
    inside the bounding box."""
    left, top, right, bottom = _bbox_to_rect(bbox)
    pixels = []

    for x in range(max(0, left), min(image.size[0], right)):
        for y in range(max(0, top), min(image.size[1], bottom)):
            px = image.getpixel((x, y))
            rgb = px[:3] if len(px) > 3 else px
            # Skip pixels close to the background color
            diff = sum(abs(a - b) for a, b in zip(rgb, bg_color))
            if diff > 60:  # Threshold: "different enough" from background
                pixels.append(rgb)

    if not pixels:
        # Fallback: use black or white, whichever contrasts more with background
        bg_brightness = sum(bg_color) / 3
        return (0, 0, 0) if bg_brightness > 128 else (255, 255, 255)

    pixels_arr = np.array(pixels)

    # Determine if text is darker or lighter than background
    bg_brightness = sum(bg_color) / 3
    pixel_brightness = np.mean(pixels_arr, axis=1)

    if np.median(pixel_brightness) < bg_brightness:
        # Dark text on light background — take the darkest 25% of pixels
        threshold = np.percentile(pixel_brightness, 25)
        dark_pixels = pixels_arr[pixel_brightness <= threshold]
        if len(dark_pixels) > 0:
            color = tuple(np.median(dark_pixels, axis=0).astype(int))
        else:
            color = (0, 0, 0)
    else:
        # Light text on dark background — take the brightest 25%
        threshold = np.percentile(pixel_brightness, 75)
        bright_pixels = pixels_arr[pixel_brightness >= threshold]
        if len(bright_pixels) > 0:
            color = tuple(np.median(bright_pixels, axis=0).astype(int))
        else:
            color = (255, 255, 255)

    # Snap near-black to pure black, near-white to pure white
    brightness = sum(color) / 3
    if brightness < 50:
        return (0, 0, 0)
    if brightness > 220:
        return (255, 255, 255)
    return color


def erase_text(
    image: Image.Image, bbox: List[List[int]], bg_color: Tuple[int, int, int],
    padding: int = 2,
) -> Image.Image:
    """Erase text by filling the bounding box with the background color."""
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = _bbox_to_rect(bbox)
    draw.rectangle(
        [left - padding, top - padding, right + padding, bottom + padding],
        fill=bg_color,
    )
    return image


def _find_font(size: int) -> ImageFont.FreeTypeFont:
    """Try to load a clean sans-serif font, falling back to default."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_text(
    image: Image.Image,
    bbox: List[List[int]],
    text: str,
    text_color: Tuple[int, int, int],
) -> Image.Image:
    """Render translated text within the bounding box, auto-sizing to fit."""
    if not text.strip():
        return image

    draw = ImageDraw.Draw(image)
    left, top, right, bottom = _bbox_to_rect(bbox)
    box_width = right - left
    box_height = bottom - top

    if box_width < 10 or box_height < 8:
        return image

    # Auto-size: start with box height, shrink until text fits width
    font_size = max(min(box_height, 200), 10)

    font = None
    for _ in range(50):
        if font_size < 8:
            font_size = 8
            break
        try:
            font = _find_font(font_size)
            bbox_text = draw.textbbox((0, 0), text, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            if text_width <= box_width and text_height <= box_height:
                break
        except OSError:
            pass
        font_size -= 1

    if font is None:
        try:
            font = _find_font(max(font_size, 8))
        except OSError:
            return image

    # Center text within the bounding box
    try:
        bbox_text = draw.textbbox((0, 0), text, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        x = left + (box_width - text_width) // 2
        y = top + (box_height - text_height) // 2
        draw.text((x, y), text, fill=text_color, font=font)
    except OSError:
        pass  # Skip rendering if font still can't handle it

    return image


def translate_image(
    image_bytes: bytes,
    translator,
    source_lang: str,
    target_lang: str,
    deck_context: str = "",
) -> Optional[bytes]:
    """Full pipeline: detect text in image, translate, re-render.

    Returns modified image bytes, or None if no text was found.
    """
    # Step 1: Upscale image for better OCR precision and rendering
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = original_image.size
    scaled_w, scaled_h = orig_w * UPSCALE_FACTOR, orig_h * UPSCALE_FACTOR
    scaled_image = original_image.resize((scaled_w, scaled_h), Image.LANCZOS)

    # Run OCR on the upscaled image
    scaled_buf = io.BytesIO()
    scaled_image.save(scaled_buf, format="PNG")
    scaled_bytes = scaled_buf.getvalue()

    regions = detect_text(scaled_bytes)
    if not regions:
        return None

    # Skip images with only a short word or two (likely logos/watermarks)
    total_text = " ".join(r.text for r in regions).strip()
    if len(regions) <= 2 and len(total_text) < 15:
        return None

    # Step 2: Get image description from Claude Vision for context
    # (send original size to save tokens)
    image_context = translator.describe_image(image_bytes)

    # Step 3: For each text region, translate and re-render on the upscaled image
    for region in regions:
        try:
            translated = translator.translate_image_text(
                region.text, source_lang, target_lang,
                deck_context=deck_context, image_context=image_context,
            )

            bg_color = sample_background_color(scaled_image, region.bbox)
            text_color = _detect_text_color(scaled_image, region.bbox, bg_color)

            scaled_image = erase_text(scaled_image, region.bbox, bg_color)
            scaled_image = render_text(scaled_image, region.bbox, translated, text_color)
        except Exception as e:
            print(f"    Warning: skipped image text region '{region.text}': {e}")

    # Step 4: Scale back down to original size
    final_image = scaled_image.resize((orig_w, orig_h), Image.LANCZOS)

    # Step 5: Export in original format
    output = io.BytesIO()
    if image_bytes[:2] == b"\xff\xd8":
        final_image.save(output, format="JPEG")
    else:
        final_image.save(output, format="PNG")
    return output.getvalue()
