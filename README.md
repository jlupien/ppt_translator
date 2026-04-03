# ppt_translator

CLI tool that translates PowerPoint (.pptx) slide decks between any languages using Claude, preserving all run-level formatting (fonts, colors, bold, italic, layout).

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Get your API key from [console.anthropic.com](https://console.anthropic.com) under **Settings > API Keys**.

> Note: API usage is billed per-token, separate from a Claude Pro/Team subscription.

## Usage

```bash
# Translate a single file (auto-detect source language)
python -m ppt_translator presentation.pptx -t French

# Specify source language
python -m ppt_translator presentation.pptx -s English -t Japanese

# Custom output path
python -m ppt_translator presentation.pptx -t Spanish -o translated.pptx

# Translate all .pptx files in a directory
python -m ppt_translator ./slides/ -t German

# Also translate text within images (diagrams, charts, labels)
python -m ppt_translator presentation.pptx -t French --translate-images
```

By default, the translated file is saved as `<original>_translated.pptx` in the same directory.

## Options

| Flag | Description |
|------|-------------|
| `-t`, `--target` | Target language (required) |
| `-s`, `--source` | Source language (default: auto-detect) |
| `-o`, `--output` | Output file path (single file only) |
| `--model` | Claude model override (default: claude-sonnet-4-20250514) |
| `--translate-images` | Also translate text embedded in images |

## How It Works

### Text translation

1. Generates a deck-wide translation glossary by sending all slide text to Claude in a single call — this ensures consistent terminology across the entire presentation
2. Extracts text from all shapes, tables, and grouped shapes on each slide
3. Sends text to Claude for translation, batched per-slide with the glossary as context
4. Writes translated text back into the original runs, preserving all formatting (font, size, color, bold, italic, alignment)
5. Saves a new .pptx file — the original is never modified

Multi-run paragraphs (e.g., mixed bold and normal text) use `<rN>` markup tags so Claude can map formatting to the correct translated words.

### Image translation (`--translate-images`)

1. Extracts images from each slide and upscales them 3x for precision
2. Runs OCR (EasyOCR) to detect text regions with bounding boxes
3. Sends the image to Claude Vision for a contextual description
4. Translates each text region using Claude, with both the deck glossary and image description as context
5. Erases original text (background color fill) and renders translated text via Pillow
6. Scales back to original size and replaces the image in the PPTX

Optimizations:
- Duplicate images (same background across slides) are detected by hash and only processed once
- Small images (< 150px) are skipped (logos, icons)
- Images with only a short word or two are skipped (watermarks)
- Translated text is cached so repeated labels are only translated once

Identical text strings are cached in-memory so repeated content (e.g., headers, footers) is only translated once.
