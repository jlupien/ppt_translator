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
```

By default, the translated file is saved as `<original>_translated.pptx` in the same directory.

## Options

| Flag | Description |
|------|-------------|
| `-t`, `--target` | Target language (required) |
| `-s`, `--source` | Source language (default: auto-detect) |
| `-o`, `--output` | Output file path (single file only) |
| `--model` | Claude model override (default: claude-sonnet-4-20250514) |

## How It Works

1. Opens the .pptx file and extracts text from all shapes, tables, and grouped shapes
2. Sends text to Claude for translation, batched per-slide for context
3. Writes translated text back into the original runs, preserving all formatting (font, size, color, bold, italic, alignment)
4. Saves a new .pptx file — the original is never modified

Identical text strings are cached in-memory so repeated content (e.g., headers, footers) is only translated once.
