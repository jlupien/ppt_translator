"""CLI entry point for the PPT translator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

from pptx import Presentation

from .pptx_handler import (
    apply_translations,
    collect_segments,
    extract_slide_text,
)
from .translator import DEFAULT_MODEL, TranslationService


def _find_pptx_files(target: Path) -> List[Path]:
    """Find .pptx files at the given path."""
    suffixes = {".pptx"}
    if target.is_file():
        if target.suffix.lower() in suffixes:
            return [target]
        return []
    if target.is_dir():
        return sorted(p for p in target.rglob("*") if p.suffix.lower() in suffixes)
    return []


def _translate_file(
    input_path: Path,
    output_path: Path,
    translator: TranslationService,
    source_lang: str,
    target_lang: str,
) -> None:
    """Translate a single PPTX file."""
    prs = Presentation(str(input_path))
    total_slides = len(prs.slides)

    for slide_number, slide in enumerate(prs.slides, start=1):
        print(f"  Slide {slide_number}/{total_slides}...", end=" ", flush=True)

        slide_text = extract_slide_text(slide, slide_number)
        segments = collect_segments(slide_text)

        if not segments:
            print("(no text)")
            continue

        translated = translator.translate_segments(segments, source_lang, target_lang)
        apply_translations(slide_text, translated)
        print(f"({len(segments)} segment(s) translated)")

    prs.save(str(output_path))
    print(f"  Saved: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ppt_translator",
        description="Translate PowerPoint slide decks using Claude.",
    )
    parser.add_argument(
        "path",
        help="Path to a .pptx file or directory containing presentations.",
    )
    parser.add_argument(
        "-t",
        "--target",
        required=True,
        help="Target language (e.g., French, Japanese, Spanish).",
    )
    parser.add_argument(
        "-s",
        "--source",
        default="",
        help="Source language. If omitted, Claude will auto-detect.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path. Only valid with a single input file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    target_path = Path(args.path).expanduser().resolve()
    files = _find_pptx_files(target_path)

    if not files:
        print(f"No .pptx files found at: {target_path}")
        sys.exit(1)

    if args.output and len(files) > 1:
        print("Error: --output can only be used with a single input file.")
        sys.exit(1)

    try:
        translator = TranslationService(model=args.model)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    for pptx_file in files:
        if args.output:
            out = Path(args.output).expanduser().resolve()
        else:
            out = pptx_file.with_name(f"{pptx_file.stem}_translated{pptx_file.suffix}")

        print(f"Translating: {pptx_file.name}")
        source_label = args.source or "auto-detect"
        print(f"  {source_label} → {args.target}")

        _translate_file(pptx_file, out, translator, args.source, args.target)

    print("Done.")
