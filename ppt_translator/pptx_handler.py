"""Read and write PowerPoint files while preserving run-level formatting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE


@dataclass
class RunInfo:
    """A single text run within a paragraph."""
    text: str
    # We don't store formatting — we modify runs in-place on the original slide


@dataclass
class ParagraphInfo:
    """A paragraph containing one or more runs."""
    runs: List[RunInfo] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(r.text for r in self.runs)


@dataclass
class TextFrameInfo:
    """All paragraphs in a text frame, with a reference back to the pptx object."""
    paragraphs: List[ParagraphInfo] = field(default_factory=list)
    text_frame: object = None  # pptx TextFrame reference

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)


@dataclass
class ShapeInfo:
    """A shape on a slide that contains translatable text."""
    shape_index: int
    text_frames: List[TextFrameInfo] = field(default_factory=list)


@dataclass
class SlideText:
    """All translatable text extracted from a single slide."""
    slide_number: int
    shapes: List[ShapeInfo] = field(default_factory=list)


def _extract_text_frames_from_shape(shape) -> List[TextFrameInfo]:
    """Extract text frame info from a shape, handling tables and groups."""
    frames = []

    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for child_shape in shape.shapes:
            frames.extend(_extract_text_frames_from_shape(child_shape))
        return frames

    if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
        for row in shape.table.rows:
            for cell in row.cells:
                tf_info = _extract_text_frame(cell.text_frame)
                if tf_info:
                    frames.append(tf_info)
        return frames

    if hasattr(shape, "text_frame"):
        tf_info = _extract_text_frame(shape.text_frame)
        if tf_info:
            frames.append(tf_info)

    return frames


def _extract_text_frame(text_frame) -> Optional[TextFrameInfo]:
    """Extract paragraph and run data from a text frame."""
    paragraphs = []
    has_text = False

    for para in text_frame.paragraphs:
        runs = []
        for run in para.runs:
            runs.append(RunInfo(text=run.text))
            if run.text.strip():
                has_text = True
        if runs:
            paragraphs.append(ParagraphInfo(runs=runs))

    if not has_text:
        return None

    return TextFrameInfo(paragraphs=paragraphs, text_frame=text_frame)


def extract_slide_text(slide, slide_number: int) -> SlideText:
    """Extract all translatable text from a slide."""
    slide_text = SlideText(slide_number=slide_number)

    for shape_index, shape in enumerate(slide.shapes):
        text_frames = _extract_text_frames_from_shape(shape)
        if text_frames:
            slide_text.shapes.append(
                ShapeInfo(shape_index=shape_index, text_frames=text_frames)
            )

    return slide_text


def collect_segments(slide_text: SlideText) -> List[str]:
    """Collect all non-empty text segments from a slide for batch translation.

    Returns a flat list of paragraph texts. Each paragraph's text is one segment.
    """
    segments = []
    for shape in slide_text.shapes:
        for tf in shape.text_frames:
            for para in tf.paragraphs:
                text = para.text
                if text.strip():
                    segments.append(text)
    return segments


def apply_translations(slide_text: SlideText, translated_segments: List[str]) -> None:
    """Write translated text back into the original pptx text frames.

    Preserves all run-level formatting (font, size, color, bold, italic, etc.)
    by only modifying the .text property of each run.
    """
    seg_idx = 0

    for shape in slide_text.shapes:
        for tf in shape.text_frames:
            for para_info, pptx_para in zip(tf.paragraphs, tf.text_frame.paragraphs):
                original_text = para_info.text
                if not original_text.strip():
                    continue

                if seg_idx >= len(translated_segments):
                    break

                translated = translated_segments[seg_idx]
                seg_idx += 1

                _distribute_text_to_runs(pptx_para, para_info.runs, translated)


def _distribute_text_to_runs(pptx_para, run_infos: List[RunInfo], translated: str) -> None:
    """Distribute translated text across the existing runs of a paragraph.

    Strategy:
    - If there's only one run, put all translated text in it.
    - If there are multiple runs, distribute proportionally based on
      original character counts, preserving each run's formatting.
    """
    pptx_runs = list(pptx_para.runs)

    if len(pptx_runs) == 1:
        pptx_runs[0].text = translated
        return

    if len(pptx_runs) == 0:
        return

    # Calculate proportional distribution
    original_lengths = [len(r.text) for r in run_infos]
    total_original = sum(original_lengths)

    if total_original == 0:
        # All runs were empty — put everything in the first run
        pptx_runs[0].text = translated
        for run in pptx_runs[1:]:
            run.text = ""
        return

    # Distribute translated text proportionally across runs
    translated_len = len(translated)
    allocated = 0

    for i, run in enumerate(pptx_runs):
        if i == len(pptx_runs) - 1:
            # Last run gets the remainder
            run.text = translated[allocated:]
        else:
            proportion = original_lengths[i] / total_original
            char_count = round(proportion * translated_len)
            # Try to break at a word boundary
            end = allocated + char_count
            end = _find_word_boundary(translated, end)
            run.text = translated[allocated:end]
            allocated = end


def _find_word_boundary(text: str, pos: int) -> int:
    """Find the nearest word boundary (space) near pos."""
    if pos >= len(text):
        return len(text)
    if pos <= 0:
        return 0

    # Look for a space within 5 characters in either direction
    for offset in range(6):
        if pos + offset < len(text) and text[pos + offset] == " ":
            return pos + offset + 1  # Include the space in the earlier run
        if pos - offset >= 0 and text[pos - offset] == " ":
            return pos - offset + 1

    # No nearby space found — just split at the calculated position
    return pos


def build_deck_summary(prs: Presentation) -> str:
    """Build a compact text summary of the entire presentation for context generation.

    Returns one line per slide with title and body text separated by pipes.
    """
    lines = []
    for slide_number, slide in enumerate(prs.slides, start=1):
        parts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                parts.append(shape.text.strip())
        if parts:
            lines.append(f"Slide {slide_number}: {' | '.join(parts)}")
        else:
            lines.append(f"Slide {slide_number}: (no text)")
    return "\n".join(lines)


def load_presentation(path: str) -> Presentation:
    """Load a PowerPoint presentation."""
    return Presentation(path)
