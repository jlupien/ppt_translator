"""Translation service using Claude API."""
from __future__ import annotations

import base64
import os
import time
from typing import Dict, List, Optional

from anthropic import Anthropic, APIError, RateLimitError

DEFAULT_MODEL = "claude-sonnet-4-20250514"
SEGMENT_DELIMITER = " ||| "


class TranslationService:
    """Translate text segments using Claude."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "Missing ANTHROPIC_API_KEY. Set it as an environment variable."
            )
        self.client = Anthropic(api_key=resolved_key)
        self.model = model
        self._cache: Dict[str, str] = {}

    def generate_context_summary(
        self, deck_text: str, source_lang: str, target_lang: str
    ) -> str:
        """Generate a translation glossary/context summary from the full deck text."""
        source_clause = f"from {source_lang} " if source_lang else ""
        system_prompt = (
            f"You are preparing to translate a presentation {source_clause}to {target_lang}.\n"
            "Below is all the text from the presentation. Create a concise translation guide "
            "(under 500 words) that includes:\n"
            "- A glossary of key terms and their preferred translations\n"
            "- The overall tone and register (formal, casual, technical, etc.)\n"
            "- Any domain-specific terminology or recurring phrases and how to translate them\n"
            "- Notes on any proper nouns, acronyms, or terms that should NOT be translated\n\n"
            "This guide will be provided as context for translating each slide individually. "
            "Be concise and practical — focus on what a translator needs to stay consistent."
        )
        return self._call_api(system_prompt, deck_text)

    def translate_segments(
        self,
        segments: List[str],
        source_lang: str,
        target_lang: str,
        context: str = "",
    ) -> List[str]:
        """Translate a list of text segments, preserving order and count.

        Sends all segments in a single API call for context and efficiency.
        Uses an in-memory cache to skip previously translated identical text.
        """
        if not segments:
            return []

        # Check cache — find which segments need translation
        cached_results: Dict[int, str] = {}
        to_translate: List[tuple[int, str]] = []

        for i, seg in enumerate(segments):
            cache_key = f"{source_lang}:{target_lang}:{seg}"
            if cache_key in self._cache:
                cached_results[i] = self._cache[cache_key]
            else:
                to_translate.append((i, seg))

        # If everything was cached, return immediately
        if not to_translate:
            return [cached_results[i] for i in range(len(segments))]

        # Build the translation request for uncached segments
        uncached_texts = [seg for _, seg in to_translate]
        joined = SEGMENT_DELIMITER.join(uncached_texts)

        source_clause = f"from {source_lang} " if source_lang else ""
        context_block = ""
        if context:
            context_block = (
                "\n\nTRANSLATION GUIDE (use for consistent terminology and tone):\n"
                f"{context}\n"
            )
        system_prompt = (
            f"You are a translation assistant. Translate text {source_clause}"
            f"to {target_lang}. Preserve tone, meaning, and formatting.\n\n"
            "RULES:\n"
            f'- The input contains text segments separated by "{SEGMENT_DELIMITER.strip()}".\n'
            f'- Return ONLY the translated segments, separated by "{SEGMENT_DELIMITER.strip()}", in the same order.\n'
            f"- You MUST return exactly {len(uncached_texts)} segment(s).\n"
            "- Do NOT add any explanation, numbering, or extra text.\n"
            "- If a segment is empty or whitespace, return it unchanged.\n"
            "- Preserve any special characters, numbers, or formatting marks.\n"
            "- IMPORTANT: Some segments contain <rN>...</rN> tags marking formatting boundaries. "
            "Each tag represents text with DIFFERENT visual formatting (e.g., bold, italic, underlined, colored). "
            "When translating, you MUST keep the SAME MEANING inside each tag. "
            "For example, if <r2>probing</r2> is the specially formatted word, then <r2> in the translation "
            "must contain the translation of 'probing' (e.g., <r2>sondage</r2>), NOT a different word. "
            "The tags mark WHICH WORDS have special formatting — map meaning to meaning, not position to position. "
            "Keep all tags in order. Every tag from the input must appear in the output."
            f"{context_block}"
        )

        translated_text = self._call_api(
            system_prompt,
            f"Translate the following:\n{joined}",
        )

        # Parse response back into segments
        translated_parts = translated_text.split(SEGMENT_DELIMITER.strip())
        # Strip whitespace from each part
        translated_parts = [p.strip() for p in translated_parts]

        # Handle mismatch: if Claude returned wrong number of segments
        if len(translated_parts) != len(uncached_texts):
            # Fallback: retry with individual translations
            translated_parts = self._translate_individually(
                uncached_texts, source_lang, target_lang, context=context
            )

        # Populate cache and build result
        for (orig_idx, orig_text), translated in zip(to_translate, translated_parts):
            cache_key = f"{source_lang}:{target_lang}:{orig_text}"
            self._cache[cache_key] = translated
            cached_results[orig_idx] = translated

        return [cached_results[i] for i in range(len(segments))]

    def _translate_individually(
        self, texts: List[str], source_lang: str, target_lang: str, context: str = ""
    ) -> List[str]:
        """Fallback: translate each segment individually."""
        results = []
        source_clause = f"from {source_lang} " if source_lang else ""
        context_block = ""
        if context:
            context_block = (
                "\n\nTRANSLATION GUIDE (use for consistent terminology and tone):\n"
                f"{context}\n"
            )
        system_prompt = (
            f"You are a translation assistant. Translate the text {source_clause}"
            f"to {target_lang}. Return ONLY the translation, nothing else."
            f"{context_block}"
        )
        for text in texts:
            if not text.strip():
                results.append(text)
                continue
            translated = self._call_api(system_prompt, f"Translate the following:\n{text}")
            results.append(translated)
        return results

    def describe_image(self, image_bytes: bytes) -> str:
        """Send an image to Claude Vision and get a description with text context."""
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        # Detect media type from bytes
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            media_type = "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            media_type = "image/jpeg"
        else:
            media_type = "image/png"  # Default

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=0.0,
                    system=(
                        "You are analyzing an image from a presentation slide. "
                        "Describe what the image shows and list all visible text in context. "
                        "Be concise (under 200 words). Focus on what the text means in context — "
                        "this description will be used to help translate the text accurately."
                    ),
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Describe this image and all visible text within it.",
                            },
                        ],
                    }],
                )
                return "".join(
                    part.text for part in response.content if hasattr(part, "text")
                ).strip()
            except RateLimitError:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"\n    Rate limited (vision), retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    raise
            except APIError as e:
                if attempt < max_retries - 1 and e.status_code and e.status_code >= 500:
                    wait = 2 ** (attempt + 1)
                    print(f"\n    API error ({e.status_code}, vision), retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    raise

    def translate_image_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        deck_context: str = "",
        image_context: str = "",
    ) -> str:
        """Translate a single text string from an image, with full context."""
        if not text.strip():
            return text

        cache_key = f"img:{source_lang}:{target_lang}:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        source_clause = f"from {source_lang} " if source_lang else ""
        context_parts = []
        if deck_context:
            context_parts.append(
                f"TRANSLATION GUIDE (deck-wide):\n{deck_context}"
            )
        if image_context:
            context_parts.append(
                f"IMAGE CONTEXT:\n{image_context}"
            )
        context_block = "\n\n".join(context_parts)

        system_prompt = (
            f"You are a translation assistant. Translate text {source_clause}"
            f"to {target_lang}. Return ONLY the translation, nothing else.\n"
            "Preserve any special characters, numbers, or abbreviations."
        )
        if context_block:
            system_prompt += f"\n\n{context_block}"

        translated = self._call_api(system_prompt, f"Translate the following:\n{text}")
        self._cache[cache_key] = translated
        return translated

    def _call_api(self, system_prompt: str, user_text: str) -> str:
        """Call the Claude API with retry on rate limits."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0.3,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_text}],
                )
                return "".join(
                    part.text for part in response.content if hasattr(part, "text")
                ).strip()
            except RateLimitError:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
            except APIError as e:
                if attempt < max_retries - 1 and e.status_code and e.status_code >= 500:
                    wait = 2 ** (attempt + 1)
                    print(f"  API error ({e.status_code}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
