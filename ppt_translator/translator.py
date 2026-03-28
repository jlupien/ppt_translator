"""Translation service using Claude API."""
from __future__ import annotations

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

    def translate_segments(
        self,
        segments: List[str],
        source_lang: str,
        target_lang: str,
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
        system_prompt = (
            f"You are a translation assistant. Translate text {source_clause}"
            f"to {target_lang}. Preserve tone, meaning, and formatting.\n\n"
            "RULES:\n"
            f'- The input contains text segments separated by "{SEGMENT_DELIMITER.strip()}".\n'
            f'- Return ONLY the translated segments, separated by "{SEGMENT_DELIMITER.strip()}", in the same order.\n'
            f"- You MUST return exactly {len(uncached_texts)} segment(s).\n"
            "- Do NOT add any explanation, numbering, or extra text.\n"
            "- If a segment is empty or whitespace, return it unchanged.\n"
            "- Preserve any special characters, numbers, or formatting marks."
        )

        translated_text = self._call_api(system_prompt, joined)

        # Parse response back into segments
        translated_parts = translated_text.split(SEGMENT_DELIMITER.strip())
        # Strip whitespace from each part
        translated_parts = [p.strip() for p in translated_parts]

        # Handle mismatch: if Claude returned wrong number of segments
        if len(translated_parts) != len(uncached_texts):
            # Fallback: retry with individual translations
            translated_parts = self._translate_individually(
                uncached_texts, source_lang, target_lang
            )

        # Populate cache and build result
        for (orig_idx, orig_text), translated in zip(to_translate, translated_parts):
            cache_key = f"{source_lang}:{target_lang}:{orig_text}"
            self._cache[cache_key] = translated
            cached_results[orig_idx] = translated

        return [cached_results[i] for i in range(len(segments))]

    def _translate_individually(
        self, texts: List[str], source_lang: str, target_lang: str
    ) -> List[str]:
        """Fallback: translate each segment individually."""
        results = []
        source_clause = f"from {source_lang} " if source_lang else ""
        system_prompt = (
            f"You are a translation assistant. Translate the text {source_clause}"
            f"to {target_lang}. Return ONLY the translation, nothing else."
        )
        for text in texts:
            if not text.strip():
                results.append(text)
                continue
            translated = self._call_api(system_prompt, text)
            results.append(translated)
        return results

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
