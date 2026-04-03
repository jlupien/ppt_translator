"""Translation service using Claude API."""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

from anthropic import Anthropic, APIError, RateLimitError

DEFAULT_MODEL = "claude-sonnet-4-20250514"


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
        slide_text: str = "",
    ) -> List[str]:
        """Translate a list of text segments one at a time with full slide context.

        Each segment is translated individually, but Claude sees the full slide
        text for context. Uses an in-memory cache to skip duplicates.
        """
        if not segments:
            return []

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
            "- Return ONLY the translation of the requested segment, nothing else.\n"
            "- Do NOT add any explanation, numbering, or extra text.\n"
            "- If the segment is empty or whitespace, return it unchanged.\n"
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

        results: List[str] = []
        for seg in segments:
            if not seg.strip():
                results.append(seg)
                continue

            cache_key = f"{source_lang}:{target_lang}:{seg}"
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
                continue

            # Build user message with slide context
            if slide_text:
                user_message = (
                    f"FULL SLIDE CONTENT (for context only — do NOT translate this):\n"
                    f"{slide_text}\n\n"
                    f"TRANSLATE THIS SEGMENT ONLY:\n{seg}"
                )
            else:
                user_message = seg

            translated = self._call_api(system_prompt, user_message)
            self._cache[cache_key] = translated
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
