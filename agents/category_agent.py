"""
category_agent.py
-----------------
Classifies free-text product descriptions into a category + subcategory pair.

Design notes identical to sentiment_agent.py — see that file for rationale.
The only difference is the output schema and validation rules.
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from openai import OpenAI
from agents.base_agent import AgentConfig, AgentResult, BaseAgent

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {
    "Electronics",
    "Clothing & Apparel",
    "Home & Garden",
    "Books & Media",
    "Sports & Outdoors",
    "Food & Beverages",
    "Health & Beauty",
    "Toys & Games",
    "Automotive",
    "E-commerce",
    "Other",
}


@dataclass
class CategoryOutput:
    category: str
    subcategory: str

class CategoryAgent(BaseAgent[CategoryOutput]):
    """
    Enriches a record with a category / subcategory classification.

    Usage:
        agent = CategoryAgent.from_env()
        result = agent.run({"review_id": "r1", "review_text": "Fone sem fio com ANC"})
        # result.data → CategoryOutput(category='Electronics', subcategory='Headphones')
    """

    PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "{version}" / "category.txt"

    def __init__(
        self,
        config: AgentConfig,
        client: OpenAI,
        prompt_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(config)
        self._client = client
        self._prompt_template = self._load_prompt(prompt_dir)
        self._last_token_count: int = 0

    @classmethod
    def from_env(cls, config: Optional[AgentConfig] = None) -> "CategoryAgent":
        import os

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        return cls(config=config or AgentConfig(), client=OpenAI(api_key=api_key))

    def _build_prompt(self, record: dict[str, Any]) -> str:
        text = record.get("review_text", "")
        if not text:
            raise ValueError(f"Record {record.get('review_id')} has no 'review_text'.")

        categories_list = "\n".join(f"- {c}" for c in sorted(VALID_CATEGORIES))
        return (
            self._prompt_template
            .replace("{{review_text}}", text)
            .replace("{{valid_categories}}", categories_list)
        )

    def _call_llm(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout_seconds,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a product taxonomy classifier. "
                        "Respond with a JSON object containing 'category' and 'subcategory'. "
                        "No prose, no markdown, no code fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        usage = response.usage
        self._last_token_count = usage.total_tokens if usage else 0
        return response.choices[0].message.content or ""

    def _parse_response(self, raw: str) -> CategoryOutput:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned non-JSON: {raw!r}") from exc

        category = str(payload.get("category", "")).strip()
        subcategory = str(payload.get("subcategory", "")).strip()

        if category not in VALID_CATEGORIES:
            matched = next(
                (c for c in VALID_CATEGORIES if c.lower() == category.lower()), None
            )
            if matched:
                logger.debug("Normalised category '%s' → '%s'", category, matched)
                category = matched
            else:
                logger.warning(
                    "Unknown category '%s'. Falling back to 'Other'.", category
                )
                category = "Other"

        return CategoryOutput(category=category, subcategory=subcategory)

    def validate(self, result: CategoryOutput) -> bool:
        return bool(result.category) and result.category in VALID_CATEGORIES

    def _load_prompt(self, override_dir: Optional[Path]) -> str:
        path = (
            override_dir / "category.txt"
            if override_dir
            else Path(str(self.PROMPT_PATH).replace("{version}", self.config.prompt_version))
        )
        if not path.exists():
            raise FileNotFoundError(f"Category prompt not found at: {path}")
        return path.read_text(encoding="utf-8")

    def enrich(self, record: dict[str, Any]) -> dict[str, Any]:
        result: AgentResult[CategoryOutput] = self.run(record)
        enriched = dict(record)

        if result.success and result.data:
            enriched["category"] = result.data.category
            enriched["subcategory"] = result.data.subcategory
        else:
            enriched["category"] = None
            enriched["subcategory"] = None

        enriched["category_tokens_used"] = result.tokens_used
        enriched["category_model"] = result.model
        return enriched
