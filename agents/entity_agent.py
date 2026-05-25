"""
entity_agent.py
---------------
Extracts named entities (brands, locations, persons) from review text using an LLM.

The output is intentionally a list per entity type rather than a flat string so that
downstream consumers (SQL queries, vector stores) can filter on individual entities
without string parsing.
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from openai import OpenAI
from agents.base_agent import AgentConfig, AgentResult, BaseAgent

logger = logging.getLogger(__name__)
@dataclass
class EntityOutput:
    brands: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    persons: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.brands or self.locations or self.persons)

    def total_entities(self) -> int:
        return len(self.brands) + len(self.locations) + len(self.persons)
class EntityAgent(BaseAgent[EntityOutput]):
    """
    Enriches a record with extracted named entities.

    Usage:
        agent = EntityAgent.from_env()
        result = agent.run({
            "review_id": "r1",
            "review_text": "Comprei na Apple Store do Shopping Iguatemi."
        })
        # result.data → EntityOutput(brands=['Apple'], locations=['Shopping Iguatemi'], persons=[])
    """

    PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "{version}" / "entity.txt"

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
    def from_env(cls, config: Optional[AgentConfig] = None) -> "EntityAgent":
        import os

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        return cls(config=config or AgentConfig(), client=OpenAI(api_key=api_key))

    def _build_prompt(self, record: dict[str, Any]) -> str:
        text = record.get("review_text", "")
        if not text:
            raise ValueError(f"Record {record.get('review_id')} has no 'review_text'.")
        return self._prompt_template.replace("{{review_text}}", text)

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
                        "You are a Named Entity Recognition (NER) engine. "
                        "Extract brands, locations, and persons from the provided text. "
                        "Return ONLY a JSON object with keys 'brands', 'locations', 'persons'. "
                        "Each value is an array of strings. "
                        "Return empty arrays when no entities are found. "
                        "No prose. No markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        usage = response.usage
        self._last_token_count = usage.total_tokens if usage else 0
        return response.choices[0].message.content or ""

    def _parse_response(self, raw: str) -> EntityOutput:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned non-JSON: {raw!r}") from exc

        def _to_str_list(value: Any) -> list[str]:
            """Coerce whatever the model returns into a clean list of strings."""
            if not isinstance(value, list):
                return []
            return [str(v).strip() for v in value if v and str(v).strip()]

        brands = _to_str_list(payload.get("brands"))
        locations = _to_str_list(payload.get("locations"))
        persons = _to_str_list(payload.get("persons"))
        return EntityOutput(
            brands=list(dict.fromkeys(brands)),
            locations=list(dict.fromkeys(locations)),
            persons=list(dict.fromkeys(persons)),
        )

    def validate(self, result: EntityOutput) -> bool:
        return (
            isinstance(result.brands, list)
            and isinstance(result.locations, list)
            and isinstance(result.persons, list)
        )

    def _load_prompt(self, override_dir: Optional[Path]) -> str:
        path = (
            override_dir / "entity.txt"
            if override_dir
            else Path(str(self.PROMPT_PATH).replace("{version}", self.config.prompt_version))
        )
        if not path.exists():
            raise FileNotFoundError(f"Entity prompt not found at: {path}")
        return path.read_text(encoding="utf-8")

    def enrich(self, record: dict[str, Any]) -> dict[str, Any]:
        result: AgentResult[EntityOutput] = self.run(record)
        enriched = dict(record)

        if result.success and result.data:
            enriched["entities_brands"] = result.data.brands
            enriched["entities_locations"] = result.data.locations
            enriched["entities_persons"] = result.data.persons
        else:
            enriched["entities_brands"] = []
            enriched["entities_locations"] = []
            enriched["entities_persons"] = []

        enriched["entity_tokens_used"] = result.tokens_used
        enriched["entity_model"] = result.model
        return enriched
