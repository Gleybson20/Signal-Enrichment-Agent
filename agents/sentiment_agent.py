"""
sentiment_agent.py
------------------
Classifies the sentiment of a free-text review as positive, negative, or neutral.

Design decisions:
- Prompt template is loaded from disk (prompts/v1/sentiment.txt) so it can be
  iterated without touching Python code.
- The LLM is expected to respond with a strict JSON object; any deviation raises
  ValueError, which triggers the retry loop in BaseAgent.
- OpenAI client is injected at construction so the class is trivially testable
  (pass a mock client, zero real API calls needed).
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

VALID_SENTIMENTS = {"positive", "negative", "neutral"}


@dataclass
class SentimentOutput:
    sentiment: str
    confidence: float
class SentimentAgent(BaseAgent[SentimentOutput]):
    """
    Enriches a record with sentiment + confidence score.

    Usage:
        agent = SentimentAgent.from_env()
        result = agent.run({"review_id": "r1", "review_text": "Great product!"})
        # result.data → SentimentOutput(sentiment='positive', confidence=0.97)
    """

    PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "{version}" / "sentiment.txt"

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
    def from_env(cls, config: Optional[AgentConfig] = None) -> "SentimentAgent":
        """Convenience constructor that reads OPENAI_API_KEY from the environment."""
        import os

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        return cls(config=config or AgentConfig(), client=OpenAI(api_key=api_key))

    def _build_prompt(self, record: dict[str, Any]) -> str:
        review_text = record.get("review_text", "")
        if not review_text:
            raise ValueError(f"Record {record.get('review_id')} has no 'review_text' field.")
        return self._prompt_template.replace("{{review_text}}", review_text)

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
                        "You are a sentiment analysis engine. "
                        "Always respond with valid JSON only. No prose, no markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        usage = response.usage
        self._last_token_count = usage.total_tokens if usage else 0
        return response.choices[0].message.content or ""

    def _parse_response(self, raw: str) -> SentimentOutput:
        """
        Safely parse the LLM JSON response.

        Strips code fences if the model adds them despite the system prompt.
        """
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned non-JSON response: {raw!r}") from exc

        sentiment = str(payload.get("sentiment", "")).lower()
        confidence = float(payload.get("confidence", 0.0))

        if sentiment not in VALID_SENTIMENTS:
            raise ValueError(
                f"Unexpected sentiment value '{sentiment}'. "
                f"Expected one of {VALID_SENTIMENTS}."
            )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence {confidence} is out of [0, 1] range.")

        return SentimentOutput(sentiment=sentiment, confidence=confidence)

    def validate(self, result: SentimentOutput) -> bool:
        return result.sentiment in VALID_SENTIMENTS and 0.0 <= result.confidence <= 1.0

    def _load_prompt(self, override_dir: Optional[Path]) -> str:
        path = (
            override_dir / "sentiment.txt"
            if override_dir
            else Path(str(self.PROMPT_PATH).replace("{version}", self.config.prompt_version))
        )
        if not path.exists():
            raise FileNotFoundError(f"Sentiment prompt not found at: {path}")
        return path.read_text(encoding="utf-8")

    def enrich(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Run the agent and merge results back into the original record dict.

        This is the interface used by batch_processor.py.
        """
        result: AgentResult[SentimentOutput] = self.run(record)
        enriched = dict(record)

        if result.success and result.data:
            enriched["sentiment"] = result.data.sentiment
            enriched["sentiment_confidence"] = result.data.confidence
        else:
            enriched["sentiment"] = None
            enriched["sentiment_confidence"] = None

        enriched["sentiment_tokens_used"] = result.tokens_used
        enriched["sentiment_model"] = result.model
        return enriched
