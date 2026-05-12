"""
test_agents.py
--------------
Unit tests for all three enrichment agents.

Zero real API calls are made. The OpenAI client is replaced with a mock
that returns pre-defined responses, making the suite fast, free, and deterministic.

Run with:  pytest tests/ -v
"""

from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from agents.base_agent import AgentConfig, AgentResult
from agents.category_agent import CategoryAgent, CategoryOutput, VALID_CATEGORIES
from agents.entity_agent import EntityAgent, EntityOutput
from agents.sentiment_agent import SentimentAgent, SentimentOutput, VALID_SENTIMENTS

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_openai_response(content: str, total_tokens: int = 100) -> MagicMock:
    """Build a minimal mock that looks like an openai.types.chat.ChatCompletion."""
    choice = MagicMock()
    choice.message.content = content

    usage = MagicMock()
    usage.total_tokens = total_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_client(response_content: str, total_tokens: int = 100) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = _make_openai_response(
        response_content, total_tokens
    )
    return client


def _config(prompt_dir: Path) -> tuple[AgentConfig, Path]:
    return AgentConfig(model="gpt-4o", max_retries=1), prompt_dir

@pytest.fixture(scope="module")
def prompt_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a minimal prompt directory used by all agents in this session."""
    d = tmp_path_factory.mktemp("prompts")
    (d / "sentiment.txt").write_text(
        "Classify the sentiment of: {{review_text}}", encoding="utf-8"
    )
    (d / "category.txt").write_text(
        "Classify the category of: {{review_text}}\n{{valid_categories}}", encoding="utf-8"
    )
    (d / "entity.txt").write_text(
        "Extract entities from: {{review_text}}", encoding="utf-8"
    )
    return d

@pytest.fixture()
def sample_records() -> list[dict]:
    path = FIXTURES_DIR / "sample_reviews.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

class TestSentimentAgent:
    def test_positive_sentiment(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        client = _make_client('{"sentiment": "positive", "confidence": 0.97}', total_tokens=90)
        agent = SentimentAgent(config=cfg, client=client, prompt_dir=pdir)

        result = agent.run({"review_id": "r1", "review_text": "Produto excelente!"})

        assert result.success is True
        assert result.data is not None
        assert result.data.sentiment == "positive"
        assert result.data.confidence == pytest.approx(0.97)
        assert result.tokens_used == 90

    def test_negative_sentiment(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        client = _make_client('{"sentiment": "negative", "confidence": 0.88}')
        agent = SentimentAgent(config=cfg, client=client, prompt_dir=pdir)

        result = agent.run({"review_id": "r2", "review_text": "Chegou quebrado."})

        assert result.success is True
        assert result.data.sentiment == "negative"

    def test_neutral_sentiment(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        client = _make_client('{"sentiment": "neutral", "confidence": 0.72}')
        agent = SentimentAgent(config=cfg, client=client, prompt_dir=pdir)

        result = agent.run({"review_id": "r3", "review_text": "Produto ok."})

        assert result.success is True
        assert result.data.sentiment == "neutral"

    def test_invalid_sentiment_triggers_retry_and_fails(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        client = _make_client('{"sentiment": "meh", "confidence": 0.5}')
        agent = SentimentAgent(config=cfg, client=client, prompt_dir=pdir)

        result = agent.run({"review_id": "r4", "review_text": "whatever"})

        assert result.success is False
        assert result.error is not None

    def test_non_json_response_fails_gracefully(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        client = _make_client("Sorry, I cannot determine sentiment.")
        agent = SentimentAgent(config=cfg, client=client, prompt_dir=pdir)

        result = agent.run({"review_id": "r5", "review_text": "ok"})

        assert result.success is False

    def test_markdown_fences_stripped(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        client = _make_client(
            "```json\n{\"sentiment\": \"positive\", \"confidence\": 0.91}\n```"
        )
        agent = SentimentAgent(config=cfg, client=client, prompt_dir=pdir)

        result = agent.run({"review_id": "r6", "review_text": "Adorei o produto!"})

        assert result.success is True
        assert result.data.sentiment == "positive"

    def test_enrich_merges_fields_into_record(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        client = _make_client('{"sentiment": "positive", "confidence": 0.95}', total_tokens=80)
        agent = SentimentAgent(config=cfg, client=client, prompt_dir=pdir)

        original = {"review_id": "r7", "review_text": "Ótimo!", "product_id": "p1"}
        enriched = agent.enrich(original)

        assert enriched["sentiment"] == "positive"
        assert enriched["sentiment_confidence"] == pytest.approx(0.95)
        assert enriched["product_id"] == "p1"
        assert enriched["sentiment_tokens_used"] == 80

    def test_missing_review_text_fails_immediately(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        client = _make_client('{"sentiment": "positive", "confidence": 0.9}')
        agent = SentimentAgent(config=cfg, client=client, prompt_dir=pdir)

        result = agent.run({"review_id": "r8"})

        assert result.success is False

    def test_all_valid_sentiment_values_pass_validation(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        agent = SentimentAgent(
            config=cfg, client=MagicMock(), prompt_dir=pdir
        )
        for sentiment in VALID_SENTIMENTS:
            output = SentimentOutput(sentiment=sentiment, confidence=0.9)
            assert agent.validate(output) is True

    def test_validate_rejects_out_of_range_confidence(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        agent = SentimentAgent(
            config=cfg, client=MagicMock(), prompt_dir=pdir
        )
        assert agent.validate(SentimentOutput(sentiment="positive", confidence=1.5)) is False
class TestCategoryAgent:
    def test_electronics_category(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        client = _make_client('{"category": "Electronics", "subcategory": "Headphones"}')
        agent = CategoryAgent(config=cfg, client=client, prompt_dir=pdir)

        result = agent.run({"review_id": "c1", "review_text": "Fone ANC sem fio"})

        assert result.success is True
        assert result.data.category == "Electronics"
        assert result.data.subcategory == "Headphones"

    def test_unknown_category_falls_back_to_other(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        client = _make_client('{"category": "Undefined Widget", "subcategory": "Unknown"}')
        agent = CategoryAgent(config=cfg, client=client, prompt_dir=pdir)

        result = agent.run({"review_id": "c2", "review_text": "something weird"})
        assert result.success is True
        assert result.data.category == "Other"

    def test_case_insensitive_normalisation(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        client = _make_client('{"category": "electronics", "subcategory": "TV"}')
        agent = CategoryAgent(config=cfg, client=client, prompt_dir=pdir)

        result = agent.run({"review_id": "c3", "review_text": "Smart TV 55\""})

        assert result.success is True
        assert result.data.category == "Electronics"

    def test_all_valid_categories_pass_validation(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        agent = CategoryAgent(config=cfg, client=MagicMock(), prompt_dir=pdir)
        for cat in VALID_CATEGORIES:
            output = CategoryOutput(category=cat, subcategory="X")
            assert agent.validate(output) is True

class TestEntityAgent:
    def test_extracts_brand_and_location(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        payload = json.dumps({
            "brands": ["Apple"],
            "locations": ["Shopping Iguatemi"],
            "persons": [],
        })
        client = _make_client(payload)
        agent = EntityAgent(config=cfg, client=client, prompt_dir=pdir)

        result = agent.run({
            "review_id": "e1",
            "review_text": "Comprei na Apple Store do Shopping Iguatemi.",
        })

        assert result.success is True
        assert result.data.brands == ["Apple"]
        assert result.data.locations == ["Shopping Iguatemi"]
        assert result.data.persons == []

    def test_empty_entities_is_valid(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        client = _make_client('{"brands": [], "locations": [], "persons": []}')
        agent = EntityAgent(config=cfg, client=client, prompt_dir=pdir)

        result = agent.run({"review_id": "e2", "review_text": "Produto ok."})

        assert result.success is True
        assert result.data.is_empty() is True

    def test_duplicate_entities_are_deduplicated(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        client = _make_client(
            '{"brands": ["Nike", "Nike", "Adidas"], "locations": [], "persons": []}'
        )
        agent = EntityAgent(config=cfg, client=client, prompt_dir=pdir)

        result = agent.run({"review_id": "e3", "review_text": "Nike e Nike e Adidas"})

        assert result.data.brands == ["Nike", "Adidas"]

    def test_enrich_populates_list_fields(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        client = _make_client(
            '{"brands": ["Samsung"], "locations": ["São Paulo"], "persons": []}'
        )
        agent = EntityAgent(config=cfg, client=client, prompt_dir=pdir)

        record = {"review_id": "e4", "review_text": "Samsung em São Paulo."}
        enriched = agent.enrich(record)

        assert enriched["entities_brands"] == ["Samsung"]
        assert enriched["entities_locations"] == ["São Paulo"]
        assert enriched["entities_persons"] == []

    def test_non_list_response_coerced_to_empty(self, prompt_dir: Path) -> None:
        cfg, pdir = _config(prompt_dir)
        client = _make_client(
            '{"brands": "Apple", "locations": [], "persons": []}'
        )
        agent = EntityAgent(config=cfg, client=client, prompt_dir=pdir)

        result = agent.run({"review_id": "e5", "review_text": "Apple"})
        assert result.success is True
        assert result.data.brands == []
class TestFixtures:
    def test_sample_reviews_fixture_loads(self) -> None:
        path = FIXTURES_DIR / "sample_reviews.json"
        assert path.exists(), f"Fixture file missing: {path}"
        with path.open(encoding="utf-8") as f:
            records = json.load(f)
        assert isinstance(records, list)
        assert len(records) > 0
        for r in records:
            assert "review_id" in r
            assert "review_text" in r
