"""Embeddings."""

from llama_index.core.callbacks.schema import CBEventType
from llama_index.core.callbacks.token_counting import TokenCountingHandler

TEST_PAYLOAD = {"chunks": ["one"], "formatted_prompt": "two", "response": "three"}
TEST_ID = "my id"


def test_on_event_start() -> None:
    """Test event start."""
    handler = TokenCountingHandler()

    event_id = handler.on_event_start(
        CBEventType.LLM, payload=TEST_PAYLOAD, event_id=TEST_ID
    )

    assert event_id == TEST_ID

    event_id = handler.on_event_start(
        CBEventType.EMBEDDING, payload=TEST_PAYLOAD, event_id=TEST_ID
    )

    assert event_id == TEST_ID
    assert len(handler.llm_token_counts) == 0
    assert len(handler.embedding_token_counts) == 0


def test_on_event_end() -> None:
    """Test event end."""
    handler = TokenCountingHandler()

    handler.on_event_end(CBEventType.LLM, payload=TEST_PAYLOAD, event_id=TEST_ID)

    assert len(handler.llm_token_counts) == 1
    assert len(handler.embedding_token_counts) == 0

    handler.on_event_end(CBEventType.EMBEDDING, payload=TEST_PAYLOAD, event_id=TEST_ID)

    assert len(handler.llm_token_counts) == 1
    assert len(handler.embedding_token_counts) == 1

    assert handler.embedding_token_counts[0].total_token_count == 1
    assert handler.llm_token_counts[0].total_token_count == 2

    # test actual counts
    # LLM should be two (prompt plus response)
    # Embedding should be one (single token chunk)
    assert handler.total_llm_token_count == 2
    assert handler.total_embedding_token_count == 1


def test_llm_model_from_event_start_when_response_has_none() -> None:
    """A provider that doesn't name the model in its raw response (Amazon Bedrock) gets it from the event start."""
    from llama_index.core.base.llms.types import ChatMessage, ChatResponse
    from llama_index.core.callbacks.schema import EventPayload

    handler = TokenCountingHandler()
    response = ChatResponse(
        message=ChatMessage(role="assistant", content="hi"),
        # Bedrock Converse reports usage with these key names, and no "model"
        raw={"usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18}},
    )

    handler.on_event_start(
        CBEventType.LLM,
        payload={EventPayload.SERIALIZED: {"model": "openai.gpt-oss-120b-1:0"}},
        event_id=TEST_ID,
    )
    handler.on_event_end(
        CBEventType.LLM,
        payload={
            EventPayload.MESSAGES: [ChatMessage(role="user", content="hello")],
            EventPayload.RESPONSE: response,
        },
        event_id=TEST_ID,
    )

    token_count = handler.llm_token_counts[0]
    assert token_count.model == "openai.gpt-oss-120b-1:0"
    # The provider's own counts are used, not a tokenizer estimate
    assert token_count.prompt_token_count == 11
    assert token_count.completion_token_count == 7
    assert handler._llm_event_models == {}


def test_llm_model_from_response_wins_over_event_start() -> None:
    """When the raw response names the model (OpenAI, Azure), that name is kept."""
    from llama_index.core.base.llms.types import ChatMessage, ChatResponse
    from llama_index.core.callbacks.schema import EventPayload

    handler = TokenCountingHandler()
    response = ChatResponse(
        message=ChatMessage(role="assistant", content="hi"),
        raw={"model": "gpt-4.1-2025-04-14", "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
    )

    handler.on_event_start(
        CBEventType.LLM,
        payload={EventPayload.SERIALIZED: {"model": "my-deployment"}},
        event_id=TEST_ID,
    )
    handler.on_event_end(
        CBEventType.LLM,
        payload={
            EventPayload.MESSAGES: [ChatMessage(role="user", content="hello")],
            EventPayload.RESPONSE: response,
        },
        event_id=TEST_ID,
    )

    token_count = handler.llm_token_counts[0]
    assert token_count.model == "gpt-4.1-2025-04-14"
    assert token_count.prompt_token_count == 5
    assert token_count.completion_token_count == 2
