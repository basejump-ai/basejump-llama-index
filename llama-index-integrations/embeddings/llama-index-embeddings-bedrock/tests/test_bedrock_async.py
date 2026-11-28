import json
import pytest

from llama_index.embeddings.bedrock import BedrockEmbedding, Models

EXP_REQUEST = "foo bar baz"
EXP_RESPONSE = {
    "embedding": [
        0.017410278,
        0.040924072,
        -0.007507324,
        0.09429932,
        0.015304565,
    ]
}


class AsyncMockStreamReader:
    async def read(self):
        return json.dumps(EXP_RESPONSE).encode()


class AsyncMockClient:
    async def __aenter__(self) -> "AsyncMockClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    async def invoke_model(self, *args, **kwargs):
        return {"contentType": "application/json", "body": AsyncMockStreamReader()}


class AsyncMockSession:
    def __init__(self, *args, **kwargs) -> "AsyncMockSession":
        pass

    def create_client(self, *args, **kwargs):
        return AsyncMockClient()


@pytest.fixture()
def mock_aiobotocore_session(monkeypatch):
    monkeypatch.setattr(
        "llama_index.embeddings.bedrock.base.create_async_session", AsyncMockSession
    )


@pytest.fixture()
def bedrock_embedding(mock_aiobotocore_session):
    return BedrockEmbedding(
        model_name=Models.TITAN_EMBEDDING,
        client=AsyncMockClient(),
    )


@pytest.mark.asyncio
async def test_aget_text_embedding(bedrock_embedding):
    response = await bedrock_embedding._aget_text_embedding(EXP_REQUEST)
    assert response == EXP_RESPONSE["embedding"]


@pytest.mark.asyncio
async def test_create_async_session_uses_given_region_and_credentials():
    from botocore.config import Config
    from llama_index.embeddings.bedrock.base import create_async_session

    session = create_async_session(
        region_name="eu-west-1",
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
        aws_session_token="test-session-token",
    )
    async with session.create_client("bedrock-runtime", config=Config()) as client:
        assert client.meta.region_name == "eu-west-1"
        assert client.meta.service_model.service_name == "bedrock-runtime"
    credentials = await session.get_credentials()
    assert credentials.access_key == "test-access-key"
    assert credentials.secret_key == "test-secret-key"
    assert credentials.token == "test-session-token"


def test_create_async_session_reuses_only_an_aiobotocore_session():
    import botocore.session
    from aiobotocore.session import get_session
    from llama_index.embeddings.bedrock.base import create_async_session

    async_session = get_session()
    assert create_async_session(botocore_session=async_session) is async_session
    # A synchronous botocore session can't make async clients, so a new session is made
    sync_session = botocore.session.get_session()
    assert create_async_session(botocore_session=sync_session) is not sync_session
