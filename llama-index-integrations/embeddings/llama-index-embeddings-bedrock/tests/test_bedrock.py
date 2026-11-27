from io import BytesIO
import json
import pytest
from unittest import TestCase

import boto3
from botocore.response import StreamingBody
from botocore.stub import Stubber
from llama_index.embeddings.bedrock import BedrockEmbedding, Models

exp_embed = [
    0.017410278,
    0.040924072,
    -0.007507324,
    0.09429932,
    0.015304565,
]


class TestBedrockEmbedding(TestCase):
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

    exp_query = "foo bar baz"
    exp_titan_response = {"embedding": exp_embed}

    def test_get_text_embedding_titan_v1(self) -> None:
        bedrock_stubber = Stubber(self.bedrock_client)

        mock_stream = BytesIO(json.dumps(self.exp_titan_response).encode())
        bedrock_stubber.add_response(
            "invoke_model",
            {
                "contentType": "application/json",
                "body": StreamingBody(
                    mock_stream, len(json.dumps(self.exp_titan_response))
                ),
            },
            expected_params={
                "accept": "application/json",
                "body": f'{{"inputText": "{self.exp_query}"}}',
                "contentType": "application/json",
                "modelId": Models.TITAN_EMBEDDING.value,
            },
        )

        bedrock_embedding = BedrockEmbedding(
            model_name=Models.TITAN_EMBEDDING,
            client=self.bedrock_client,
        )
        assert bedrock_embedding.model_name == Models.TITAN_EMBEDDING

        bedrock_stubber.activate()
        embedding = bedrock_embedding.get_text_embedding(text=self.exp_query)
        bedrock_stubber.deactivate()

        bedrock_stubber.assert_no_pending_responses()
        self.assertEqual(embedding, self.exp_titan_response["embedding"])

    def test_get_text_embedding_titan_v1_bad_params(self) -> None:
        bedrock_stubber = Stubber(self.bedrock_client)

        bedrock_embedding_dim = BedrockEmbedding(
            model_name=Models.TITAN_EMBEDDING,
            client=self.bedrock_client,
            additional_kwargs={"dimensions": 512},
        )
        bedrock_embedding_norm = BedrockEmbedding(
            model_name=Models.TITAN_EMBEDDING,
            client=self.bedrock_client,
            additional_kwargs={"normalize": False},
        )

        bedrock_stubber.activate()
        for embedder in [bedrock_embedding_dim, bedrock_embedding_norm]:
            with pytest.raises(ValueError):
                embedder.get_text_embedding(text=self.exp_query)
        bedrock_stubber.deactivate()

        bedrock_stubber.assert_no_pending_responses()

    def test_get_text_embedding_titan_v2(self) -> None:
        bedrock_stubber = Stubber(self.bedrock_client)

        exp_body_request_param = json.dumps(
            {"inputText": self.exp_query, "dimensions": 512, "normalize": True}
        )

        mock_stream = BytesIO(json.dumps(self.exp_titan_response).encode())
        bedrock_stubber.add_response(
            "invoke_model",
            {
                "contentType": "application/json",
                "body": StreamingBody(
                    mock_stream, len(json.dumps(self.exp_titan_response))
                ),
            },
            expected_params={
                "accept": "application/json",
                "body": exp_body_request_param,
                "contentType": "application/json",
                "modelId": Models.TITAN_EMBEDDING_V2_0.value,
            },
        )

        bedrock_embedding = BedrockEmbedding(
            model_name=Models.TITAN_EMBEDDING_V2_0,
            client=self.bedrock_client,
            additional_kwargs={"dimensions": 512, "normalize": True},
        )
        assert bedrock_embedding.model_name == Models.TITAN_EMBEDDING_V2_0

        bedrock_stubber.activate()
        embedding = bedrock_embedding.get_text_embedding(text=self.exp_query)
        bedrock_stubber.deactivate()

        bedrock_stubber.assert_no_pending_responses()
        self.assertEqual(embedding, self.exp_titan_response["embedding"])

    def test_get_text_embedding_cohere(self) -> None:
        bedrock_stubber = Stubber(self.bedrock_client)

        mock_response = {"embeddings": [exp_embed]}

        mock_stream = BytesIO(json.dumps(mock_response).encode())

        bedrock_stubber.add_response(
            "invoke_model",
            {
                "contentType": "application/json",
                "body": StreamingBody(mock_stream, len(json.dumps(mock_response))),
            },
        )

        bedrock_embedding = BedrockEmbedding(
            model_name=Models.COHERE_EMBED_ENGLISH_V3,
            client=self.bedrock_client,
        )

        bedrock_stubber.activate()
        embedding = bedrock_embedding.get_text_embedding(text=self.exp_query)
        bedrock_stubber.deactivate()

        bedrock_stubber.assert_no_pending_responses()
        self.assertEqual(embedding, mock_response["embeddings"][0])

    def test_get_text_embedding_batch_cohere(self) -> None:
        bedrock_stubber = Stubber(self.bedrock_client)

        mock_response = {"embeddings": [exp_embed, exp_embed]}
        mock_request = [self.exp_query, self.exp_query]

        mock_stream = BytesIO(json.dumps(mock_response).encode())

        bedrock_stubber.add_response(
            "invoke_model",
            {
                "contentType": "application/json",
                "body": StreamingBody(mock_stream, len(json.dumps(mock_response))),
            },
        )

        bedrock_embedding = BedrockEmbedding(
            model_name=Models.COHERE_EMBED_ENGLISH_V3,
            client=self.bedrock_client,
        )

        bedrock_stubber.activate()
        embedding = bedrock_embedding.get_text_embedding_batch(texts=mock_request)

        bedrock_stubber.deactivate()

        self.assertEqual(len(embedding), 2)

        for i in range(2):
            self.assertEqual(embedding[i], mock_response["embeddings"][i])

    def test_get_text_embedding_cohere_v4_inference_profile(self) -> None:
        """Embed v4 is called through an inference profile and returns typed embeddings."""
        bedrock_stubber = Stubber(self.bedrock_client)

        mock_response = {"embeddings": {"float": [exp_embed, exp_embed]}}
        long_text = "x" * 5000
        mock_request = [self.exp_query, long_text]

        mock_stream = BytesIO(json.dumps(mock_response).encode())

        bedrock_stubber.add_response(
            "invoke_model",
            {
                "contentType": "application/json",
                "body": StreamingBody(mock_stream, len(json.dumps(mock_response))),
            },
            expected_params={
                "accept": "application/json",
                "contentType": "application/json",
                "modelId": "us.cohere.embed-v4:0",
                "body": json.dumps(
                    {
                        "input_type": "search_document",
                        "embedding_types": ["float"],
                        "output_dimension": 1536,
                        # v4 inputs are not truncated to the v3 limit of 2048 characters
                        "texts": mock_request,
                    }
                ),
            },
        )

        bedrock_embedding = BedrockEmbedding(
            model_name="us.cohere.embed-v4:0",
            client=self.bedrock_client,
            additional_kwargs={"output_dimension": 1536},
        )

        bedrock_stubber.activate()
        embedding = bedrock_embedding.get_text_embedding_batch(texts=mock_request)
        bedrock_stubber.deactivate()

        bedrock_stubber.assert_no_pending_responses()
        self.assertEqual(embedding, [exp_embed, exp_embed])

    def test_get_query_embedding_cohere_v4(self) -> None:
        bedrock_stubber = Stubber(self.bedrock_client)

        mock_response = {"embeddings": {"float": [exp_embed]}}
        mock_stream = BytesIO(json.dumps(mock_response).encode())

        bedrock_stubber.add_response(
            "invoke_model",
            {
                "contentType": "application/json",
                "body": StreamingBody(mock_stream, len(json.dumps(mock_response))),
            },
            expected_params={
                "accept": "application/json",
                "contentType": "application/json",
                "modelId": Models.COHERE_EMBED_V4.value,
                "body": json.dumps(
                    {
                        "input_type": "search_query",
                        "embedding_types": ["float"],
                        "texts": [self.exp_query],
                    }
                ),
            },
        )

        bedrock_embedding = BedrockEmbedding(
            model_name=Models.COHERE_EMBED_V4,
            client=self.bedrock_client,
        )

        bedrock_stubber.activate()
        embedding = bedrock_embedding.get_query_embedding(query=self.exp_query)
        bedrock_stubber.deactivate()

        bedrock_stubber.assert_no_pending_responses()
        self.assertEqual(embedding, exp_embed)

    def test_output_dimension_rejected_for_cohere_v3(self) -> None:
        bedrock_embedding = BedrockEmbedding(
            model_name=Models.COHERE_EMBED_ENGLISH_V3,
            client=self.bedrock_client,
            additional_kwargs={"output_dimension": 1536},
        )

        with pytest.raises(ValueError, match="output_dimension"):
            bedrock_embedding.get_text_embedding(text=self.exp_query)

    def test_list_supported_models(self):
        exp_dict = {
            "amazon": [
                "amazon.titan-embed-text-v1",
                "amazon.titan-embed-text-v2:0",
                "amazon.titan-embed-g1-text-02",
            ],
            "cohere": [
                "cohere.embed-english-v3",
                "cohere.embed-multilingual-v3",
                "cohere.embed-v4:0",
            ],
        }

        bedrock_embedding = BedrockEmbedding(
            model_name=Models.COHERE_EMBED_ENGLISH_V3,
            client=self.bedrock_client,
        )

        assert bedrock_embedding.list_supported_models() == exp_dict

    def test_optional_args_in_json_schema(self) -> None:
        json_schema = BedrockEmbedding.model_json_schema()
        assert "botocore_session" in json_schema["properties"]
        assert json_schema["properties"]["botocore_session"].get("default") is None
        assert "botocore_session" not in json_schema.get("required", [])
