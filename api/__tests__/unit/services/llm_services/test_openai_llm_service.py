import pytest
from unittest.mock import Mock, AsyncMock, patch
from langchain_core.messages import SystemMessage, HumanMessage
from api.infra.services.llm_services.openai_llm_service import OpenaiLLMService
from api.infra.config.env import ConfigEnvs


@pytest.fixture
def mock_envs():
    """Mock ConfigEnvs"""
    envs = Mock(spec=ConfigEnvs)
    envs.OPENAI_API_KEY = "test-api-key"
    envs.OPENAI_MODEL = "gpt-4o-mini"
    return envs


@pytest.fixture
def openai_service(mock_envs):
    """Create OpenaiLLMService instance"""
    return OpenaiLLMService(envs=mock_envs)


class TestOpenaiLLMService:
    """Test suite for OpenaiLLMService"""

    def test_init_with_defaults(self, mock_envs):
        """Test initialization with default values"""
        service = OpenaiLLMService(envs=mock_envs)

        assert service.model_name == "gpt-4o-mini"
        assert service.openai_api_key == "test-api-key"
        assert service.stream is True

    def test_init_with_custom_model(self, mock_envs):
        """Test initialization with custom model"""
        service = OpenaiLLMService(envs=mock_envs, model="gpt-4o")

        assert service.model_name == "gpt-4o"
        assert service.openai_api_key == "test-api-key"

    @patch("api.infra.services.llm_services.openai_llm_service.ChatOpenAI")
    def test_create_chat_model(self, mock_chat_openai, openai_service):
        """Test _create_chat_model method"""
        mock_model = Mock()
        mock_chat_openai.return_value = mock_model

        result = openai_service._create_chat_model(temperature=0.7, top_p=0.8)

        mock_chat_openai.assert_called_once_with(
            model=openai_service.model_name,
            streaming=openai_service.stream,
            openai_api_key=openai_service.openai_api_key,
            temperature=0.7,
            top_p=0.8,
        )
        assert result == mock_model

    @patch("api.infra.services.llm_services.openai_llm_service.ChatOpenAI")
    def test_model_property(self, mock_chat_openai, openai_service):
        """Test model property"""
        mock_model = Mock()
        mock_chat_openai.return_value = mock_model

        result = openai_service.model

        assert result == mock_model
        mock_chat_openai.assert_called_once()

    @pytest.mark.asyncio
    @patch("api.infra.services.llm_services.openai_llm_service.ChatOpenAI")
    async def test_generate_response(self, mock_chat_openai, openai_service):
        """Test generate_response async method"""
        mock_chunk1 = Mock()
        mock_chunk1.content = "Hello"
        mock_chunk2 = Mock()
        mock_chunk2.content = " World"

        async def async_gen(messages):
            yield mock_chunk1
            yield mock_chunk2

        mock_model = Mock()
        mock_model.astream = async_gen
        mock_chat_openai.return_value = mock_model

        chunks = []
        async for chunk in openai_service.generate_response(
            user_input="Test prompt",
            system_prompt="You are a helpful assistant",
            temperature=0.7,
            top_p=0.8,
        ):
            chunks.append(chunk)

        assert chunks == ["Hello", " World"]
        mock_chat_openai.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_without_system_prompt(self, openai_service):
        """Test generate_response without system prompt"""
        with patch(
            "api.infra.services.llm_services.openai_llm_service.ChatOpenAI"
        ) as mock_chat_openai:
            mock_chunk = Mock()
            mock_chunk.content = "Response"

            async def async_gen(messages):
                yield mock_chunk

            mock_model = Mock()
            mock_model.astream = async_gen
            mock_chat_openai.return_value = mock_model

            chunks = []
            async for chunk in openai_service.generate_response(
                user_input="Test",
                system_prompt="",
            ):
                chunks.append(chunk)

            assert chunks == ["Response"]

    @pytest.mark.asyncio
    async def test_get_available_models(self, openai_service):
        """Test get_available_models method"""
        models = await openai_service.get_available_models()

        assert isinstance(models, list)
        assert "gpt-4o-mini" in models
        assert "gpt-4o" in models
        assert "gpt-3.5-turbo" in models

    @patch("api.infra.services.llm_services.openai_llm_service.ChatOpenAI")
    def test_invoke(self, mock_chat_openai, openai_service):
        """Test invoke synchronous method"""
        mock_response = Mock()
        mock_response.content = "  Test response  "

        mock_model = Mock()
        mock_model.invoke = Mock(return_value=mock_response)
        mock_chat_openai.return_value = mock_model

        result = openai_service.invoke(
            prompt="Test prompt",
            system_prompt="System prompt",
            temperature=0.7,
            top_p=0.8,
        )

        assert result == "Test response"
        mock_model.invoke.assert_called_once()
        # Verify messages were created correctly
        call_args = mock_model.invoke.call_args[0][0]
        assert len(call_args) == 2
        assert isinstance(call_args[0], SystemMessage)
        assert isinstance(call_args[1], HumanMessage)

    @pytest.mark.asyncio
    @patch("api.infra.services.llm_services.openai_llm_service.ChatOpenAI")
    async def test_get_model_response(self, mock_chat_openai, openai_service):
        """Test get_model_response method"""
        mock_chunk = Mock()
        mock_chunk.content = "Chunk"

        async def async_gen(messages):
            yield mock_chunk

        mock_model = Mock()
        mock_model.astream = async_gen
        mock_chat_openai.return_value = mock_model

        chunks = []
        async for chunk in openai_service.get_model_response("input", "system"):
            chunks.append(chunk)

        assert chunks == ["Chunk"]
