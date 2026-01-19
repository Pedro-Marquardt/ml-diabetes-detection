import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from langchain_core.messages import SystemMessage, HumanMessage
from api.infra.services.llm_services.ollama_llm_service import OllamaLLMService
from api.infra.config.env import ConfigEnvs


@pytest.fixture
def mock_envs():
    """Mock ConfigEnvs"""
    envs = Mock(spec=ConfigEnvs)
    envs.OLLAMA_HOST = "http://localhost:11434"
    envs.OLLAMA_MODEL = "llama3.2:1b"
    return envs


@pytest.fixture
def ollama_service(mock_envs):
    """Create OllamaLLMService instance"""
    return OllamaLLMService(envs=mock_envs)


class TestOllamaLLMService:
    """Test suite for OllamaLLMService"""

    def test_init_with_defaults(self, mock_envs):
        """Test initialization with default values"""
        service = OllamaLLMService(envs=mock_envs)

        assert service.model_name == "llama3.2:1b"
        assert service.ollama_host == "http://localhost:11434"
        assert service.stream is True

    def test_init_with_custom_model(self, mock_envs):
        """Test initialization with custom model"""
        service = OllamaLLMService(envs=mock_envs, model="custom-model")

        assert service.model_name == "custom-model"
        assert service.ollama_host == "http://localhost:11434"

    def test_init_with_custom_host(self, mock_envs):
        """Test initialization with custom host"""
        mock_envs.OLLAMA_HOST = "http://custom-host:11434"
        service = OllamaLLMService(envs=mock_envs)

        assert service.ollama_host == "http://custom-host:11434"

    @patch("api.infra.services.llm_services.ollama_llm_service.ChatOllama")
    def test_create_chat_model(self, mock_chat_ollama, ollama_service):
        """Test _create_chat_model method"""
        mock_model = Mock()
        mock_chat_ollama.return_value = mock_model

        result = ollama_service._create_chat_model(temperature=0.7, top_p=0.8)

        mock_chat_ollama.assert_called_once_with(
            model=ollama_service.model_name,
            base_url=ollama_service.ollama_host,
            streaming=ollama_service.stream,
            temperature=0.7,
            top_p=0.8,
        )
        assert result == mock_model

    @patch("api.infra.services.llm_services.ollama_llm_service.ChatOllama")
    def test_model_property(self, mock_chat_ollama, ollama_service):
        """Test model property"""
        mock_model = Mock()
        mock_chat_ollama.return_value = mock_model

        result = ollama_service.model

        assert result == mock_model
        mock_chat_ollama.assert_called_once()

    @pytest.mark.asyncio
    @patch("api.infra.services.llm_services.ollama_llm_service.ChatOllama")
    async def test_generate_response(self, mock_chat_ollama, ollama_service):
        """Test generate_response async method"""
        # Mock chat model and streaming
        mock_chunk1 = Mock()
        mock_chunk1.content = "Hello"
        mock_chunk2 = Mock()
        mock_chunk2.content = " World"

        # Create async generator that accepts messages argument
        async def async_gen(messages):
            yield mock_chunk1
            yield mock_chunk2

        mock_model = Mock()
        mock_model.astream = async_gen
        mock_chat_ollama.return_value = mock_model

        chunks = []
        async for chunk in ollama_service.generate_response(
            user_input="Test prompt",
            system_prompt="You are a helpful assistant",
            temperature=0.7,
            top_p=0.8,
        ):
            chunks.append(chunk)

        assert chunks == ["Hello", " World"]
        mock_chat_ollama.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_without_system_prompt(self, ollama_service):
        """Test generate_response without system prompt"""
        with patch(
            "api.infra.services.llm_services.ollama_llm_service.ChatOllama"
        ) as mock_chat_ollama:
            mock_chunk = Mock()
            mock_chunk.content = "Response"

            async def async_gen(messages):
                yield mock_chunk

            mock_model = Mock()
            mock_model.astream = async_gen
            mock_chat_ollama.return_value = mock_model

            chunks = []
            async for chunk in ollama_service.generate_response(
                user_input="Test",
                system_prompt="",
            ):
                chunks.append(chunk)

            assert chunks == ["Response"]

    @pytest.mark.asyncio
    async def test_get_available_models(self, ollama_service):
        """Test get_available_models method"""
        models = await ollama_service.get_available_models()

        assert isinstance(models, list)
        assert "llama3.2:1b" in models

    @patch("api.infra.services.llm_services.ollama_llm_service.ChatOllama")
    def test_invoke(self, mock_chat_ollama, ollama_service):
        """Test invoke synchronous method"""
        mock_response = Mock()
        mock_response.content = "  Test response  "

        mock_model = Mock()
        mock_model.invoke = Mock(return_value=mock_response)
        mock_chat_ollama.return_value = mock_model

        result = ollama_service.invoke(
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

    @patch("api.infra.services.llm_services.ollama_llm_service.ChatOllama")
    def test_invoke_with_custom_model(self, mock_chat_ollama, ollama_service):
        """Test invoke with custom model parameter"""
        mock_response = Mock()
        mock_response.content = "Response"

        mock_model = Mock()
        mock_model.invoke = Mock(return_value=mock_response)
        mock_chat_ollama.return_value = mock_model

        ollama_service.invoke(
            prompt="Test",
            model="custom-model",
        )

        # Verify custom model was used
        call_kwargs = mock_chat_ollama.call_args[1]
        assert call_kwargs["model"] == "custom-model"

    @pytest.mark.asyncio
    @patch("api.infra.services.llm_services.ollama_llm_service.ChatOllama")
    async def test_get_model_response(self, mock_chat_ollama, ollama_service):
        """Test get_model_response method"""
        mock_chunk = Mock()
        mock_chunk.content = "Chunk"

        async def async_gen(messages):
            yield mock_chunk

        mock_model = Mock()
        mock_model.astream = async_gen
        mock_chat_ollama.return_value = mock_model

        chunks = []
        async for chunk in ollama_service.get_model_response("input", "system"):
            chunks.append(chunk)

        assert chunks == ["Chunk"]
