from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from api.application.services.llm_service import LLMService
from api.infra.config.env import ConfigEnvs


class OllamaLLMService(LLMService):
    def __init__(
        self,
        envs: ConfigEnvs,
        model: Optional[str] = None,
        stream: bool = True,
        **kwargs,
    ):
        super().__init__(envs)
        # Use model from env or parameter, default to llama3.2:1b
        self.model_name = model or envs.OLLAMA_MODEL or "llama3.2:1b"
        self.stream = stream
        self.ollama_host = envs.OLLAMA_HOST or "http://localhost:11434"

    @property
    def model(self) -> BaseChatModel:
        return self._create_chat_model(temperature=0.5, top_p=0.9)

    def _create_chat_model(
        self, temperature: float = 0.5, top_p: float = 0.9
    ) -> BaseChatModel:
        return ChatOllama(
            model=self.model_name,
            base_url=self.ollama_host,
            streaming=self.stream,
            temperature=temperature,
            top_p=top_p,
        )

    async def generate_response(
        self,
        user_input: str,
        system_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.5,
        top_p: float = 0.9,
    ):
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_input))

        # Use provided model or default
        model_to_use = model or self.model_name

        # Create model with temperature and top_p
        chat_model = ChatOllama(
            model=model_to_use,
            base_url=self.ollama_host,
            streaming=self.stream,
            temperature=temperature,
            top_p=top_p,
        )

        async for chunk in chat_model.astream(messages):
            yield chunk.content

    async def get_available_models(self) -> list[str]:
        """Get list of available models"""
        # Common Ollama models
        return [
            "llama3.2:1b",
        ]

    def invoke(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.5,
        top_p: float = 0.9,
        model: Optional[str] = None,
    ) -> str:
        """
        Synchronous method to invoke LLM and get complete response.

        Args:
            prompt: User prompt
            system_prompt: System prompt (instructions for the LLM)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            model: Optional model name to override default

        Returns:
            Complete response string
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        # Use provided model or default
        model_to_use = model or self.model_name

        # Create model with temperature and top_p
        chat_model = ChatOllama(
            model=model_to_use,
            base_url=self.ollama_host,
            streaming=False,
            temperature=temperature,
            top_p=top_p,
        )
        response = chat_model.invoke(messages)

        return response.content.strip()

    async def get_model_response(self, user_input: str, system_prompt: str):
        async for chunk in self.generate_response(user_input, system_prompt):
            yield chunk
