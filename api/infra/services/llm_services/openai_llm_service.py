from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from api.application.services.llm_service import LLMService
from api.infra.config.env import ConfigEnvs


class OpenaiLLMService(LLMService):
    def __init__(
        self,
        envs: ConfigEnvs,
        model: str = "gpt-4o-mini",
        stream: bool = True,
        **kwargs,
    ):
        super().__init__(envs)
        self.model_name = model
        self.stream = stream
        self.openai_api_key = envs.OPENAI_API_KEY

    @property
    def model(self) -> BaseChatModel:
        return self._create_chat_model(temperature=0.5, top_p=0.9)

    def _create_chat_model(
        self, temperature: float = 0.5, top_p: float = 0.9
    ) -> BaseChatModel:
        return ChatOpenAI(
            model=self.model_name,
            streaming=self.stream,
            openai_api_key=self.openai_api_key,
            temperature=temperature,
            top_p=top_p,
        )

    async def generate_response(
        self,
        user_input: str,
        system_prompt: str,
        model: str = "default",
        temperature: float = 0.5,
        top_p: float = 0.9,
    ):
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_input))

        # Create model with temperature and top_p
        chat_model = self._create_chat_model(temperature=temperature, top_p=top_p)

        async for chunk in chat_model.astream(messages):
            yield chunk.content

    async def get_available_models(self) -> list[str]:
        """Get list of available models"""
        return ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

    def invoke(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.5,
        top_p: float = 0.9,
    ) -> str:
        """
        Synchronous method to invoke LLM and get complete response.

        Args:
            prompt: User prompt
            system_prompt: System prompt (instructions for the LLM)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Complete response string
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        # Create model with temperature and top_p
        chat_model = self._create_chat_model(temperature=temperature, top_p=top_p)
        response = chat_model.invoke(messages)

        return response.content.strip()

    async def get_model_response(self, user_input: str, system_prompt: str):
        async for chunk in self.generate_response(user_input, system_prompt):
            yield chunk
