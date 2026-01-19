from abc import ABC, abstractmethod
from typing import AsyncGenerator

from langchain_core.language_models import BaseChatModel
from api.infra.config.env import ConfigEnvs


class LLMService(ABC):
    """Interface for Large Language Model services"""

    def __init__(self, envs: ConfigEnvs):
        self.envs = envs

    @property
    @abstractmethod
    def model(self) -> BaseChatModel:
        pass

    @abstractmethod
    async def generate_response(
        self,
        user_input: str,
        system_prompt: str,
        model: str = "default",
        temperature: float = 0.5,
        top_p: float = 0.9,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from LLM"""
        pass

    @abstractmethod
    def invoke(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.5,
        top_p: float = 0.9,
    ) -> str:
        """Invoke LLM and get complete response"""
        pass

    @abstractmethod
    async def get_available_models(self) -> list[str]:
        """Get list of available models"""
        pass
