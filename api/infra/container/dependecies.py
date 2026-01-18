from pathlib import Path
from dependency_injector import containers, providers
from api.infra.services.llm_services.ollama_llm_service import OllamaLLMService
from api.infra.services.llm_services.openai_llm_service import OpenaiLLMService
from api.infra.services.predict_services.diabetes_prediction_service import (
    DiabetesPredictionService,
)
from api.infra.services.diagnostic_service import DiabetesDiagnosticService
from api.infra.config.env import ConfigEnvs
from api.application.enum.llm_model import LLMModels


def _create_llm_service(envs: ConfigEnvs):
    provider = envs.LLM_PROVIDER or "ollama"

    try:
        llm_model = LLMModels(provider.lower())
        if llm_model == LLMModels.OLLAMA:
            return OllamaLLMService(envs=envs)
        elif llm_model == LLMModels.OPENAI:
            return OpenaiLLMService(envs=envs)
    except ValueError:
        # If invalid value, default to Ollama
        return OllamaLLMService(envs=envs)


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    envs = providers.Singleton(ConfigEnvs)

    llm_service = providers.Factory(
        _create_llm_service,
        envs=envs,
    )

    # Serviço de predição de diabetes (Singleton - carrega modelo uma vez)
    # O model_path é calculado internamente pelo serviço
    prediction_service = providers.Singleton(DiabetesPredictionService)

    # Serviço de diagnóstico (Singleton - combina predição + LLM para relatórios)
    diagnostic_service = providers.Singleton(
        DiabetesDiagnosticService,
        prediction_service=prediction_service,
        llm_service=llm_service,
    )
