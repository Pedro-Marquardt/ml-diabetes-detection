from typing import Dict, Any
from api.application.services.diagnostic_service import DiagnosticService
from api.application.services.llm_service import LLMService
from api.infra.services.predict_services.diabetes_prediction_service import (
    DiabetesPredictionService,
)
from api.infra.utils.prompt_builder import create_system_prompt, create_user_prompt


class DiabetesDiagnosticService(DiagnosticService):
    def __init__(
        self,
        prediction_service: DiabetesPredictionService,
        llm_service: LLMService,
    ):
        self.prediction_service = prediction_service
        self.llm_service = llm_service

    def generate_diagnostic_report(
        self,
        patient_data: Dict[str, Any],
    ) -> str:
        prediction_result = self.prediction_service.predict(patient_data)

        system_prompt = create_system_prompt()
        user_prompt = create_user_prompt(patient_data, prediction_result)

        return self.llm_service.invoke(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            top_p=0.9,
        )

    async def generate_diagnostic_report_stream(
        self,
        patient_data: Dict[str, Any],
    ):
        prediction_result = self.prediction_service.predict(patient_data)

        system_prompt = create_system_prompt()
        user_prompt = create_user_prompt(patient_data, prediction_result)

        async for chunk in self.llm_service.generate_response(
            user_input=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            top_p=0.9,
        ):
            yield chunk
