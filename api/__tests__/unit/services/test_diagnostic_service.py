import pytest
from unittest.mock import Mock, AsyncMock, patch
from api.infra.services.diagnostic_service import DiabetesDiagnosticService
from api.infra.services.predict_services.diabetes_prediction_service import (
    DiabetesPredictionService,
)
from api.application.services.llm_service import LLMService


@pytest.fixture
def mock_prediction_service():
    """Mock DiabetesPredictionService"""
    service = Mock(spec=DiabetesPredictionService)
    service.predict = Mock(
        return_value={
            "has_diabetes": False,
            "probability": 0.35,
            "threshold_used": 0.59,
            "confidence": "high",
        }
    )
    return service


@pytest.fixture
def mock_llm_service():
    """Mock LLMService"""
    service = Mock(spec=LLMService)
    service.invoke = Mock(return_value="This is a diagnostic report.")
    service.generate_response = AsyncMock()
    return service


@pytest.fixture
def diagnostic_service(mock_prediction_service, mock_llm_service):
    """Create DiabetesDiagnosticService instance"""
    return DiabetesDiagnosticService(
        prediction_service=mock_prediction_service, llm_service=mock_llm_service
    )


@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing"""
    return {
        "age": 45.0,
        "education_level": "Graduate",
        "income_level": "Middle",
        "physical_activity_minutes_per_week": 150.0,
        "diet_score": 7.5,
        "family_history_diabetes": 1,
        "bmi": 28.5,
        "waist_to_hip_ratio": 0.92,
        "systolic_bp": 130.0,
        "cholesterol_total": 220.0,
        "hdl_cholesterol": 45.0,
        "ldl_cholesterol": 140.0,
        "triglycerides": 180.0,
        "glucose_fasting": 95.0,
        "glucose_postprandial": 140.0,
        "insulin_level": 12.0,
        "hba1c": 5.8,
        "diabetes_risk_score": 6.5,
    }


class TestDiabetesDiagnosticService:
    """Test suite for DiabetesDiagnosticService"""

    def test_init(self, mock_prediction_service, mock_llm_service):
        """Test initialization"""
        service = DiabetesDiagnosticService(
            prediction_service=mock_prediction_service, llm_service=mock_llm_service
        )

        assert service.prediction_service == mock_prediction_service
        assert service.llm_service == mock_llm_service

    @patch("api.infra.services.diagnostic_service.create_system_prompt")
    @patch("api.infra.services.diagnostic_service.create_user_prompt")
    def test_generate_diagnostic_report(
        self,
        mock_create_user_prompt,
        mock_create_system_prompt,
        diagnostic_service,
        mock_prediction_service,
        mock_llm_service,
        sample_patient_data,
    ):
        """Test generate_diagnostic_report method"""
        mock_create_system_prompt.return_value = "System prompt"
        mock_create_user_prompt.return_value = "User prompt"
        mock_llm_service.invoke.return_value = "Diagnostic report text"

        result = diagnostic_service.generate_diagnostic_report(sample_patient_data)

        # Verify prediction service was called
        mock_prediction_service.predict.assert_called_once_with(sample_patient_data)

        # Verify prompts were created
        mock_create_system_prompt.assert_called_once()
        mock_create_user_prompt.assert_called_once_with(
            sample_patient_data, mock_prediction_service.predict.return_value
        )

        # Verify LLM service was called
        mock_llm_service.invoke.assert_called_once_with(
            prompt="User prompt",
            system_prompt="System prompt",
            temperature=0.7,
            top_p=0.9,
        )

        assert result == "Diagnostic report text"

    @pytest.mark.asyncio
    @patch("api.infra.services.diagnostic_service.create_system_prompt")
    @patch("api.infra.services.diagnostic_service.create_user_prompt")
    async def test_generate_diagnostic_report_stream(
        self,
        mock_create_user_prompt,
        mock_create_system_prompt,
        diagnostic_service,
        mock_prediction_service,
        mock_llm_service,
        sample_patient_data,
    ):
        """Test generate_diagnostic_report_stream async method"""
        mock_create_system_prompt.return_value = "System prompt"
        mock_create_user_prompt.return_value = "User prompt"

        # Mock async generator that accepts arguments
        async def mock_generator(user_input, system_prompt, temperature=0.7, top_p=0.9):
            yield "Chunk 1"
            yield "Chunk 2"
            yield "Chunk 3"

        mock_llm_service.generate_response = mock_generator

        chunks = []
        async for chunk in diagnostic_service.generate_diagnostic_report_stream(
            sample_patient_data
        ):
            chunks.append(chunk)

        # Verify prediction service was called
        mock_prediction_service.predict.assert_called_once_with(sample_patient_data)

        # Verify prompts were created
        mock_create_system_prompt.assert_called_once()
        mock_create_user_prompt.assert_called_once_with(
            sample_patient_data, mock_prediction_service.predict.return_value
        )

        assert chunks == ["Chunk 1", "Chunk 2", "Chunk 3"]

    def test_generate_diagnostic_report_with_diabetes_positive(
        self, mock_prediction_service, mock_llm_service, sample_patient_data
    ):
        """Test generate_diagnostic_report with positive diabetes prediction"""
        # Mock positive prediction
        mock_prediction_service.predict.return_value = {
            "has_diabetes": True,
            "probability": 0.85,
            "threshold_used": 0.59,
            "confidence": "high",
        }
        mock_llm_service.invoke.return_value = "High risk report"

        service = DiabetesDiagnosticService(
            prediction_service=mock_prediction_service, llm_service=mock_llm_service
        )

        with (
            patch(
                "api.infra.services.diagnostic_service.create_system_prompt"
            ) as mock_sys,
            patch(
                "api.infra.services.diagnostic_service.create_user_prompt"
            ) as mock_usr,
        ):
            mock_sys.return_value = "System"
            mock_usr.return_value = "User"

            result = service.generate_diagnostic_report(sample_patient_data)

            assert result == "High risk report"
            mock_prediction_service.predict.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_diagnostic_report_stream_empty_response(
        self, mock_prediction_service, mock_llm_service, sample_patient_data
    ):
        """Test generate_diagnostic_report_stream with empty response"""

        async def empty_generator(
            user_input, system_prompt, temperature=0.7, top_p=0.9
        ):
            return
            yield  # Make it a generator

        mock_llm_service.generate_response = empty_generator

        service = DiabetesDiagnosticService(
            prediction_service=mock_prediction_service, llm_service=mock_llm_service
        )

        with (
            patch("api.infra.services.diagnostic_service.create_system_prompt"),
            patch("api.infra.services.diagnostic_service.create_user_prompt"),
        ):
            chunks = []
            async for chunk in service.generate_diagnostic_report_stream(
                sample_patient_data
            ):
                chunks.append(chunk)

            assert chunks == []
