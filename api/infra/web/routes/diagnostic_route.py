"""
Diagnostic routes - Generate diagnostic reports with ML prediction + LLM explanation
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from api.application.dto.diabetes_prediction import PatientData, DiagnosticReportResponse, PredictionResponse
from api.application.services.diagnostic_service import DiagnosticService
from api.infra.container.dependecies import Container

router = APIRouter(prefix="/diagnostic", tags=["Diagnostic"])

container = Container()


def get_diagnostic_service() -> DiagnosticService:
    return container.diagnostic_service()


@router.post("/invoke", response_model=DiagnosticReportResponse)
def invoke_diagnostic(
    patient_data: PatientData,
    diagnostic_service: DiagnosticService = Depends(get_diagnostic_service),
):

    try:
        patient_dict = patient_data.model_dump(mode='json')
        
        diagnostic_report = diagnostic_service.generate_diagnostic_report(patient_dict)
        
        prediction_result = container.prediction_service().predict(patient_dict)
        prediction_response = PredictionResponse(**prediction_result)
        
        return DiagnosticReportResponse(
            prediction=prediction_response,
            diagnostic_report=diagnostic_report
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating diagnostic report: {str(e)}"
        )


@router.post("/stream")
async def stream_diagnostic(
    patient_data: PatientData,
    diagnostic_service: DiagnosticService = Depends(get_diagnostic_service),
):

    try:
        patient_dict = patient_data.model_dump(mode='json')
        
        async def generate():
            async for chunk in diagnostic_service.generate_diagnostic_report_stream(patient_dict):
                yield chunk
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error streaming diagnostic report: {str(e)}"
        )
