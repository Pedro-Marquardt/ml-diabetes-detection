from abc import ABC, abstractmethod
from typing import Dict, Any


class DiagnosticService(ABC):
    """Interface para serviços de diagnóstico médico"""

    @abstractmethod
    def generate_diagnostic_report(
        self,
        patient_data: Dict[str, Any],
    ) -> str:
        """
        Gera um relatório diagnóstico explicativo baseado nos dados do paciente.
        Faz a predição internamente e gera o relatório com explicação.

        Args:
            patient_data: Dados do paciente (18 features)

        Returns:
            Relatório diagnóstico em texto explicativo
        """
        pass

    @abstractmethod
    async def generate_diagnostic_report_stream(
        self,
        patient_data: Dict[str, Any],
    ):
        """
        Gera um relatório diagnóstico explicativo de forma assíncrona (streaming).
        Faz a predição internamente e gera o relatório com explicação.

        Args:
            patient_data: Dados do paciente (18 features)

        Yields:
            Chunks do relatório diagnóstico
        """
        pass
