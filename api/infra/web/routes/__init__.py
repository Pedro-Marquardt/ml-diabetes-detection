"""
Routes module - Importa todas as rotas
"""

from api.infra.web.routes.health_route import router as health_router
from api.infra.web.routes.diagnostic_route import router as diagnostic_router

__all__ = ["health_router", "diagnostic_router"]
