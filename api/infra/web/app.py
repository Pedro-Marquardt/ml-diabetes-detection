"""
FastAPI Application
"""

from fastapi import FastAPI
from api.infra.web.routes import health_router, diagnostic_router

# Criar instância do FastAPI
app = FastAPI(
    title="Diabetes Detection API",
    description="API para predição de diabetes usando Machine Learning",
    version="1.0.0",
)

# Registrar rotas
app.include_router(health_router)
app.include_router(diagnostic_router)


@app.get("/")
async def root():
    """
    Rota raiz
    """
    return {"message": "Diabetes Detection API", "version": "1.0.0", "docs": "/docs"}
