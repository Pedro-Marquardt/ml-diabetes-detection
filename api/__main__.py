"""
Entry point para rodar o servidor uvicorn
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.infra.web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
