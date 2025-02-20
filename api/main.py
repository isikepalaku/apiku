from fastapi import FastAPI, Depends
from starlette.middleware.cors import CORSMiddleware

from api.settings import api_settings
from api.routes.v1_router import v1_router
from api.dependencies.auth import verify_api_key
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address


def create_app() -> FastAPI:
    """Create a FastAPI App

    Returns:
        FastAPI: FastAPI App
    """

    # Create FastAPI App
    app: FastAPI = FastAPI(
        title=api_settings.title,
        version=api_settings.version,
        docs_url="/docs" if api_settings.docs_enabled else None,
        redoc_url="/redoc" if api_settings.docs_enabled else None,
        openapi_url="/openapi.json" if api_settings.docs_enabled else None,
    )

    # Add v1 router with API key dependency
    app.include_router(v1_router, dependencies=[Depends(verify_api_key)])

    # Add Middlewares
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

        # Konfigurasi rate limiting global
    limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(429, _rate_limit_exceeded_handler)

    return app


# Create FastAPI app
app = create_app()
