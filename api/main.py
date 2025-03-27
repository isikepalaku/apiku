from fastapi import FastAPI, Depends, HTTPException, UploadFile
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from api.settings import api_settings
from api.routes.v1_router import v1_router
from api.dependencies.auth import verify_api_key

def create_app() -> FastAPI:
    """Create a FastAPI App

    Returns:
        FastAPI: FastAPI App
    """

    # Create FastAPI App with custom max request size
    app: FastAPI = FastAPI(
        title=api_settings.title,
        version=api_settings.version,
        docs_url="/docs" if api_settings.docs_enabled else None,
        redoc_url="/redoc" if api_settings.docs_enabled else None,
        openapi_url="/openapi.json" if api_settings.docs_enabled else None,
    )

    # Add v1 router with API key dependency
    app.include_router(v1_router, dependencies=[Depends(verify_api_key)])
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Configure maximum upload size
    @app.middleware("http")
    async def max_body_size(request, call_next):
        if request.method == "POST":
            content_length = int(request.headers.get("content-length", 0))
            if content_length > api_settings.max_upload_size:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"File too large. Maximum size allowed is {api_settings.max_upload_size} bytes"}
                )
        return await call_next(request)

    return app

# Create FastAPI app
app = create_app()

@app.get("/status")
async def health_check():
    """Health check endpoint that is rate limited"""
    return {"status": "healthy"}
