from fastapi import FastAPI, Depends, HTTPException, UploadFile, Request
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from redis import Redis, ReadOnlyError
import logging
import os

from api.settings import api_settings
from api.routes.v1_router import v1_router
from api.dependencies.auth import verify_api_key

# Setup logging
logger = logging.getLogger("rate_limiter")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Konfigurasi Redis - menggunakan variabel environment atau default
REDIS_HOST = os.getenv("REDIS_HOST", "agno-demo-app-dev-redis")  # Gunakan nama container sebagai default
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_URI = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

logger.info(f"Connecting to Redis at: {REDIS_URI}")

# Function untuk mendapatkan user_id dari request
def get_user_id(request: Request):
    # Dapatkan user_id dari request (bisa dari header, query param, atau token)
    # Di sini contoh mengambil dari header "X-User-ID", bisa disesuaikan
    user_id = request.headers.get("X-User-ID", "anonymous")
    
    # Jika tidak ada user_id, gunakan IP address sebagai fallback
    if user_id == "anonymous":
        user_id = request.client.host
    
    logger.info(f"Rate limit check for user: {user_id}")
    return user_id

# Inisialisasi limiter dengan In-Memory storage sebagai default
# Ini akan digunakan sebagai fallback jika Redis gagal
limiter = Limiter(key_func=get_user_id, default_limits=["4/120second"])
redis_status = "not_configured"

try:
    # Inisialisasi Redis client
    redis_client = Redis(
        host=REDIS_HOST, 
        port=REDIS_PORT, 
        db=REDIS_DB,
        socket_connect_timeout=5,  # 5 detik timeout
        socket_timeout=5,          # 5 detik timeout untuk operasi
        retry_on_timeout=True      # Retry pada timeout
    )
    
    # Test koneksi ke Redis dan coba operasi write
    test_key = "rate_limit_test_key"
    redis_client.set(test_key, "test_value")
    redis_client.delete(test_key)
    
    logger.info("Successfully connected to Redis server and write operation successful")
    redis_status = "connected_readwrite"
    
    # Inisialisasi limiter dengan Redis sebagai backend storage dan fungsi get_user_id
    limiter = Limiter(
        key_func=get_user_id,
        storage_uri=REDIS_URI,
        default_limits=["4/120second"]  # Default rate limit untuk semua endpoint
    )
except ReadOnlyError as e:
    logger.warning(f"Redis server is in read-only mode: {e}. Using in-memory storage for rate limiting.")
    redis_status = "connected_readonly"
    # Tetap gunakan in-memory limiter yang sudah diinisialisasi sebelumnya
except Exception as e:
    logger.error(f"Failed to connect to Redis server: {e}. Using in-memory storage for rate limiting.")
    redis_status = "connection_failed"
    # Tetap gunakan in-memory limiter yang sudah diinisialisasi sebelumnya

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

    # Tambahkan limiter ke app.state
    app.state.limiter = limiter
    
    # Tambahkan exception handler untuk rate limit exceeded
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Override _rate_limit_exceeded_handler untuk logging dan respons yang lebih baik
    @app.exception_handler(RateLimitExceeded)
    async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
        logger.warning(f"Rate limit exceeded for: {get_user_id(request)}, path: {request.url.path}")
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Terlalu banyak permintaan, coba lagi nanti.",
                "retry_after": getattr(exc, "retry_after", 120)  # 120 detik default
            }
        )

    # Tambahkan exception handler untuk ConnectionError
    @app.exception_handler(ConnectionError)
    async def connection_error_handler(request: Request, exc: ConnectionError):
        logger.error(f"Connection error for: {get_user_id(request)}, path: {request.url.path}, error: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Kesalahan koneksi ke Redis server. Silakan coba lagi nanti."
            }
        )
    
    # Tambahkan exception handler untuk ReadOnlyError
    @app.exception_handler(ReadOnlyError)
    async def readonly_error_handler(request: Request, exc: ReadOnlyError):
        logger.error(f"Redis ReadOnly error for: {get_user_id(request)}, path: {request.url.path}, error: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Redis server dalam mode read-only. Rate limiting menggunakan in-memory storage."
            }
        )
    
    # Tambahkan exception handler umum untuk semua exception
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception for: {get_user_id(request)}, path: {request.url.path}, error: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Terjadi kesalahan pada server. Silakan coba lagi nanti."
            }
        )

    # Tambahkan middleware untuk rate limiting - HARUS ditambahkan SEBELUM include_router
    try:
        app.add_middleware(SlowAPIMiddleware)
        logger.info("SlowAPIMiddleware successfully added")
    except Exception as e:
        logger.error(f"Failed to add SlowAPIMiddleware: {e}")

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

# Debug endpoint untuk status dengan limiter eksplisit
@app.get("/status")
@limiter.limit("4/120second")  # 2 request per 2 menit (120 detik)
async def health_check(request: Request):
    """Health check endpoint that is rate limited"""
    return {"status": "healthy", "redis_status": redis_status}

# Endpoint contoh dengan rate limiting berbasis user_id
@app.get("/rate-status")
async def rate_limited_endpoint(request: Request):
    """Contoh endpoint dengan rate limiting berbasis user_id menggunakan default rate limit"""
    # Rate limiting sudah diterapkan melalui default_limits di limiter
    user_id = get_user_id(request)
    
    # Get current redis status
    current_status = redis_status
    
    # Jika status connected, coba ping sebagai double-check
    if current_status == "connected_readwrite":
        try:
            if not redis_client.ping():
                current_status = "ping_failed"
        except Exception as e:
            current_status = f"error: {type(e).__name__}"
    
    # Get limiter info
    limiter_info = {
        "type": "redis" if "redis://" in str(limiter._storage_uri) else "in-memory",
        "default_limits": limiter._default_limits
    }
    
    return {
        "status": "healthy", 
        "rate_limit": {
            "user_id": user_id,
            "limit": "4 request per 2 menit",
            "redis_status": current_status,
            "limiter_info": limiter_info
        }
    }
