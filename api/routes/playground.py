from os import getenv
from fastapi import Depends, UploadFile, File, Form, HTTPException # Removed alias, ensure File is fastapi.File
from typing import List, Optional
from fastapi.responses import StreamingResponse
from agno.playground import Playground
from agno.media import File as AgnoFile
import os # Import os for file operations
import mimetypes # Import mimetypes for MIME type detection
import re # Import re for regex operations
import hashlib # Import hashlib for generating file hashes
import time # Import time for timestamps
from pathlib import Path # Import Path for better file handling
from google import genai # Re-import genai
from time import sleep # Re-import sleep
from agno.utils.log import logger # Menggunakan logger dari agno
from agents.agen_perkaba import get_perkaba_agent
from agents.agen_bantek import get_perkaba_bantek_agent
from agents.agen_emp import get_emp_agent
from agents.agen_wassidik import get_wassidik_agent
from agents.wassidik_chat import get_wassidik_chat_agent
from agents.fidusia import get_corruption_investigator
from agents.hoax import fact_checker_agent
from agents.perbankan import get_perbankan_agent
from agents.image import get_geo_agent
from agents.agen_maps import get_maps_agent
from agents.tipidkor_chat import get_tipidkor_agent
from agents.research import get_research_agent
from agents.dokpol import get_medis_agent
from agents.forensic import get_forensic_agent
from agents.trend_kejahatan import get_crime_trend_agent
from agents.p2sk_chat import get_p2sk_agent
from agents.indagsi_chat import get_ipi_agent
from agents.tipidter_chat import get_tipidter_agent
from agents.kuhp_chat import get_kuhp_agent # Existing KUHP (UU 1/2023) agent
from agents.kuhap_chat import get_kuhap_agent # New KUHAP (UU 8/1981) agent
from agents.fismondev_chat import get_fismondev_agent
from agents.ite_chat import get_ite_agent
from agents.siber_chat import get_siber_agent
from agents.ciptakerja_chat import get_cipta_kerja_agent
from agents.kesehatan_chat import get_kesehatan_agent
from agents.sentiment_analyzer import get_sentiment_team
from agents.narkotika_chat import get_narkotika_agent
from agents.ppa_ppo_chat import get_ppa_ppo_agent
from agents.dit_reskrimum_chat import get_dit_reskrimum_agent # Import Dit Reskrimum agent
from agents.ahli_hukum_pidana import get_ahli_hukum_pidana_agent # Import Ahli Hukum Pidana agent
from workflows.modus_operandi import get_analisator_tren_kejahatan
from workflows.sentiment_analysis import get_sentiment_analyzer
from workflows.analisis_hukum import get_sistem_penelitian_hukum
from teams.penelititipidkor import get_sentiment_analysis_team # Import the new sentiment analysis team
from teams.ahli_hukum_pidana import get_ahli_hukum_pidana_team # Import the criminal law expert team

######################################################
## Router for the agent playground
######################################################
trend_kejahatan = get_crime_trend_agent(debug_mode=True)
agen_p2sk = get_p2sk_agent(debug_mode=True)
agen_kuhp = get_kuhp_agent(debug_mode=True) # Existing KUHP (UU 1/2023) agent
agen_kuhap = get_kuhap_agent(debug_mode=True) # New KUHAP (UU 8/1981) agent
agen_ite = get_ite_agent(debug_mode=True)
agen_cipta_kerja = get_cipta_kerja_agent(debug_mode=True)
agen_kesehatan = get_kesehatan_agent(debug_mode=True)
agen_indagsi = get_ipi_agent(debug_mode=True)
agen_emp = get_emp_agent(debug_mode=True)
agen_wassidik = get_wassidik_agent(debug_mode=True)
agen_wassidik_chat = get_wassidik_chat_agent(debug_mode=True)
agen_perkaba = get_perkaba_agent(debug_mode=True)
agen_bantek = get_perkaba_bantek_agent(debug_mode=True)
geo_agent = get_geo_agent(debug_mode=True)
penyidik_polri = get_research_agent(debug_mode=True)
penyidik_tipikor = get_corruption_investigator(debug_mode=True)
agen_tipidkor = get_tipidkor_agent(debug_mode=True)
analisator_kejahatan = get_analisator_tren_kejahatan(debug_mode=True)
sentiment_analyzer = get_sentiment_analyzer(debug_mode=True)
sentiment_team = get_sentiment_team(debug_mode=True)
sistem_penelitian_hukum = get_sistem_penelitian_hukum(debug_mode=True)
agen_dokpol = get_medis_agent(debug_mode=True)
agen_forensic = get_forensic_agent(debug_mode=True)
agen_maps = get_maps_agent(debug_mode=True)
agen_fismondev = get_fismondev_agent(debug_mode=True)
# Inisialisasi agen_siber tanpa file sehingga bisa muncul di Swagger
agen_siber = get_siber_agent(debug_mode=True)
agen_perbankan = get_perbankan_agent(debug_mode=True)
agen_tipidter = get_tipidter_agent(debug_mode=True)
agen_narkotika = get_narkotika_agent(debug_mode=True)
agen_ppa_ppo = get_ppa_ppo_agent(debug_mode=True)
agen_dit_reskrimum = get_dit_reskrimum_agent(debug_mode=True) # Instantiate Dit Reskrimum agent
agen_ahli_hukum_pidana = get_ahli_hukum_pidana_agent(debug_mode=True) # Instantiate Ahli Hukum Pidana agent
sentiment_analysis_team_instance = get_sentiment_analysis_team(debug_mode=True)
ahli_hukum_pidana_team_instance = get_ahli_hukum_pidana_team(debug_mode=True)
# penyidik_tipikor_team telah dihapus karena fungsinya tidak tersedia lagi

playground = Playground(
    agents=[
        fact_checker_agent,
        agen_maps,
        geo_agent,
        penyidik_polri,
        penyidik_tipikor,
        trend_kejahatan,
        agen_perkaba,
        agen_bantek,
        agen_emp,
        agen_wassidik,
        agen_wassidik_chat,
        agen_tipidkor,
        agen_p2sk,
        agen_kuhp, # Existing KUHP (UU 1/2023) agent
        agen_kuhap, # New KUHAP (UU 8/1981) agent
        agen_ite,
        agen_cipta_kerja,
        agen_kesehatan,
        agen_indagsi,
        agen_dokpol,
        agen_forensic,
        agen_fismondev,
        agen_siber,  # Mengembalikan agen_siber ke daftar
        agen_perbankan,
        agen_tipidter,
        sentiment_team,
        agen_narkotika,
        agen_ppa_ppo,
        agen_dit_reskrimum, # Add Dit Reskrimum agent instance
        agen_ahli_hukum_pidana # Add Ahli Hukum Pidana agent instance
    ],
    workflows=[
        analisator_kejahatan,
        sentiment_analyzer,
        sistem_penelitian_hukum
    ],
    teams=[ 
        sentiment_analysis_team_instance, # Gunakan instance tim yang telah diganti namanya
        ahli_hukum_pidana_team_instance # Add the criminal law expert team
    ],
)

# Log the playground endpoint with phidata.app
# if getenv("RUNTIME_ENV") == "dev":
if getenv("RUNTIME_ENV") == "dev":
    playground.register_app_on_platform()

# Create the router from the playground
playground_router = playground.get_async_router()

# Mapping eksplisit untuk file types yang didukung Google GenAI
GENAI_MIME_TYPE_MAP = {
    # Document formats
    '.pdf': 'application/pdf',
    '.txt': 'text/plain',
    '.html': 'text/html',
    '.css': 'text/css',
    '.md': 'text/markdown',
    '.csv': 'text/csv',
    '.xml': 'text/xml',
    '.rtf': 'application/rtf',
    
    # Code formats
    '.js': 'application/javascript',
    '.py': 'text/x-python',
    '.java': 'text/x-java-source',
    '.cpp': 'text/x-c',
    '.c': 'text/x-c',
    '.php': 'text/x-php',
    '.go': 'text/x-go',
    '.rs': 'text/x-rust',
    '.kt': 'text/x-kotlin',
    '.swift': 'text/x-swift',
    '.rb': 'text/x-ruby',
    '.sql': 'text/x-sql',
    '.json': 'application/json',
    '.yaml': 'text/yaml',
    '.yml': 'text/yaml',
    
    # Office formats (yang sering bermasalah)
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.xls': 'application/vnd.ms-excel',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.ppt': 'application/vnd.ms-powerpoint',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    
    # Image formats
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.bmp': 'image/bmp',
    '.webp': 'image/webp',
    '.tiff': 'image/tiff',
    '.tif': 'image/tiff',
    
    # Audio formats
    '.mp3': 'audio/mpeg',
    '.wav': 'audio/wav',
    '.flac': 'audio/flac',
    '.aac': 'audio/aac',
    '.ogg': 'audio/ogg',
    '.m4a': 'audio/mp4',
    
    # Video formats
    '.mp4': 'video/mp4',
    '.avi': 'video/x-msvideo',
    '.mov': 'video/quicktime',
    '.wmv': 'video/x-ms-wmv',
    '.flv': 'video/x-flv',
    '.webm': 'video/webm',
    '.mkv': 'video/x-matroska',
    
    # Archive formats
    '.zip': 'application/zip',
    '.rar': 'application/x-rar-compressed',
    '.7z': 'application/x-7z-compressed',
    '.tar': 'application/x-tar',
    '.gz': 'application/gzip',
}

def get_mime_type_for_genai(file_path: str, original_filename: str = None) -> str:
    """
    Get MIME type untuk Google GenAI dengan fallback yang reliable
    
    Args:
        file_path: Path ke file yang akan diupload
        original_filename: Nama file asli (untuk extension detection)
        
    Returns:
        MIME type string yang sesuai untuk Google GenAI
    """
    # Gunakan original filename jika ada, fallback ke file_path
    filename_for_ext = original_filename or file_path
    extension = Path(filename_for_ext).suffix.lower()
    
    logger.info(f"Determining MIME type for file: {filename_for_ext}")
    logger.info(f"Detected extension: {extension}")
    
    # Check mapping eksplisit dulu (prioritas tertinggi)
    if extension in GENAI_MIME_TYPE_MAP:
        mime_type = GENAI_MIME_TYPE_MAP[extension]
        logger.info(f"Using explicit mapping for {extension}: {mime_type}")
        return mime_type
    
    # Fallback ke mimetypes.guess_type
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        logger.info(f"Using mimetypes.guess_type for {extension}: {mime_type}")
        return mime_type
    
    # Last resort: berdasarkan kategori extension
    logger.warning(f"Could not determine MIME type for {file_path}, using fallback")
    
    # Fallback untuk extensions yang tidak dikenal
    if extension in ['.txt', '.text']:
        return 'text/plain'
    elif extension in ['.htm', '.html']:
        return 'text/html'
    elif extension in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    elif extension in ['.png']:
        return 'image/png'
    elif extension in ['.pdf']:
        return 'application/pdf'
    else:
        # Default fallback
        logger.warning(f"Using application/octet-stream for unknown extension: {extension}")
        return 'application/octet-stream'

def validate_file_for_genai(file_path: str, original_filename: str, max_size_mb: int = 100) -> tuple[bool, str]:
    """
    Validate file untuk Google GenAI upload
    
    Args:
        file_path: Path ke file temporer
        original_filename: Nama file asli
        max_size_mb: Maksimum ukuran file dalam MB
        
    Returns:
        Tuple (is_valid: bool, error_message: str)
    """
    try:
        # Check file exists
        if not os.path.exists(file_path):
            return False, f"File tidak ditemukan: {file_path}"
        
        # Check file size
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        logger.info(f"File validation - Size: {file_size_mb:.2f}MB")
        
        if file_size_mb > max_size_mb:
            return False, f"File terlalu besar: {file_size_mb:.2f}MB. Maksimum: {max_size_mb}MB"
        
        # Check minimum file size (anti corruption)
        if file_size < 10:  # File kurang dari 10 bytes kemungkinan corrupt
            return False, f"File terlalu kecil atau corrupt: {file_size} bytes"
        
        # Check extension
        extension = Path(original_filename).suffix.lower()
        if not extension:
            return False, "File harus memiliki extension"
        
        # Check if extension is commonly supported
        supported_extensions = {
            # Documents
            '.pdf', '.txt', '.html', '.css', '.md', '.csv', '.xml', '.rtf',
            '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            # Code files
            '.js', '.py', '.java', '.cpp', '.c', '.php', '.go', '.rs', '.kt', '.swift', '.rb',
            '.sql', '.json', '.yaml', '.yml',
            # Images
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif',
            # Audio
            '.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a',
            # Video  
            '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv'
        }
        
        if extension not in supported_extensions and extension not in ['.bin', '.exe', '.app']:
            logger.warning(f"File extension {extension} mungkin tidak didukung oleh Google GenAI")
        
        # Special validation untuk Office files
        if extension in ['.docx', '.xlsx', '.pptx', '.doc', '.xls', '.ppt']:
            logger.info(f"Office file detected: {extension}")
            # Additional validation bisa ditambahkan di sini jika perlu
        
        return True, "File valid"
        
    except Exception as e:
        logger.error(f"Error validating file {original_filename}: {e}")
        return False, f"Error validating file: {str(e)}"

def upload_file_to_genai_with_retry(file_path: str, original_filename: str, max_retries: int = 3) -> any:
    """
    Upload file ke Google GenAI dengan retry mechanism dan comprehensive error handling
    
    Args:
        file_path: Path ke file temporer
        original_filename: Nama file asli untuk display
        max_retries: Maksimum retry attempts
        
    Returns:
        Google GenAI file object
    """
    # Validate file first
    is_valid, error_msg = validate_file_for_genai(file_path, original_filename)
    if not is_valid:
        raise ValueError(f"File validation failed: {error_msg}")
    
    # Log comprehensive file info  
    file_size = os.path.getsize(file_path)
    extension = Path(original_filename).suffix.lower()
    
    logger.info(f"=== Google GenAI Upload Details ===")
    logger.info(f"Original filename: {original_filename}")
    logger.info(f"File extension: {extension}")
    logger.info(f"File size: {file_size / (1024*1024):.2f}MB")
    logger.info(f"Temp file path: {file_path}")
    logger.info(f"Note: MIME type will be auto-detected by Google GenAI API")
    
    genai_client = genai.Client()
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Upload attempt {attempt + 1}/{max_retries}")
            
            # Generate valid file name untuk Google GenAI API
            # API requirements: lowercase alphanumeric + dashes only, no start/end dash
            # IMPORTANT: nama maksimal 40 karakter (excluding 'files/')
            
            clean_filename = Path(original_filename).stem.lower()
            # Replace spaces and special chars with dashes, keep only alphanumeric and dashes
            clean_filename = re.sub(r'[^a-z0-9\-]', '-', clean_filename)
            # Remove multiple consecutive dashes
            clean_filename = re.sub(r'-+', '-', clean_filename)
            # Remove leading/trailing dashes
            clean_filename = clean_filename.strip('-')
            
            # Ensure filename is not empty
            if not clean_filename:
                clean_filename = 'uploaded-file'
            
            # ALWAYS add uniqueness to prevent 409 ALREADY_EXISTS errors
            # Generate unique timestamp and hash
            timestamp = str(int(time.time()))[-6:]  # Last 6 digits of timestamp
            hash_suffix = hashlib.md5(f"{original_filename}{time.time()}".encode()).hexdigest()[:4]
            
            # Limit to 40 characters (Google GenAI constraint)
            MAX_FILENAME_LENGTH = 40
            # Reserve space for uniqueness suffix: -XXXXXX-YYYY = 12 chars
            reserved_space = 12
            available_length = MAX_FILENAME_LENGTH - reserved_space
            
            if len(clean_filename) > available_length:
                # Truncate if too long
                truncated_name = clean_filename[:available_length].rstrip('-')
                clean_filename = f"{truncated_name}-{timestamp}-{hash_suffix}"
                logger.info(f"Filename truncated for uniqueness: {len(clean_filename)} characters")
            else:
                # Add uniqueness suffix even if name is short enough
                clean_filename = f"{clean_filename}-{timestamp}-{hash_suffix}"
            
            # Final safety check
            if len(clean_filename) > MAX_FILENAME_LENGTH:
                # Emergency fallback: use only timestamp and hash
                timestamp = str(int(time.time()))[-8:]
                hash_suffix = hashlib.md5(f"{original_filename}{time.time()}".encode()).hexdigest()[:6]
                clean_filename = f"file-{timestamp}-{hash_suffix}"
            
            genai_file_name = f"files/{clean_filename}"
            logger.info(f"Generated file name for Google GenAI: {genai_file_name} (length: {len(clean_filename)})")
            
            # Upload dengan parameter yang benar sesuai dokumentasi agno
            upload_result = genai_client.files.upload(
                file=file_path,
                config=dict(name=genai_file_name)
            )
            
            logger.info(f"Upload initiated successfully. File name: {upload_result.name}")
            
            # Wait for the file to finish processing (seperti contoh agno)
            while upload_result.state.name == "PROCESSING":
                logger.info("Waiting for file to be processed...")
                sleep(2)
                upload_result = genai_client.files.get(name=upload_result.name)
            
            logger.info(f"File processing complete: {upload_result.uri}")
            return upload_result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Upload attempt {attempt + 1} failed: {error_msg}")
            
            # Specific error handling
            if "Unknown mime type" in error_msg or "Could not determine the mimetype" in error_msg:
                # MIME type error biasanya terjadi jika file corrupt atau format tidak supported
                logger.error(f"MIME type detection failed for {original_filename}")
                raise ValueError(f"File format tidak didukung atau file corrupt: {extension}. Google GenAI tidak dapat memproses file ini.")
            
            elif "File too large" in error_msg or "size" in error_msg.lower():
                raise ValueError(f"File terlalu besar: {original_filename}. Maksimum ukuran file yang didukung adalah 100MB.")
            
            elif "ALREADY_EXISTS" in error_msg or "already exists" in error_msg:
                # File sudah ada - generate nama yang lebih unik untuk retry
                if attempt < max_retries - 1:
                    logger.info(f"File name conflict detected, will retry with more unique name...")
                    # Will retry with different timestamp/hash in next iteration
                    sleep(0.5)  # Brief pause to ensure different timestamp
                    continue
                else:
                    raise ValueError(f"File dengan nama serupa sudah ada di Google GenAI. Gagal membuat nama unik setelah {max_retries} percobaan.")
            
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    logger.info(f"Rate limit hit, waiting {wait_time}s before retry...")
                    sleep(wait_time)
                    continue
                else:
                    raise ValueError("Quota limit tercapai. Silakan coba lagi nanti.")
            
            # Jika ini attempt terakhir, raise error
            if attempt == max_retries - 1:
                raise ValueError(f"Gagal upload file {original_filename} setelah {max_retries} percobaan: {error_msg}")
            
            # Wait sebelum retry
            sleep(1 * (attempt + 1))
    
    raise ValueError(f"Upload failed after {max_retries} attempts")

# Universal endpoint for all agents to handle file uploads
@playground_router.post("/agents/{agent_id}/runs-with-files", tags=["Agents"])
async def agent_with_files(
    agent_id: str,
    message: str = Form(...),
    stream: bool = Form(False),
    monitor: bool = Form(False),
    session_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None), # Correct type hint and default using fastapi.File
):
    """Universal endpoint handler for any agent that supports file uploads"""
    agno_files = []

    # Process uploaded files if any
    if files:
        temp_file_paths = [] # Keep track of temp files for cleanup
        for file in files:
            logger.info(f"Processing uploaded file: {file.filename}")
            # Create a temporary file path to save the content
            temp_file_path = f"/tmp/{file.filename}"
            logger.info(f"Saving file temporarily to: {temp_file_path}")
            temp_file_paths.append(temp_file_path) # Add to list for cleanup

            try:
                # Read the file content
                file_content = await file.read()
                # Save the file content to the temporary path
                with open(temp_file_path, "wb") as f:
                    f.write(file_content)
                logger.info(f"File saved successfully to {temp_file_path}")

                # --- Google GenAI Upload Logic (Enhanced) ---
                logger.info(f"Uploading {temp_file_path} to Google GenAI...")
                
                # Use enhanced upload function with comprehensive error handling
                upload_result = upload_file_to_genai_with_retry(
                    file_path=temp_file_path,
                    original_filename=file.filename,
                    max_retries=3
                )

                # Get the file from Google GenAI, retry if not ready
                retrieved_google_file = None
                retries = 0
                wait_time = 5
                genai_client = genai.Client()  # Create client instance in correct scope
                while retrieved_google_file is None and retries < 4: # Increased retries
                    logger.info(f"Attempt {retries + 1} to get file status...")
                    try:
                        retrieved_google_file = genai_client.files.get(name=upload_result.name)
                        # Check the state explicitly
                        if retrieved_google_file.state.name != 'ACTIVE':
                            logger.warning(f"File state is {retrieved_google_file.state.name}. Retrying in {wait_time}s...")
                            retrieved_google_file = None # Reset to trigger retry
                            sleep(wait_time)
                        else:
                            logger.info(f"File is ACTIVE.")
                            break # Exit loop if active
                    except Exception as e_get:
                        logger.error(f"Error getting file status: {e_get}. Retrying...")
                        sleep(wait_time)
                    retries += 1

                if retrieved_google_file and retrieved_google_file.state.name == 'ACTIVE':
                    logger.info(f"File {retrieved_google_file.name} is ready. Adding to agno_files using external object.")
                    agno_files.append(AgnoFile(external=retrieved_google_file)) # Use external object
                else:
                    logger.error(f"File {upload_result.name} was not ready after multiple attempts or failed processing. State: {retrieved_google_file.state.name if retrieved_google_file else 'N/A'}")
                    # Optionally raise an exception or return an error response
                    # raise HTTPException(status_code=500, detail=f"File processing failed for {file.filename}")
                # --- End Google GenAI Upload Logic ---

            except ValueError as ve:
                # Handle validation errors dengan user-friendly messages
                logger.error(f"File validation error for {file.filename}: {ve}")
                # Clean up any files created so far in this request before raising
                for path in temp_file_paths:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception as cleanup_e:
                            logger.error(f"Error removing temporary file {path} during exception handling: {cleanup_e}")
                raise HTTPException(status_code=400, detail=f"File {file.filename}: {str(ve)}")
            
            except Exception as e:
                # Handle unexpected errors
                error_msg = str(e)
                logger.error(f"Unexpected error processing file {file.filename}: {error_msg}")
                
                # Clean up any files created so far in this request before raising
                for path in temp_file_paths:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception as cleanup_e:
                            logger.error(f"Error removing temporary file {path} during exception handling: {cleanup_e}")
                
                # Provide specific error messages based on error content
                if "Unknown mime type" in error_msg:
                    extension = Path(file.filename).suffix.lower()
                    raise HTTPException(
                        status_code=400, 
                        detail=f"File {file.filename}: Format file {extension} tidak didukung. Silakan gunakan PDF, DOCX, TXT, atau format gambar."
                    )
                elif "File too large" in error_msg or "size" in error_msg.lower():
                    raise HTTPException(
                        status_code=400, 
                        detail=f"File {file.filename}: File terlalu besar. Maksimum ukuran file yang didukung adalah 100MB."
                    )
                elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                    raise HTTPException(
                        status_code=429, 
                        detail="Quota upload tercapai. Silakan coba lagi dalam beberapa menit."
                    )
                else:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Error processing file {file.filename}: {error_msg}"
                    )

    # Dynamically get the correct agent function based on agent_id
    agent_functions = {
        "siber-chat": get_siber_agent,
        "tipidkor-chat": get_tipidkor_agent,
        "p2sk-chat": get_p2sk_agent,
        "kuhp-chat": get_kuhp_agent, # Existing KUHP (UU 1/2023) agent
        "kuhap-chat": get_kuhap_agent, # New KUHAP (UU 8/1981) agent
        "ite-chat": get_ite_agent,
        "cipta-kerja-chat": get_cipta_kerja_agent,
        "kesehatan-chat": get_kesehatan_agent,
        "indagsi-chat": get_ipi_agent,
        "tipidter-chat": get_tipidter_agent,
        "fismondev-chat": get_fismondev_agent,
        "perbankan-chat": get_perbankan_agent,
        "narkotika-chat": get_narkotika_agent,
        "ppa-ppo-chat": get_ppa_ppo_agent,
        "research-agent": get_research_agent,
        "corruption-investigator": get_corruption_investigator,
        "geo-agent": get_geo_agent,
        "fact-checker": lambda **kwargs: fact_checker_agent,
        "medis-agent": get_medis_agent,
        "forensic-agent": get_forensic_agent,
        "maps-agent": get_maps_agent,
        "perkaba-agent": get_perkaba_agent,
        "bantek-agent": get_perkaba_bantek_agent,
        "emp-agent": get_emp_agent,
        "wassidik-agent": get_wassidik_agent,
        "wassidik-chat": get_wassidik_chat_agent,
        "dit-reskrimum-chat": get_dit_reskrimum_agent, # Add Dit Reskrimum agent function mapping
        "ahli-hukum-pidana": get_ahli_hukum_pidana_agent, # Add Ahli Hukum Pidana agent function mapping
    }
    
    # Get the agent creation function or return 404 if not found
    if agent_id not in agent_functions:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Create the agent
    agent_function = agent_functions[agent_id]
    # Create the agent, passing files if available
    agent = agent_function(
        user_id=user_id,
        session_id=session_id,
        debug_mode=True
        # Files should NOT be passed during agent init
    )

    try:
        # Process the request using the agent
        if stream:
            # For streaming responses, use arun with stream=True
            logger.info(f"Starting streaming response for agent {agent_id}")
            
            async def generate_stream():
                try:
                    # Use arun with stream=True - this returns an async generator
                    logger.info(f"Calling agent.arun with stream=True for {agent_id}")
                    run_response = agent.arun(
                        message=message,
                        files=agno_files if agno_files else None,
                        stream=True
                    )
                    
                    # The response should be an async generator/iterator
                    if run_response is None:
                        logger.error(f"Agent {agent_id} returned None for streaming request")
                        yield f"data: Error: Agent tidak dapat memproses permintaan streaming\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    
                    # Check if it's awaitable first
                    if hasattr(run_response, '__await__'):
                        logger.info(f"run_response is awaitable, awaiting it first")
                        run_response = await run_response
                    
                    # Check if it has __aiter__ method (async iterator)
                    if not hasattr(run_response, '__aiter__'):
                        logger.error(f"Agent {agent_id} returned non-async-iterable response: {type(run_response)}")
                        # Try to extract content from the response
                        if hasattr(run_response, 'content'):
                            yield f"data: {run_response.content}\n\n"
                        elif hasattr(run_response, 'output'):
                            yield f"data: {run_response.output}\n\n"
                        else:
                            yield f"data: {str(run_response)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    
                    # Now we can safely iterate through the async generator
                    logger.info(f"Starting async iteration for {agent_id}")
                    async for chunk in run_response:
                        # chunk should have a 'content' attribute based on the agents.py example
                        if chunk and hasattr(chunk, 'content'):
                            content = chunk.content
                            if content:
                                yield f"data: {content}\n\n"
                        elif chunk:
                            # Fallback if chunk is just a string
                            yield f"data: {chunk}\n\n"
                    
                    yield "data: [DONE]\n\n"
                    logger.info(f"Finished streaming for {agent_id}")
                    
                except Exception as e:
                    logger.error(f"Error in streaming generation for agent {agent_id}: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    yield f"data: Error: {str(e)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            # For non-streaming responses
            run_result = await agent.arun(
                message=message,
                files=agno_files if agno_files else None # Pass files here
            )
            # Extract the output string from the result object
            response_output = run_result.output if hasattr(run_result, 'output') else str(run_result)

            return {"response": response_output}
    finally:
        # Clean up the temporary files after the request is processed
        if 'temp_file_paths' in locals():
            for path in temp_file_paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        logger.info(f"Cleaned up temporary file: {path}")
                    except Exception as e:
                        logger.error(f"Error removing temporary file {path} during cleanup: {e}")
