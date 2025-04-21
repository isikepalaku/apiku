import os
from typing import Optional
from pathlib import Path

from agno.agent import Agent
from agno.media import Image
from agno.models.google import Gemini
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from dotenv import load_dotenv
from google.generativeai import upload_file
from google.generativeai.types import file_types
from agno.tools.googlesearch import GoogleSearchTools

# Load environment variables
load_dotenv()

def get_geo_agent(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    """Initialize the geography analysis agent."""
    
    geo_query = """
    # Ahli Geografi: Analisis Lokasi dari Gambar

    PENTING: Berikan jawaban dalam Bahasa Indonesia.

    ## Petunjuk Visual yang Dianalisis
    - **Landmark:** Tempat-tempat atau bangunan ikonik yang dapat dikenali
    - **Arsitektur:** Gaya bangunan atau desain yang muncul dalam gambar
    - **Fitur Alam:** Elemen alam seperti gunung, sungai, dan garis pantai
    - **Bahasa atau Simbol:** Teks, rambu jalan, papan iklan, atau nama yang terlihat dalam gambar
    - **Pakaian atau Aspek Budaya:** Cara berpakaian atau elemen budaya masyarakat yang terlihat
    - **Petunjuk Lingkungan:** Kondisi cuaca, waktu (siang/malam), dan aspek lingkungan lainnya

    ## Instruksi Analisis
    1. **Periksa Gambar Secara Menyeluruh:**  
       Lihat semua detail yang tersedia dalam gambar.

    2. **Buat Dugaan Lokasi:**  
       Lakukan pencarian 'DuckDuckGoTools' untuk mengetahui informasi lokasi yang relevan. Berikan dugaan lokasi yang mencakup nama jalan (jika terlihat), kota, provinsi (jika relevan), dan negara.

    3. **Jelaskan Alasan Secara Detail:**  
       Sertakan penjelasan mendalam mengenai petunjuk visual yang mendukung kesimpulan Anda.

    4. **Berikan Opsi Jika Tidak Pasti:**  
       Jika tidak yakin, tawarkan beberapa kemungkinan lokasi beserta alasan yang mendukung masing-masing dugaan.

    ## Format Output
    Berikan hasil analisis dalam format:
    Nama Lokasi, Kota, Negara

    Disertai dengan penjelasan yang mendetail dalam format markdown menggunakan Bahasa Indonesia.

    **Catatan:** Jika gambar tidak tersedia, beritahukan bahwa gambar diperlukan untuk analisis.
    """

    return Agent(
        name="Agen Analisis Lokasi",
        agent_id="geo-image-agent",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash"),
        tools=[
            GoogleSearchTools(cache_results=True), 
            Newspaper4kTools(),
            MCPTools(
                server_params=StdioServerParameters(
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-sequential-thinking"]
                )
            )
        ],
        description="Anda adalah ahli geografi yang menganalisis gambar untuk menentukan lokasi berdasarkan petunjuk visual yang tersedia. Semua analisis akan diberikan dalam Bahasa Indonesia.",
        instructions=[geo_query],
        markdown=True,
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        monitoring=True,
        debug_mode=debug_mode,
    )

# Function to analyze the image and return location information
def analyze_image(image_path: Path) -> Optional[str]:
    """
    Analyze an image to determine its geographic location.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        String containing location analysis in markdown format
        
    Raises:
        RuntimeError: If there's an error during image analysis
    """
    try:
        agent = get_geo_agent()
        prompt = """
        Mohon analisis gambar ini dan berikan:
        1. Lokasi yang terlihat dalam gambar
        2. Penjelasan detail mengapa Anda memilih lokasi tersebut
        3. Jika tidak yakin, berikan beberapa kemungkinan lokasi

        PENTING: Berikan jawaban dalam Bahasa Indonesia.
        """
        response = agent.run(prompt, images=[Image(filepath=image_path)])
        return response.content
    except Exception as e:
        raise RuntimeError(f"Terjadi kesalahan saat menganalisis gambar: {e}")
