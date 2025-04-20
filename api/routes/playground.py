from os import getenv
from fastapi import Depends, UploadFile, File, Form, HTTPException # Removed alias, ensure File is fastapi.File
from typing import List, Optional
from fastapi.responses import StreamingResponse
from agno.playground import Playground
from agno.media import File as AgnoFile
import os # Import os for file operations
from google import genai # Re-import genai
from time import sleep # Re-import sleep
import logging # Import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from agents.agen_perkaba import get_perkaba_agent
from agents.agen_bantek import get_perkaba_bantek_agent
from agents.agen_emp import get_emp_agent
from agents.agen_wassidik import get_wassidik_agent
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
from workflows.modus_operandi import get_analisator_tren_kejahatan
from workflows.sentiment_analysis import get_sentiment_analyzer
from workflows.analisis_hukum import get_sistem_penelitian_hukum
#from teams.penelititipidkor import get_corruption_investigator_team # Import the team

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
agen_siber = get_siber_agent(debug_mode=True, files=None)
agen_perbankan = get_perbankan_agent(debug_mode=True)
agen_tipidter = get_tipidter_agent(debug_mode=True)
agen_narkotika = get_narkotika_agent(debug_mode=True)
agen_ppa_ppo = get_ppa_ppo_agent(debug_mode=True)
#penyidik_tipikor_team = get_corruption_investigator_team(debug_mode=True) # Instantiate the team

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
        agen_ppa_ppo
    ],
    workflows=[
        analisator_kejahatan,
        sentiment_analyzer,
        sistem_penelitian_hukum
    ],
    teams=[ # Add the teams parameter
        #penyidik_tipikor_team
    ],
)

# Log the playground endpoint with phidata.app
if getenv("RUNTIME_ENV") == "dev":
    playground.create_endpoint("http://localhost:8000")

# Create the router from the playground
playground_router = playground.get_router()

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
        genai_client = genai.Client() # Initialize client
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

                # --- Google GenAI Upload Logic ---
                logger.info(f"Uploading {temp_file_path} to Google GenAI...")
                upload_result = genai_client.files.upload(file=temp_file_path)
                logger.info(f"Upload initiated. File name: {upload_result.name}")

                # Get the file from Google GenAI, retry if not ready
                retrieved_google_file = None
                retries = 0
                wait_time = 5
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

            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                # Clean up any files created so far in this request before raising
                for path in temp_file_paths:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception as cleanup_e:
                            logger.error(f"Error removing temporary file {path} during exception handling: {cleanup_e}")
                raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}")

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
            # For streaming responses
            response = await agent.aprint_response(
                message=message,
                files=agno_files if agno_files else None,
                stream=True
            )

            async def generate_stream():
                async for chunk in response:
                    yield f"data: {chunk}\n\n"
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
