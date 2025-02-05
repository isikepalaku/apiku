from os import getenv
from agno.playground import Playground
from agents.example import get_example_agent
from agents.agen_polisi import get_police_agent
from agents.web_search import get_web_search_agent
from agents.hoax import fact_checker_agent
from agents.image import get_geo_agent
from agents.research import get_research_agent
from workflows.modus_operandi import get_analisator_tren_kejahatan

######################################################
## Router for the agent playground
######################################################

example_agent = get_example_agent(debug_mode=True)
agen_polisi = get_police_agent(debug_mode=True)
web_search_agent = get_web_search_agent(debug_mode=True)
geo_agent = get_geo_agent(debug_mode=True)
penyidik_polri = get_research_agent(debug_mode=True)
analisator_kejahatan = get_analisator_tren_kejahatan(debug_mode=True)

# Create a playground instance
playground = Playground(
    agents=[
        example_agent,
        agen_polisi,
        web_search_agent,
        fact_checker_agent,
        geo_agent,
        penyidik_polri
    ],
    workflows=[analisator_kejahatan],
)

# Log the playground endpoint with phidata.app
if getenv("RUNTIME_ENV") == "dev":
    playground.create_endpoint("http://localhost:8000")

playground_router = playground.get_router()
