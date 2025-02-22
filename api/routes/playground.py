from os import getenv
from agno.playground import Playground
from agents.agen_polisi import get_police_agent
from agents.agen_perkaba import get_perkaba_agent
from agents.agen_bantek import get_perkaba_bantek_agent
from agents.agen_emp import get_emp_agent
from agents.agen_wassidik import get_wassidik_agent
from agents.hoax import fact_checker_agent
from agents.image import get_geo_agent
from agents.research import get_research_agent
from agents.trend_kejahatan import get_crime_trend_agent
from agents.fidusia import get_fidusia_agent
from agents.sentiment_analyzer import get_sentiment_team
from workflows.modus_operandi import get_analisator_tren_kejahatan
from workflows.sentiment_analysis import get_sentiment_analyzer

######################################################
## Router for the agent playground
######################################################
trend_kejahatan = get_crime_trend_agent(debug_mode=True)
agen_emp = get_emp_agent(debug_mode=True)
agen_wassidik = get_wassidik_agent(debug_mode=True)
agen_perkaba = get_perkaba_agent(debug_mode=True)
agen_bantek = get_perkaba_bantek_agent(debug_mode=True)
agen_polisi = get_police_agent(debug_mode=True)
geo_agent = get_geo_agent(debug_mode=True)
penyidik_polri = get_research_agent(debug_mode=True)
analisator_kejahatan = get_analisator_tren_kejahatan(debug_mode=True)
fidusia_agent = get_fidusia_agent(debug_mode=True)
tim_analisis_sentimen = get_sentiment_team(debug_mode=True)
sentiment_analyzer = get_sentiment_analyzer(debug_mode=True)

# Create a playground instance
playground = Playground(
    agents=[
        agen_polisi,
        fact_checker_agent,
        geo_agent,
        penyidik_polri,
        trend_kejahatan,
        fidusia_agent,
        tim_analisis_sentimen,
        agen_perkaba,
        agen_bantek,
        agen_emp,
        agen_wassidik
    ],
    workflows=[
        analisator_kejahatan,
        sentiment_analyzer
    ],
)

# Log the playground endpoint with phidata.app
if getenv("RUNTIME_ENV") == "dev":
    playground.create_endpoint("http://localhost:8000")

playground_router = playground.get_router()
