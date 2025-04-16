from os import getenv
from agno.playground import Playground
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
from agents.kuhp_chat import get_kuhp_agent
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
agen_kuhp = get_kuhp_agent(debug_mode=True)
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
agen_siber = get_siber_agent(debug_mode=True)
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
        agen_kuhp,
        agen_ite,
        agen_cipta_kerja,
        agen_kesehatan,
        agen_indagsi,
        agen_dokpol,
        agen_forensic,
        agen_fismondev,
        agen_siber,
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

playground_router = playground.get_router()
