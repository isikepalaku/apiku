from os import getenv
from phi.playground import Playground
from agents.example import get_example_agent
from agents.agen_polisi import get_police_agent
from agents.web_search import get_web_search_agent
from agents.hoax import fact_checker_agent

######################################################
## Router for the agent playground
######################################################

example_agent = get_example_agent(debug_mode=True)
agen_polisi = get_police_agent(debug_mode=True)
web_search_agent = get_web_search_agent(debug_mode=True)

# Create a playground instance
playground = Playground(agents=[example_agent, agen_polisi, web_search_agent, fact_checker_agent])

# Log the playground endpoint with phidata.app
if getenv("RUNTIME_ENV") == "dev":
    playground.create_endpoint("http://localhost:8000")

playground_router = playground.get_router()
