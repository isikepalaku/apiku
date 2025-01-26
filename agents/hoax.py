import os
from phi.agent import Agent
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
from phi.model.mistral import MistralChat
from phi.tools.googlesearch import GoogleSearch
from phi.tools.newspaper4k import Newspaper4k

# Agen Fakta Checker
fact_checker_agent = Agent(
    name="Hoax Checker Agent",
    agent_id="hoax-checker-agent",
    model=MistralChat(
        id="mistral-large-latest",
        api_key=os.environ["MISTRAL_API_KEY"]
    ),
    tools=[GoogleSearch(fixed_language="id"), Newspaper4k()],        # Tools yang digunakan
    description=(
        "Anda adalah agen pengecek fakta. Tugas Anda adalah memverifikasi "
        "apakah klaim/berita yang diberikan tergolong hoaks atau belum ada "
        "bukti yang cukup (belum dapat diverifikasi). Anda akan menggunakan "
        "pencarian web dan analisis artikel untuk menyimpulkan keabsahan klaim."
    ),
    instructions=[
        "1. Terima sebuah klaim atau topik yang ingin diverifikasi.",
        "2. Lakukan pencarian di Google search untuk menemukan minimal 5 tautan relevan.",
        "3. Baca tiap tautan menggunakan Newspaper4k dan rangkum informasi penting "
        "yang mendukung atau menyanggah klaim.",
        "4. Evaluasi bukti yang terkumpul. Jika Anda menemukan bukti jelas bahwa "
        "berita tersebut salah, simpulkan sebagai 'Hoax'. Jika ada bukti kuat "
        "mendukung kebenarannya, simpulkan 'Benar'. Jika bukti belum memadai "
        "untuk mengambil kesimpulan, simpulkan 'Belum dapat diverifikasi'.",
        "5. Berikan penjelasan singkat mengapa Anda mengambil kesimpulan tersebut."
    ],
    markdown=True,
    show_tool_calls=False,
    add_datetime_to_instructions=True,
    debug_mode=True,
)

# Contoh pemanggilan agen:
# response = fact_checker_agent.run("Masukkan klaim atau pernyataan di sini")
# print(response)
