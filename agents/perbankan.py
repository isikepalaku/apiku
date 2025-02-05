"""🤔 DeepLegal - An AI Agent that iteratively searches a legal knowledge base to answer legal questions in the financial services sector

This agent performs iterative searches through its legal knowledge base, breaking down complex
queries into sub-questions, and synthesizing comprehensive answers. It's designed to explore
legal topics deeply and thoroughly by following chains of reasoning.

In this example, the agent uses the legal documentation provided in UUNomor4Tahun2023.txt

Key Features:
- Iteratively searches a legal knowledge base
- Source attribution and citations

Run `pip install openai lancedb tantivy inquirer agno` to install dependencies.
"""

from textwrap import dedent
from typing import List, Optional

import inquirer
import typer
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.url import UrlKnowledge
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.vectordb.pgvector import PgVector, SearchType
from db.session import db_url
from rich import print


def initialize_knowledge_base():
    """Initialize the legal knowledge base with the provided documentation.
    Here we use legal documentation as an example, but you can replace with any relevant legal sources
    """
    agent_knowledge = UrlKnowledge(
        urls=["https://celebesbot.com/pdf/UUNomor4Tahun2023.txt"],
        vector_db=PgVector(
            table_name="legal_knowledge",
            db_url=db_url,
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id="text-embedding-3-small"),
        ),
    )
    # Load the knowledge base (comment out after first run)
    agent_knowledge.load()
    return agent_knowledge


def get_agent_storage():
    """Return agent storage"""
    return SqliteAgentStorage(
        table_name="deep_knowledge_sessions", db_file="tmp/agents.db"
    )


def create_agent(session_id: Optional[str] = None) -> Agent:
    """Create and return a configured DeepLegal agent."""
    agent_knowledge = initialize_knowledge_base()
    agent_storage = get_agent_storage()
    return Agent(
        name="Agen Ahli Hukum Jasa Keuangan",
        session_id=session_id,
        model=OpenAIChat(id="gpt-4o"),
        description=dedent("""\
        Anda adalah Agen Ahli Hukum Jasa Keuangan, seorang pakar hukum yang mendalam dengan spesialisasi
        dalam sektor jasa keuangan. Anda memiliki pengetahuan yang luas mengenai peraturan perundang-undangan,
        kasus hukum, dan kebijakan yang mengatur industri keuangan. Keahlian Anda mencakup analisis peraturan,
        penafsiran kontrak, dan pemberian pendapat hukum yang akurat dan komprehensif.
        """),
        instructions=dedent("""\
        Misi Anda adalah memberikan jawaban yang mendalam dan tepat mengenai setiap pertanyaan hukum di sektor jasa keuangan.
        
        Langkah-langkah yang harus diikuti:
        1. **Analisis Pertanyaan:** Pahami dan uraikan pertanyaan yang diajukan.
        2. **Pencarian Awal:** Lakukan setidaknya 3 pencarian mendetail dalam basis pengetahuan hukum untuk mengumpulkan informasi relevan.
        3. **Evaluasi Informasi:** Jika jawaban dari basis pengetahuan belum lengkap atau ambigu, minta klarifikasi lebih lanjut.
        4. **Proses Iteratif:**
           - Teruskan pencarian dalam basis pengetahuan hingga semua aspek pertanyaan terjawab dengan tuntas.
           - Evaluasi kembali kelengkapan jawaban setelah setiap iterasi pencarian.
           - Ulangi proses pencarian hingga yakin tidak ada sudut yang terlewat.
        5. **Dokumentasi Alasan:** Catat proses pencarian dan sumber-sumber hukum yang digunakan.
        6. **Sintesis Akhir:** Hasilkan jawaban final yang komprehensif dengan referensi hukum yang tepat.
        7. **Peningkatan Berkelanjutan:** Jika ada informasi hukum baru yang muncul, perbarui jawaban Anda sesuai kebutuhan.
        """),
        additional_context=dedent("""\
        Anda harus menjawab dengan final dan lengkap berdasarkan:
        - Dokumen hukum: UUNomor4Tahun2023.txt
        - Mempertimbangkan peraturan perundang-undangan dan pedoman hukum terkini.
        - Riwayat interaksi dan pencarian sebelumnya.
        """),
        knowledge=agent_knowledge,
        storage=agent_storage,
        add_history_to_messages=True,
        num_history_responses=3,
        show_tool_calls=True,
        read_chat_history=True,
        markdown=True,
    )


def get_example_topics() -> List[str]:
    """Return a list of example topics for the agent."""
    return [
        "Apa saja tantangan hukum dalam sektor jasa keuangan?",
        "Bagaimana peraturan perundang-undangan mengatur kontrak perbankan?",
        "Apa saja risiko hukum yang harus diperhatikan dalam investasi keuangan?",
        "Bagaimana mekanisme penyelesaian sengketa dalam sektor keuangan?",
        "Apa peran regulator dalam menjaga integritas sistem keuangan?",
    ]


def handle_session_selection() -> Optional[str]:
    """Handle session selection and return the selected session ID."""
    agent_storage = get_agent_storage()

    new = typer.confirm("Apakah Anda ingin memulai sesi baru?", default=True)
    if new:
        return None

    existing_sessions: List[str] = agent_storage.get_all_session_ids()
    if not existing_sessions:
        print("Tidak ditemukan sesi yang ada. Memulai sesi baru.")
        return None

    print("\nSesi yang ada:")
    for i, session in enumerate(existing_sessions, 1):
        print(f"{i}. {session}")

    session_idx = typer.prompt(
        "Pilih nomor sesi untuk melanjutkan (atau tekan Enter untuk sesi terbaru)",
        default=1,
    )

    try:
        return existing_sessions[int(session_idx) - 1]
    except (ValueError, IndexError):
        return existing_sessions[0]


def run_interactive_loop(agent: Agent):
    """Run the interactive question-answering loop."""
    example_topics = get_example_topics()

    while True:
        choices = [f"{i + 1}. {topic}" for i, topic in enumerate(example_topics)]
        choices.extend(["Masukkan pertanyaan khusus...", "Keluar"])

        questions = [
            inquirer.List(
                "topic",
                message="Pilih topik atau ajukan pertanyaan:",
                choices=choices,
            )
        ]
        answer = inquirer.prompt(questions)

        if answer["topic"] == "Keluar":
            break

        if answer["topic"] == "Masukkan pertanyaan khusus...":
            questions = [inquirer.Text("custom", message="Masukkan pertanyaan:")]
            custom_answer = inquirer.prompt(questions)
            topic = custom_answer["custom"]
        else:
            topic = example_topics[int(answer["topic"].split(".")[0]) - 1]

        agent.print_response(topic, stream=True)


def deep_knowledge_agent():
    """Main function to run the DeepLegal agent."""

    session_id = handle_session_selection()
    agent = create_agent(session_id)

    print("\n🤔 Selamat datang di DeepLegal - Asisten Hukum Jasa Keuangan!")
    if session_id is None:
        session_id = agent.session_id
        if session_id is not None:
            print(f"[bold green]Memulai Sesi Baru: {session_id}[/bold green]\n")
        else:
            print("[bold green]Memulai Sesi Baru[/bold green]\n")
    else:
        print(f"[bold blue]Melanjutkan Sesi: {session_id}[/bold blue]\n")

    run_interactive_loop(agent)


def get_legal_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False
) -> Agent:
    """
    Return a configured instance of the Legal Financial Services Agent.
    
    Args:
        user_id: Optional user identifier
        session_id: Optional session identifier
        debug_mode: Enable debug mode if True
    
    Returns:
        Agent: Configured legal agent instance
    """
    return create_agent(session_id)

if __name__ == "__main__":
    typer.run(deep_knowledge_agent)

# Contoh prompt untuk dicoba:
"""
Jelajahi kemampuan DeepLegal dengan query berikut:
1. "Apa saja tantangan hukum dalam sektor jasa keuangan?"
2. "Bagaimana peraturan perundang-undangan mengatur kontrak dan pinjaman perbankan?"
3. "Apa risiko hukum dalam investasi dan asuransi?"
4. "Bagaimana mekanisme penyelesaian sengketa hukum perbankan?"
5. "Apa peran OJK dalam mengatur industri jasa keuangan?"
"""