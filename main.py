import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sqlalchemy import create_engine, text

from crewai import Agent, Crew, Process, Task, BaseLLM
from crewai.tools import BaseTool

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = BASE_DIR / "knowledge" / "handbook.txt"

OLLAMA_OPENAI_BASE_URL = os.getenv(
    "OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1"
)
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gpt-3.5-turbo")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "company_docs")

DB_URL = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER', 'appuser')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'apppassword')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'appdb')}"
)

openai_client = OpenAI(
    base_url=OLLAMA_OPENAI_BASE_URL,
    api_key="ollama",
)

qdrant = QdrantClient(url=QDRANT_URL)
engine = create_engine(DB_URL, pool_pre_ping=True)


class OllamaOpenAICompatLLM(BaseLLM):
    def __init__(self, model: str, endpoint: str, temperature: float = 0):
        super().__init__(model=model, temperature=temperature)
        self.endpoint = endpoint.rstrip("/")

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[str, Any]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        response = requests.post(
            f"{self.endpoint}/chat/completions",
            headers={
                "Authorization": "Bearer ollama",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        try:
            content = result["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(
                f"Réponse LLM invalide: {json.dumps(result, ensure_ascii=False)}"
            ) from e

        if content is None:
            raise RuntimeError(
                f"Réponse LLM invalide: {json.dumps(result, ensure_ascii=False)}"
            )

        content = str(content).strip()

        if not content:
            return "Aucune réponse générée."

        return content

    def supports_function_calling(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192


llm = OllamaOpenAICompatLLM(
    model=OLLAMA_LLM_MODEL,
    endpoint=OLLAMA_OPENAI_BASE_URL,
    temperature=0,
)


def healthcheck_llm() -> None:
    result = llm.call("Réponds uniquement par OK")
    if not result or not result.strip():
        raise RuntimeError("Healthcheck LLM vide.")
    print(f"[LLM CHECK] {result}")


def healthcheck_db() -> None:
    try:
        with engine.connect() as conn:
            ping = conn.execute(text("SELECT 1")).scalar()
            count = conn.execute(text("SELECT COUNT(*) FROM employees")).scalar()
            sample = (
                conn.execute(
                    text(
                        "SELECT id, name, department, role, location FROM employees ORDER BY id LIMIT 3"
                    )
                )
                .mappings()
                .all()
            )

        print(f"[DB CHECK] Connexion OK | SELECT 1 = {ping}")
        print(f"[DB CHECK] employees count = {count}")
        print("[DB CHECK] sample rows:")
        for row in sample:
            print(dict(row))
    except Exception as e:
        raise RuntimeError(f"Erreur connexion ou lecture PostgreSQL: {e}") from e


def ollama_embed(text_value: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=OLLAMA_EMBED_MODEL,
        input=text_value,
    )
    embedding = response.data[0].embedding
    if not embedding:
        raise RuntimeError("Embedding vide retourné par Ollama.")
    return embedding


def chunk_text(text_value: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    chunks = []
    start = 0
    while start < len(text_value):
        end = min(len(text_value), start + chunk_size)
        chunks.append(text_value[start:end])
        if end == len(text_value):
            break
        start = end - overlap
    return chunks


def ensure_vector_store() -> None:
    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(f"Knowledge file introuvable: {KNOWLEDGE_FILE}")

    text_content = KNOWLEDGE_FILE.read_text(encoding="utf-8")
    chunks = chunk_text(text_content)

    if not chunks:
        raise ValueError("Le fichier handbook.txt est vide.")

    vectors = [ollama_embed(chunk) for chunk in chunks]

    collections = [c.name for c in qdrant.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=len(vectors[0]),
                distance=models.Distance.COSINE,
            ),
        )

    qdrant.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            models.PointStruct(
                id=i + 1,
                vector=vectors[i],
                payload={"text": chunk},
            )
            for i, chunk in enumerate(chunks)
        ],
    )


def rag_lookup(query: str, limit: int = 3) -> str:
    query_vector = ollama_embed(query)

    try:
        results = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
        )
        points = results.points if hasattr(results, "points") else results
    except Exception:
        points = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=limit,
        )

    if not points:
        return "Aucun résultat RAG trouvé."

    lines = []
    for i, p in enumerate(points, start=1):
        payload = getattr(p, "payload", {}) or {}
        score = getattr(p, "score", 0.0)
        lines.append(f"[{i}] score={score:.4f} | extrait={payload.get('text', '')}")

    return "\n".join(lines)


def normalize_question(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("question", "query", "input", "text", "description"):
            if key in value and isinstance(value[key], str):
                return value[key].strip()
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value).strip()


def sql_lookup(question: Any) -> str:
    q_raw = normalize_question(question)
    q = q_raw.lower()

    if not q:
        q = "liste les employés"

    if "infra" in q and ("salaire" in q or "salary" in q):
        sql = """
        SELECT name, role, salary, location
        FROM employees
        WHERE department = 'Infra'
        ORDER BY salary DESC;
        """
    elif "infra" in q:
        sql = """
        SELECT name, department, role, location
        FROM employees
        WHERE department = 'Infra'
        ORDER BY name
        LIMIT 10;
        """
    elif "reims" in q:
        sql = """
        SELECT name, department, role
        FROM employees
        WHERE location ILIKE 'Reims'
        LIMIT 10;
        """
    elif "sql" in q and ("compétence" in q or "competence" in q or "skill" in q):
        sql = """
        SELECT name, department, skills
        FROM employees
        WHERE 'sql' = ANY(skills)
        LIMIT 10;
        """
    elif "nom" in q or "agent" in q or "employé" in q or "employee" in q:
        sql = """
        SELECT id, name, department, role, location
        FROM employees
        ORDER BY id
        LIMIT 10;
        """
    else:
        sql = """
        SELECT id, name, department, role, location
        FROM employees
        ORDER BY id
        LIMIT 10;
        """

    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).mappings().all()
    except Exception as e:
        return f"Erreur SQL lors de l'exécution de la requête:\n{sql}\nErreur: {str(e)}"

    if not rows:
        return (
            f"Question interprétée: {q_raw}\nRequête exécutée:\n{sql}\nAucun résultat."
        )

    rendered = "\n".join(str(dict(r)) for r in rows)
    return f"Question interprétée: {q_raw}\nRequête exécutée:\n{sql}\nRésultats:\n{rendered}"


class RagSearchInput(BaseModel):
    query: str = Field(
        ..., description="Question à rechercher dans la base vectorielle"
    )
    limit: int = Field(3, description="Nombre maximum de résultats")


class RagSearchTool(BaseTool):
    name: str = "rag_search_tool"
    description: str = "Recherche sémantique dans Qdrant avec embeddings Ollama."
    args_schema: type[BaseModel] = RagSearchInput

    def _run(self, query: str, limit: int = 3) -> str:
        try:
            return rag_lookup(query, limit=limit)
        except Exception as e:
            return f"Erreur RAG: {str(e)}"


class SqlSearchInput(BaseModel):
    question: str = Field(..., description="Question reformulée pour la base SQL")


class SqlQueryTool(BaseTool):
    name: str = "sql_query_tool"
    description: str = (
        "Exécute des requêtes SELECT sécurisées sur PostgreSQL. "
        "Attends une question claire et reformulée pour interroger la table employees."
    )
    args_schema: type[BaseModel] = SqlSearchInput

    def _run(self, question: Any = None, **kwargs) -> str:
        try:
            effective_question = question
            if effective_question is None:
                effective_question = kwargs.get("question", kwargs.get("query", kwargs))
            return sql_lookup(effective_question)
        except Exception as e:
            return f"Erreur dans sql_query_tool: {str(e)} | input={question} | kwargs={kwargs}"


def build_crew() -> Crew:
    rewriter = Agent(
        role="Question Rewriter",
        goal="Reformuler la question utilisateur pour qu'elle soit comprise par l'agent SQL.",
        backstory="Spécialiste de la reformulation orientée base de données relationnelle.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    rag_agent = Agent(
        role="RAG Specialist",
        goal="Répondre aux questions à partir de la base vectorielle Qdrant.",
        backstory="Expert RAG utilisant embeddings Ollama et Qdrant pour retrouver des extraits du handbook.",
        tools=[RagSearchTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    sql_agent = Agent(
        role="SQL Analyst",
        goal="Répondre aux questions à partir de la base PostgreSQL.",
        backstory="Analyste SQL spécialisé en lecture sécurisée de la table employees.",
        tools=[SqlQueryTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    coordinator = Agent(
        role="Coordinator",
        goal="Synthétiser les réponses des agents RAG et SQL.",
        backstory="Coordinateur chargé d'assembler les résultats spécialisés.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    rewrite_task = Task(
        description=(
            "Reformule la question utilisateur '{question}' en une question simple, claire et exploitable "
            "par un agent SQL qui interroge la table employees. "
            "Réponds uniquement avec la question reformulée, sans explication."
        ),
        expected_output="Une question reformulée pour SQL.",
        agent=rewriter,
    )

    rag_task = Task(
        description=(
            "Utilise uniquement rag_search_tool pour répondre à la question originale '{question}'. "
            "Retourne les extraits les plus pertinents et une courte interprétation."
        ),
        expected_output="Résultats provenant de la base vectorielle.",
        agent=rag_agent,
    )

    sql_task = Task(
        description=(
            "Utilise uniquement sql_query_tool. "
            "Base-toi d'abord sur la reformulation produite dans le contexte précédent pour répondre. "
            "Si la reformulation parle d'infra, de localisation, de rôle ou d'employés, "
            "interroge la table employees avec une requête adaptée."
        ),
        expected_output="Résultats provenant de PostgreSQL à partir de la question reformulée.",
        agent=sql_agent,
        context=[rewrite_task],
    )

    final_task = Task(
        description=(
            "À partir des résultats du RAG et du SQL, réponds à la question utilisateur '{question}' "
            "de façon courte, claire et structurée."
        ),
        expected_output="Réponse finale consolidée.",
        agent=coordinator,
        context=[rewrite_task, rag_task, sql_task],
    )

    return Crew(
        agents=[rewriter, rag_agent, sql_agent, coordinator],
        tasks=[rewrite_task, rag_task, sql_task, final_task],
        process=Process.sequential,
        verbose=True,
    )


def main():
    healthcheck_llm()
    healthcheck_db()
    ensure_vector_store()

    crew = build_crew()
    question = os.getenv(
        "DEMO_QUESTION",
        "donne moi le nom de 1 agent dans l'infra? en te servant de l'outil sql",
    )

    result = crew.kickoff(inputs={"question": question})
    print(result)


if __name__ == "__main__":
    main()
