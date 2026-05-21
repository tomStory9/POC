import os

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    KeywordTableIndex,
    get_response_synthesizer,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore

DATA_DIR = "data"
STORAGE_DIR = "storage"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)


LLM = Ollama(model="llama3.2", request_timeout=120.0)

EMBED = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2",
    trust_remote_code=True,
)

Settings.llm = LLM
Settings.embed_model = EMBED


reader = SimpleDirectoryReader(
    DATA_DIR,
    recursive=True,
    required_exts=[".pdf", ".txt", ".md"],
)
documents = reader.load_data()

print(f"Loaded {len(documents)} docs.")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        EMBED,
    ],
)

nodes = pipeline.run(documents=documents)

print(f"Split into {len(nodes)} chunks.")


vector_store = LanceDBVectorStore(
    uri=os.path.join(STORAGE_DIR, "lancedb"),
    mode="overwrite",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
keyword_index = KeywordTableIndex(nodes)


vector_index.storage_context.persist(STORAGE_DIR)


class HybridRetriever(BaseRetriever):
    def __init__(self, vector_index, keyword_index, vector_top_k=5, keyword_top_k=5):
        self._vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=vector_top_k,
        )
        self._keyword_retriever = KeywordTableSimpleRetriever(
            index=keyword_index,
            similarity_top_k=keyword_top_k,
        )
        super().__init__()

    def _retrieve(self, query: str) -> list[NodeWithScore]:
        vector_nodes = self._vector_retriever.retrieve(query)
        keyword_nodes = self._keyword_retriever.retrieve(query)

        # Union + dédoublonnage
        combined = {}
        for node in vector_nodes + keyword_nodes:
            combined[node.node.node_id] = node

        return list(combined.values())


hybrid_retriever = HybridRetriever(vector_index, keyword_index)


text_qa_template = PromptTemplate("""
Tu es un assistant sur la documentation interne de l'entreprise.
Voici le contexte extrait :
{context_str}
---
Question : {query_str}
Réponse détaillée en français, claire et structurée :
""")

response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT,
    text_qa_template=text_qa_template,
)

query_engine = RetrieverQueryEngine(
    retriever=hybrid_retriever,
    response_synthesizer=response_synthesizer,
)

chat_engine = vector_index.as_chat_engine(
    chat_mode="condense_question",
    verbose=True,
)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("LlamaIndex POC local – exemple d’usage")
    print("=" * 50)

    # 1. QnA simple RAG via HybridRetriever
    print("\n[1] QnA RAG hybride (vector + keyword)")
    response = query_engine.query("Qu’est‑ce que contient le fichier conges.txt ?")
    print("Résultat :\n", response.response)

    # 2. ChatEngine (mode condense_question) basé sur l’index vectoriel
    print("\n[2] ChatEngine (mode condense_question)")
    query = "Quelles sont les principales informations du fichier politique.txt ?"
    response = chat_engine.chat(query)
    print("ChatEngine :\n", response.response)
