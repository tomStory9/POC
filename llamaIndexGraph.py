from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from IPython.display import Markdown, display

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

documents = SimpleDirectoryReader("./data").load_data()

# define LLM
# NOTE: at the time of demo, text-davinci-002 did not have rate-limit errors

llm = Ollama(model="llama3.2", request_timeout=120.0)
Settings.llm = llm
Settings.chunk_size = 512
# pip install
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
from llama_index.core import StorageContext

graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# NOTE: can take a while!
index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    storage_context=storage_context,
)

query_engine = index.as_query_engine(include_text=False, response_mode="tree_summarize")
response = query_engine.query(
    "Tell me more about the holidays",
)


## create graph
from pyvis.network import Network

g = index.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
html = net.generate_html()
with open("example.html", "w", encoding="utf-8") as f:
    f.write(html)
