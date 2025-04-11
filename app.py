
import logging
import sys

from llama_index.core import (Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate, Document)
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from qdrant_client import QdrantClient
from llama_index.core import load_index_from_storage
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.indices.postprocessor import LLMRerank 
import logging
import sys

from config import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


global query_engine
query_engine = None

global index
index = None

global vector_store
vector_store = None

def init_llm():
    # llm = Ollama
    # embed_model = OpenAIEmbedding(model_name="text-embedding-3-large")
    llm = Ollama(base_url=f"{OLLAMA_URL}", model=f"{LLM_MODEL}", temperature=0.8, request_timeout=300,
             system_prompt="You are an agent who consider the context passed "
                           "in, to answer any questions dont consider your prior "
                           "knowledge to answer and if you dont find the answer "
                           "please respond that you dont know.")
    embed_model = OllamaEmbedding(base_url=f"{OLLAMA_URL}", model_name=f"{EMBED_MODEL}")
    Settings.llm = llm
    Settings.embed_model = embed_model


def init_index():
    global index
    global vector_store
    # create qdrant client
    qdrant_client = QdrantClient(
        url=f"{QDRANT_HOST}", 
        api_key=f"{QDRANT_API_KEY}",
    )
    
    # qdrant vector store with enabling hybrid search
    vector_store = QdrantVectorStore(
        collection_name=f"{QDRANT_COLLECTION_NAME}",
        client=qdrant_client,
        enable_hybrid=True,
        batch_size=20
    )

    upload_files = len(os.listdir(f"{UPLOAD_FILE_PATH}"))
    if upload_files > 1:
        index_storage_files = len(os.listdir(f"{INDEX_STORAGE_PATH}"))
        if index_storage_files > 1:
            new_storage_context = StorageContext.from_defaults( vector_store=vector_store, persist_dir=f"{INDEX_STORAGE_PATH}")
            index = load_index_from_storage(new_storage_context)
        else:
            load_index()  

def load_index():
    global vector_store
    reader = SimpleDirectoryReader(input_dir=f"{UPLOAD_FILE_PATH}", recursive=True)
    documents = reader.load_data()

    logging.info("index creating with `%d` documents", len(documents))

    # create large document with documents for better text balancing
    document = Document(text="\n\n".join([doc.text for doc in documents]))

    # sentece window node parser
    # window_size = 3, the resulting window will be three sentences long
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
  
    # storage context and service context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # initialize vector store index with qdrant
    index = VectorStoreIndex.from_documents(
        [document],
        #service_context=service_context,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
        node_parser=node_parser,
    )
    index.storage_context.persist(persist_dir=f"{INDEX_STORAGE_PATH}")
    
def init_query_engine():
    global query_engine
    global index

    # after retrieval, we need to replace the sentence with the entire window from the metadata by defining a
    # MetadataReplacementPostProcessor and using it in the list of node_postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

    # re-ranker with BAAI/bge-reranker-base model
    rerank = SentenceTransformerRerank(
       top_n=2,
       model="BAAI/bge-reranker-base"
    )
    #reranker = LLMRerank(choice_batch_size=5, top_n=5) 
    #reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)
    new_storage_context = StorageContext.from_defaults( vector_store=vector_store, persist_dir=f"{INDEX_STORAGE_PATH}")

    index = load_index_from_storage(new_storage_context)
    # similarity_top_k configure the retriever to return the top 3 most similar documents, the default value of similarity_top_k is 2
    # use meta data post processor and re-ranker as post processors
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        node_postprocessors=[postproc, rerank],
    )


def chat(input_question, user):
    global query_engine

    response = query_engine.query(input_question)
    logging.info("response from llm - %s", response)

    # view sentece window retrieval window and origianl text
    logging.info("sentence window retrieval window - %s", response.source_nodes[0].node)
    logging.info("sentence window retrieval orginal_text - %s", response.source_nodes)

    return response.response