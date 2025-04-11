import os

# http api port
HTTP_PORT = os.getenv('HTTP_PORT', 7654)

#ollama
OLLAMA_URL= os.getenv('OLLAMA_URL','http://localhost:11434')
LLM_MODEL = os.getenv('LLM_MODEL','llama3.2')
EMBED_MODEL = os.getenv('EMBED_MODEL','mxbai-embed-large:latest')

# qdrant vector store config
QDRANT_HOST = os.getenv('QDRANT_URL','https://2ed75e54-a92a-4c25-9ba0-4cb1c04f22ee.us-west-2-0.aws.cloud.qdrant.io:6333')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY','eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ5Mzk3NjcwfQ.iVZvbpWEl4hlaKS2AlcZS1lcvA8qwK1NK4vQA8zV4Uw')
QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME','advrag3')
UPLOAD_FILE_PATH = os.getenv('UPLOAD_FILE_PATH','/Users/srikanth/Documents/work/file_storage')
INDEX_STORAGE_PATH = os.getenv('INDEX_STORAGE_PATH','/Users/srikanth/Documents/work/index_storage')