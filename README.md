# Automatic RAG system evaluation pipeline

This repository consists of three RAG pipelines that support holistic testing and evaluation of various RAG optimization techniques, iterating over every possible combination of techniques specified. This main system consists of these three pipelines:

1. **Generation** - Builds vector indexes/databases in Azure AI Search based on your selection of embedding models and search algorithms.
2. **Inference** - Orchestrates the in-retrieval and post-retrieval workflows to enable user inference on the RAG system, based on your selection of LLMs, agentic retrieval, and reranking methods.
3. **Evaluation** - Evaluates the output of the inference pipeline using two metrics: ROUGE and RAGAs. 

## Install dependencies


```bash
pip install -r requirements.txt
```

## Azure configuration

Create a config.json file with the Azure endpoints, keys, and connections.

## RAG techniques configuration 
This system supports the interchangability and testing of the following techniques:
- Embedding model
- Search algorithm
- LLM
- Agentic retrieval
- Reranking

Embedding models, search algorithms, and LLMs provided by Azure can be seamlessly interchanged by specifying their names in the corresponding configuration variables.

Custom models are accessed via online endpoints and require registration and deployment to be used.

Agentic retrieval and reranking methods can be enabled or disabled. Using alternative agentic retrieval or reranking techniques requires manual implementation. 

# Pipelines

## 1. Generation pipeline - RAG_generation_pipeline.ipynb
This notebook handles the creation and configuration of Azure AI Search components. Specifically, it:
- Connects to Azure AI Search and links the specified database.
- Iterates over various combinations of:
  - Embedding models (e.g., `text-embedding-3-large`, `custom-embedding-model`)
  - Search algorithms (e.g., `HNSW`, `ExhaustiveKNN`)
- Constructs and uploads the following for each combination of techniques:
  - Search index (vector database)
  - Indexer configurations
  - Skillsets
 
### Output
It produces multiple AI Search indexes that are ready to use through retrieval.
