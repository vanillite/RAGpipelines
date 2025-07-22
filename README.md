# Automatic RAG system evaluation pipeline

This repository consists of RAG pipelines that support holistic testing and evaluation of various RAG optimization techniques, iterating over every possible combination of techniques specified. This main system consists of these three pipelines:

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
- Agentic retrieval with query transformation
- Reranking

Embedding models, search algorithms, and LLMs provided by Azure can be seamlessly interchanged by specifying their names in the corresponding configuration variables.

Custom models are accessed via online endpoints and require registration and deployment to be used.

Agentic retrieval and reranking methods can be enabled or disabled. Using alternative agentic retrieval or reranking techniques requires manual implementation. 

# Pipelines

## 1. `RAG_generation_pipeline.ipynb`

### Description
This notebook handles the creation and configuration of Azure AI Search components. Specifically, it:
- Connects to Azure AI Search and links the specified database.
- Iterates over various techniques to create the vector database:
  - Embedding models (e.g., `text-embedding-3-large`, `custom-embedding-model`)
  - Search algorithms (e.g., `HNSW`, `ExhaustiveKNN`)
- Constructs and uploads the following for each combination of techniques:
  - Search index (vector database)
  - Indexer configurations
  - Skillsets
 
### Output
It produces multiple AI Search indexes that are ready to use through retrieval.

## 2. `RAG_inference_pipeline.ipynb`

### Description
This notebook handles the retrieval and inference process. It:
- Contains the prompts and strucutered output schemas for LLMs
- Routes user queries to the correct workflow via agentic retrieval
- Embeds user queries and connects to the Azure AI Search indexes to use the queries for document retrieval
- Reranks the top N retrieved documents
- Forwards the top K reranked documents for final LLM response generation
- Stores the results in a dataframe with pickle checkpoints for evaluation
- Iterates this process over all possible combination of techniques:
  - LLMs (e.g., `gpt-4.1`, `gpt-4o`)
  - Query transformation (e.g., query rewriting, query decomposition, query expansion)
  - Agentic retrieval (`True` or `False`)
  - Reranking (e.g., `reciprocal rank fusion`, `bge-reranker-v2-m3`)

### Output
It produces a `full_dataframe_pickle` containing:
- Original query
- Ground truth answer
- Generated response
- Run time
- Additional metadata
