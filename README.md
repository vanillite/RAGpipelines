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

Create a config.json file to store the Azure endpoints, access keys, and connection details.

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
The notebook produces multiple AI Search indexes that are ready to use through retrieval.

## 2. `RAG_inference_pipeline.ipynb`

### Description
This notebook handles the retrieval and inference process. It:
- Contains the prompts and strucutered output schemas for LLMs.
- Routes user queries to the correct workflow via agentic retrieval.
- Embeds user queries and connects to the Azure AI Search indexes to use the queries for document retrieval.
- Reranks the top N retrieved documents.
- Forwards the top K reranked documents for final LLM response generation.
- Stores the results in a dataframe with pickle checkpoints for evaluation.
- Iterates this process over all possible combination of techniques:
  - LLMs (e.g., `gpt-4.1`, `gpt-4o`)
  - Query transformation (e.g., query rewriting, query decomposition, query expansion)
  - Agentic retrieval (`True` or `False`)
  - Reranking (e.g., `reciprocal rank fusion`, `bge-reranker-v2-m3`)

### Output
The notebook produces a dataframe containing:
- Original query
- Ground truth answer
- Generated response
- Run time
- Additional metadata

### Optional usage
For cases where the objective is to utilize the RAG system without performing evaluations, the `retrieve` function can be invoked directly, and subsequent cells in the notebook can be skipped. 

## 3. `RAG_evaluation_pipeline.ipynb`

### Description
This notebook evaluates the inference results by comparing them against ground truths. It:
- Loads:
  - Inference results dataframe.
  - Document containing user query and ground truths.
- Merges the two datasets based on queries.
- Calculates:
  - **ROUGE-1** and **ROUGE-L** scores.
  - **RAGAs** metrics (faithfulness, context recall, and cobtext precision).
- Groups results by by various methods to output averaged scores.

### Output
The notebook generates multiple dataframes for the evaluation metrics and various groupby methods. Results can optionally be used for additional significance testing (`statistical_tests.py`).

## Miscellaneous

The resulting dataframe from the evaluation pipeline can be optionally used for additional significance testing using `statistical_tests.py`. This file performs t-tests and Tukey's HSD tests to compare the mean score of different groups.

The `score.py` script is used to deploy a custom embedding model in Azure Machine Learning Studio. It ensures that data is correctly formatted when sent to and received from AI Search.
