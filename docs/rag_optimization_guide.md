# RAG Optimization & Architecture Guide

This guide explains the underlying models, architecture, and improvements implemented in the **MSA Compliance Automation** pipeline to achieve high F1 and Exact Match (EM) scores.

---

## 1. Core Language & Embedding Models

The RAG pipeline utilizes two primary models designed to run locally, ensuring zero-latency external network dependencies and complete data privacy:

### A. Local Inference Model: Qwen 2.5 7B Instruct (4-bit Quantized)
*   **Repository / Type**: `mlx-community/Qwen2.5-7B-Instruct-4bit`
*   **Architecture**: Transformer-based decoder-only model with Grouped-Query Attention (GQA) and SwiGLU activations.
*   **Quantization**: 4-bit Group Quantization, reducing the 14GB 16-bit weight footprint down to ~4.3GB. This allows the model to fit comfortably within the VRAM boundaries of Apple Silicon Unified Memory architectures (M1/M2/M3/M4 series Macs).
*   **Context Window**: Supports up to 128K context length (though we restrict retrieval context to maintain prompt focus and prevent performance degradation).
*   **Role in Pipeline**: Evaluates compliance rules against retrieved contract text segments and outputs structured compliance audits in JSON format.

### B. Dense Embeddings Model: all-MiniLM-L6-v2
*   **Library**: `SentenceTransformers` / Hugging Face.
*   **Dimension**: 384-dimensional dense vectors.
*   **Context Limit**: 256 input tokens.
*   **Role in Pipeline**: Generates semantic embeddings for each contract text segment, allowing high-performance cosine similarity queries within the local ChromaDB database.

---

## 2. RAG Pipeline Architecture

The system operates as an end-to-end Retrieval-Augmented Generation (RAG) framework:

```
+------------------+     +------------------+     +--------------------+
|  Input Contract  | --> | ContractChunker  | --> | Parent-Child Nodes |
|      (.txt)      |     | (Custom Segment) |     | (Child / Parent)   |
+------------------+     +------------------+     +--------------------+
                                                            |
                                                            v
+------------------+     +------------------+     +--------------------+
|   ChromaDB       | <-- |  MiniLM Embed    | <-- | Index child vector |
|   Vector DB      |     |  (all-MiniLM)    |     | (Keep parent text) |
+------------------+     +------------------+     +--------------------+
         |
         v
+------------------+     +------------------+     +--------------------+
| Query ChromaDB   | --> | Retrieve matches | --> | Map to Parent text |
| for Rule Category|     |   (Top Child)    |     | (Get full segment) |
+------------------+     +------------------+     +--------------------+
                                                            |
                                                            v
+------------------+     +------------------+     +--------------------+
| Qwen 2.5 LLM     | <-- | Construct ChatML | <-- | Compile Context &  |
|  (Inference)     |     |   Prompt (Sys)   |     | Verbatim Rule      |
+------------------+     +------------------+     +--------------------+
         |
         v
+------------------+     +------------------+
| Output JSON Card | --> | Vector Cleanup   |
| (F1 / EM Score)  |     | (Delete indices) |
+------------------+     +------------------+
```

---

## 3. High F1 & Exact Match (EM) Optimization Strategy

Achieving high F1 and EM metrics in legal document extraction requires balancing precise semantic search with complete generation contexts.

### A. Parent-Child Chunking Strategy (Fully Implemented)
*   **The Problem**:
    *   *Large chunks* (e.g., 2000 characters) capture complete clauses but muddy the vector space, leading to less precise similarity matches.
    *   *Small chunks* (e.g., 400 characters) generate high-precision vector matches but cut legal sentences in half, causing the LLM to generate incomplete clauses (lowering F1 and EM).
*   **The Parent-Child Solution**:
    1.  **Chunking**: Slice the text into small **Child Chunks** (~400 characters) and large overlapping **Parent Chunks** (~2000 characters). This is implemented in `ContractChunker.split_text_parent_child()` in [chunker.py](file:///Users/siputroj/Desktop/react/msa-compliance-automation/backend/src/chunker.py).
    2.  **Indexing**: Embed and insert the *child* chunks into ChromaDB. Attach the corresponding *parent* chunk text, start character index, and end character index as metadata properties of the child vector. This is implemented in `ContractVectorStore.add_contract_chunks()` in [vector_store.py](file:///Users/siputroj/Desktop/react/msa-compliance-automation/backend/src/vector_store.py).
    3.  **Retrieval**: When a query matches a child chunk, fetch its metadata and return the complete **parent chunk** text to the LLM. This is implemented in `ContractVectorStore.search_relevant_chunks()` in [vector_store.py](file:///Users/siputroj/Desktop/react/msa-compliance-automation/backend/src/vector_store.py).
*   **Why it increases F1/EM**: The retriever matching remains highly granular (matching specific rules to small child paragraphs), while the LLM receives the full paragraph context necessary to extract clean, whole clauses.

### B. Strict Verbatim Prompt Constraints (Fully Implemented)
*   **The Problem**: Default LLM generation models tend to summarize, rephrase, or add preamble formatting (e.g., *"Here is the extracted clause..."*). Any variation in word boundaries immediately degrades F1 and EM.
*   **The Solution**: We inject explicit instructions in the System template inside [compliance_engine.py](file:///Users/siputroj/Desktop/react/msa-compliance-automation/backend/src/compliance_engine.py):
    1.  Instruct the model to copy matching clauses **word-for-word, verbatim** from the context.
    2.  Strictly prohibit summarizing, paraphrasing, or introducing text.
    3.  Treat absence of context as an empty string `""` response.

### C. Benchmarking & Parameter Sweeps
The pipeline is verified using a local evaluation script in [evaluate_rag_performance.py](file:///Users/siputroj/Desktop/react/msa-compliance-automation/backend/evaluate_rag_performance.py). It runs a hyperparameter sweep over different configuration settings:
*   **Configuration 1**: Chunk Size 1000, Overlap 100, Top K 3
*   **Configuration 2**: Chunk Size 2000, Overlap 200, Top K 3
*   **Configuration 3**: Chunk Size 2000, Overlap 200, Top K 5

To run the pipeline benchmark on sample contracts:
```bash
# In backend/ directory with active venv:
# Runs with dry-run heuristic models
python evaluate_rag_performance.py --samples 3

# Runs with actual local GPU Qwen-Instruct model
python evaluate_rag_performance.py --samples 3 --mlx
```

### D. Advanced Future Roadmap (Lexical Hybrid Search + Re-ranking)
*   **Lexical BM25 Hybrid Retrieval**:
    *   Dense embeddings are strong for semantic concepts but weak at keyword lookups. 
    *   Integrating BM25 keyword matching ensures that queries containing exact terms (such as specific jurisdictions like `"Delaware"` or notice durations like `"30 days"`) score higher.
*   **Cross-Encoder Re-ranking**:
    *   A bi-encoder retrieves the top 10 candidates. A local cross-encoder re-ranks them based on exact query-candidate interaction.
    *   This selects the absolute best context blocks, preventing the performance drops associated with large, noisy contexts.
