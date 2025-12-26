# RAG Module for Knowledge Graph Triplets

This module provides Retrieval-Augmented Generation (RAG) capabilities for knowledge graph triplets extracted by the EDC pipeline.

## Overview

The RAG module indexes triplets from EDC pipeline output (`canon_kg.txt`) into a FAISS vector store for fast semantic retrieval, and can generate answers using multiple LLM providers.

### Supported LLM Providers

| Provider | Configuration | Notes |
|----------|--------------|-------|
| **Google AI Studio** | `source export_google_ai.sh` | Free tier available, uses Gemini models |
| **SambaNova** | `source export_sambanova.sh` | Free tier available, uses Llama models |
| **Local LLM** | `source export_local_llm.sh` | Requires GPU + bitsandbytes |

### Pipeline Steps

1. **Load Triplets** - Parse triplets from `canon_kg.txt`
2. **Represent** - Convert triplets to embeddable text
3. **Embed** - Generate embeddings using sentence-transformers
4. **Store** - Index in FAISS for similarity search
5. **Retrieve** - Find relevant triplets for a query
6. **Augment Prompt** - Build prompt with retrieved KG facts
7. **Generate** - Produce answer using configured LLM provider

## Quick Start

### 1. Run EDC Pipeline First

```bash
python run.py --input_text_file_path ./datasets/webnlg.txt
```

This creates `./output/tmp/iter0/canon_kg.txt`.

### 2. Index the Triplets

```bash
python index_rag.py --input ./webnlg_triplets_110_samples --output_dir ./webnlg_triplets_110_samples/rag
```

The indexer automatically finds the latest `canon_kg.txt` in the output directory.

### 3. Search (Retrieval Only)

```bash
# Single query - retrieve relevant triplets
python index_rag.py --load ./webnlg_triplets_110_samples/rag --query "Where is Trane located?"

# Interactive retrieval mode
python index_rag.py --load ./webnlg_triplets_110_samples/rag --interactive
```

### 4. Generate Answers (Full RAG with LLM)

Choose one of the following LLM providers:

```bash
# Option 1: Google AI Studio (recommended - free, no GPU required)
source export_google_ai.sh

# Option 2: SambaNova (free, no GPU required)
source export_sambanova.sh

# Option 3: Local LLM (requires GPU + bitsandbytes)
source export_local_llm.sh
```

Then run generation:

```bash
# Single query with LLM answer
python index_rag.py --load ./webnlg_triplets_110_samples/rag --generate --query "Where is Trane located?"

# Interactive Q&A with LLM
python index_rag.py --load ./webnlg_triplets_110_samples/rag --generate --interactive
```

## CLI Reference

### Indexing Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | - | Path to EDC output directory or `canon_kg.txt` |
| `--output_dir` | `./output/rag` | Directory to save index files |
| `--mode` | `triplet_text` | Representation mode (see below) |
| `--prefix` | `kg_triplets` | Filename prefix for saved files |
| `--batch_size` | `32` | Batch size for embedding |

### Model Options

| Option | Default | Description |
|--------|---------|-------------|
| `--embedding_model` | `BAAI/bge-small-en-v1.5` | Sentence transformer model |
| `--device` | auto | Device for embeddings (`cuda` or `cpu`) |
| `--gpu_faiss` | `False` | Use GPU for FAISS |

### Search Options

| Option | Default | Description |
|--------|---------|-------------|
| `--load` | - | Path to load existing index |
| `--query` | - | Query string |
| `--interactive` | `False` | Interactive mode |
| `--top_k` | `10` | Number of triplets to retrieve |

### Generation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--generate` | `False` | Enable LLM generation mode |
| `--temperature` | `0.1` | LLM temperature |
| `--max_tokens` | `256` | Maximum tokens to generate |

## Representation Modes

| Mode | Description | Example Output |
|------|-------------|----------------|
| `triplet_text` | Simple text format (default) | `"Trane location Swords Dublin"` |
| `entity_context` | All facts grouped by entity | `"Trane: location Swords Dublin, type Company"` |

## Programmatic Usage

### Search Only (Steps 1-5)

```python
from rag import KGRagIndexer

# Index triplets
indexer = KGRagIndexer()
indexer.index_from_path("./output/tmp", mode="triplet_text")
indexer.save("./output/rag")

# Load and search
indexer = KGRagIndexer.load("./output/rag")
results = indexer.search("Where is Trane located?", top_k=5)

for r in results:
    print(f"Score: {r.score:.4f}")
    print(f"  Triplet: ({r.metadata['subject']}, {r.metadata['predicate']}, {r.metadata['object']})")
```

### Full RAG with LLM (Steps 1-7)

First configure your LLM provider by sourcing one of the export scripts:
- `source export_google_ai.sh` - Google AI Studio
- `source export_sambanova.sh` - SambaNova  
- `source export_local_llm.sh` - Local LLM

```python
from rag import KGRagIndexer, KGRagGenerator

# Load index
indexer = KGRagIndexer.load("./output/rag")

# Create generator (uses provider from environment)
generator = KGRagGenerator(indexer)

# Generate answer
result = generator.generate("Where is Trane located?")
print(result.answer)    # "Trane is located in Swords, Dublin."
print(result.sources)   # [(Trane, location, Swords_Dublin)]
```

### Using Individual Components

```python
from rag import KGRetriever, KGPromptBuilder, KGRagIndexer

# Retrieval
indexer = KGRagIndexer.load("./output/rag")
retriever = KGRetriever(indexer)
context = retriever.retrieve("Where is Trane located?", top_k=5)
print(context.formatted_context)

# Prompt building
builder = KGPromptBuilder()
prompt = builder.build("Where is Trane located?", context)
print(prompt)
```

## Input Format

The module reads `canon_kg.txt` which contains one line per input text:

```
[]
[['Ciudad_Ayala', 'country', 'Morelos'], ['Ciudad_Ayala', 'contains settlement', 'Council-manager_government']]
[['Alan_B._Miller_Hall', 'country', 'Virginia'], ['Mason_School_of_Business', 'country', 'United_States']]
```

Each line is a Python list of triplets in `[subject, predicate, object]` format.

## Output Files

After indexing, the following files are created:

```
output/rag/
├── kg_triplets.faiss        # FAISS vector index
├── kg_triplets_meta.json    # Metadata for each triplet
└── kg_triplets_config.json  # Index configuration
```

## Module Structure

```
rag/
├── __init__.py              # Module exports
├── triplet_loader.py        # Step 1: Load triplets from canon_kg.txt
├── representation.py        # Step 2: Convert triplets to text
├── embedder.py              # Step 3: Generate embeddings
├── faiss_store.py           # Step 4: FAISS vector store
├── retriever.py             # Step 5: High-level retrieval interface
├── prompt_builder.py        # Step 6: Prompt augmentation
├── prompt_templates/        # Prompt templates
│   └── kg_qa.txt            # KG Q&A template
├── generator.py             # Step 7: End-to-end RAG generator
├── kg_rag_indexer.py        # Main orchestrator (Steps 1-4)
└── README.md                # This file
```

## Dependencies

Requires `faiss-cpu` (or `faiss-gpu` for GPU support):

```bash
pip install faiss-cpu
# or
pip install faiss-gpu
```

### LLM Provider Setup

For LLM generation (`--generate` mode), configure one of the following providers:

#### Google AI Studio (Recommended)

No additional dependencies. Just configure:

```bash
source export_google_ai.sh
```

#### SambaNova

No additional dependencies. Just configure:

```bash
source export_sambanova.sh
```

#### Local LLM

Requires GPU and additional dependencies:

```bash
pip install bitsandbytes
source export_local_llm.sh
```

Other dependencies (`sentence-transformers`, `numpy`, `tqdm`, `openai`) are already in the EDC environment.
