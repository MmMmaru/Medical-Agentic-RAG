# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical-Agentic-RAG is a multimodal medical question-answering system that integrates text, images, and structured data for healthcare information retrieval. It implements an agentic RAG (Retrieval-Augmented Generation) architecture with reinforcement learning capabilities.

- **Language**: Python 3.10+
- **Framework**: PyTorch, vLLM, Transformers
- **Vector DB**: Milvus Lite (default), FAISS
- **Models**: Qwen3-VL-Embedding-2B, Qwen3-VL-4B/8B-Instruct

## Common Commands

### Environment Setup
```bash
conda create -n medrag python=3.10
conda activate medrag
pip install -r requirements.txt
```

### Start Model Services (Required before running RAG)
```bash
# Terminal 1: Start Embedding Service (vLLM on port 8001)
bash scripts/embedding_server.sh

# Terminal 2: Start VLM Service (vLLM on port 8000)
bash scripts/vlm_server.sh
```

### Run RAG System
```bash
# Basic retrieval test
python MMRAG/MMRAG.py

# Embedding service test
python MMRAG/model_service/embedding_service.py
```

### Training
```bash
# FSDP multi-GPU training for embedding fine-tuning
cd train/finetune_embedding
torchrun --nproc_per_node=4 train_fsdp.py \
    --model_name Qwen/Qwen3-VL-Embedding-2B \
    --train_data <path> \
    --output_dir <path> \
    --ckpt <path>
```

## Architecture

### Layer Structure
```
Application Layer (Web UI, API Server, Evaluation)
    ↓
Agentic RAG Layer (RL Agent: Planner, Tool Executor, Memory)
    ↓
Core RAG Layer (MMRAG Engine, Retrieval Pipeline)
    ↓
Model Service Layer (Embedding Service, VLM Service, Reranker)
    ↓
Data Layer (Milvus/FAISS Vector DB, Dataset Cache)
```

### Key Components

**MMRAG Engine** (`MMRAG/MMRAG.py`)
- Main orchestration class for document ingestion and retrieval
- Key methods: `ingest_dataset()`, `retrieve()`, `delete_document()`
- Uses async/await pattern throughout

**DataChunk** (`MMRAG/base.py`)
- Core data structure: `doc_id`, `chunk_id`, `content`, `image_paths`, `vector`, `metadata`
- `dict_to_datachunk()` converts dataset dicts to DataChunk objects

**Vector Storage** (`MMRAG/DB/milvus_vectorDB.py`)
- MilvusVectorStorage: Local file-based vector DB (`.db` files)
- Schema: chunk_id (PK), vector (2048-dim), doc_id, content, image_paths, metadata
- Alternative: FAISS implementation in `faiss_vectorDB.py`

**Embedding Service** (`MMRAG/model_service/embedding_service.py`)
- `OpenaiEmbeddingService`: Connects to local vLLM server (port 8001)
- `VllmEmbeddingService`: Direct vLLM integration
- Input format: `[{"text": "...", "image": "path_or_url"}]`
- Supports base64 encoding for local images, URL fetching for remote images

**VLM Service** (`MMRAG/model_service/vlm_service.py`)
- Connects to vLLM server on port 8000 for generation
- OpenAI-compatible API

### Data Flow

**Ingestion**: Raw Dataset → DataChunk → Embedding (batch) → Vector Store
**Retrieval**: Query → Embedding → Vector Search → Ranked DataChunks

### Dataset Processing

Dataset loaders in `data/` directory:
- `medmax.py`: MedMax multi-task medical dataset
- `IU_Xray.py`: Radiology chest X-ray
- `Harvard_fairVLMed.py`: Ophthalmology VQA
- `pmc_oa.py`: Pathology open access
- `pmc_vqa.py`: Pathology VQA

## Configuration

### Environment Variables (`.env`)
```bash
HF_HOME=E:/PROJECT/Medical-agentic-rag/.cache/huggingface
HF_TOKEN=<your_token>
```

### Service Configuration
- Embedding service runs on `127.0.0.1:8001`
- VLM service runs on `127.0.0.1:8000`
- GPU memory utilization: 0.8 for embedding, 0.4 for VLM
- Max model length: 4096 tokens

## Important Implementation Details

**Async Pattern**: All I/O operations use async/await. Main methods:
- `MMRAG.ingest_dataset()` - async
- `MMRAG.retrieve()` - async
- `EmbeddingService.async_embed_batch()` - async

**Image Handling**: Images can be:
- Local paths (converted to base64)
- URLs (http/https/oss, fetched at runtime)
- PIL Image objects

**Vector Dimension**: 2048 (from Qwen3-VL-Embedding-2B)

**Batch Processing**: Default batch size 128 for embedding, configurable in `MMRAG.insert_batch_size`

## Extension Points

**New Vector Database**: Inherit from `BaseVectorStorage` (`MMRAG/base.py`)
**New Embedding Service**: Inherit from `EmbeddingService`
**New Dataset**: Create loader in `data/` following existing patterns

## Documentation

- `docs/architecture.md`: System design and data flow
- `docs/datasets.md`: Dataset preparation
- `docs/training.md`: Model fine-tuning
- `docs/evaluation.md`: Performance evaluation
