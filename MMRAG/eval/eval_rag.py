"""
MMRAG System Test Suite

This module provides comprehensive testing for the Medical Multi-modal RAG system,
including retrieval performance, end-to-end RAG, and hybrid retrieval tests.

Usage:
    python test_rag.py

Environment Variables:
    - MMRAG_TEST_WORKSPACE: Test workspace directory (default: ./test_workspace)
    - MMRAG_TEST_DATASET_SIZE: Default dataset size for testing (default: 100)
    - EMBEDDING_SERVICE_URL: URL for embedding service (default: http://localhost:8001/v1)
    - VLM_SERVICE_URL: URL for VLM service (default: http://localhost:8000/v1)
"""

import asyncio
import sys
import os
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from MMRAG.MMRAG import MMRAG
from MMRAG.model_service.embedding_service import OpenaiEmbeddingService
from MMRAG.model_service.vlm_service import OpenAIVLMService
from MMRAG.model_service.rerank_service import OpenAIRerankerService
from MMRAG.base import DataChunk
from MMRAG.utils import logger, compute_mdhash_id
from data.pmc_oa import PMCOADataset


@dataclass
class RetrievalMetrics:
    """Container for retrieval performance metrics."""
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    total_queries: int = 0
    successful_queries: int = 0
    avg_query_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mrr": self.mrr,
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "avg_query_time": self.avg_query_time,
        }

    def __str__(self) -> str:
        return (
            f"Recall@1: {self.recall_at_1:.4f} | "
            f"Recall@5: {self.recall_at_5:.4f} | "
            f"Recall@10: {self.recall_at_10:.4f} | "
            f"MRR: {self.mrr:.4f} | "
            f"Avg Time: {self.avg_query_time:.4f}s"
        )


@dataclass
class E2ETestResult:
    """Container for end-to-end RAG test results."""
    query: str = ""
    ground_truth: str = ""
    predicted_answer: str = ""
    retrieved_chunks: List[DataChunk] = field(default_factory=list)
    accuracy: float = 0.0
    query_time: float = 0.0


class TestMMRAG:
    """
    MMRAG System Test Suite

    Provides comprehensive testing capabilities for:
    1. Retrieval performance (Recall@K, MRR)
    2. End-to-end RAG pipeline
    3. Hybrid retrieval comparison
    """

    def __init__(
        self,
        workspace: str = None,
        embedding_url: str = "http://localhost:8001/v1",
        vlm_url: str = "http://localhost:8000/v1",
        vlm_model: str = "Qwen/Qwen3-VL-4B-Instruct",
        embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B",
        reranker_model: str = "Qwen/Qwen3-VL-Reranker-2B",
        use_reranker: bool = False,
    ):
        """
        Initialize the MMRAG test suite.

        Args:
            workspace: Test workspace directory
            embedding_url: URL for embedding service
            vlm_url: URL for VLM service
            vlm_model: VLM model name
            embedding_model: Embedding model name
            reranker_model: Reranker model name
            use_reranker: Whether to use reranker
        """
        self.workspace = workspace or os.getenv("MMRAG_TEST_WORKSPACE", "./test_workspace")
        self.embedding_url = embedding_url
        self.vlm_url = vlm_url
        self.vlm_model = vlm_model
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.use_reranker = use_reranker

        # Initialize services
        self.embedding_service = OpenaiEmbeddingService(
            model_name=embedding_model,
            api_key="EMPTY"
        )
        self.embedding_service.base_url = embedding_url

        self.vlm_service = OpenAIVLMService(
            model_name=vlm_model,
            api_key="EMPTY",
            url=vlm_url
        )

        self.rerank_service = None
        if use_reranker:
            self.rerank_service = OpenAIRerankerService(
                model_name=reranker_model,
                api_key="EMPTY",
                url=f"{vlm_url}/score"
            )

        # Initialize RAG system
        self.rag = MMRAG(
            embedding_service=self.embedding_service,
            rerank_service=self.rerank_service,
            llm_service=self.vlm_service,
            workspace=self.workspace
        )

        logger.info(f"TestMMRAG initialized with workspace: {self.workspace}")

    async def test_retrieval_performance(
        self,
        dataset_name: str = "pmc-oa",
        dataset_size: int = None,
        top_k_values: List[int] = None,
        save_results: bool = True
    ) -> RetrievalMetrics:
        """
        Test retrieval performance using Recall@K and MRR metrics.

        This test evaluates how well the retrieval system can find the correct
        document (chunk_id) given a query (question).

        Args:
            dataset_name: Name of the dataset to test on
            dataset_size: Number of samples to test (None for all)
            top_k_values: List of K values for Recall@K (default: [1, 5, 10])
            save_results: Whether to save results to file

        Returns:
            RetrievalMetrics object containing test results
        """
        dataset_size = dataset_size or int(os.getenv("MMRAG_TEST_DATASET_SIZE", "100"))
        top_k_values = top_k_values or [1, 5, 10]
        max_k = max(top_k_values)

        logger.info(f"Starting retrieval performance test on {dataset_name} (n={dataset_size})")

        # Load dataset
        try:
            dataset = PMCOADataset(dataset_name, dataset_size=dataset_size)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

        if len(dataset) == 0:
            raise ValueError(f"Dataset {dataset_name} is empty")

        # Metrics tracking
        recall_counts = {k: 0 for k in top_k_values}
        mrr_sum = 0.0
        query_times = []
        successful_queries = 0

        # Test each sample
        for i in range(len(dataset)):
            item = dataset[i]
            query_text = item.get('question', '')
            gold_chunk_id = item.get('chunk_id')

            if not query_text or not gold_chunk_id:
                logger.warning(f"Skipping sample {i}: missing question or chunk_id")
                continue

            # Perform retrieval
            start_time = time.time()
            try:
                results = await self.rag.naive_retrieve(
                    query=[{"text": query_text}],
                    top_k=max_k
                )
                query_time = time.time() - start_time
                query_times.append(query_time)
                successful_queries += 1

                # Find rank of gold chunk
                gold_rank = None
                for rank, chunk in enumerate(results, start=1):
                    if chunk.chunk_id == gold_chunk_id:
                        gold_rank = rank
                        break

                # Update metrics
                if gold_rank:
                    for k in top_k_values:
                        if gold_rank <= k:
                            recall_counts[k] += 1
                    mrr_sum += 1.0 / gold_rank
                else:
                    # Gold chunk not found in top-k
                    mrr_sum += 0.0

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(dataset)} queries...")

            except Exception as e:
                logger.error(f"Error processing query {i}: {e}")
                continue

        # Calculate final metrics
        total = len(dataset)
        metrics = RetrievalMetrics(
            recall_at_1=recall_counts.get(1, 0) / total if total > 0 else 0,
            recall_at_5=recall_counts.get(5, 0) / total if total > 0 else 0,
            recall_at_10=recall_counts.get(10, 0) / total if total > 0 else 0,
            mrr=mrr_sum / total if total > 0 else 0,
            total_queries=total,
            successful_queries=successful_queries,
            avg_query_time=sum(query_times) / len(query_times) if query_times else 0
        )

        logger.info(f"Retrieval Performance Results: {metrics}")

        # Save results
        if save_results:
            results_path = os.path.join(self.workspace, f"retrieval_metrics_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            os.makedirs(self.workspace, exist_ok=True)
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {results_path}")

        return metrics

    async def test_e2e_rag(
        self,
        dataset_name: str = "pmc-oa",
        dataset_size: int = None,
        top_k: int = 5,
        save_results: bool = True
    ) -> List[E2ETestResult]:
        """
        Test end-to-end RAG pipeline.

        This test evaluates the complete RAG flow:
        Query -> Retrieval -> Context Building -> Answer Generation

        Args:
            dataset_name: Name of the dataset to test on
            dataset_size: Number of samples to test
            top_k: Number of documents to retrieve
            save_results: Whether to save results to file

        Returns:
            List of E2ETestResult objects
        """
        dataset_size = dataset_size or int(os.getenv("MMRAG_TEST_DATASET_SIZE", "50"))
        logger.info(f"Starting E2E RAG test on {dataset_name} (n={dataset_size})")

        # Load dataset
        try:
            dataset = PMCOADataset(dataset_name, dataset_size=dataset_size)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

        results = []
        correct_count = 0

        for i in range(min(dataset_size, len(dataset))):
            item = dataset[i]
            query_text = item.get('question', '')
            ground_truth = item.get('answer', '')
            image_paths = item.get('image_paths', [])

            if not query_text:
                logger.warning(f"Skipping sample {i}: missing question")
                continue

            start_time = time.time()
            try:
                # Step 1: Hybrid retrieval
                retrieved_chunks = await self.rag.hybrid_retrieve(
                    query_text=query_text,
                    query_image_paths=image_paths if image_paths else None,
                    top_k=top_k
                )

                # Step 2: Optional reranking
                if self.use_reranker and self.rerank_service and retrieved_chunks:
                    try:
                        retrieved_chunks = await self.rag.rerank(
                            query=query_text,
                            chunks=retrieved_chunks
                        )
                    except Exception as e:
                        logger.warning(f"Reranking failed: {e}")

                # Step 3: Build context from retrieved chunks
                context = self._build_context(retrieved_chunks)

                # Step 4: Generate answer using VLM
                messages = self._build_rag_prompt(query_text, context, image_paths)
                predicted_answers = await self.vlm_service.async_generate_batch(
                    contents=[messages],
                    temperature=0.7
                )
                predicted_answer = predicted_answers[0] if predicted_answers else ""

                query_time = time.time() - start_time

                # Evaluate accuracy (simple matching for now)
                accuracy = self._evaluate_answer(predicted_answer, ground_truth)
                if accuracy > 0.5:
                    correct_count += 1

                result = E2ETestResult(
                    query=query_text,
                    ground_truth=ground_truth,
                    predicted_answer=predicted_answer,
                    retrieved_chunks=retrieved_chunks,
                    accuracy=accuracy,
                    query_time=query_time
                )
                results.append(result)

                if (i + 1) % 5 == 0:
                    logger.info(f"Processed {i + 1}/{min(dataset_size, len(dataset))} E2E queries...")

            except Exception as e:
                logger.error(f"Error in E2E test for sample {i}: {e}")
                continue

        # Calculate overall accuracy
        overall_accuracy = correct_count / len(results) if results else 0
        logger.info(f"E2E RAG Test Complete: {len(results)} samples, Accuracy: {overall_accuracy:.4f}")

        # Save results
        if save_results and results:
            results_path = os.path.join(self.workspace, f"e2e_results_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            os.makedirs(self.workspace, exist_ok=True)
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "overall_accuracy": overall_accuracy,
                    "total_samples": len(results),
                    "results": [
                        {
                            "query": r.query,
                            "ground_truth": r.ground_truth,
                            "predicted_answer": r.predicted_answer,
                            "accuracy": r.accuracy,
                            "query_time": r.query_time,
                            "num_retrieved": len(r.retrieved_chunks)
                        }
                        for r in results
                    ]
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"E2E results saved to {results_path}")

        return results

    async def test_hybrid_retrieval(
        self,
        query_text: str,
        query_image_paths: List[str] = None,
        gold_chunk_id: str = None,
        top_k: int = 10,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Test and compare different retrieval methods.

        Compares:
        - Dense retrieval (vector-based)
        - Sparse retrieval (BM25)
        - Hybrid retrieval (RRF fusion)

        Args:
            query_text: Query text
            query_image_paths: Optional query image paths
            gold_chunk_id: Optional gold standard chunk ID for evaluation
            top_k: Number of results to retrieve
            save_results: Whether to save results to file

        Returns:
            Dictionary containing comparison results
        """
        logger.info(f"Testing hybrid retrieval for query: {query_text[:50]}...")

        results = {
            "query": query_text,
            "timestamp": datetime.now().isoformat(),
            "top_k": top_k
        }

        # Test 1: Dense Retrieval
        start_time = time.time()
        try:
            query_vectors = await self.embedding_service.async_embed_batch([{"text": query_text}])
            dense_results = await self.rag.vector_storage.search(query_vectors[0], top_k=top_k)
            dense_time = time.time() - start_time

            results["dense"] = {
                "retrieval_time": dense_time,
                "num_results": len(dense_results),
                "results": [
                    {"chunk_id": r.chunk_id, "score": r.metadata.get("score", 0)}
                    for r in dense_results
                ]
            }

            if gold_chunk_id:
                dense_rank = next((i + 1 for i, r in enumerate(dense_results) if r.chunk_id == gold_chunk_id), None)
                results["dense"]["gold_rank"] = dense_rank
                results["dense"]["recall"] = 1 if dense_rank and dense_rank <= top_k else 0

        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            results["dense"] = {"error": str(e)}

        # Test 2: Sparse Retrieval (BM25)
        start_time = time.time()
        try:
            sparse_results = await self.rag.bm25_storage.search(query_text, top_k=top_k)
            sparse_time = time.time() - start_time

            results["sparse"] = {
                "retrieval_time": sparse_time,
                "num_results": len(sparse_results),
                "results": [
                    {"chunk_id": chunk_id, "score": score}
                    for chunk_id, score in sparse_results
                ]
            }

            if gold_chunk_id:
                sparse_rank = next((i + 1 for i, (cid, _) in enumerate(sparse_results) if cid == gold_chunk_id), None)
                results["sparse"]["gold_rank"] = sparse_rank
                results["sparse"]["recall"] = 1 if sparse_rank and sparse_rank <= top_k else 0

        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            results["sparse"] = {"error": str(e)}

        # Test 3: Hybrid Retrieval
        start_time = time.time()
        try:
            hybrid_results = await self.rag.hybrid_retrieve(
                query_text=query_text,
                query_image_paths=query_image_paths,
                top_k=top_k
            )
            hybrid_time = time.time() - start_time

            results["hybrid"] = {
                "retrieval_time": hybrid_time,
                "num_results": len(hybrid_results),
                "results": [
                    {"chunk_id": r.chunk_id, "rrf_score": r.metadata.get("rrf_score", 0)}
                    for r in hybrid_results
                ]
            }

            if gold_chunk_id:
                hybrid_rank = next((i + 1 for i, r in enumerate(hybrid_results) if r.chunk_id == gold_chunk_id), None)
                results["hybrid"]["gold_rank"] = hybrid_rank
                results["hybrid"]["recall"] = 1 if hybrid_rank and hybrid_rank <= top_k else 0

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            results["hybrid"] = {"error": str(e)}

        # Comparison summary
        logger.info("=" * 60)
        logger.info("Retrieval Method Comparison:")
        logger.info(f"  Dense:   {results.get('dense', {}).get('num_results', 0)} results, "
                   f"time={results.get('dense', {}).get('retrieval_time', 0):.4f}s")
        logger.info(f"  Sparse:  {results.get('sparse', {}).get('num_results', 0)} results, "
                   f"time={results.get('sparse', {}).get('retrieval_time', 0):.4f}s")
        logger.info(f"  Hybrid:  {results.get('hybrid', {}).get('num_results', 0)} results, "
                   f"time={results.get('hybrid', {}).get('retrieval_time', 0):.4f}s")

        if gold_chunk_id:
            logger.info(f"  Gold chunk rank - Dense: {results.get('dense', {}).get('gold_rank', 'N/A')}, "
                       f"Sparse: {results.get('sparse', {}).get('gold_rank', 'N/A')}, "
                       f"Hybrid: {results.get('hybrid', {}).get('gold_rank', 'N/A')}")
        logger.info("=" * 60)

        # Save results
        if save_results:
            results_path = os.path.join(self.workspace, f"hybrid_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            os.makedirs(self.workspace, exist_ok=True)
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Hybrid comparison results saved to {results_path}")

        return results

    async def test_hybrid_retrieval_batch(
        self,
        dataset_name: str = "pmc-oa",
        dataset_size: int = None,
        top_k: int = 10,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Batch test hybrid retrieval comparison on a dataset.

        Args:
            dataset_name: Name of the dataset
            dataset_size: Number of samples to test
            top_k: Number of results to retrieve
            save_results: Whether to save results

        Returns:
            Aggregated comparison statistics
        """
        dataset_size = dataset_size or int(os.getenv("MMRAG_TEST_DATASET_SIZE", "50"))
        logger.info(f"Starting batch hybrid retrieval test on {dataset_name} (n={dataset_size})")

        # Load dataset
        try:
            dataset = PMCOADataset(dataset_name, dataset_size=dataset_size)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

        # Statistics tracking
        stats = {
            "dense": {"recall": 0, "total_time": 0, "gold_ranks": []},
            "sparse": {"recall": 0, "total_time": 0, "gold_ranks": []},
            "hybrid": {"recall": 0, "total_time": 0, "gold_ranks": []},
        }

        for i in range(min(dataset_size, len(dataset))):
            item = dataset[i]
            query_text = item.get('question', '')
            gold_chunk_id = item.get('chunk_id')
            image_paths = item.get('image_paths', [])

            if not query_text or not gold_chunk_id:
                continue

            try:
                result = await self.test_hybrid_retrieval(
                    query_text=query_text,
                    query_image_paths=image_paths if image_paths else None,
                    gold_chunk_id=gold_chunk_id,
                    top_k=top_k,
                    save_results=False
                )

                # Aggregate statistics
                for method in ["dense", "sparse", "hybrid"]:
                    if method in result and "error" not in result[method]:
                        stats[method]["total_time"] += result[method].get("retrieval_time", 0)
                        stats[method]["recall"] += result[method].get("recall", 0)
                        if result[method].get("gold_rank"):
                            stats[method]["gold_ranks"].append(result[method]["gold_rank"])

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{min(dataset_size, len(dataset))} comparison queries...")

            except Exception as e:
                logger.error(f"Error in batch comparison for sample {i}: {e}")
                continue

        # Calculate averages
        total = min(dataset_size, len(dataset))
        summary = {
            "dataset": dataset_name,
            "total_queries": total,
            "top_k": top_k,
            "methods": {}
        }

        for method in ["dense", "sparse", "hybrid"]:
            method_stats = stats[method]
            count = len(method_stats["gold_ranks"])
            summary["methods"][method] = {
                "recall_at_k": method_stats["recall"] / total if total > 0 else 0,
                "avg_retrieval_time": method_stats["total_time"] / total if total > 0 else 0,
                "avg_gold_rank": sum(method_stats["gold_ranks"]) / count if count > 0 else None,
                "mrr": sum(1.0 / r for r in method_stats["gold_ranks"]) / count if count > 0 else 0
            }

        logger.info("=" * 60)
        logger.info("Batch Hybrid Retrieval Comparison Summary:")
        for method, method_summary in summary["methods"].items():
            logger.info(f"  {method.capitalize()}:")
            logger.info(f"    Recall@{top_k}: {method_summary['recall_at_k']:.4f}")
            logger.info(f"    MRR: {method_summary['mrr']:.4f}")
            logger.info(f"    Avg Time: {method_summary['avg_retrieval_time']:.4f}s")
        logger.info("=" * 60)

        # Save results
        if save_results:
            results_path = os.path.join(self.workspace, f"hybrid_batch_summary_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            os.makedirs(self.workspace, exist_ok=True)
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Batch summary saved to {results_path}")

        return summary

    def _build_context(self, chunks: List[DataChunk]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.content or ""
            context_parts.append(f"[Document {i}]: {content}")
        return "\n\n".join(context_parts)

    def _build_rag_prompt(self, query: str, context: str, image_paths: List[str] = None) -> List[Dict]:
        """Build RAG prompt for VLM."""
        prompt_text = f"""Based on the following retrieved documents, please answer the question.

Retrieved Documents:
{context}

Question: {query}

Please provide a comprehensive answer based on the retrieved documents."""

        content = []

        # Add images if available
        if image_paths:
            from MMRAG.utils import encode_image_paths_to_base64
            image_content = encode_image_paths_to_base64(image_paths)
            content.extend(image_content)

        content.append({"type": "text", "text": prompt_text})

        return content

    def _evaluate_answer(self, predicted: str, ground_truth: str) -> float:
        """
        Evaluate answer accuracy.

        Simple implementation - can be extended with:
        - Exact match
        - F1 score
        - Semantic similarity
        - Medical entity matching
        """
        if not predicted or not ground_truth:
            return 0.0

        pred_clean = predicted.lower().strip()
        gt_clean = ground_truth.lower().strip()

        # Exact match
        if pred_clean == gt_clean:
            return 1.0

        # Contains match
        if gt_clean in pred_clean or pred_clean in gt_clean:
            return 0.8

        # Token overlap (simple F1)
        pred_tokens = set(pred_clean.split())
        gt_tokens = set(gt_clean.split())

        if not pred_tokens or not gt_tokens:
            return 0.0

        intersection = pred_tokens & gt_tokens
        precision = len(intersection) / len(pred_tokens) if pred_tokens else 0
        recall = len(intersection) / len(gt_tokens) if gt_tokens else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    async def cleanup(self):
        """Clean up resources."""
        await self.rag.shutdown()
        logger.info("TestMMRAG cleanup complete")


async def main():
    """Main entry point for running tests."""
    import argparse

    parser = argparse.ArgumentParser(description="MMRAG Test Suite")
    parser.add_argument("--test", choices=["retrieval", "e2e", "hybrid", "all"], default="all",
                       help="Test type to run")
    parser.add_argument("--workspace", default="./workspace", help="Test workspace directory")
    parser.add_argument("--dataset", default="pmc-oa", help="Dataset name")
    parser.add_argument("--dataset-size", type=int, default=None, help="Dataset size")
    parser.add_argument("--embedding-url", default="http://localhost:8001/v1", help="Embedding service URL")
    parser.add_argument("--vlm-url", default="http://localhost:8000/v1", help="VLM service URL")
    parser.add_argument("--use-reranker", action="store_true", help="Enable reranker")
    parser.add_argument("--query", default=None, help="Single query for hybrid test")

    args = parser.parse_args()

    # Initialize tester
    tester = TestMMRAG(
        workspace=args.workspace,
        embedding_url=args.embedding_url,
        vlm_url=args.vlm_url,
        use_reranker=args.use_reranker
    )

    try:
        if args.test in ["retrieval", "all"]:
            logger.info("=" * 60)
            logger.info("Running Retrieval Performance Test")
            logger.info("=" * 60)
            metrics = await tester.test_retrieval_performance(
                dataset_name=args.dataset,
                dataset_size=args.dataset_size
            )
            print(f"\nFinal Results: {metrics}\n")

        if args.test in ["e2e", "all"]:
            logger.info("=" * 60)
            logger.info("Running End-to-End RAG Test")
            logger.info("=" * 60)
            results = await tester.test_e2e_rag(
                dataset_name=args.dataset,
                dataset_size=args.dataset_size
            )
            print(f"\nE2E Test Complete: {len(results)} samples processed\n")

        if args.test in ["hybrid", "all"]:
            logger.info("=" * 60)
            logger.info("Running Hybrid Retrieval Test")
            logger.info("=" * 60)

            if args.query:
                # Single query test
                result = await tester.test_hybrid_retrieval(query_text=args.query)
                print(f"\nHybrid Test Complete for query: {args.query[:50]}...\n")
            else:
                # Batch test
                summary = await tester.test_hybrid_retrieval_batch(
                    dataset_name=args.dataset,
                    dataset_size=args.dataset_size
                )
                print(f"\nBatch Hybrid Test Complete\n")

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
