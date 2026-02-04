"""
BM25 Sparse Retrieval Storage Implementation

This module provides BM25-based sparse retrieval for text documents,
supporting both Chinese (via jieba) and English text tokenization.
"""

import os
import re
import json
import pickle
import asyncio
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict

from ..utils import logger
from ..base import DataChunk

# Optional imports with error handling
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    logger.error("rank_bm25 not installed. Please install it via `pip install rank-bm25`")
    raise

try:
    import jieba
except ImportError:
    logger.error("jieba not installed. Please install it via `pip install jieba`")
    raise


# Basic English stop words list
ENGLISH_STOP_WORDS: Set[str] = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they',
    'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do',
    'how', 'their', 'if', 'up', 'out', 'many', 'then', 'them',
    'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into',
    'him', 'has', 'two', 'more', 'very', 'after', 'words', 'just',
    'where', 'most', 'know', 'get', 'through', 'back', 'much',
    'go', 'good', 'new', 'write', 'our', 'me', 'man', 'too',
    'any', 'day', 'same', 'right', 'look', 'think', 'also',
    'around', 'another', 'came', 'come', 'work', 'three',
    'must', 'because', 'does', 'part', 'even', 'place',
    'well', 'such', 'here', 'take', 'why', 'things', 'help',
    'put', 'years', 'different', 'away', 'again', 'off',
    'went', 'old', 'number', 'great', 'tell', 'men', 'say',
    'small', 'every', 'found', 'still', 'between', 'name',
    'should', 'home', 'big', 'give', 'air', 'line', 'set',
    'own', 'under', 'read', 'last', 'never', 'us', 'left',
    'end', 'along', 'while', 'might', 'next', 'sound',
    'below', 'saw', 'something', 'thought', 'both',
    'few', 'those', 'always', 'show', 'large', 'often',
    'together', 'asked', 'house', 'don\'t', 'world', 'going',
    'want', 'school', 'important', 'until', 'form', 'food',
    'keep', 'children', 'feet', 'land', 'side', 'without',
    'boy', 'once', 'animal', 'life', 'enough', 'took',
    'four', 'head', 'above', 'kind', 'began', 'almost',
    'live', 'page', 'got', 'built', 'grow', 'cut', 'knew',
    'earth', 'father', 'head', 'stand', 'own', 'course',
    'stay', 'wheel', 'full', 'force', 'blue', 'object',
    'decide', 'surface', 'deep', 'moon', 'island', 'foot',
    'system', 'busy', 'test', 'record', 'boat', 'common',
    'gold', 'possible', 'plane', 'stead', 'dry', 'wonder',
    'laugh', 'thousands', 'ago', 'ran', 'check', 'game',
    'shape', 'equate', 'hot', 'miss', 'brought', 'heat',
    'snow', 'tire', 'bring', 'yes', 'distant', 'fill',
    'east', 'paint', 'language', 'among'
}


class BM25Storage:
    """
    BM25 sparse retrieval storage implementation.

    Supports both Chinese (via jieba) and English text tokenization.
    Uses rank_bm25.BM25Okapi for the core BM25 algorithm.

    Attributes:
        workspace: Directory path for storing index files
        k1: BM25 k1 parameter (term frequency saturation)
        b: BM25 b parameter (length normalization)
    """

    def __init__(self, workspace: str, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 storage.

        Args:
            workspace: Directory path for storing index files
            k1: BM25 k1 parameter (default: 1.5)
            b: BM25 b parameter (default: 0.75)
        """
        self.workspace = workspace
        self.k1 = k1
        self.b = b

        # Create storage directory
        self.storage_dir = os.path.join(workspace, "bm25")
        os.makedirs(self.storage_dir, exist_ok=True)

        # File paths
        self.index_path = os.path.join(self.storage_dir, "bm25_index.pkl")
        self.corpus_path = os.path.join(self.storage_dir, "corpus.pkl")
        self.metadata_path = os.path.join(self.storage_dir, "metadata.json")

        # Data structures
        self.bm25_index: Optional[BM25Okapi] = None
        self.corpus: List[List[str]] = []  # Tokenized documents
        self.chunk_id_to_idx: Dict[str, int] = {}  # chunk_id -> index position
        self.idx_to_chunk_id: Dict[int, str] = {}  # index position -> chunk_id
        self.chunk_texts: Dict[str, str] = {}  # chunk_id -> raw text content
        self.deleted_chunk_ids: Set[str] = set()  # Track deleted chunks

        # Load existing index if available
        if os.path.exists(self.metadata_path):
            asyncio.run(self.load())

        logger.info(f"BM25Storage initialized at {self.storage_dir} (k1={k1}, b={b})")

    def _is_chinese(self, text: str) -> bool:
        """
        Check if text contains Chinese characters.

        Args:
            text: Input text

        Returns:
            True if text contains Chinese characters
        """
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def _tokenize_chinese(self, text: str) -> List[str]:
        """
        Tokenize Chinese text using jieba.

        Args:
            text: Input Chinese text

        Returns:
            List of tokens
        """
        # Use jieba for Chinese word segmentation
        tokens = list(jieba.cut(text))
        # Filter out stop words and empty tokens
        filtered_tokens = [
            token.strip().lower()
            for token in tokens
            if token.strip() and len(token.strip()) > 1
        ]
        return filtered_tokens

    def _tokenize_english(self, text: str) -> List[str]:
        """
        Tokenize English text using regex word extraction.

        Args:
            text: Input English text

        Returns:
            List of tokens
        """
        # Extract words using regex (alphanumeric sequences)
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # Filter out stop words
        filtered_tokens = [
            token for token in tokens
            if token not in ENGLISH_STOP_WORDS and len(token) > 1
        ]
        return filtered_tokens

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text based on language detection.

        Supports mixed Chinese-English content.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if not text or not isinstance(text, str):
            return []

        # Check if text contains Chinese
        if self._is_chinese(text):
            # For mixed content, jieba handles both Chinese and English
            return self._tokenize_chinese(text)
        else:
            return self._tokenize_english(text)

    def _rebuild_index(self) -> None:
        """
        Rebuild BM25 index from current corpus.

        This is called after deletions or when loading from disk.
        """
        if not self.corpus:
            self.bm25_index = None
            return

        # Filter out deleted documents
        active_corpus = []
        active_idx_mapping = {}  # old_idx -> new_idx
        new_idx = 0

        for old_idx, tokens in enumerate(self.corpus):
            chunk_id = self.idx_to_chunk_id.get(old_idx)
            if chunk_id and chunk_id not in self.deleted_chunk_ids:
                active_corpus.append(tokens)
                active_idx_mapping[old_idx] = new_idx
                new_idx += 1

        if active_corpus:
            self.bm25_index = BM25Okapi(active_corpus, k1=self.k1, b=self.b)
            self.corpus = active_corpus

            # Update mappings
            new_chunk_id_to_idx = {}
            new_idx_to_chunk_id = {}

            for old_idx, new_idx in active_idx_mapping.items():
                chunk_id = self.idx_to_chunk_id[old_idx]
                new_chunk_id_to_idx[chunk_id] = new_idx
                new_idx_to_chunk_id[new_idx] = chunk_id

            self.chunk_id_to_idx = new_chunk_id_to_idx
            self.idx_to_chunk_id = new_idx_to_chunk_id

            # Clean up deleted texts
            for chunk_id in list(self.deleted_chunk_ids):
                self.chunk_texts.pop(chunk_id, None)

            self.deleted_chunk_ids.clear()
        else:
            self.bm25_index = None
            self.corpus = []
            self.chunk_id_to_idx = {}
            self.idx_to_chunk_id = {}

    async def upsert(self, chunks: List[DataChunk]) -> None:
        """
        Add or update documents in the BM25 index.

        Args:
            chunks: List of DataChunk objects to add/update
        """
        if not chunks:
            return

        # Process each chunk
        new_corpus_entries = []
        new_chunk_ids = []

        for chunk in chunks:
            chunk_id = getattr(chunk, 'chunk_id', None)
            if not chunk_id:
                logger.warning("Skipping chunk without chunk_id")
                continue

            content = getattr(chunk, 'content', None)
            if not content:
                logger.warning(f"Skipping chunk {chunk_id} without content")
                continue

            # Check if this is an update (chunk_id already exists)
            if chunk_id in self.chunk_id_to_idx:
                # Mark old entry as deleted - will be rebuilt later
                self.deleted_chunk_ids.add(chunk_id)

            # Tokenize content
            tokens = self._tokenize(content)
            if not tokens:
                logger.warning(f"Chunk {chunk_id} produced no tokens after filtering")
                continue

            # Store text content
            self.chunk_texts[chunk_id] = content

            # Add to new entries list
            new_corpus_entries.append(tokens)
            new_chunk_ids.append(chunk_id)

        if not new_corpus_entries:
            logger.info("No valid chunks to upsert")
            return

        # If there are deleted entries, rebuild index with new entries included
        if self.deleted_chunk_ids:
            self.corpus.extend(new_corpus_entries)
            for chunk_id in new_chunk_ids:
                idx = len(self.corpus) - len(new_chunk_ids) + new_chunk_ids.index(chunk_id)
                self.idx_to_chunk_id[idx] = chunk_id
                self.chunk_id_to_idx[chunk_id] = idx
            self._rebuild_index()
        else:
            # Just append to existing index
            start_idx = len(self.corpus)
            self.corpus.extend(new_corpus_entries)

            for i, chunk_id in enumerate(new_chunk_ids):
                idx = start_idx + i
                self.chunk_id_to_idx[chunk_id] = idx
                self.idx_to_chunk_id[idx] = chunk_id

            # Rebuild BM25 index with all documents
            if self.corpus:
                self.bm25_index = BM25Okapi(self.corpus, k1=self.k1, b=self.b)

        logger.info(f"Upserted {len(new_corpus_entries)} chunks to BM25 index")

    async def delete(self, chunk_ids: List[str]) -> None:
        """
        Delete documents from the BM25 index.

        Since BM25Okapi doesn't support deletion, this marks entries as deleted
        and rebuilds the index.

        Args:
            chunk_ids: List of chunk IDs to delete
        """
        if not chunk_ids:
            return

        # Track which chunks actually exist
        existing_ids = []
        for chunk_id in chunk_ids:
            if chunk_id in self.chunk_id_to_idx:
                self.deleted_chunk_ids.add(chunk_id)
                existing_ids.append(chunk_id)

        if not existing_ids:
            logger.info("No matching chunks found for deletion")
            return

        # Rebuild index without deleted entries
        self._rebuild_index()

        logger.info(f"Deleted {len(existing_ids)} chunks from BM25 index")

    async def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for documents using BM25 scoring.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of (chunk_id, score) tuples, sorted by score descending
        """
        if not self.bm25_index or not self.corpus:
            logger.warning("BM25 index is empty, cannot search")
            return []

        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided")
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            logger.warning("Query produced no tokens after filtering")
            return []

        try:
            # Get BM25 scores for all documents
            scores = self.bm25_index.get_scores(query_tokens)

            # Get top-k indices
            # scores is a numpy array or list of floats
            if hasattr(scores, 'argsort'):
                # numpy array
                top_indices = scores.argsort()[-top_k:][::-1]
                results = [
                    (self.idx_to_chunk_id[int(idx)], float(scores[idx]))
                    for idx in top_indices
                    if float(scores[idx]) > 0
                ]
            else:
                # Regular list
                indexed_scores = [(i, score) for i, score in enumerate(scores)]
                indexed_scores.sort(key=lambda x: x[1], reverse=True)
                results = [
                    (self.idx_to_chunk_id[idx], float(score))
                    for idx, score in indexed_scores[:top_k]
                    if score > 0
                ]

            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    async def persist(self) -> None:
        """
        Save BM25 index and metadata to disk.

        Files saved:
        - bm25_index.pkl: Pickled BM25 index
        - corpus.pkl: Tokenized corpus
        - metadata.json: chunk_id mappings and raw texts
        """
        if not self.corpus:
            logger.warning("No data to persist")
            return

        try:
            # Save BM25 index
            with open(self.index_path, 'wb') as f:
                pickle.dump(self.bm25_index, f)

            # Save corpus
            with open(self.corpus_path, 'wb') as f:
                pickle.dump(self.corpus, f)

            # Save metadata
            metadata = {
                'chunk_id_to_idx': self.chunk_id_to_idx,
                'idx_to_chunk_id': {str(k): v for k, v in self.idx_to_chunk_id.items()},
                'chunk_texts': self.chunk_texts,
                'deleted_chunk_ids': list(self.deleted_chunk_ids),
                'k1': self.k1,
                'b': self.b
            }
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"BM25 index persisted to {self.storage_dir}")

        except Exception as e:
            logger.error(f"Failed to persist BM25 index: {e}")
            raise

    async def load(self) -> None:
        """
        Load BM25 index and metadata from disk.

        Loads files from:
        - bm25_index.pkl: Pickled BM25 index
        - corpus.pkl: Tokenized corpus
        - metadata.json: chunk_id mappings and raw texts
        """
        if not os.path.exists(self.metadata_path):
            logger.info("No existing BM25 index found, starting fresh")
            return

        try:
            # Load metadata
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            self.chunk_id_to_idx = metadata.get('chunk_id_to_idx', {})
            self.idx_to_chunk_id = {int(k): v for k, v in metadata.get('idx_to_chunk_id', {}).items()}
            self.chunk_texts = metadata.get('chunk_texts', {})
            self.deleted_chunk_ids = set(metadata.get('deleted_chunk_ids', []))

            # Load corpus
            if os.path.exists(self.corpus_path):
                with open(self.corpus_path, 'rb') as f:
                    self.corpus = pickle.load(f)

            # Load BM25 index or rebuild if needed
            if os.path.exists(self.index_path):
                with open(self.index_path, 'rb') as f:
                    self.bm25_index = pickle.load(f)
            elif self.corpus:
                # Rebuild index if pickle file is missing
                self._rebuild_index()

            # Handle any pending deletions
            if self.deleted_chunk_ids:
                self._rebuild_index()

            logger.info(f"BM25 index loaded from {self.storage_dir} "
                       f"({len(self.corpus)} documents)")

        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            # Reset to empty state
            self.bm25_index = None
            self.corpus = []
            self.chunk_id_to_idx = {}
            self.idx_to_chunk_id = {}
            self.chunk_texts = {}
            self.deleted_chunk_ids = set()
            raise

    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the BM25 storage.

        Returns:
            Dictionary with storage statistics
        """
        return {
            'num_documents': len(self.corpus),
            'num_deleted': len(self.deleted_chunk_ids),
            'total_chunks': len(self.chunk_id_to_idx),
            'k1': self.k1,
            'b': self.b,
            'storage_dir': self.storage_dir,
            'has_index': self.bm25_index is not None
        }

    def get_document(self, chunk_id: str) -> Optional[str]:
        """
        Get the raw text content of a document by chunk_id.

        Args:
            chunk_id: Document chunk ID

        Returns:
            Raw text content or None if not found
        """
        return self.chunk_texts.get(chunk_id)
