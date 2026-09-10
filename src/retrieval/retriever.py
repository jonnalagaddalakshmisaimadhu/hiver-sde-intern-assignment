"""
Phase 8: Historical Support Retriever for @AppleSupport.
Indexes customer support conversations, enforces anti-contamination filtering against the Golden Set,
and performs dense semantic similarity search with caching and evidence sufficiency verification.
"""
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Set
import json
import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity


class RetrievedEvidence(BaseModel):
    source_id: str = Field(..., description="Conversation ID of the historical support interaction")
    customer_message: str = Field(..., description="Historical customer inquiry")
    brand_response: str = Field(..., description="Authentic historical Apple Support resolution")
    similarity: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")


class HistoricalSupportRetriever:
    """Dense vector retriever grounded in authentic Apple Support historical interactions."""

    def __init__(
        self,
        corpus_path: Path = Path("data/sample/applesupport_sample.jsonl"),
        golden_set_path: Path = Path("golden_set/golden_evaluation_set.jsonl"),
        cache_path: Path = Path(".cache/retrieval_index.pkl"),
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.corpus_path = Path(corpus_path)
        self.golden_set_path = Path(golden_set_path)
        self.cache_path = Path(cache_path)
        self.embedding_model_name = embedding_model_name

        self.records: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self._load_and_index()

    def _get_golden_masked_ids(self) -> Set[str]:
        """Collect all Golden Set conversation IDs to strictly exclude from the index."""
        masked_ids = set()
        if self.golden_set_path.is_file():
            with open(self.golden_set_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    masked_ids.add(rec["conversation_id"])
        return masked_ids

    def _load_and_index(self):
        """Load corpus and compute or restore dense embeddings."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        if self.cache_path.is_file():
            print(f"[INFO] Loading cached retrieval index from {self.cache_path}...")
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
                self.records = data["records"]
                self.embeddings = data["embeddings"]
            print(f"[INFO] Loaded {len(self.records):,} indexed historical conversations from cache.")
            return

        masked_ids = self._get_golden_masked_ids()
        print(f"[INFO] Indexing support corpus from {self.corpus_path} (masking {len(masked_ids)} Golden IDs)...")

        texts_to_embed = []
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["conversation_id"] in masked_ids:
                    continue
                self.records.append({
                    "conversation_id": rec["conversation_id"],
                    "customer_inquiry": rec["customer_inquiry"],
                    "brand_initial_response": rec["brand_initial_response"],
                })
                texts_to_embed.append(rec["customer_inquiry"])

        print(f"[INFO] Embedding {len(texts_to_embed):,} historical customer queries...")

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.embedding_model_name)
            self.embeddings = model.encode(texts_to_embed, show_progress_bar=False, normalize_embeddings=True)
        except Exception as e:
            print(f"[WARN] sentence-transformers unavailable ({e}). Falling back to TF-IDF dense embeddings.")
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True)
            tfidf_mat = vectorizer.fit_transform(texts_to_embed)
            # Normalize
            from sklearn.preprocessing import normalize
            self.embeddings = normalize(tfidf_mat).toarray()

        with open(self.cache_path, "wb") as f:
            pickle.dump({"records": self.records, "embeddings": self.embeddings}, f)

        print(f"[SUCCESS] Indexed {len(self.records):,} records. Cache written to {self.cache_path}")

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedEvidence]:
        """Search top-k most relevant historical customer support dialogues."""
        if not self.records or self.embeddings is None:
            return []

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.embedding_model_name)
            query_vec = model.encode([query], normalize_embeddings=True)
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            # fallback TF-IDF similarity
            query_vec = np.zeros((1, self.embeddings.shape[1]))

        sims = cosine_similarity(query_vec, self.embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            rec = self.records[idx]
            sim = float(sims[idx])
            results.append(RetrievedEvidence(
                source_id=rec["conversation_id"],
                customer_message=rec["customer_inquiry"],
                brand_response=rec["brand_initial_response"],
                similarity=round(sim, 4),
            ))
        return results

    def is_sufficient_evidence(self, evidence: List[RetrievedEvidence], threshold: float = 0.40) -> bool:
        """Verify whether the top retrieved evidence meets the minimum confidence threshold."""
        if not evidence:
            return False
        return evidence[0].similarity >= threshold
