import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.config import Config

class VectorIndex:
    def __init__(self, index_file: Path = Config.PROCESSED_DIR / "vector_index.pkl"):
        self.index_file = index_file
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, stop_words='english')
        self.doc_ids: List[str] = []
        self.chunks: List[Dict[str, Any]] = []
        self.tfidf_matrix = None

    def build_from_corpus(self, corpus_file: Path = Config.CORPUS_FILE):
        print(f"Building vector index from {corpus_file}...")
        texts = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                doc = json.loads(line)
                doc_id = doc.get('doc_id')
                title = doc.get('title', '')
                text = doc.get('text', '')

                # Chunking: Title + Infobox + first 1000 chars of body
                chunk_text = f"{title}\n{text[:1500]}"
                self.chunks.append({
                    'doc_id': doc_id,
                    'title': title,
                    'text': chunk_text
                })
                self.doc_ids.append(doc_id)
                texts.append(chunk_text)

        print(f"Vectorizing {len(texts)} chunks...")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.save()
        print(f"Vector index built and saved to {self.index_file}")

    def save(self):
        Config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.index_file, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'doc_ids': self.doc_ids,
                'chunks': self.chunks,
                'tfidf_matrix': self.tfidf_matrix
            }, f)

    def load(self) -> bool:
        if not self.index_file.exists():
            return False
        with open(self.index_file, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.doc_ids = data['doc_ids']
            self.chunks = data['chunks']
            self.tfidf_matrix = data['tfidf_matrix']
        return True

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if self.tfidf_matrix is None:
            if not self.load():
                self.build_from_corpus()

        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))
        return results

if __name__ == "__main__":
    idx = VectorIndex()
    idx.build_from_corpus()
