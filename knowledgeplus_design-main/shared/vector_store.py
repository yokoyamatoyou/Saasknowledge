import os
import json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

class VectorStore:
    """シンプルなベクトルストアクラス"""
    
    def __init__(self, kb_path):
        """
        ベクトルストアの初期化
        
        Args:
            kb_path: ナレッジベースのパス
        """
        self.kb_path = Path(kb_path)
        self.chunks_path = self.kb_path / "chunks"
        self.metadata_path = self.kb_path / "metadata"
        self.embeddings_path = self.kb_path / "embeddings"
        
        # データを読み込み
        self.chunks = self._load_chunks()
        self.embeddings = self._load_embeddings()
    
    def _load_chunks(self):
        """チャンクテキストとメタデータを読み込み"""
        chunks = {}
        
        if self.chunks_path.exists():
            for chunk_file in self.chunks_path.glob("*.txt"):
                chunk_id = chunk_file.stem
                
                # チャンクテキストの読み込み
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_text = f.read()
                
                # メタデータの読み込み
                metadata = {}
                metadata_file = self.metadata_path / f"{chunk_id}.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                
                chunks[chunk_id] = {
                    'text': chunk_text,
                    'metadata': metadata
                }
        
        return chunks
    
    def _load_embeddings(self):
        """埋め込みベクトルを読み込み"""
        embeddings = {}
        
        if self.embeddings_path.exists():
            for emb_file in self.embeddings_path.glob("*.pkl"):
                chunk_id = emb_file.stem
                
                with open(emb_file, 'rb') as f:
                    embeddings[chunk_id] = pickle.load(f)
        
        return embeddings
    
    def search(self, query_vector, top_k=5, threshold=0.6):
        """
        クエリベクトルに最も類似するチャンクを検索
        
        Args:
            query_vector: クエリの埋め込みベクトル
            top_k: 返す結果の最大数
            threshold: 最小類似度閾値
        
        Returns:
            類似度順にソートされた検索結果のリスト
        """
        results = []
        
        for chunk_id, embedding in self.embeddings.items():
            if chunk_id in self.chunks:
                # コサイン類似度を計算
                similarity = cosine_similarity([query_vector], [embedding])[0][0]
                
                if similarity >= threshold:
                    results.append({
                        'id': chunk_id,
                        'text': self.chunks[chunk_id]['text'],
                        'metadata': self.chunks[chunk_id]['metadata'],
                        'similarity': float(similarity)
                    })
        
        # 類似度でソート（降順）
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # top_k件を返す
        return results[:top_k]


def initialize_vector_store(kb_path):
    """
    ベクトルストアを初期化
    
    Args:
        kb_path: ナレッジベースのパス
    
    Returns:
        VectorStoreインスタンス
    """
    return VectorStore(kb_path)
