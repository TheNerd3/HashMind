import numpy as np
import streamlit as st

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False


@st.cache_resource
def build_faiss_index(embeddings: np.ndarray, metric: str = 'ip'):
    # embeddings: (N, D) float32
    if embeddings is None or len(embeddings) == 0:
        return None
    emb = embeddings.astype('float32')
    if FAISS_AVAILABLE:
        if metric == 'ip':
            index = faiss.IndexFlatIP(emb.shape[1])
        else:
            index = faiss.IndexFlatL2(emb.shape[1])
        index.add(emb)
        return index
    else:
        # store embeddings and fall back to numpy search
        return emb


def search_index(query_emb, index, top_k=5):
    if index is None:
        return np.array([], dtype=int), np.array([], dtype=float)
    q = np.asarray(query_emb, dtype='float32')
    if FAISS_AVAILABLE and hasattr(index, 'search'):
        if q.ndim == 1:
            q = q.reshape(1, -1)
        distances, indices = index.search(q, top_k)
        # For IP metric, distances are similarity
        return indices[0], distances[0]
    else:
        # index is embeddings matrix
        emb = index
        sims = (emb @ q.T).ravel()
        top_idx = np.argsort(sims)[-top_k:][::-1]
        return top_idx, sims[top_idx]


def buscar_posts_similares(embedding_query, embeddings_db, df, top_k=5):
    """Search for similar posts in the database.
    
    Args:
        embedding_query: query embedding vector
        embeddings_db: numpy array of embeddings or None
        df: reference dataframe
        top_k: number of top results
    
    Returns:
        DataFrame with top-k similar posts
    """
    if embeddings_db is None or len(embeddings_db) == 0:
        return df.head(0)
    
    # Build/use index
    if isinstance(embeddings_db, np.ndarray):
        index = build_faiss_index(embeddings_db)
    else:
        # Assume it's already an index
        index = embeddings_db
    
    idxs, scores = search_index(embedding_query, index, top_k=top_k)
    if len(idxs) == 0:
        return df.head(0)
    top_indices = np.array(idxs, dtype=int)
    # Ensure indices are within bounds
    top_indices = top_indices[top_indices < len(df)]
    if len(top_indices) == 0:
        return df.head(0)
    similar_posts = df.iloc[top_indices].copy()
    similar_posts['similarity_score'] = scores[:len(top_indices)]
    return similar_posts
