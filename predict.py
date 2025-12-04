import numpy as np
import streamlit as st
from scipy.sparse import hstack, csr_matrix
from similarity import buscar_posts_similares
from embeddings import generar_embeddings_texto


def predecir_hashtags_mejorado(model_pack, texto, embeddings_model, df_reference, top_k=15):
    clf = model_pack['hashtag_predictor']
    mlb = model_pack['mlb']

    if embeddings_model is not None:
        embedding = generar_embeddings_texto([texto], embeddings_model)[0]
    else:
        embedding = np.zeros(768, dtype=float)

    # Text features (same as training in models.py)
    text_len = len(texto)
    word_count = len(texto.split())
    has_url = 1 if 'http' in texto.lower() else 0
    has_mention = 1 if '@' in texto else 0
    emoji_count = sum(1 for c in texto if ord(c) > 127)
    upper_ratio = sum(1 for c in texto if c.isupper()) / max(len(texto), 1)

    text_features = np.array([
        text_len,
        word_count,
        has_url,
        has_mention,
        emoji_count,
        upper_ratio
    ])

    # Numeric features (same as training: popularity_score, hashtag_count, engagement_z, log_likes, hashtag_density)
    numeric_features = np.array([
        df_reference['popularity_score'].median(),
        2.0,  # hashtag_count (default: assume 2 hashtags)
        0.0,  # engagement_z (normalized, default to 0)
        df_reference['log_likes'].median(),
        0.05  # hashtag_density
    ])

    # Stack: embedding (768) + text_features (6) + numeric_features (5) = 779 dimensions
    X = np.hstack([embedding, text_features, numeric_features]).reshape(1, -1)

    y_pred = clf.predict(X)[0]

    top_indices = np.argsort(y_pred)[-top_k:][::-1]
    hashtags_pred = []
    for idx in top_indices:
        score = float(y_pred[idx])
        confidence = 'Alta' if score > 0.6 else 'Media' if score > 0.3 else 'Baja'
        hashtags_pred.append({'hashtag': mlb.classes_[idx], 'score': score, 'confidence': confidence})

    return hashtags_pred


def generar_hashtags_semanticos(texto, df, embeddings_db, embeddings_model, top_k=10):
    if embeddings_model is None:
        return []
    embedding_query = generar_embeddings_texto([texto], embeddings_model)[0]
    posts_similares = buscar_posts_similares(embedding_query, embeddings_db, df, top_k=20)
    hashtag_scores = {}
    for idx, row in posts_similares.iterrows():
        similarity = float(row.get('similarity_score', 0.0))
        engagement_factor = np.log1p(row.get('likes', 0)) / max(1.0, np.log1p(df['likes'].max()))
        combined_score = similarity * (0.7 + 0.3 * engagement_factor)
        for hashtag in row.get('hashtag_list', []) or []:
            hashtag_scores[hashtag] = hashtag_scores.get(hashtag, 0.0) + combined_score

    sorted_hashtags = sorted(hashtag_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    max_score = sorted_hashtags[0][1] if sorted_hashtags else 1
    result = []
    for hashtag, score in sorted_hashtags:
        normalized_score = score / max_score
        result.append({'hashtag': hashtag, 'score': normalized_score, 'confidence': 'Alta' if normalized_score > 0.7 else 'Media' if normalized_score > 0.4 else 'Baja'})
    return result
