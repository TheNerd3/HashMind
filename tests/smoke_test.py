import sys
import os

proj_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, proj_root)

print("Running smoke test for HashMind...")

try:
    import pandas as pd
    import numpy as np
    import streamlit as st
    print("- core libraries imported: pandas, numpy, streamlit")
except Exception as e:
    print("WARN: core import failed:", e)

try:
    from preprocess import ensure_schema, compute_metrics, parse_hashtags
    from embeddings import load_embedding_models, generar_embeddings_texto
    from similarity import buscar_posts_similares
    from models import entrenar_predictor_hashtags_mejorado
    from predict import predecir_hashtags_mejorado, generar_hashtags_semanticos
    print("- project modules imported OK")
except Exception as e:
    print("ERROR importing project modules:", e)
    raise

# Minimal dummy dataframe
df = pd.DataFrame([
    {"user": "alice", "text": "Hello world #test", "hashtags": "test", "likes": 10, "timestamp": "2024-01-01 12:00:00"},
    {"user": "bob", "text": "Another post about AI", "hashtags": "ai ml", "likes": 5, "timestamp": "2024-01-02 13:00:00"}
])

print("- ensuring schema...")
df = ensure_schema(df)
print(df.dtypes)

print("- computing metrics...")
df = compute_metrics(df)
print(df[['hashtag_list','hashtag_count','popularity_score']].head())

# Try embeddings (may fallback)
text_model, clip_models = load_embedding_models()
texts = df['text'].fillna("").tolist()
embs = generar_embeddings_texto(texts, text_model)
print(f"- generated embeddings shape: {np.array(embs).shape}")

# Semantic hashtags
res = generar_hashtags_semanticos("Testing embeddings and hashtags", df, embs, text_model, top_k=5)
print("- semantic hashtags result:", res)

print("Smoke test finished. If you saw no exceptions, core functions imported and ran.")
