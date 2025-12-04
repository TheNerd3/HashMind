import numpy as np
import streamlit as st

try:
    from sentence_transformers import SentenceTransformer
    import torch
    from transformers import CLIPProcessor, CLIPModel
    EMBEDDINGS_AVAILABLE = True
except Exception:
    EMBEDDINGS_AVAILABLE = False


@st.cache_resource
def load_embedding_models():
    if not EMBEDDINGS_AVAILABLE:
        return None, None
    try:
        text_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return text_model, (clip_model, clip_processor)
    except Exception:
        return None, None


def _batch_encode(model, texts, batch_size=64):
    # batching to avoid OOM and speed up large lists
    if model is None:
        return np.zeros((len(texts), 768), dtype=np.float32)
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embs.append(emb)
    return np.vstack(embs)


@st.cache_data
def generar_embeddings_texto(texts, _model=None):
    # texts: list of str
    # Prefix model argument with underscore so Streamlit does not try to hash the model object
    return _batch_encode(_model, texts)


def generar_embedding_imagen(image_pil, clip_models):
    if clip_models is None:
        return np.zeros(512, dtype=np.float32)
    clip_model, clip_processor = clip_models
    inputs = clip_processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features.numpy().flatten()


def combinar_embeddings(text_emb, image_emb=None, text_weight=0.7):
    if image_emb is None or len(image_emb) == 0:
        return text_emb
    text_norm = text_emb / (np.linalg.norm(text_emb) + 1e-10)
    image_norm = image_emb / (np.linalg.norm(image_emb) + 1e-10)
    combined = np.concatenate([
        text_norm * text_weight,
        image_norm * (1 - text_weight)
    ])
    return combined
