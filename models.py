import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def entrenar_modelo_relevancia_mejorado(df, embeddings, label_mode, label_value):
    # This function expects 'df' already labeled (i.e., df contains a 'label' column)
    if 'label' not in df.columns:
        raise ValueError('label missing in df; call label_by_strategy before training')

    text_data = df["text_plus"].fillna("").astype(str).replace("", "empty_post")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
    X_text = tfidf.fit_transform(text_data)

    X_embeddings = embeddings

    numeric_features = df[["popularity_score", "hashtag_count", "engagement_z", 
                           "hashtag_density", "log_likes"]].fillna(0).replace([np.inf, -np.inf], 0).values

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(numeric_features)

    X = hstack([X_text, csr_matrix(X_embeddings), csr_matrix(X_num_scaled)])
    y = (df["label"] == "alta").astype(int).to_numpy()

    if y.sum() == 0 or y.sum() == len(y):
        st.error("❌ Todas las etiquetas son iguales")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    pred_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, pred),
        'f1': f1_score(y_test, pred, zero_division=0),
        'precision': precision_score(y_test, pred, zero_division=0),
        'recall': recall_score(y_test, pred, zero_division=0)
    }

    model_pack = {
        'clf': clf,
        'tfidf': tfidf,
        'scaler': scaler,
        'embeddings_dim': X_embeddings.shape[1],
        'metrics': metrics,
        'y_test': y_test,
        'y_pred': pred,
        'y_proba': pred_proba
    }

    return model_pack


def entrenar_predictor_hashtags_mejorado(df, embeddings, top_n_hashtags=50, min_hashtag_freq=3):
    import numpy as _np
    from sklearn.linear_model import Ridge
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, jaccard_score
    from preprocess import explode_hashtags

    exploded = explode_hashtags(df)
    hashtag_freq = exploded['hashtag_list'].value_counts()
    valid_hashtags = hashtag_freq[hashtag_freq >= min_hashtag_freq].head(top_n_hashtags).index.tolist()
    if len(valid_hashtags) < 5:
        st.error(f"❌ Solo {len(valid_hashtags)} hashtags válidos. Necesitas más datos.")
        return None, None, None

    df_train = df.copy()
    df_train['target_hashtags'] = df_train['hashtag_list'].apply(lambda ls: [h for h in (ls or []) if h in valid_hashtags])

    mlb = MultiLabelBinarizer(classes=valid_hashtags)
    y_multi = mlb.fit_transform(df_train['target_hashtags'])

    X_embeddings = embeddings[:len(df_train)]

    # Vectorized text features
    texts = df_train['text'].fillna("")
    text_len = texts.str.len().to_numpy().reshape(-1,1)
    word_count = texts.str.split().apply(len).to_numpy().reshape(-1,1)
    has_url = texts.str.contains('http', na=False).astype(int).to_numpy().reshape(-1,1)
    has_mention = texts.str.contains('@', na=False).astype(int).to_numpy().reshape(-1,1)
    emoji_count = texts.apply(lambda s: sum(1 for c in s if ord(c) > 127)).to_numpy().reshape(-1,1)
    upper_ratio = texts.apply(lambda s: sum(1 for c in s if c.isupper())/max(len(s),1)).to_numpy().reshape(-1,1)

    text_features = _np.hstack([text_len, word_count, has_url, has_mention, emoji_count, upper_ratio])

    numeric_features = df_train[[
        'popularity_score', 'hashtag_count', 'engagement_z', 'log_likes', 'hashtag_density'
    ]].fillna(0).replace([_np.inf, -_np.inf], 0).values

    X = _np.hstack([X_embeddings, text_features, numeric_features])

    if _np.isnan(X).any() or _np.isinf(X).any():
        st.error("❌ Hay valores NaN o Inf en las features")
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y_multi, test_size=0.2, random_state=42)

    base_clf = Ridge(alpha=1.0, random_state=42)
    clf = OneVsRestClassifier(base_clf, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict(X_test)
    threshold = 0.3
    y_pred = (y_pred_proba > threshold).astype(int)

    precision = precision_score(y_test, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
    hamming = hamming_loss(y_test, y_pred)
    try:
        jaccard = jaccard_score(y_test, y_pred, average='samples', zero_division=0)
    except Exception:
        jaccard = 0.0

    correct_predictions = (y_test * y_pred).sum(axis=1) > 0
    coverage = correct_predictions.mean()

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hamming_loss': hamming,
        'jaccard': jaccard,
        'n_hashtags': len(valid_hashtags),
        'threshold': threshold,
        'coverage': coverage
    }

    return clf, mlb, metrics
