import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import streamlit as st

HASHTAG_SPLIT = re.compile(r"[#,;\s]+")


def parse_hashtags(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    toks = [t.strip().lower() for t in HASHTAG_SPLIT.split(str(s)) if t.strip()]
    return toks


def ensure_schema(df):
    expected = {"user", "text", "hashtags", "likes", "timestamp"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"❌ Faltan columnas: {', '.join(sorted(missing))}")
    df = df.copy()
    df["user"] = df["user"].astype(str)
    df["text"] = df["text"].astype(str)
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def detect_lang_safe(text):
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "unk"


def build_normalizer(lang_code):
    lang_map = {"es": "spanish", "en": "english"}
    snow_lang = lang_map.get(lang_code, "english")
    stemmer = SnowballStemmer(snow_lang)
    try:
        stop = set(stopwords.words(snow_lang))
    except LookupError:
        nltk.download('stopwords')
        stop = set(stopwords.words(snow_lang))
    return stemmer, stop


def normalize_text(row):
    txt = (row.get("text") or "")
    hs = row.get("hashtag_list") or []
    lang = row.get("lang", "en")
    stemmer, stop = build_normalizer(lang if lang in ("es", "en") else "en")
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9_]+", txt.lower())
    tokens = [t for t in tokens if t not in stop and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    htokens = [f"#{h}" for h in hs]
    return " ".join(tokens + htokens)


@st.cache_data
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hashtag_list"] = df["hashtags"].apply(parse_hashtags)
    df["hashtag_count"] = df["hashtag_list"].apply(len)
    df["lang"] = df["text"].fillna("").apply(detect_lang_safe)
    df["log_likes"] = np.log1p(df["likes"].astype(float))

    approx_tokens = df["text"].fillna("").apply(lambda s: max(1, len(re.findall(r"[A-Za-zÀ-ÿ0-9_]+", s))))
    df["hashtag_density"] = df["hashtag_count"] / approx_tokens
    df["popularity_score"] = df["log_likes"] * (1 + 0.15 * df["hashtag_count"])

    def zscore_user(group):
        x = group["likes"].to_numpy(dtype=float)
        if len(x) <= 1:
            return np.zeros_like(x)
        mu = x.mean()
        sd = x.std()
        if sd == 0 or np.isnan(sd) or np.isinf(sd):
            return np.zeros_like(x)
        result = (x - mu) / sd
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    df["engagement_z"] = df.groupby("user", group_keys=False).apply(zscore_user)
    df["text_plus"] = df.apply(normalize_text, axis=1)

    numeric_cols = ["log_likes", "hashtag_density", "popularity_score", "engagement_z", "hashtag_count"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df[col] = df[col].replace([np.inf, -np.inf], 0)

    return df


@st.cache_data
def explode_hashtags(df: pd.DataFrame) -> pd.DataFrame:
    exploded = df.explode("hashtag_list")
    exploded = exploded.dropna(subset=["hashtag_list"])
    return exploded


def ensure_nltk_min():
    """Ensure minimal NLTK data is available (stopwords).
    
    Downloads stopwords if not already present. Shows progress to user via Streamlit.
    """
    try:
        from nltk.corpus import stopwords
        _ = stopwords.words('english')
        return True
    except Exception:
        try:
            import nltk
            # Show progress to user
            progress_placeholder = st.empty()
            progress_placeholder.info("📥 Descargando recursos NLTK (primera vez)...")
            nltk.download('stopwords', quiet=True)
            progress_placeholder.success("✅ Recursos NLTK descargados")
            return True
        except Exception as e:
            # best-effort, caller will handle missing resources
            st.warning(f"⚠️ No se pudieron descargar recursos NLTK: {str(e)}")
            return False


def daily_norm_series(exploded_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with columns: date, hashtag_list, count, norm

    norm is the proportion of posts that day that include the hashtag.
    exploded_df must contain 'timestamp' and 'hashtag_list'.
    """
    if exploded_df is None or exploded_df.empty:
        return pd.DataFrame(columns=['date', 'hashtag_list', 'count', 'norm'])

    df = exploded_df.copy()
    if 'timestamp' not in df.columns:
        return pd.DataFrame(columns=['date', 'hashtag_list', 'count', 'norm'])
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    # count occurrences per date/hashtag
    grp = df.groupby(['date', 'hashtag_list']).size().reset_index(name='count')
    # total posts per date (unique post ids per date) - approximate by counting rows per date in original df
    total_per_date = df.drop_duplicates(subset=['user','text','timestamp']).groupby('date').size().rename('total')
    grp = grp.join(total_per_date, on='date')
    grp['norm'] = grp['count'] / grp['total'].replace({0:1})
    return grp[['date','hashtag_list','count','norm']]


def moving_average(series, k=7):
    """Compute moving average of a pandas Series or array-like."""
    try:
        return pd.Series(series).rolling(window=k, min_periods=1).mean().to_numpy()
    except Exception:
        # fallback simple numpy
        arr = np.asarray(series, dtype=float)
        if len(arr) == 0:
            return arr
        out = np.convolve(arr, np.ones(k)/k, mode='same')
        return out


def label_by_strategy(df: pd.DataFrame, mode: str, value) -> tuple:
    """Label posts as 'alta' or 'baja' according to strategy.

    Returns (df_labeled, detail_dict)
    """
    df2 = df.copy()
    # Ensure metrics exist
    if 'popularity_score' not in df2.columns or 'likes' not in df2.columns:
        try:
            df2 = compute_metrics(df2)
        except Exception:
            pass

    detail = {}
    mode = (mode or 'percentil').lower()
    if mode == 'percentil':
        q = float(value) if value is not None else 0.75
        thresh = df2['likes'].quantile(q)
        df2['label'] = np.where(df2['likes'] >= thresh, 'alta', 'baja')
        detail['threshold'] = float(thresh)
        detail['mode'] = 'percentil'
        detail['q'] = q
    elif mode == 'absoluto':
        thresh = float(value) if value is not None else 100.0
        df2['label'] = np.where(df2['likes'] >= thresh, 'alta', 'baja')
        detail['threshold'] = thresh
        detail['mode'] = 'absoluto'
    else:
        # zscore: treat posts with engagement_z greater than value as alta
        try:
            zthr = float(value)
            if 'engagement_z' not in df2.columns:
                df2 = compute_metrics(df2)
            df2['label'] = np.where(df2['engagement_z'] >= zthr, 'alta', 'baja')
            detail['threshold'] = zthr
            detail['mode'] = 'zscore'
        except Exception:
            # fallback: use median
            med = df2['likes'].median()
            df2['label'] = np.where(df2['likes'] >= med, 'alta', 'baja')
            detail['threshold'] = float(med)
            detail['mode'] = 'median_fallback'

    return df2, detail
