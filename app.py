import io
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime
from collections import Counter

# NLP
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42
import nltk

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# ML
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from scipy.sparse import hstack, csr_matrix

import joblib

# ===================== CONFIGURACIÓN MEJORADA =====================
st.set_page_config(
    page_title="Social Trends AI Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejor apariencia
st.markdown("""
<style>
    /* Títulos más elegantes */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Métricas mejoradas */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Sidebar más atractivo */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Botones mejorados */
    .stButton>button {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Tabs personalizados */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        background-color: rgba(102, 126, 234, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Dataframes mejorados */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ===================== FUNCIONES AUXILIARES =====================
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
    df["user"] = df["user"].astype(str)
    df["text"] = df["text"].astype(str)
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def detect_lang_safe(text):
    try:
        return detect(text)
    except:
        return "unk"

def build_normalizer(lang_code):
    lang_map = {"es":"spanish", "en":"english"}
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
    stemmer, stop = build_normalizer(lang if lang in ("es","en") else "en")
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9_]+", txt.lower())
    tokens = [t for t in tokens if t not in stop and len(t)>2]
    tokens = [stemmer.stem(t) for t in tokens]
    htokens = [f"#{h}" for h in hs]
    return " ".join(tokens + htokens)

def compute_metrics(df):
    df["hashtag_list"] = df["hashtags"].apply(parse_hashtags)
    df["hashtag_count"] = df["hashtag_list"].apply(len)
    df["lang"] = df["text"].fillna("").apply(detect_lang_safe)
    df["log_likes"] = np.log1p(df["likes"].astype(float))
    
    approx_tokens = df["text"].fillna("").apply(lambda s: max(1, len(re.findall(r"[A-Za-zÀ-ÿ0-9_]+", s))))
    df["hashtag_density"] = df["hashtag_count"] / approx_tokens
    df["popularity_score"] = df["log_likes"] * (1 + 0.15 * df["hashtag_count"])
    
    def zscore_user(group):
        x = group["likes"].to_numpy(dtype=float)
        mu = x.mean()
        sd = x.std() if x.std() > 0 else 1.0
        return (x - mu) / sd
    
    df["engagement_z"] = df.groupby("user", group_keys=False).apply(zscore_user)
    df["text_plus"] = df.apply(normalize_text, axis=1)
    return df

def explode_hashtags(df):
    exploded = df.explode("hashtag_list")
    exploded = exploded.dropna(subset=["hashtag_list"])
    return exploded

def label_by_strategy(df, strategy, value):
    if strategy == "percentil":
        thr = df["likes"].quantile(value)
        y = np.where(df["likes"] >= thr, "alta", "baja")
        detail = f"percentil={int(value*100)}% → umbral={thr:.0f} likes"
    elif strategy == "absoluto":
        thr = float(value)
        y = np.where(df["likes"] >= thr, "alta", "baja")
        detail = f"likes ≥ {thr:.0f}"
    elif strategy == "zscore":
        thr = float(value)
        y = np.where(df["engagement_z"] >= thr, "alta", "baja")
        detail = f"engagement_z ≥ {thr:.2f}"
    else:
        raise ValueError("Estrategia no válida.")
    df = df.copy()
    df["label"] = y
    return df, detail

def daily_norm_series(exploded):
    exploded = exploded.copy()
    exploded["date"] = exploded["timestamp"].dt.date
    daily_total = exploded.groupby("date").size().rename("all_posts")
    per_tag = exploded.groupby(["date","hashtag_list"]).size().rename("count").reset_index()
    per_tag = per_tag.merge(daily_total, on="date", how="left")
    per_tag["norm"] = per_tag["count"] / per_tag["all_posts"].clip(lower=1)
    return per_tag

def moving_average(series, k=7):
    return series.rolling(k, min_periods=1).mean()

def ensure_nltk_min():
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download('stopwords')

# ===================== HEADER =====================
st.markdown('<h1 class="main-title">🚀 Social Trends AI Analyzer</h1>', unsafe_allow_html=True)
st.markdown("**Análisis avanzado de redes sociales con IA** | Pandas + NumPy + ML + Visualizaciones interactivas")
st.divider()

# ===================== SIDEBAR MEJORADO =====================
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    
    with st.expander("📁 Cargar Datos", expanded=True):
        files = st.file_uploader(
            "Subir CSV(s)", 
            type=["csv"], 
            accept_multiple_files=True,
            help="Formato: user, text, hashtags, likes, timestamp"
        )
        if files:
            st.success(f"✅ {len(files)} archivo(s) cargado(s)")
    
    st.divider()
    
    with st.expander("🎯 Parámetros de Análisis", expanded=True):
        top_n = st.slider("Top N elementos", 5, 50, 15, 5)
    
    st.divider()
    
    with st.expander("🏷️ Clasificación Alta/Baja", expanded=True):
        label_mode = st.selectbox(
            "Estrategia",
            ["percentil", "absoluto", "zscore"],
            help="Método para clasificar relevancia"
        )
        
        if label_mode == "percentil":
            label_value = st.slider("Percentil", 0.50, 0.95, 0.75, 0.05)
        elif label_mode == "absoluto":
            label_value = st.number_input("Umbral de likes", min_value=0, value=100)
        else:
            label_value = st.slider("Z-score", -2.0, 3.0, 1.0, 0.1)
    
    st.divider()
    
    with st.expander("🤖 Modelo de IA", expanded=True):
        model_name = st.selectbox(
            "Algoritmo",
            ["LogisticRegression", "LinearSVC", "RandomForest"]
        )
        use_grid = st.checkbox("GridSearch CV", value=False)
    
    st.divider()
    
    want_csv = st.checkbox("📥 Habilitar descargas", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 Leyenda")
    st.markdown("""
    - **Engagement Z**: Desviación estándar por usuario
    - **Popularity Score**: log(likes) × hashtags
    - **Hashtag Density**: hashtags/tokens
    """)

# ===================== VALIDACIÓN DE DATOS =====================
if not files:
    st.info("👆 Sube al menos un archivo CSV para comenzar el análisis")
    st.markdown("### 📋 Formato esperado del CSV:")
    st.code("""
user,text,hashtags,likes,timestamp
john_doe,"Post sobre IA",ai machinelearning,150,2024-01-15 10:30:00
jane_smith,"Tutorial de Python",python coding,200,2024-01-16 14:20:00
    """, language="csv")
    st.stop()

# ===================== CARGA Y PROCESAMIENTO =====================
with st.spinner("🔄 Procesando datos..."):
    dfs = []
    for f in files:
        df_raw = pd.read_csv(f)
        df_raw["__source"] = f.name
        try:
            df_i = ensure_schema(df_raw)
        except Exception as e:
            st.error(f"❌ Error en [{f.name}]: {e}")
            st.stop()
        dfs.append(df_i)
    
    df = pd.concat(dfs, ignore_index=True)
    df = compute_metrics(df)
    
st.success("✅ Datos procesados correctamente")

# ===================== TABS PRINCIPALES =====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", "📈 Tendencias", "🧠 Modelado IA", "🔮 Predicción", "📂 Campañas"
])

# ===================== TAB 1: DASHBOARD =====================
with tab1:
    st.markdown("## 📊 Resumen General")
    
    # Métricas principales con iconos
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📝 Posts Totales",
            value=f"{len(df):,}",
            delta=f"{len(df)//7} posts/semana (aprox)"
        )
    
    with col2:
        st.metric(
            label="👥 Usuarios Únicos",
            value=f"{df['user'].nunique():,}",
            delta=f"{len(df)/df['user'].nunique():.1f} posts/usuario"
        )
    
    with col3:
        st.metric(
            label="❤️ Likes Totales",
            value=f"{int(df['likes'].sum()):,}",
            delta=f"{int(df['likes'].mean()):.0f} promedio"
        )
    
    with col4:
        hashtag_unicos = len(set([h for hs in df['hashtag_list'] for h in hs]))
        st.metric(
            label="#️⃣ Hashtags Únicos",
            value=f"{hashtag_unicos:,}",
            delta=f"{df['hashtag_count'].mean():.1f} por post"
        )
    
    st.divider()
    
    # Gráficos interactivos con Plotly
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 🏆 Top Hashtags")
        exploded = explode_hashtags(df)
        top_hashtags = exploded.groupby("hashtag_list").size().sort_values(ascending=False).head(top_n)
        
        fig = px.bar(
            x=top_hashtags.values,
            y=[f"#{h}" for h in top_hashtags.index],
            orientation='h',
            labels={'x': 'Menciones', 'y': 'Hashtag'},
            color=top_hashtags.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            showlegend=False,
            height=400,
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown("### 👤 Usuarios Más Activos")
        top_users = df["user"].value_counts().head(top_n)
        
        fig = px.pie(
            values=top_users.values,
            names=top_users.index,
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Distribución de idiomas
    col_lang, col_likes = st.columns(2)
    
    with col_lang:
        st.markdown("### 🌍 Distribución de Idiomas")
        lang_counts = df["lang"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=lang_counts.index,
            values=lang_counts.values,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_likes:
        st.markdown("### 📊 Distribución de Likes")
        fig = px.histogram(
            df,
            x="likes",
            nbins=30,
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(
            xaxis_title="Likes",
            yaxis_title="Frecuencia",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Tabla de datos
    st.markdown("### 📋 Vista Previa de Datos")
    display_cols = ["user", "text", "hashtags", "likes", "lang", "popularity_score", "timestamp"]
    st.dataframe(
        df[display_cols].head(20),
        use_container_width=True,
        hide_index=True
    )
    
    if want_csv:
        buff = io.StringIO()
        df.to_csv(buff, index=False)
        st.download_button(
            "⬇️ Descargar Dataset Completo",
            buff.getvalue(),
            file_name="dataset_enriquecido.csv",
            mime="text/csv"
        )

# ===================== TAB 2: TENDENCIAS =====================
with tab2:
    st.markdown("## 📈 Análisis de Tendencias Temporales")
    
    min_date = pd.to_datetime(df["timestamp"].min())
    max_date = pd.to_datetime(df["timestamp"].max())
    
    if pd.isna(min_date) or pd.isna(max_date):
        st.warning("⚠️ No hay timestamps válidos")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            date_range = st.date_input(
                "📅 Rango de Fechas",
                (min_date.date(), max_date.date())
            )
        
        with col2:
            k = st.slider("🔄 Media Móvil (días)", 3, 30, 7, 1)
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            d1, d2 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
            dff = df[(df["timestamp"] >= d1) & (df["timestamp"] < d2)].copy()
        else:
            dff = df.copy()
        
        exploded_f = explode_hashtags(dff)
        freq = exploded_f["hashtag_list"].value_counts()
        default_tags = list(freq.head(5).index)
        options = list(freq.index)
        
        selected_tags = st.multiselect(
            "🏷️ Seleccionar Hashtags",
            options=options,
            default=default_tags
        )
        
        if selected_tags:
            per_tag = daily_norm_series(exploded_f)
            
            # Gráfico interactivo con Plotly
            fig = go.Figure()
            
            for tag in selected_tags:
                t = per_tag[per_tag["hashtag_list"] == tag].sort_values("date")
                t["ma"] = moving_average(t["norm"], k=k)
                
                fig.add_trace(go.Scatter(
                    x=t["date"],
                    y=t["ma"],
                    mode='lines+markers',
                    name=f"#{tag}",
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title="Tendencia Normalizada con Media Móvil",
                xaxis_title="Fecha",
                yaxis_title="Proporción de Posts",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla de estadísticas
            st.markdown("### 📊 Estadísticas por Hashtag")
            stats_data = []
            for tag in selected_tags:
                t = per_tag[per_tag["hashtag_list"] == tag]
                stats_data.append({
                    "Hashtag": f"#{tag}",
                    "Total Posts": len(t),
                    "Promedio Diario": f"{t['norm'].mean():.4f}",
                    "Máximo": f"{t['norm'].max():.4f}",
                    "Tendencia": "📈" if t['norm'].iloc[-1] > t['norm'].mean() else "📉"
                })
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

# ===================== TAB 3: MODELADO IA =====================
with tab3:
    st.markdown("## 🧠 Entrenamiento del Modelo de IA")
    
    ensure_nltk_min()
    
    df_lab, detail = label_by_strategy(df, label_mode, label_value)
    
    st.info(f"📌 **Estrategia de etiquetado:** {label_mode} ({detail})")
    
    # Distribución de clases
    col1, col2 = st.columns([1, 2])
    
    with col1:
        label_dist = df_lab["label"].value_counts()
        fig = px.pie(
            values=label_dist.values,
            names=label_dist.index,
            color=label_dist.index,
            color_discrete_map={'alta': '#667eea', 'baja': '#764ba2'},
            hole=0.4
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Distribución de Clases")
        st.dataframe(
            label_dist.rename_axis("Clase").to_frame("Cantidad"),
            use_container_width=True
        )
        balance = min(label_dist.values) / max(label_dist.values)
        if balance < 0.3:
            st.warning("⚠️ Dataset desbalanceado. Considera ajustar el umbral.")
        else:
            st.success(f"✅ Balance aceptable ({balance:.2%})")
    
    st.divider()
    
    # Preparación de datos
    with st.spinner("🔧 Preparando datos para entrenamiento..."):
        tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2), min_df=2)
        X_text = tfidf.fit_transform(df_lab["text_plus"].fillna(""))
        
        X_num = df_lab[["popularity_score", "hashtag_count", "engagement_z", 
                        "hashtag_density", "log_likes"]].to_numpy(dtype=float)
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
        
        X = hstack([X_text, csr_matrix(X_num_scaled)])
        y = (df_lab["label"] == "alta").astype(int).to_numpy()
    
    if y.sum() == 0 or y.sum() == len(y):
        st.error("❌ Todas las etiquetas son iguales. Ajusta la estrategia.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenamiento
        with st.spinner(f"🚀 Entrenando modelo {model_name}..."):
            if model_name == "LogisticRegression":
                base = LogisticRegression(max_iter=300)
                grid = {"C":[0.1,1,3], "penalty":["l2"], "class_weight":[None,"balanced"]}
            elif model_name == "LinearSVC":
                base = LinearSVC()
                grid = {"C":[0.1,1,3], "class_weight":[None,"balanced"]}
            else:
                base = RandomForestClassifier(random_state=42)
                grid = {"n_estimators":[200,400], "max_depth":[None,10,20], 
                       "min_samples_split":[2,5]}
            
            if use_grid:
                clf = GridSearchCV(base, grid, scoring="f1", cv=3, n_jobs=-1)
            else:
                clf = base
            
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
        
        st.success("✅ Modelo entrenado exitosamente")
        
        # Métricas
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Accuracy", f"{acc:.3f}")
        col2.metric("📊 F1-Score", f"{f1:.3f}")
        col3.metric("📈 Modelo", model_name)
        
        st.divider()
        
        # Visualizaciones
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 📋 Reporte de Clasificación")
            report = classification_report(y_test, pred, target_names=["baja","alta"])
            st.code(report, language="text")
        
        with col_right:
            st.markdown("### 🎯 Matriz de Confusión")
            cm = confusion_matrix(y_test, pred)
            
            fig = px.imshow(
                cm,
                labels=dict(x="Predicción", y="Real", color="Cantidad"),
                x=['Baja', 'Alta'],
                y=['Baja', 'Alta'],
                color_continuous_scale='Blues',
                text_auto=True
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Guardar modelo
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Guardar Modelo", use_container_width=True):
                model_pack = {
                    "clf": clf,
                    "tfidf": tfidf,
                    "scaler": scaler,
                    "label_mode": label_mode,
                    "label_value": label_value,
                }
                joblib.dump(model_pack, "model_pack.joblib")
                st.success("✅ Modelo guardado")
                
                with open("model_pack.joblib", "rb") as f:
                    st.download_button(
                        "⬇️ Descargar Modelo",
                        f,
                        file_name="model_pack.joblib",
                        use_container_width=True
                    )
        
        with col2:
            uploaded_model = st.file_uploader("📂 Cargar Modelo", type=["joblib"])
            if uploaded_model:
                model_pack = joblib.load(uploaded_model)
                st.success("✅ Modelo cargado")
                st.session_state["loaded_model"] = model_pack

# ===================== TAB 4: PREDICCIÓN =====================
with tab4:
    st.markdown("## 🔮 Predicción de Relevancia")
    
    model_pack = st.session_state.get("loaded_model")
    if model_pack is None:
        possible = ["clf","tfidf","scaler","label_mode","label_value"]
        if all(name in globals() for name in possible):
            model_pack = {k: globals()[k] for k in possible}
    
    if not model_pack:
        st.info("⚠️ Entrena o carga un modelo en la pestaña 'Modelado IA'")
    else:
        st.markdown("### ✍️ Ingresa un nuevo post")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            example_text = st.text_area(
                "Texto del post",
                "Análisis de datos con Python: numpy, pandas y visualizaciones interactivas 🚀",
                height=100
            )
        
        with col2:
            example_ht = st.text_input(
                "Hashtags",
                "python, datascience, analytics"
            )
        
        if st.button("🔮 Predecir Relevancia", use_container_width=True):
            with st.spinner("Analizando..."):
                hs = [h.strip().lower() for h in re.split(r"[#,;\s]+", example_ht) if h.strip()]
                lang = detect_lang_safe(example_text)
                stemmer, stop = build_normalizer(lang if lang in ("es","en") else "en")
                tokens = re.findall(r"[A-Za-zÀ-ÿ0-9_]+", example_text.lower())
                tokens = [t for t in tokens if t not in stop and len(t)>2]
                tokens = [stemmer.stem(t) for t in tokens]
                text_plus = " ".join(tokens + [f"#{h}" for h in hs])
                
                tfidf = model_pack["tfidf"]
                scaler = model_pack["scaler"]
                clf = model_pack["clf"]
                
                X_text_new = tfidf.transform([text_plus])
                X_num_new = np.array([[0.0, len(hs), 0.0, len(hs)/max(1,len(tokens)+len(hs)), 0.0]])
                X_num_new_scaled = scaler.transform(X_num_new)
                X_new = hstack([csr_matrix(X_text_new), csr_matrix(X_num_new_scaled)])
            
            # Resultado con diseño mejorado
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_new)[0,1]
                label = "alta" if proba >= 0.5 else "baja"
                
                st.markdown("### 🎯 Resultado de la Predicción")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if label == "alta":
                        st.success(f"### ✅ RELEVANCIA ALTA")
                    else:
                        st.warning(f"### ⚠️ RELEVANCIA BAJA")
                
                with col2:
                    st.metric("Probabilidad Alta", f"{proba:.1%}")
                
                with col3:
                    confidence = max(proba, 1-proba)
                    st.metric("Confianza", f"{confidence:.1%}")
                
                # Gráfico de probabilidad
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = proba * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probabilidad de Alta Relevancia"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 30], 'color': "#ffcccc"},
                            {'range': [30, 70], 'color': "#ffffcc"},
                            {'range': [70, 100], 'color': "#ccffcc"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Análisis del post
                st.markdown("### 📊 Análisis del Post")
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    st.markdown("**Características detectadas:**")
                    st.markdown(f"- 🔤 Idioma: `{lang.upper()}`")
                    st.markdown(f"- #️⃣ Hashtags: `{len(hs)}`")
                    st.markdown(f"- 📝 Tokens: `{len(tokens)}`")
                    st.markdown(f"- 📏 Densidad hashtags: `{len(hs)/max(1,len(tokens)+len(hs)):.3f}`")
                
                with analysis_col2:
                    st.markdown("**Recomendaciones:**")
                    if len(hs) < 3:
                        st.markdown("- ⚠️ Agrega más hashtags (ideal: 3-5)")
                    elif len(hs) > 7:
                        st.markdown("- ⚠️ Reduce hashtags (pueden parecer spam)")
                    else:
                        st.markdown("- ✅ Buena cantidad de hashtags")
                    
                    if len(tokens) < 10:
                        st.markdown("- ⚠️ El texto es muy corto")
                    else:
                        st.markdown("- ✅ Longitud de texto adecuada")
            
            else:
                pred = clf.predict(X_new)[0]
                label = "ALTA" if pred == 1 else "BAJA"
                
                if pred == 1:
                    st.success(f"### ✅ RELEVANCIA {label}")
                else:
                    st.warning(f"### ⚠️ RELEVANCIA {label}")

# ===================== TAB 5: COMPARAR CAMPAÑAS =====================
with tab5:
    st.markdown("## 📂 Comparativa de Campañas")
    
    if "__source" not in df.columns or df["__source"].nunique() == 1:
        st.info("ℹ️ Sube múltiples archivos para comparar campañas")
    else:
        # Resumen por campaña
        agg = df.groupby("__source").agg(
            posts=("user","count"),
            users=("user","nunique"),
            likes_tot=("likes","sum"),
            likes_med=("likes","mean"),
            htags_med=("hashtag_count","mean"),
            pop_med=("popularity_score","mean")
        ).sort_values("posts", ascending=False)
        
        st.markdown("### 📊 Métricas por Campaña")
        
        # Gráficos comparativos
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                agg,
                x=agg.index,
                y="posts",
                title="Posts por Campaña",
                color="posts",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(xaxis_title="Campaña", yaxis_title="Posts")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                agg,
                x=agg.index,
                y="likes_tot",
                title="Likes Totales por Campaña",
                color="likes_tot",
                color_continuous_scale="Blues"
            )
            fig.update_layout(xaxis_title="Campaña", yaxis_title="Likes")
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabla detallada
        st.markdown("### 📋 Tabla Comparativa")
        agg_display = agg.copy()
        agg_display.columns = ["Posts", "Usuarios", "Likes Total", "Likes Promedio", 
                               "Hashtags Promedio", "Popularidad Promedio"]
        agg_display = agg_display.round(2)
        st.dataframe(agg_display, use_container_width=True)
        
        st.divider()
        
        # Top hashtags por campaña
        st.markdown("### 🏷️ Top Hashtags por Campaña")
        
        src_sel = st.selectbox(
            "Seleccionar campaña",
            sorted(df["__source"].unique().tolist())
        )
        
        exploded_all = explode_hashtags(df)
        top_by_src = (exploded_all.groupby(["__source","hashtag_list"])
                     .size().rename("count")
                     .reset_index()
                     .sort_values(["__source","count"], ascending=[True,False]))
        
        top_src = top_by_src[top_by_src["__source"]==src_sel].head(20)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                top_src,
                x="count",
                y=[f"#{h}" for h in top_src["hashtag_list"]],
                orientation='h',
                title=f"Top Hashtags en {src_sel}",
                color="count",
                color_continuous_scale="Plasma"
            )
            fig.update_layout(
                yaxis={'categoryorder':'total ascending'},
                xaxis_title="Menciones",
                yaxis_title="Hashtag"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"**Estadísticas de {src_sel}:**")
            campaign_data = df[df["__source"] == src_sel]
            st.metric("Posts", f"{len(campaign_data):,}")
            st.metric("Engagement Promedio", f"{campaign_data['likes'].mean():.1f}")
            st.metric("Hashtags Únicos", 
                     f"{len(set([h for hs in campaign_data['hashtag_list'] for h in hs])):,}")
        
        # Comparación temporal
        st.divider()
        st.markdown("### 📈 Evolución Temporal por Campaña")
        
        df_with_date = df.copy()
        df_with_date["date"] = df_with_date["timestamp"].dt.date
        
        daily_by_campaign = df_with_date.groupby(["date", "__source"]).size().reset_index(name="posts")
        
        fig = px.line(
            daily_by_campaign,
            x="date",
            y="posts",
            color="__source",
            title="Posts Diarios por Campaña",
            markers=True
        )
        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Cantidad de Posts",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Descarga
        if want_csv:
            buff = io.StringIO()
            agg.to_csv(buff)
            st.download_button(
                "⬇️ Descargar Comparativa (CSV)",
                buff.getvalue(),
                file_name="campanias_resumen.csv",
                mime="text/csv",
                use_container_width=True
            )

# ===================== FOOTER =====================
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 <strong>Social Trends AI Analyzer</strong> | Análisis Avanzado de Redes Sociales</p>
    <p>Powered by Pandas • NumPy • Scikit-learn • Plotly • Streamlit</p>
</div>
""", unsafe_allow_html=True)