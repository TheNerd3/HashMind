import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime
from PIL import Image
from scipy.sparse import csr_matrix, hstack

# Import modularized helpers
from preprocess import (
    parse_hashtags, ensure_schema, compute_metrics, explode_hashtags, daily_norm_series, moving_average, label_by_strategy, ensure_nltk_min
)
from embeddings import (
    load_embedding_models, generar_embeddings_texto, generar_embedding_imagen, combinar_embeddings
)
from similarity import buscar_posts_similares
from models import (
    entrenar_modelo_relevancia_mejorado, entrenar_predictor_hashtags_mejorado
)
from predict import predecir_hashtags_mejorado, generar_hashtags_semanticos

import joblib
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ================ CONFIG ================
st.set_page_config(
    page_title="Hashmind AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado (mantener exactamente)
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
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
    
    .hashtag-score {
        display: inline-block;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        margin: 5px;
        font-weight: 600;
    }
    
    .similar-post {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        color:#222 !important;
    }
            
</style>
""", unsafe_allow_html=True)

# ================ SESSION KEYS (compat) ================
# All session keys are initialized on-demand.

# ================ HELPERS: load CSVs (cached) ================
@st.cache_data
def load_uploaded_files(_files):
    dfs = []
    for f in _files:
        try:
            df_raw = pd.read_csv(f)
        except Exception:
            df_raw = pd.read_csv(f, encoding='latin1')
        df_raw['__source'] = getattr(f, 'name', 'uploaded')
        df_i = ensure_schema(df_raw)
        dfs.append(df_i)
    if not dfs:
        return pd.DataFrame(columns=['user','text','hashtags','likes','timestamp'])
    df = pd.concat(dfs, ignore_index=True)
    return df

# ================ SIDEBAR ================
st.markdown('<h1 class="main-title">🚀 Hashmind AI</h1>', unsafe_allow_html=True)
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
    with st.expander("🎯 Parámetros", expanded=True):
        top_n = st.slider("Top N elementos", 5, 50, 15, 5)
    st.divider()
    with st.expander("🏷️ Clasificación Alta/Baja", expanded=True):
        label_mode = st.selectbox("Estrategia", ["percentil", "absoluto", "zscore"])
        if label_mode == "percentil":
            label_value = st.slider("Percentil", 0.50, 0.95, 0.75, 0.05)
        elif label_mode == "absoluto":
            label_value = st.number_input("Umbral de likes", min_value=0, value=100)
        else:
            label_value = st.slider("Z-score", -2.0, 3.0, 1.0, 0.1)
    st.divider()
    text_model_tmp, clip_tmp = load_embedding_models()
    if text_model_tmp is not None:
        st.success("✅ Embeddings habilitados")
    else:
        st.error("❌ Instala dependencias para embeddings avanzados")

# If no files, show example and stop
if 'files' not in locals() or not files:
    st.info("👆 Sube al menos un archivo CSV para comenzar")
    st.code('''
user,text,hashtags,likes,timestamp
john_doe,"Post sobre IA",ai machinelearning,150,2024-01-15 10:30:00
    ''', language="csv")
    st.stop()

# ================ LOAD & PROCESS DATA ================
with st.spinner("🔄 Procesando datos..."):
    df = load_uploaded_files(files)
    df = compute_metrics(df)
    text_model, clip_models = load_embedding_models()
    if text_model is not None:
        texts = df['text'].fillna("").tolist()
        df_embeddings = generar_embeddings_texto(texts, text_model)
        st.session_state['df_embeddings'] = df_embeddings
    else:
        df_embeddings = np.zeros((len(df), 768), dtype=float)

st.success("✅ Datos procesados correctamente")

# ================ TABS ================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", "📈 Tendencias", "🧠 Modelos IA", "🔮 Predicción", "📂 Campañas"
])

# ---------------- TAB 1: DASHBOARD ----------------
with tab1:
    st.markdown("## 📊 Resumen General")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="📝 Posts Totales", value=f"{len(df):,}", delta=f"{len(df)//7} posts/semana (aprox)")
    with col2:
        st.metric(label="👥 Usuarios Únicos", value=f"{df['user'].nunique():,}", delta=f"{len(df)/df['user'].nunique():.1f} posts/usuario")
    with col3:
        st.metric(label="❤️ Likes Totales", value=f"{int(df['likes'].sum()):,}", delta=f"{int(df['likes'].mean()):.0f} promedio")
    with col4:
        hashtag_unicos = len(set([h for hs in df['hashtag_list'] for h in hs]))
        st.metric(label="#️⃣ Hashtags Únicos", value=f"{hashtag_unicos:,}", delta=f"{df['hashtag_count'].mean():.1f} por post")
    st.divider()
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### 🏆 Top Hashtags")
        exploded = explode_hashtags(df)
        top_hashtags = exploded.groupby("hashtag_list").size().sort_values(ascending=False).head(top_n)
        fig = px.bar(x=top_hashtags.values, y=[f"#{h}" for h in top_hashtags.index], orientation='h', labels={'x':'Menciones','y':'Hashtag'}, color=top_hashtags.values, color_continuous_scale='Viridis')
        fig.update_layout(showlegend=False, height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, width='stretch', key='plotly_1')
    with col_right:
        st.markdown("### 👤 Usuarios Más Activos")
        top_users = df["user"].value_counts().head(top_n)
        fig = px.pie(values=top_users.values, names=top_users.index, hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch', key='plotly_2')
    st.divider()
    col_lang, col_likes = st.columns(2)
    with col_lang:
        st.markdown("### 🌍 Distribución de Idiomas")
        lang_counts = df["lang"].value_counts()
        fig = go.Figure(data=[go.Pie(labels=lang_counts.index, values=lang_counts.values, marker=dict(colors=px.colors.qualitative.Set3))])
        fig.update_layout(height=350)
        st.plotly_chart(fig, width='stretch', key='plotly_3')
    with col_likes:
        st.markdown("### 📊 Distribución de Likes")
        fig = px.histogram(df, x="likes", nbins=30, color_discrete_sequence=['#667eea'])
        fig.update_layout(xaxis_title="Likes", yaxis_title="Frecuencia", height=350)
        st.plotly_chart(fig, width='stretch', key='plotly_4')
    st.divider()
    st.markdown("### 📋 Vista Previa de Datos")
    display_cols = ["user", "text", "hashtags", "likes", "lang", "popularity_score", "timestamp"]
    st.dataframe(df[display_cols].head(20), width='stretch', hide_index=True)
    st.divider()
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    st.download_button("⬇️ Descargar Dataset Completo", buff.getvalue(), file_name="dataset_enriquecido.csv", mime="text/csv", key="download_dataset_complete_tab1")

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
                (min_date.date(), max_date.date()),
                key="date_range_tab2_b"
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
            
            st.plotly_chart(fig, width='stretch', key='plotly_13')
            
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
            
            st.dataframe(pd.DataFrame(stats_data), width='stretch', hide_index=True)
            
            st.divider()
            
            # Gráfico adicional: Comparación de engagement
            st.markdown("### 💡 Engagement por Hashtag")
            
            engagement_data = []
            for tag in selected_tags:
                tag_posts = exploded_f[exploded_f['hashtag_list'] == tag]
                engagement_data.append({
                    'Hashtag': f"#{tag}",
                    'Avg Likes': tag_posts['likes'].mean(),
                    'Total Posts': len(tag_posts),
                    'Max Likes': tag_posts['likes'].max()
                })
            
            eng_df = pd.DataFrame(engagement_data)
            
            fig = px.scatter(
                eng_df,
                x='Total Posts',
                y='Avg Likes',
                size='Max Likes',
                color='Hashtag',
                hover_data=['Max Likes'],
                title='Relación Posts vs Engagement'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch', key='plotly_14')
            
            # Descarga de datos de tendencias
            st.divider()
            buff_trends = io.StringIO()
            per_tag[per_tag['hashtag_list'].isin(selected_tags)].to_csv(buff_trends, index=False)
            
            st.download_button(
                "⬇️ Descargar Datos de Tendencias",
                buff_trends.getvalue(),
                file_name="tendencias_hashtags.csv",
                mime="text/csv"
            )
        else:
            st.info("👆 Selecciona al menos un hashtag para ver tendencias")
            
            # Mostrar top trending hashtags
            st.markdown("### 🔥 Hashtags Trending (últimos 7 días)")
            
            recent_date = df['timestamp'].max() - pd.Timedelta(days=7)
            recent_df = df[df['timestamp'] >= recent_date]
            
            if len(recent_df) > 0:
                recent_exploded = explode_hashtags(recent_df)
                trending = recent_exploded.groupby('hashtag_list').agg(
                    count=('hashtag_list', 'size'),
                    avg_likes=('likes', 'mean')
                ).sort_values('count', ascending=False).head(10)
                
                fig = px.bar(
                    trending,
                    x=trending.index,
                    y='count',
                    color='avg_likes',
                    color_continuous_scale='Reds',
                    title='Top 10 Hashtags Últimos 7 Días'
                )
                
                fig.update_layout(
                    xaxis_title="Hashtag",
                    yaxis_title="Menciones",
                    xaxis={'tickangle': -45}
                )
                
                st.plotly_chart(fig, width='stretch', key='plotly_15')

# ===================== TAB 3: MODELOS IA MEJORADOS =====================
with tab3:
    st.markdown("## 🧠 Entrenamiento de Modelos Avanzados")
    ensure_nltk_min()
    st.info("🚀 Sistema mejorado con: XGBoost + Embeddings + Multi-Label")
    col1, col2 = st.columns(2)
    with col1:
        # Correcting the width parameter for the training buttons
        if st.button("🎯 Entrenar Modelo de Relevancia (XGBoost)", width='content', type="primary"):
            with st.spinner("🔥 Entrenando XGBoost..."):
                # Ensure the dataframe has a 'label' column before training.
                # If missing, apply the existing labeling strategy from preprocess.py
                if 'label' not in df.columns:
                    df_labeled, detail = label_by_strategy(df.copy(), label_mode, label_value)
                    st.info(f"🔖 Etiquetado automático aplicado: {detail}")
                else:
                    df_labeled = df

                model_pack = entrenar_modelo_relevancia_mejorado(
                    df_labeled, df_embeddings, label_mode, label_value
                )
                
                if model_pack:
                    st.session_state['relevance_model'] = model_pack
                    
                    metrics = model_pack['metrics']
                    
                    st.success("✅ Modelo entrenado!")
                    
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    mcol1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                    mcol2.metric("F1-Score", f"{metrics['f1']:.3f}")
                    mcol3.metric("Precision", f"{metrics['precision']:.3f}")
                    mcol4.metric("Recall", f"{metrics['recall']:.3f}")
                    
                    # Matriz de confusión
                    cm = confusion_matrix(model_pack['y_test'], model_pack['y_pred'])
                    fig = px.imshow(
                        cm,
                        labels=dict(x="Predicción", y="Real"),
                        x=['Baja', 'Alta'],
                        y=['Baja', 'Alta'],
                        color_continuous_scale='Blues',
                        text_auto=True
                    )
                    st.plotly_chart(fig, width='stretch', key='plotly_16')
                    
                    # Feature importance
                    st.markdown("### 📊 Importancia de Features")
                    feature_importance = model_pack['clf'].feature_importances_[:20]
                    fig = px.bar(
                        x=feature_importance,
                        y=[f"Feature {i}" for i in range(len(feature_importance))],
                        orientation='h',
                        title="Top 20 Features más importantes"
                    )
                    st.plotly_chart(fig, width='stretch', key='plotly_17')
    
    with col2:
        st.markdown("### 🏷️ Predictor de Hashtags")
        
        # Parámetros de entrenamiento
        col_params1, col_params2 = st.columns(2)
        
        with col_params1:
            top_n_hashtags = st.number_input(
                "Top N hashtags",
                min_value=10,
                max_value=100,
                value=50,
                help="Número máximo de hashtags a predecir"
            )
        
        with col_params2:
            min_freq = st.number_input(
                "Frecuencia mínima",
                min_value=1,
                max_value=10,
                value=3,
                help="Mínimo de veces que debe aparecer un hashtag"
            )
        
        # Button moved below the selectors
        if st.button("🏷️ Entrenar Predictor de Hashtags", width='content', type="primary"):
            with st.spinner("🔄 Entrenando modelo multi-label mejorado..."):
                clf, mlb, metrics = entrenar_predictor_hashtags_mejorado(
                    df, 
                    df_embeddings, 
                    top_n_hashtags=top_n_hashtags,
                    min_hashtag_freq=min_freq
                )
                
                if clf is not None:
                    st.session_state['hashtag_predictor'] = clf
                    st.session_state['mlb'] = mlb
                    st.session_state['hashtag_metrics'] = metrics
                    
                    st.success("✅ Predictor de hashtags entrenado!")
                    
                    # Métricas principales
                    mcol1, mcol2, mcol3 = st.columns(3)
                    mcol1.metric("Precision", f"{metrics['precision']:.3f}")
                    mcol2.metric("Recall", f"{metrics['recall']:.3f}")
                    mcol3.metric("F1-Score", f"{metrics['f1']:.3f}")
                    
                    # Métricas adicionales
                    st.markdown("#### 📊 Métricas Detalladas")
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    with col_m1:
                        st.metric(
                            "Hamming Loss",
                            f"{metrics['hamming_loss']:.3f}",
                            help="Fracción de etiquetas incorrectamente predichas"
                        )
                    
                    with col_m2:
                        st.metric(
                            "Jaccard Score",
                            f"{metrics['jaccard']:.3f}",
                            help="Similitud entre predicciones y etiquetas reales"
                        )
                    
                    with col_m3:
                        st.metric(
                            "Coverage",
                            f"{metrics['coverage']:.1%}",
                            help="% de posts con al menos 1 predicción correcta"
                        )
                    
                    st.info(f"📋 Modelo entrenado para {metrics['n_hashtags']} hashtags (threshold: {metrics['threshold']})")
                    
                    # Mostrar hashtags entrenados
                    with st.expander("📝 Ver lista de hashtags entrenados"):
                        hashtags_list = list(mlb.classes_)
                        st.write(f"Total: {len(hashtags_list)} hashtags")
                        
                        # Mostrar en 3 columnas
                        cols = st.columns(3)
                        for i, ht in enumerate(hashtags_list):
                            cols[i % 3].write(f"#{ht}")
    
    st.divider()
    
    # Guardar/Cargar modelos
    st.markdown("### 💾 Gestión de Modelos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Guardar Todos los Modelos", width='stretch'):
            model_bundle = {
                'relevance_model': st.session_state.get('relevance_model'),
                'hashtag_predictor': st.session_state.get('hashtag_predictor'),
                'mlb': st.session_state.get('mlb'),
                'hashtag_metrics': st.session_state.get('hashtag_metrics'),
                'df_embeddings': st.session_state.get('df_embeddings')
            }

            joblib.dump(model_bundle, "hashmind_models_full.joblib")
            st.success("✅ Modelos guardados")

            with open("hashmind_models_full.joblib", "rb") as f:
                st.download_button(
                    "⬇️ Descargar Bundle Completo",
                    f,
                    file_name="hashmind_models_full.joblib",
                    width='stretch',
                    key="download_models_bundle_tab3_copy"
                )
    
    with col2:
        uploaded_bundle = st.file_uploader("📂 Cargar Bundle de Modelos", type=["joblib"], key="model_upload")
        if uploaded_bundle:
            bundle = joblib.load(uploaded_bundle)
            st.session_state.update(bundle)
            st.success("✅ Modelos cargados")
            
            # Mostrar métricas de modelo cargado
            if 'hashtag_metrics' in bundle and bundle['hashtag_metrics']:
                metrics = bundle['hashtag_metrics']
                st.info(f"📊 Modelo: {metrics['n_hashtags']} hashtags | F1: {metrics['f1']:.3f}")

# ===================== TAB 4: PREDICCIÓN AVANZADA - VERSIÓN CORREGIDA =====================
with tab4:
    st.markdown("## 🔮 Predicción Avanzada con IA")
    
    # Verificar modelos disponibles
    has_relevance = 'relevance_model' in st.session_state
    has_hashtag = 'hashtag_predictor' in st.session_state
    
    if not has_relevance and not has_hashtag:
        st.warning("⚠️ Entrena los modelos en la pestaña 'Modelos IA Mejorados'")
        st.stop()
    
    # Selector de modo de predicción
    col_mode, col_method = st.columns([2, 1])
    
    with col_mode:
        prediction_mode = st.radio(
            "📋 Modo de predicción",
            ["✏️ Texto Manual", "📸 Análisis de Imagen con IA"],
            horizontal=True,
            key="prediction_mode_selector"
        )
    
    with col_method:
        if has_hashtag:
            hashtag_method = st.radio(
                "🏷️ Método hashtags",
                ["🤖 Modelo ML", "🔍 Semántico"],
                help="Modelo ML: usa el modelo entrenado. Semántico: busca por similitud"
            )
        else:
            hashtag_method = "🔍 Semántico"
            st.info("Usando método semántico (no hay modelo entrenado)")
    
    st.divider()
    
    # Variables compartidas
    texto_prediccion = ""
    hashtags_input = ""
    imagen_pil = None
    categoria_manual = "Tecnología"  # default
    categoria_imagen = "Lifestyle"   # default
    
    # ============ MODO TEXTO MANUAL ============
    if prediction_mode == "✏️ Texto Manual":
        st.markdown("### ✏️ Ingresa tu post")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            texto_prediccion = st.text_area(
                "Texto del post",
                "Análisis de datos con Python: numpy, pandas y visualizaciones interactivas 🚀",
                height=120,
                key="manual_text_2"
            )
        
        with col2:
            hashtags_input = st.text_input(
                "Hashtags actuales (opcional)",
                "python, datascience",
                key="manual_hashtags_v1"
            )
            
            categoria_manual = st.selectbox(
                "Categoría",
                ["Tecnología", "Lifestyle", "Negocios", "Educación", "Arte"],
                key="cat_manual_v1"
            )
        
        ejecutar_prediccion = st.button("🔮 Analizar y Predecir", width='stretch', type="primary")
    
    # ============ MODO IMAGEN CON IA ============
    else:
        st.markdown("### 📸 Análisis Inteligente de Imágenes")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_image = st.file_uploader(
                "📤 Sube tu imagen",
                type=["jpg", "jpeg", "png", "webp"],
                help="Vista previa de tu contenido visual"
            )
            
            if uploaded_image:
                imagen_pil = Image.open(uploaded_image)
                st.image(imagen_pil, caption="📷 Vista previa", use_container_width=True)
        
        with col2:
            st.markdown("**✏️ Describe tu contenido:**")
            image_description = st.text_area(
                "",
                placeholder="Ej: Paisaje de montaña al atardecer...",
                height=120,
                key="image_desc_v1"
            )
            
            categoria_imagen = st.selectbox(
                "🎯 Categoría principal",
                ["Lifestyle", "Tecnología", "Comida", "Viajes", "Moda", "Fitness", 
                 "Negocios", "Naturaleza", "Arte", "Fotografía"],
                key="image_category_v1"
            )
        
        ejecutar_prediccion = st.button("✨ Analizar Imagen", width='stretch', type="primary")
        
        if ejecutar_prediccion:
            texto_prediccion = image_description
            hashtags_input = ""
    
    # ============ PROCESAMIENTO DE PREDICCIÓN ============
    if ejecutar_prediccion and texto_prediccion.strip():
        st.divider()
        
        with st.spinner("🔍 Analizando contenido..."):
            # 1. Generar embeddings
            text_model, clip_models = load_embedding_models()
            
            if text_model is not None:
                embedding_texto = generar_embeddings_texto([texto_prediccion], text_model)[0]
            else:
                embedding_texto = np.zeros(768)
            
            # Si hay imagen
            embedding_imagen = None
            if imagen_pil is not None and clip_models is not None:
                embedding_imagen = generar_embedding_imagen(imagen_pil, clip_models)
            
            # Combinar embeddings
            embedding_final = combinar_embeddings(embedding_texto, embedding_imagen)
            
            # 2. Buscar posts similares
            st.markdown("### 🔍 Posts Similares en tu Dataset")
            
            if 'df_embeddings' in st.session_state:
                posts_similares = buscar_posts_similares(
                    embedding_texto,
                    st.session_state['df_embeddings'],
                    df,
                    top_k=5
                )
                
                for idx, row in posts_similares.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="similar-post">
                            <strong>👤 {row['user']}</strong> | 
                            ❤️ {int(row['likes'])} likes | 
                            🎯 Similitud: {row['similarity_score']:.2%}
                            <br>
                            <em>"{row['text'][:150]}..."</em>
                            <br>
                            <small>#{' #'.join(row['hashtag_list'][:5])}</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.divider()
            
            # 3. PREDICCIÓN DE HASHTAGS
            st.markdown("### 🏷️ Hashtags Recomendados por IA")
            
            # Elegir método
            if hashtag_method == "🤖 Modelo ML" and has_hashtag:
                st.info("🤖 Usando modelo entrenado")
                
                model_pack_ht = {
                    'hashtag_predictor': st.session_state['hashtag_predictor'],
                    'mlb': st.session_state['mlb']
                }
                
                hashtags_predichos = predecir_hashtags_mejorado(
                    model_pack_ht,
                    texto_prediccion,
                    text_model,
                    df,
                    top_k=15
                )
            
            else:
                st.info("🔍 Usando método semántico (similitud con posts)")
                
                hashtags_predichos = generar_hashtags_semanticos(
                    texto_prediccion,
                    df,
                    st.session_state.get('df_embeddings', df_embeddings),
                    text_model,
                    top_k=15
                )
            
            if hashtags_predichos:
                # Mostrar con gráfico
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Gráfico de barras
                    hashtags_names = [f"#{h['hashtag']}" for h in hashtags_predichos]
                    hashtags_scores = [h['score'] for h in hashtags_predichos]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=hashtags_scores,
                            y=hashtags_names,
                            orientation='h',
                            marker=dict(
                                color=hashtags_scores,
                                colorscale='Viridis',
                                showscale=True
                            ),
                            text=[f"{s:.2%}" for s in hashtags_scores],
                            textposition='auto'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Score de Predicción por Hashtag",
                        xaxis_title="Probabilidad/Relevancia",
                        yaxis_title="Hashtag",
                        yaxis={'categoryorder':'total ascending'},
                        height=500
                    )
                    
                    st.plotly_chart(fig, width='stretch', key='plotly_18')
                
                with col2:
                    st.markdown("**Top 10 Hashtags:**")
                    
                    for ht_data in hashtags_predichos[:10]:
                        color_map = {
                            'Alta': '#28a745',
                            'Media': '#ffc107',
                            'Baja': '#dc3545'
                        }
                        color = color_map.get(ht_data['confidence'], '#6c757d')
                        
                        st.markdown(f"""
                        <div style="background: {color}; color: white; padding: 10px; 
                             margin: 5px 0; border-radius: 8px;">
                            <strong>#{ht_data['hashtag']}</strong><br>
                            Score: {ht_data['score']:.2%} | {ht_data['confidence']}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Botón de copiar
                    hashtags_copy = " ".join([f"#{h['hashtag']}" for h in hashtags_predichos[:10]])
                    st.code(hashtags_copy, language=None)
            
            else:
                st.warning("⚠️ No se pudieron generar hashtags. Verifica el dataset.")
            
            st.divider()
            
            # 4. PREDICCIÓN DE RELEVANCIA (mantener como está)
            st.markdown("### 📊 Predicción de Relevancia")
            
            if has_relevance:
                model_pack = st.session_state['relevance_model']
                
                # Preparar features
                tfidf = model_pack['tfidf']
                scaler = model_pack['scaler']
                clf = model_pack['clf']
                
                # TF-IDF
                text_normalized = texto_prediccion.lower()
                X_text = tfidf.transform([text_normalized])
                
                # Embeddings
                X_emb = csr_matrix(embedding_texto.reshape(1, -1))
                
                # Numéricos
                numeric_vals = np.array([[
                    df['popularity_score'].mean(),
                    len(parse_hashtags(hashtags_input)),
                    0,
                    0.01,
                    np.log1p(100)
                ]])
                X_num = scaler.transform(numeric_vals)
                
                # Combinar
                X_pred = hstack([X_text, X_emb, csr_matrix(X_num)])
                
                # Predecir
                pred_proba = clf.predict_proba(X_pred)[0, 1]
                pred_class = "Alta Relevancia" if pred_proba > 0.5 else "Baja Relevancia"
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "🎯 Predicción",
                        pred_class,
                        delta=f"{pred_proba:.1%} confianza"
                    )
                
                with col2:
                    st.metric(
                        "📈 Score de Éxito",
                        f"{pred_proba:.1%}",
                        delta="Alta" if pred_proba > 0.7 else "Media"
                    )
                
                with col3:
                    engagement_estimado = int(pred_proba * df['likes'].quantile(0.75))
                    st.metric(
                        "❤️ Engagement Estimado",
                        f"{engagement_estimado:,} likes"
                    )
                
                # Gráfico de probabilidad
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pred_proba * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilidad de Alta Relevancia"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "gray"},
                            {'range': [70, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                st.plotly_chart(fig, width='stretch', key='plotly_19')
            
            else:
                st.info("⚠️ Entrena el modelo de relevancia primero")
            
            st.divider()
            
            # 5. GENERADOR DE TEXTO OPTIMIZADO
            st.markdown("### ✨ Texto Optimizado Sugerido")
            
            if hashtags_predichos:
                top_5_hashtags = [h['hashtag'] for h in hashtags_predichos[:5]]
                
                # Template simple de optimización
                emoji_map = {
                    "Tecnología": "💻🚀",
                    "Lifestyle": "✨💫",
                    "Negocios": "💼📊",
                    "Educación": "📚🎓",
                    "Arte": "🎨🖌️"
                }
                
                categoria_actual = categoria_manual if prediction_mode == "✏️ Texto Manual" else categoria_imagen
                emoji = emoji_map.get(categoria_actual, "✨")
                
                texto_optimizado = f"{emoji} {texto_prediccion[:200]}...\n\n"
                texto_optimizado += f"#{' #'.join(top_5_hashtags)} {emoji}"
                
                st.text_area(
                    "Texto optimizado para copiar:",
                    texto_optimizado,
                    height=150
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.code(texto_optimizado, language=None)
                
                with col2:
                    st.markdown("**💡 Tips:**")
                    st.markdown("""
                    - Usa los 3-5 hashtags principales
                    - Añade emojis relevantes
                    - Publica en horarios pico
                    - Interactúa con comentarios
                    """)
    

# ===================== TAB 5: CAMPAÑAS (resumido) =====================
with tab5:
    st.markdown("## 📂 Comparativa de Campañas")

    if "__source" not in df.columns or df["__source"].nunique() == 1:
        st.info("ℹ️ Sube múltiples archivos CSV para comparar campañas")
    else:
        agg = df.groupby("__source").agg(
            posts=("user","count"),
            likes_tot=("likes","sum"),
            likes_med=("likes","mean")
        )

        fig = px.bar(agg, x=agg.index, y="posts", title="Posts por Campaña")
        st.plotly_chart(fig, width='stretch', key='plotly_22')
        st.dataframe(agg, width='stretch')

# ===================== FOOTER =====================
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 <strong>Hashmind AI</strong> | Sistema Avanzado de Predicción</p>
    <p>XGBoost • Sentence Transformers • CLIP • Multi-Label Classification</p>
</div>
""", unsafe_allow_html=True)