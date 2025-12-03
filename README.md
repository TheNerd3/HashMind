# 🚀 hashmind
**Detección de Tendencias con IA y Análisis de Hashtags**

Una aplicación web interactiva para analizar datos de redes sociales, detectar tendencias de hashtags y predecir la relevancia de posts usando inteligencia artificial.

## 📋 Características Principales

- 📊 **Dashboard Interactivo**: Métricas en tiempo real y visualizaciones dinámicas
- 📈 **Análisis de Tendencias**: Seguimiento temporal de hashtags populares
- 🧠 **IA Integrada**: Modelo de regresión logística para predecir relevancia de posts
- 🔮 **Predicción**: Evalúa qué tan viral será un post antes de publicarlo
- 📂 **Comparativa de Campañas**: Analiza múltiples datasets simultáneamente

## 🛠️ Instalación y Configuración

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clona o descarga el repositorio**
   ```bash
   git clone https://github.com/TheNerd3/HashMind.git
   cd HashMind
   ```

2. **Instala las dependencias**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecuta la aplicación**
   ```bash
   streamlit run app.py
   ```

4. **Abre tu navegador**
   - La aplicación se abrirá automáticamente en: `http://localhost:8501`
   - Si no se abre automáticamente, copia y pega la URL en tu navegador

## 📁 Formato de Datos

Para usar la aplicación, necesitas archivos CSV con el siguiente formato:

```csv
user,text,hashtags,likes,timestamp
john_doe,"Nuevo tutorial de Python!",python programming tutorial,150,2024-01-15 10:30:00
jane_smith,"Tips de machine learning",ml ai datascience,280,2024-01-16 14:20:00
```

**Columnas requeridas:**
- `user`: Nombre del usuario
- `text`: Contenido del post
- `hashtags`: Hashtags separados por espacios (sin #)
- `likes`: Número de likes/interacciones
- `timestamp`: Fecha y hora (formato: YYYY-MM-DD HH:MM:SS)

## 🚀 Guía de Uso

1. **Sube tus datos**: Usa el panel lateral para cargar uno o más archivos CSV
2. **Explora el Dashboard**: Visualiza métricas generales y top hashtags
3. **Analiza Tendencias**: Revisa la evolución temporal de hashtags
4. **Entrena el Modelo**: Configura parámetros y entrena el modelo de IA
5. **Haz Predicciones**: Evalúa nuevos posts antes de publicarlos
6. **Compara Campañas**: Analiza múltiples datasets simultáneamente

## 🤖 Modelo de IA

La aplicación utiliza **Regresión Logística optimizada** con:
- Vectorización TF-IDF para análisis de texto
- Características numéricas (engagement, popularidad, densidad de hashtags)
- Balanceo automático de clases
- Validación cruzada integrada

## 📊 Tecnologías

- **Frontend**: Streamlit
- **Análisis de Datos**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualizaciones**: Plotly, Matplotlib
- **Procesamiento de Texto**: NLTK, LangDetect

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError"
```bash
pip install --upgrade -r requirements.txt
```

### Error: "NLTK data not found"
La aplicación descarga automáticamente los datos necesarios de NLTK la primera vez.

### La aplicación no se abre
Verifica que el puerto 8501 no esté en uso y reinicia la aplicación.

## 📞 Soporte

Si encuentras algún problema:
1. Verifica que todas las dependencias estén instaladas
2. Asegúrate de que tus archivos CSV tengan el formato correcto
3. Revisa que Python 3.8+ esté instalado

---

**¡Listo para analizar tus datos de redes sociales con IA! 🎉**
