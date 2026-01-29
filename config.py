import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
VIZ_DIR = OUTPUT_DIR / "visualizations"

# Ollama Config
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_TEMPERATURE = 0.1
OLLAMA_NUM_PREDICT = 512

# Streamlit Config
PAGE_TITLE = "Data Analysis Assistant"
PAGE_ICON = "📊"
LAYOUT = "wide"
