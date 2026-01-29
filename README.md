# Data Analysis & Visualization Assistant

An AI-powered data analysis and visualization assistant built with LangChain, Ollama, and Streamlit. Users can upload datasets and ask natural language questions to get insights and visualizations.

## Features

- **Data Loading**: Support for CSV, Excel, and JSON files
- **Automatic Profiling**: Instant data overview when files are uploaded
- **Natural Language Queries**: Ask questions in plain English
- **Statistical Analysis**: Correlations, group-by operations, outlier detection
- **Rich Visualizations**: Histograms, scatter plots, heatmaps, box plots, time series, and more
- **Conversational Interface**: Chat-based UI powered by Streamlit

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- qwen3:8b model pulled in Ollama

## Setup

1. **Install Ollama and pull the model**:
```bash
# Install Ollama from https://ollama.ai/

# Pull the qwen3:8b model
ollama pull qwen3:8b

# Start Ollama server (if not running)
ollama serve
```

2. **Create and activate virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Running the Application

1. **Ensure Ollama is running**:
```bash
ollama serve
```

2. **Run the Streamlit app**:
```bash
streamlit run app.py
```

3. **Open your browser** to `http://localhost:8501`

## Usage

1. **Upload Data**: Use the file uploader in the sidebar to upload a CSV, Excel, or JSON file
2. **Review Profile**: The system automatically analyzes your data and shows a summary
3. **Ask Questions**: Type natural language questions in the chat input
4. **Get Insights**: Receive answers with visualizations and analysis

### Example Queries

- "Show me summary statistics"
- "Are there any missing values?"
- "Plot a correlation heatmap"
- "Show the distribution of revenue"
- "Create a scatter plot of price vs quantity"
- "Which region has the highest sales?"
- "Detect outliers in price"
- "Group by product_category and sum revenue"

## Available Tools

### Data Exploration
- Get data overview (shape, columns, types)
- Check missing values
- Get summary statistics
- Get column information
- List all columns

### Statistical Analysis
- Calculate correlation matrix
- Perform group-by aggregations
- Detect outliers (IQR method)

### Visualizations
- Distribution plots (histogram/KDE)
- Correlation heatmap
- Scatter plots
- Bar charts
- Box plots
- Time series plots

## Sample Datasets

Two sample datasets are included in the `data/` directory:

1. **sample_sales.csv** - E-commerce sales data (1000 rows)
   - Columns: date, product_category, price, quantity, revenue, region
   - Includes some missing values

2. **sample_iris.csv** - Classic Iris dataset (150 rows)
   - Columns: sepal_length, sepal_width, petal_length, petal_width, species
   - No missing values

## Project Structure

```
DVA/
├── app.py                      # Streamlit main app
├── config.py                   # Configuration
├── requirements.txt            # Dependencies
├── agents/
│   ├── data_agent.py          # Data profiling agent
│   └── manager_agent.py       # Query processing agent
├── tools/
│   ├── data_loading.py        # Data loading tools
│   ├── data_exploration.py    # Exploration tools
│   ├── statistical_analysis.py # Statistical tools
│   ├── visualization.py       # Visualization tools
│   └── tool_registry.py       # Tool registration
├── utils/
│   ├── data_state.py          # Data state management
│   └── streamlit_helpers.py   # UI helpers
└── data/
    ├── sample_sales.csv       # Sample dataset
    └── sample_iris.csv        # Sample dataset
```

## Architecture

- **Manager Agent**: LangChain ReAct agent using Ollama qwen3:8b
- **Data Agent**: Automatic profiling when data is loaded
- **Data State**: Singleton managing current dataset
- **Tools**: 14 specialized tools for analysis and visualization
- **UI**: Streamlit chat interface with inline visualizations

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Model Not Found
```bash
# Pull the model
ollama pull qwen3:8b

# List available models
ollama list
```

### Port Already in Use
```bash
# Run on different port
streamlit run app.py --server.port 8502
```

## License

MIT

## Contributing

This is a hackathon MVP. Contributions welcome!
