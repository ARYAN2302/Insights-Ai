# Insights AI - AI/ML Layer

AI and machine learning components for the Conversational GenAI Business Insights Suite.

## Features

- **Data Processing**: Automated data profiling and analysis
- **LLM Integration**: Groq API integration with LangChain
- **Vector Search**: ChromaDB for semantic search and RAG
- **Automated Insights**: Proactive anomaly detection and trend analysis
- **Report Generation**: AI-powered business report creation

## Components

### 1. Data Processor (`data_processor.py`)

Handles data parsing, profiling, and analysis:

- **File Processing**: Support for CSV, Excel, and JSON formats
- **Metadata Generation**: Automatic column profiling and statistics
- **Anomaly Detection**: Statistical outlier detection using IQR method
- **Data Quality Analysis**: Missing data patterns and quality metrics
- **Insights Generation**: Automated business insights from data patterns

### 2. LLM Service (`llm_service.py`)

Manages interactions with Groq LLM and vector storage:

- **Groq Integration**: Llama 3 70B model for conversational analytics
- **Vector Embeddings**: HuggingFace embeddings for semantic search
- **RAG Implementation**: Retrieval-augmented generation for dataset queries
- **Report Generation**: Automated business report creation
- **Proactive Insights**: AI-powered trend and anomaly detection

## Setup Instructions

### 1. Install Dependencies

```bash
cd ai
pip install -r requirements.txt
```

### 2. Environment Configuration

Make sure you have these environment variables set:

```env
# Groq API
GROQ_API_KEY=your-groq-api-key

# Vector Store
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

### 3. Get Groq API Key

1. Visit [Groq Console](https://console.groq.com)
2. Create an account or sign in
3. Generate an API key
4. Add it to your environment variables

## Usage Examples

### Data Processing

```python
from data_processor import DataProcessor

# Initialize processor
processor = DataProcessor()

# Process a dataset
metadata, row_count = processor.process_file("data.csv", "csv")

# Get sample data
sample = processor.get_sample_data("data.csv", "csv", n_rows=10)

# Detect anomalies
anomalies = processor.detect_anomalies("data.csv", "csv")

# Generate insights
insights = processor.generate_insights("data.csv", "csv")
```

### LLM Service

```python
from llm_service import GroqLLMService

# Initialize service
llm_service = GroqLLMService()

# Embed dataset for RAG
success = llm_service.embed_dataset(
    dataset_id=1,
    file_path="data.csv",
    file_type="csv",
    columns_info=metadata
)

# Query dataset
result = llm_service.query_dataset(
    dataset_id=1,
    question="What are the top 5 products by sales?",
    columns_info=metadata
)

# Generate report
report = llm_service.generate_report(
    dataset_id=1,
    report_type="summary",
    columns_info=metadata
)

# Detect insights
insights = llm_service.detect_insights(
    dataset_id=1,
    columns_info=metadata
)
```

## Data Processing Pipeline

### 1. File Upload & Parsing

```
Upload → File Validation → Data Loading → Profiling → Metadata Generation
```

### 2. Vector Embedding

```
Data Processing → Text Chunking → Embedding Generation → Vector Storage
```

### 3. Query Processing

```
User Query → Vector Search → Context Retrieval → LLM Processing → Response
```

### 4. Report Generation

```
Dataset Analysis → Template Selection → Content Generation → Chart Suggestions
```

## Supported File Formats

### CSV Files
- Comma-separated values
- Automatic delimiter detection
- Header row detection
- Data type inference

### Excel Files
- `.xlsx` and `.xls` formats
- Multiple sheet support (uses first sheet)
- Automatic data type detection
- Missing value handling

### JSON Files
- Structured JSON data
- Nested object flattening
- Array handling
- Schema inference

## AI Capabilities

### Conversational Analytics

The system can answer questions like:
- "What are the trends in our sales data?"
- "Show me the top performing products"
- "Are there any anomalies in the data?"
- "What insights can you provide about customer behavior?"

### Automated Reports

Generate comprehensive reports including:
- **Summary Reports**: Overview of dataset characteristics
- **Trend Reports**: Time-series analysis and forecasting
- **Custom Reports**: Tailored analysis based on specific requirements

### Proactive Insights

Automatically detect and alert on:
- **Anomalies**: Statistical outliers and unusual patterns
- **Trends**: Growth, decline, and seasonal patterns
- **Data Quality**: Missing data and consistency issues
- **Business Opportunities**: Revenue optimization suggestions

## Vector Store Configuration

### ChromaDB Setup

```python
# Default configuration
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

# Custom configuration
vectorstore = Chroma(
    collection_name="dataset_1",
    embedding_function=embeddings,
    persist_directory=CHROMA_PERSIST_DIRECTORY
)
```

### Embedding Models

Currently using HuggingFace embeddings:
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Performance**: Fast inference, good quality

## Performance Optimization

### Chunking Strategy

- **Chunk Size**: 1000 characters
- **Overlap**: 200 characters
- **Metadata Preservation**: Dataset ID and content type

### Caching

- Vector embeddings are persisted to disk
- Dataset metadata is cached in database
- Query results can be cached for common questions

## Error Handling

The AI layer includes comprehensive error handling:

- **File Processing Errors**: Invalid formats, corrupted files
- **API Errors**: Groq API rate limits, network issues
- **Vector Store Errors**: ChromaDB connection issues
- **Memory Errors**: Large dataset processing

## Development

### Adding New File Formats

1. Extend `DataProcessor.process_file()` method
2. Add format-specific parsing logic
3. Update metadata generation
4. Add tests for new format

### Custom LLM Models

To use different LLM providers:

1. Implement new service class
2. Follow the same interface as `GroqLLMService`
3. Update dependency injection in main app

### Testing

```bash
# Run AI component tests
pytest tests/test_data_processor.py
pytest tests/test_llm_service.py
```

## Monitoring

### Metrics to Track

- **Processing Time**: File processing and query response times
- **API Usage**: Groq API calls and token consumption
- **Vector Store**: Embedding generation and search performance
- **Error Rates**: Failed processing and API errors

### Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log processing events
logger.info(f"Processing dataset {dataset_id}")
logger.error(f"Failed to process file: {error}")
```

## Security Considerations

- **Data Privacy**: Uploaded data is processed locally
- **API Keys**: Secure storage of Groq API credentials
- **File Validation**: Comprehensive file format validation
- **Access Control**: User-based data isolation

## Troubleshooting

### Common Issues

1. **Groq API Errors**: Check API key and rate limits
2. **ChromaDB Issues**: Ensure write permissions for persist directory
3. **Memory Issues**: Large files may require processing optimization
4. **Import Errors**: Verify all dependencies are installed

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual components
processor = DataProcessor()
metadata, rows = processor.process_file("test.csv", "csv")
print(f"Processed {rows} rows with {len(metadata['columns'])} columns")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

## License

This project is licensed under the MIT License.
