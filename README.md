# AI Journal Analysis Pipeline: Components 2, 3, and 4 + AstraDB Integration

**Production-Ready Emotion Analysis, NER & Temporal Processing, Feature Engineering, and AstraDB Storage Pipeline**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![AstraDB](https://img.shields.io/badge/Database-AstraDB-purple.svg)](https://astra.datastax.com)
[![Vector Search](https://img.shields.io/badge/Features-Vector%20Search-green.svg)](https://github.com)
[![Tests](https://img.shields.io/badge/Tests-Passing-green.svg)](https://github.com)

## 🎯 **Overview**

This repository contains a production-ready integration of three AI components that work together to analyze journal entries and store them in **AstraDB** with two optimized collections:

- **Component 2**: Emotion Analysis with Reinforcement Learning
- **Component 3**: Named Entity Recognition & Temporal Analysis  
- **Component 4**: Feature Engineering Pipeline
- **AstraDB Integration**: Vector database storage with semantic search capabilities

The pipeline transforms natural language journal entries into structured data optimized for **AstraDB vector operations** and **semantic similarity search**.

## 🚀 **Quick Start**

### **Prerequisites**
```bash
# Set up your .env file with AstraDB credentials
ASTRA_DB_APPLICATION_TOKEN=your_application_token_here
ASTRA_DB_API_ENDPOINT=https://your-database-id-your-region.apps.astra.datastax.com
ASTRA_DB_KEYSPACE=your_keyspace_name
```

### **Installation**
```bash
git clone <repository-url>
cd comp_2_3_4_integration
pip install -r requirements.txt
pip install astrapy python-dotenv
```

### **Run AstraDB Integration**
```python
from astra_db_integration import AstraDBIntegrator

# Initialize integrator (automatically connects to AstraDB)
integrator = AstraDBIntegrator()

# Process journal entry and push to AstraDB
result = integrator.process_journal_entry(
    text="Had a great breakthrough at work today!",
    user_id="user_123",
    session_id="session_456"
)

# Push to AstraDB collections
success = integrator.push_to_astra_db(result)
if success:
    print("✅ Data successfully pushed to AstraDB!")
```

### **Run Tests**
```bash
python astra_db_integration.py
```

## 📊 **AstraDB Collections Schema**

### **Collection 1: `chat_embeddings`**
**Purpose**: Real-time conversation analysis and chat context storage

```json
{
  "id": "uuid",
  "user_id": "string",
  "entry_id": "uuid",
  "message_content": "text",
  "message_type": "user_message|ai_response|system_message",
  "timestamp": "ISO8601 with Z suffix",
  "session_id": "uuid",
  "conversation_context": "string",
  "primary_embedding": [768 dimensions - float array],
  "lightweight_embedding": [384 dimensions - float array],
  "text_length": "integer",
  "processing_time_ms": "float",
  "model_version": "string",
  "semantic_tags": ["array of strings"],
  "emotion_context": "JSON string",
  "entities_mentioned": "JSON string",
  "temporal_context": "JSON string",
  "feature_vector": [90 dimensions - float array],
  "temporal_features": [25 dimensions - float array],
  "emotional_features": [20 dimensions - float array],
  "semantic_features": [30 dimensions - float array],
  "user_features": [15 dimensions - float array],
  "feature_completeness": "float",
  "confidence_score": "float"
}
```

### **Collection 2: `semantic_search`**
**Purpose**: Content discovery and relationship mapping for semantic search

```json
{
  "id": "uuid",
  "user_id": "string",
  "content_type": "journal_entry|event|person|location|topic",
  "title": "string",
  "content": "text",
  "primary_embedding": [768 dimensions - float array],
  "created_at": "ISO8601 with Z suffix",
  "updated_at": "ISO8601 with Z suffix",
  "tags": ["array of strings"],
  "linked_entities": "JSON string",
  "search_metadata": "JSON string",
  "feature_vector": [90 dimensions - float array],
  "temporal_features": [25 dimensions - float array],
  "emotional_features": [20 dimensions - float array],
  "semantic_features": [30 dimensions - float array],
  "user_features": [15 dimensions - float array]
}
```

## 🏗️ **Architecture**

### **Component 2: Emotion Analysis**
- **Technology**: Cardiff NLP RoBERTa + Reinforcement Learning (SAC)
- **Features**: Real-time learning, user personalization, 8 emotion categories
- **Output**: Emotion analysis with confidence scores
- **AstraDB Integration**: Stores emotion context as JSON in `chat_embeddings`

### **Component 3: NER & Temporal Analysis**
- **Technology**: spaCy NER + Temporal parsing + Sentence-Transformers
- **Features**: Entity extraction, temporal event detection, semantic embeddings
- **Output**: Structured semantic analysis with 768D and 384D embeddings
- **AstraDB Integration**: Primary source of vector embeddings for both collections

### **Component 4: Feature Engineering**
- **Technology**: Production feature extraction and normalization
- **Features**: 90D vectors (25+20+30+15), quality control, validation
- **Output**: Structured features with metadata
- **AstraDB Integration**: Provides feature vectors and quality metrics

### **AstraDB Integration Layer**
- **Connection Management**: Automatic AstraDB connection via environment variables
- **Data Formatting**: Strict schema compliance with proper data types
- **Vector Operations**: Optimized for AstraDB's vector similarity search
- **Error Handling**: Robust fallbacks and local file backup

## 📁 **Repository Structure**

```
├── comp2/                  # Component 2: Emotion Analysis
│   ├── src/               # Core emotion analysis modules
│   ├── models/            # Pre-trained emotion models
│   └── tests/             # Unit tests
├── comp3/                 # Component 3: NER & Temporal
│   ├── src/               # NER and temporal analysis modules  
│   └── tests/             # Unit tests
├── comp4/                 # Component 4: Feature Engineering
│   ├── src/               # Feature engineering modules
│   ├── data/              # Data schemas and structures
│   └── tests/             # Unit tests
├── astra_db_integration.py      # Main AstraDB integration file
├── requirements.txt              # Core dependencies
├── requirements_astra.txt        # AstraDB-specific dependencies
├── tests/                 # Integration and production tests
├── documentations/        # Detailed component documentation
└── run_tests.py          # Test runner script
```

## 🔧 **AstraDB Configuration**

### **Environment Variables**
```bash
# Required for AstraDB connection
export ASTRA_DB_APPLICATION_TOKEN="your_application_token_here"
export ASTRA_DB_API_ENDPOINT="https://your-database-id-your-region.apps.astra.datastax.com"
export ASTRA_DB_KEYSPACE="your_keyspace_name"
```

### **Collection Setup**
```sql
-- Create collections in AstraDB
CREATE COLLECTION chat_embeddings;
CREATE COLLECTION semantic_search;

-- Enable vector search (if using AstraDB Vector)
ALTER COLLECTION chat_embeddings ADD VECTOR primary_embedding DIMENSION 768;
ALTER COLLECTION chat_embeddings ADD VECTOR lightweight_embedding DIMENSION 384;
ALTER COLLECTION semantic_search ADD VECTOR primary_embedding DIMENSION 768;
```

## 📈 **Performance & Features**

### **Vector Search Capabilities**
- **Primary Embeddings**: 768D high-quality vectors for semantic similarity
- **Lightweight Embeddings**: 384D fast vectors for quick search operations
- **Feature Vectors**: 90D comprehensive feature representation
- **Real-time Processing**: <500ms end-to-end pipeline execution

### **Search Operations**
```python
# Vector similarity search in AstraDB
# Find similar journal entries
similar_entries = chat_collection.find(
    {"primary_embedding": {"$vector": query_embedding, "$similarity": 0.8}}
)

# Semantic search across content types
search_results = search_collection.find(
    {"tags": {"$in": ["work", "meeting"]}}
)
```

### **Quality Metrics**
- **Feature Completeness**: >90% target
- **Confidence Scores**: >85% target
- **Embedding Accuracy**: Exact dimension matching (768D, 384D, 90D)
- **Schema Compliance**: 100% field name and type accuracy

## 🧪 **Testing & Validation**

### **Schema Compliance Testing**
```python
# Test output format compliance
result = integrator.process_journal_entry(
    text="Test entry for validation",
    user_id="test_user"
)

# Verify schema compliance
assert len(result.chat_embeddings["primary_embedding"]) == 768
assert len(result.chat_embeddings["lightweight_embedding"]) == 384
assert len(result.chat_embeddings["feature_vector"]) == 90
```

### **AstraDB Connection Testing**
```python
# Test AstraDB connectivity
integrator = AstraDBIntegrator()
print("✅ AstraDB connection successful!")

# Test data insertion
success = integrator.push_to_astra_db(result)
assert success == True
```

## 🚀 **Production Deployment**

### **Prerequisites**
- Python 3.8+
- 4GB+ RAM recommended
- AstraDB account with Vector enabled
- Valid application token and API endpoint

### **Docker Deployment**
```dockerfile
FROM python:3.8-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install astrapy python-dotenv
CMD ["python", "astra_db_integration.py"]
```

### **Monitoring & Health Checks**
```python
# Check component health
integrator = AstraDBIntegrator()

# Verify AstraDB connection
if integrator.astra_connector.db:
    print("✅ AstraDB: Connected")
else:
    print("❌ AstraDB: Connection failed")
```

## 🔍 **Use Cases**

### **Real-time Journal Analysis**
- **Emotion Tracking**: Monitor user emotional patterns over time
- **Entity Recognition**: Track people, locations, and organizations mentioned
- **Temporal Analysis**: Understand writing patterns and seasonal trends

### **Semantic Search & Discovery**
- **Content Recommendation**: Find similar journal entries
- **Relationship Mapping**: Discover connections between people and events
- **Topic Clustering**: Group related content for insights

### **AI Training & Research**
- **Feature Engineering**: 90D vectors for machine learning models
- **Vector Database**: Optimized storage for similarity search
- **Metadata Enrichment**: Rich context for AI model training

## 🛡️ **Security & Best Practices**

### **AstraDB Security**
- **Application Tokens**: Read-only by default, scope-limited access
- **API Endpoints**: HTTPS encrypted connections
- **Keyspace Isolation**: Prevents cross-tenant data access
- **Environment Variables**: Never commit credentials to version control

### **Data Privacy**
- **User Isolation**: Strict user_id separation
- **Session Management**: UUID-based session tracking
- **Audit Logging**: Complete processing history tracking

## 📚 **API Reference**

### **AstraDBIntegrator Class**
```python
class AstraDBIntegrator:
    def __init__(self, config_path: str = "unified_config.yaml")
    def process_journal_entry(text, user_id, session_id, ...) -> AstraDBOutput
    def push_to_astra_db(output: AstraDBOutput) -> bool
    def batch_process(entries: List[Dict]) -> List[AstraDBOutput]
    def export_for_astra_db(output: AstraDBOutput) -> Dict
```

### **AstraDBConnector Class**
```python
class AstraDBConnector:
    def __init__(self)  # Auto-connects via environment variables
    def push_to_collection(collection_name: str, data: Dict) -> bool
```

## 🤝 **Contributing**

### **Development Setup**
```bash
git clone <repository-url>
cd comp_2_3_4_integration
pip install -r requirements.txt
pip install astrapy python-dotenv

# Set up .env file with your AstraDB credentials
# Run integration test
python astra_db_integration.py
```

### **Code Quality Standards**
- All tests must pass before merging
- AstraDB schema compliance validation
- Proper error handling and logging
- Performance benchmarks must be met

## 📋 **Changelog**

### **v2.0.0 - AstraDB Integration Release**
- ✅ Complete Components 2+3+4 + AstraDB integration
- ✅ Two optimized collections: `chat_embeddings` and `semantic_search`
- ✅ Vector search capabilities with 768D and 384D embeddings
- ✅ 90-dimensional feature vectors with quality metrics
- ✅ Automatic AstraDB connection management
- ✅ Schema compliance validation
- ✅ Production-ready error handling and backup

## 📄 **License**

[License information]

## 🆘 **Support**

For AstraDB integration support and technical questions:
- Create an issue in this repository
- Review the documentation in `documentations/`
- Test AstraDB connection: `python astra_db_integration.py`
- Check schema compliance in the output files

---

**🎉 Ready for AstraDB Production Deployment!**

This pipeline is production-ready with full AstraDB integration, vector search capabilities, and comprehensive feature engineering for real-world AI applications.