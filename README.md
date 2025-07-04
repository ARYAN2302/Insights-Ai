# Insights AI - Conversational GenAI Business Insights Suite

A fullstack GenAI-powered platform for business data analytics. Users upload datasets and interact with an LLM-powered assistant for conversational analytics, automated reports, and proactive business insights.

## 🚀 Features

### Core Analytics
- **Conversational Analytics**: Ask questions about your data in natural language
- **Automated Reports**: Generate comprehensive business reports with AI
- **Proactive Insights**: Automatic anomaly detection and trend analysis
- **Data Visualization**: Interactive charts and dashboards

### AI-Powered Capabilities
- **RAG (Retrieval-Augmented Generation)**: Context-aware responses using your data
- **Multi-format Support**: CSV, Excel, and JSON file uploads
- **Semantic Search**: Find relevant information across your datasets
- **Smart Recommendations**: AI suggests relevant analyses and visualizations

### User Experience
- **Dark Chat UI**: Modern, responsive interface
- **Real-time Processing**: Instant feedback and results
- **History Tracking**: Access previous queries and reports
- **Alert System**: Notifications for important insights

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   AI Layer      │
│   (Next.js)     │◄──►│   (FastAPI)     │◄──►│ (Groq + LangChain)│
│                 │    │                 │    │                 │
│ • Chat Interface│    │ • REST API      │    │ • LLM Service   │
│ • Data Views    │    │ • Authentication│    │ • Vector Store  │
│ • Reports       │    │ • File Upload   │    │ • Data Analysis │
│ • Alerts        │    │ • Database      │    │ • Insights Gen  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety and developer experience
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - Modern UI components
- **Recharts** - Data visualization library

### Backend
- **FastAPI** - High-performance Python web framework
- **SQLAlchemy** - Python SQL toolkit and ORM
- **SQLite** - Lightweight database for development
- **JWT** - JSON Web Token authentication
- **Pydantic** - Data validation and serialization

### AI/ML
- **Groq API** - High-performance LLM inference
- **LangChain** - LLM application framework
- **ChromaDB** - Vector database for embeddings
- **HuggingFace** - Pre-trained embedding models
- **pandas** - Data manipulation and analysis

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or pnpm

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/insights-ai.git
cd insights-ai
```

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your configuration
uvicorn main:app --reload
```

### 3. Frontend Setup
```bash
cd frontend
npm install
# or
pnpm install

npm run dev
# or
pnpm dev
```

### 4. Environment Configuration

#### Backend (.env)
```env
# Database
DATABASE_URL=sqlite:///./insights_ai.db

# JWT
SECRET_KEY=your-super-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Groq API (Get from https://console.groq.com)
GROQ_API_KEY=your-groq-api-key-here

# Vector Store
CHROMA_PERSIST_DIRECTORY=./chroma_db

# File Storage
STORAGE_PATH=../storage
```

#### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🚀 Quick Start

1. **Start the Backend**
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. **Start the Frontend**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access the Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs


## 🐛 Known Issues

- Large file uploads may timeout (>100MB)
- ChromaDB persistence issues on some systems
- Rate limiting with Groq API

