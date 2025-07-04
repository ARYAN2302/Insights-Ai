# Insights AI - Conversational GenAI Business Insights Suite

A fullstack GenAI-powered platform for business data analytics. Users upload datasets and interact with an LLM-powered assistant for conversational analytics, automated reports, and proactive business insights.

![Project Status](https://img.shields.io/badge/Status-Active-green)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Node.js](https://img.shields.io/badge/Node.js-16+-green)

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

4. **Create an Account**
   - Register a new user account
   - Upload your first dataset
   - Start asking questions about your data!

## 📊 Usage Examples

### Upload Data
```bash
# Supported formats: CSV, Excel, JSON
curl -X POST "http://localhost:8000/api/datasets/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@sales_data.csv" \
  -F "name=Sales Data" \
  -F "description=Monthly sales report"
```

### Chat with Your Data
```bash
curl -X POST "http://localhost:8000/api/chat/query" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the top 5 products by revenue?",
    "dataset_id": 1
  }'
```

### Generate Reports
```bash
curl -X POST "http://localhost:8000/api/reports" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Monthly Sales Summary",
    "report_type": "summary",
    "dataset_id": 1
  }'
```

## 🔧 API Documentation

### Authentication Endpoints
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user

### Dataset Endpoints
- `POST /api/datasets/upload` - Upload dataset
- `GET /api/datasets` - List datasets
- `GET /api/datasets/{id}` - Get dataset details
- `DELETE /api/datasets/{id}` - Delete dataset

### Chat Endpoints
- `POST /api/chat/query` - Ask questions
- `GET /api/chat/history` - Get chat history

### Report Endpoints
- `POST /api/reports` - Create report
- `GET /api/reports` - List reports
- `GET /api/reports/{id}` - Get report

### Alert Endpoints
- `GET /api/alerts` - List alerts
- `GET /api/alerts/unread` - Get unread alerts

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

### End-to-End Tests
```bash
cd frontend
npm run test:e2e
```

## 📚 Documentation

- [Backend API Documentation](./backend/README.md)
- [AI/ML Layer Documentation](./ai/README.md)
- [Frontend Documentation](./frontend/README.md)
- [API Reference](http://localhost:8000/docs) (when running)

## 🚀 Deployment

### Development
```bash
# Backend
cd backend && uvicorn main:app --reload

# Frontend
cd frontend && npm run dev
```

### Production

#### Using Docker
```bash
# Build and run with Docker Compose
docker-compose up --build
```

#### Using Vercel (Frontend) + Render (Backend)
1. Deploy frontend to Vercel
2. Deploy backend to Render
3. Update environment variables

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🛡️ Security

- JWT-based authentication
- File upload validation
- SQL injection prevention
- CORS configuration
- Rate limiting (recommended for production)

## 📋 Roadmap

### Phase 1 (Current)
- ✅ Basic chat interface
- ✅ File upload and processing
- ✅ LLM integration
- ✅ Vector search and RAG

### Phase 2 (Upcoming)
- [ ] Advanced visualizations
- [ ] Email notifications
- [ ] Report scheduling
- [ ] Multi-user workspaces

### Phase 3 (Future)
- [ ] Real-time data streaming
- [ ] Advanced AI agents
- [ ] Custom model training
- [ ] Enterprise features

## 🐛 Known Issues

- Large file uploads may timeout (>100MB)
- ChromaDB persistence issues on some systems
- Rate limiting with Groq API

## 📞 Support

- Create an issue for bugs or feature requests
- Check existing issues before creating new ones
- Join our community discussions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Groq](https://groq.com) for high-performance LLM inference
- [LangChain](https://langchain.com) for LLM application framework
- [ChromaDB](https://www.trychroma.com) for vector database
- [shadcn/ui](https://ui.shadcn.com) for UI components
- [FastAPI](https://fastapi.tiangolo.com) for backend framework

## 📊 Project Stats

- **Lines of Code**: ~5,000+
- **Components**: 15+ React components
- **API Endpoints**: 20+ endpoints
- **Database Tables**: 6 tables
- **File Formats**: 3 supported formats

---

**Built with ❤️ for data-driven businesses**
