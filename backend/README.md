# Insights AI Backend

FastAPI-based backend for the Conversational GenAI Business Insights Suite.

## Features

- **User Authentication**: JWT-based authentication system
- **Dataset Management**: Upload, process, and manage CSV/Excel/JSON datasets
- **AI-Powered Chat**: Conversational analytics using Groq LLM with RAG
- **Automated Reports**: Generate comprehensive business reports
- **Proactive Insights**: Detect anomalies and trends automatically
- **Vector Search**: ChromaDB integration for semantic search

## Setup Instructions

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Update the `.env` file with your configuration:

```env
# Database
DATABASE_URL=sqlite:///./insights_ai.db

# JWT
SECRET_KEY=your-super-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Groq API (Get your API key from https://console.groq.com)
GROQ_API_KEY=your-groq-api-key-here

# Vector Store
CHROMA_PERSIST_DIRECTORY=./chroma_db

# File Storage
STORAGE_PATH=../storage

# Environment
ENVIRONMENT=development
```

### 3. Run the Server

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 4. API Documentation

Once the server is running, you can access:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user profile

### Datasets
- `POST /api/datasets/upload` - Upload dataset
- `GET /api/datasets` - List user datasets
- `GET /api/datasets/{id}` - Get dataset details
- `GET /api/datasets/{id}/sample` - Get dataset sample
- `DELETE /api/datasets/{id}` - Delete dataset
- `POST /api/datasets/{id}/insights` - Generate insights

### Chat & Analytics
- `POST /api/chat/query` - Ask questions about data
- `GET /api/chat/history` - Get chat history

### Reports
- `POST /api/reports` - Create report
- `GET /api/reports` - List reports
- `GET /api/reports/{id}` - Get report details

### Alerts
- `GET /api/alerts` - List alerts
- `GET /api/alerts/unread` - Get unread alerts
- `POST /api/alerts/{id}/mark-read` - Mark alert as read

## Database Schema

The application uses SQLite with the following main tables:

- **users**: User accounts and authentication
- **datasets**: Uploaded datasets and metadata
- **queries**: Chat history and Q&A sessions
- **reports**: Generated reports and analytics
- **alerts**: System notifications and insights
- **user_preferences**: User settings and preferences

## File Structure

```
backend/
├── main.py              # FastAPI application
├── models.py            # SQLAlchemy models
├── schemas.py           # Pydantic schemas
├── crud.py              # Database operations
├── auth.py              # Authentication utilities
├── database.py          # Database configuration
├── requirements.txt     # Python dependencies
├── .env.example        # Environment template
└── README.md           # This file
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Code Quality

```bash
# Install development dependencies
pip install black isort flake8

# Format code
black .
isort .

# Lint code
flake8 .
```

## Deployment

### Using Docker

```bash
# Build image
docker build -t insights-ai-backend .

# Run container
docker run -p 8000:8000 insights-ai-backend
```

### Using Render

1. Connect your GitHub repository to Render
2. Set environment variables in Render dashboard
3. Deploy as a Web Service

### Environment Variables for Production

- `DATABASE_URL`: Production database URL
- `SECRET_KEY`: Strong random secret key
- `GROQ_API_KEY`: Your Groq API key
- `ENVIRONMENT`: Set to "production"

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
2. **Database Errors**: Check if the database file is writable
3. **API Key Errors**: Verify your Groq API key is valid
4. **CORS Issues**: Update the allowed origins in main.py

### Logs

Check the application logs for detailed error information:

```bash
# View logs in development
uvicorn main:app --reload --log-level debug
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
