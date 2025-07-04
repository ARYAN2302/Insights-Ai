Conversational GenAI Business Insights Suite

Overview

Build a fullstack GenAI-powered platform for business data analytics. Users upload datasets and interact with an LLM-powered assistant for conversational analytics, automated reports, and proactive business insights. The frontend (Next.js, dark chat UI) is already built. This document details the backend, AI, and integration requirements, with detailed explanations of all features and their interactions.

Core Features & Detailed Workflow

1. User Authentication & Profiles
 • Users register/login with email and password (JWT-based).
 • User profile stores uploaded datasets, query history, and report preferences.

2. Dataset Upload & Processing
 • Users upload business data (CSV, Excel, or JSON).
 • Backend parses and profiles data using ‎`pandas`.
 • Extracts columns, types, summary statistics, and stores metadata in SQLite.
 • On upload, the backend:
 ▫ Embeds key data/text for semantic search (ChromaDB/Pinecone).
 ▫ Prepares data for analytics and visualization.

3. Datasets Management
 • Users view a list of uploaded datasets with metadata.
 • Selecting a dataset loads its context into the chat.
 • Users can delete, rename, or update datasets.

4. Chat-Based Conversational Analytics (with RAG)
 • Users interact with a chat interface to ask questions about their data (e.g., “What were last quarter’s top products?”).
 • When a question is asked:
 ▫ LangChain retrieves relevant data/context.
 ▫ Groq LLM analyzes and answers, possibly generating summary tables or charts (as JSON for frontend rendering).
 ▫ Chat supports follow-ups and maintains context.
 • All queries and responses are stored for audit/history.

5. Automated Reports & Visualizations
 • Users can request or schedule reports (e.g., weekly sales summary).
 • Backend generates reports using LLM (Groq) and data analysis libraries (‎`pandas`, ‎`plotly`).
 • Reports include summaries, charts, trends, and actionable insights.
 • Reports are accessible in a dedicated section and can be sent via email (optional).

6. Alerts, Trends & Proactive Agentic Insights
 • The platform proactively:
 ▫ Detects anomalies, trends, or significant changes in the uploaded data.
 ▫ Sends alerts or notifications to the user (dashboard and optional email).
 ▫ Suggests questions or analyses the user might not have considered.
 • Insights and alerts are displayed in a sidebar and can be loaded into the chat for discussion.

How Features Work Together
 • Agentic Analytics: The system doesn’t just wait for user input—it analyzes uploaded data and proactively surfaces important findings or alerts.
 • Conversational Hub: All analytics, report requests, and follow-up discussions happen through the chat, making complex data analysis accessible and interactive.
 • Semantic Search & RAG: Every question leverages embedded data for precise, context-aware answers tailored to the user’s datasets.
 • Proactive Business Value: Automated reporting, trend detection, and smart suggestions ensure users get value even without specific questions.

Tech Stack
 • Frontend: Next.js (TypeScript, dark chat UI, shadcn/ui or Chakra UI)
 • Backend: FastAPI (Python), SQLite, SQLAlchemy
 • AI/LLM: Groq API (Llama 3/Mistral), LangChain
 • Vector Store: ChromaDB or Pinecone
 • Data Processing: pandas, plotly
 • Auth: JWT
 • File Storage: Local ‎`/storage`

1. Project Structure
 • ‎`/frontend` — Next.js frontend (already built)
 • ‎`/backend` — FastAPI backend (to be built)
 • ‎`/ai` — AI/ML scripts and utilities (to be built)
 • ‎`/storage` — Uploaded datasets and assets

2. Backend API
 • Framework: FastAPI (Python)
 • Features:
 ▫ User authentication (JWT-based)
 ▫ File upload (CSV, Excel, JSON) with secure local storage
 ▫ Metadata extraction (columns, types, stats)
 ▫ Dataset management (CRUD)
 ▫ Report scheduler/generator endpoints
 ▫ Q&A endpoint (user question, AI answer)
 ▫ Alerts, trends, anomaly endpoints
 ▫ User preferences/settings
 • Persistence: SQLite with SQLAlchemy ORM

3. AI/ML Layer
 • Data Parsing: ‎`pandas` for CSV/Excel
 • Conversational Analytics/Q&A: Groq API (Llama 3/Mistral) via LangChain
 • Vector Embeddings & RAG: sentence-transformers or Groq embeddings; vector store with ChromaDB or Pinecone
 • Agentic Features: LangChain agents to automate report generation, detect anomalies/trends, and suggest analyses

4. Integration
 • Document endpoints with OpenAPI
 • All responses as JSON matching frontend expectations
 • Replace frontend mock data with real API calls (JWT auth)
 • File uploads via multipart/form-data
 • Handle loading/error states in frontend

5. Deployment & Local Dev
 • Frontend: Vercel
 • Backend: Render
 • Use environment variables for all secrets

6. Documentation
 • API docs, setup, and usage in ‎`/docs`
 • README in ‎`/backend` and ‎`/ai` with setup instructions

7. Tech Stack
 • Frontend: Next.js (TypeScript, shadcn/ui/Chakra UI, dark chat UI)
 • Backend: FastAPI, SQLite, SQLAlchemy
 • AI/LLM: Groq API, LangChain
 • Vector Store: ChromaDB or Pinecone

 FastAPI backend with all required endpoints and SQLite models
 2. File upload and parsing
 3. LLM-powered features (summarization, Q&A, analytics) via Groq + LangChain
 4. Vector search and RAG
 5. Connect frontend to backend, replacing mock data
 6. Polish UI/UX for chat, planner/reports, and insights
 7. Deploy and document

