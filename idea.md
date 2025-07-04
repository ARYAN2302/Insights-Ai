Conversational GenAI Business Insights Suite

Overview

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

