import os
import json
from typing import Dict, Any, List, Optional
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
import numpy as np
import pickle
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVectorStore:
    """Simple vector store for RAG functionality using scikit-learn"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        """Add documents to the vector store"""
        if metadata is None:
            metadata = [{}] * len(texts)
            
        for text, meta in zip(texts, metadata):
            self.documents.append(text)
            self.metadata.append(meta)
            
            # Generate embedding
            embedding = self.embedding_model.encode(text)
            self.embeddings.append(embedding)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if not self.embeddings:
            return []
            
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate cosine similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top k similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for similarity
                results.append({
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'similarity_score': similarities[idx]
                })
        
        return results
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.embeddings = data['embeddings']
                self.metadata = data['metadata']

class GroqLLMService:
    """Service for interacting with Groq LLM API."""
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name="llama3-70b-8192",
            temperature=0.1
        )
        
        # Initialize vector store
        self.vector_store = SimpleVectorStore()
        self.vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_stores")
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Text splitter for chunking data
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def embed_dataset(self, dataset_id: int, file_path: str, file_type: str, 
                     columns_info: Dict[str, Any]) -> bool:
        """Embed dataset content for RAG."""
        try:
            # Load data
            if file_type == 'csv':
                df = pd.read_csv(file_path)
            elif file_type == 'excel':
                df = pd.read_excel(file_path)
            elif file_type == 'json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Create text representations of the data
            documents = []
            
            # Add column information
            columns_text = f"Dataset has {len(df.columns)} columns: {', '.join(df.columns)}"
            documents.append(columns_text)
            
            # Add statistical summary
            stats_text = f"Dataset statistics:\n{df.describe().to_string()}"
            documents.append(stats_text)
            
            # Add sample data
            sample_text = f"Sample data:\n{df.head(10).to_string()}"
            documents.append(sample_text)
            
            # Add data types info
            dtypes_text = f"Data types:\n{df.dtypes.to_string()}"
            documents.append(dtypes_text)
            
            # Add null values info
            nulls_text = f"Null values:\n{df.isnull().sum().to_string()}"
            documents.append(nulls_text)
            
            # Add unique values info for categorical columns
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].nunique() < 20:
                    unique_text = f"Unique values in {col}: {df[col].unique().tolist()}"
                    documents.append(unique_text)
            
            # Create metadata
            metadata = [{'dataset_id': dataset_id, 'type': 'dataset_info'}] * len(documents)
            
            # Add to vector store
            self.vector_store.add_documents(documents, metadata)
            
            # Save vector store
            store_path = os.path.join(self.vector_store_path, f"dataset_{dataset_id}.pkl")
            self.vector_store.save(store_path)
            
            logger.info(f"Successfully embedded dataset {dataset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error embedding dataset {dataset_id}: {str(e)}")
            return False
    
    def load_dataset_embeddings(self, dataset_id: int) -> bool:
        """Load embeddings for a specific dataset."""
        try:
            store_path = os.path.join(self.vector_store_path, f"dataset_{dataset_id}.pkl")
            if os.path.exists(store_path):
                self.vector_store.load(store_path)
                logger.info(f"Loaded embeddings for dataset {dataset_id}")
                return True
            else:
                logger.warning(f"No embeddings found for dataset {dataset_id}")
                return False
        except Exception as e:
            logger.error(f"Error loading embeddings for dataset {dataset_id}: {str(e)}")
            return False
    
    def chat_with_data(self, query: str, dataset_id: int, conversation_history: List[Dict] = None) -> str:
        """Chat with dataset using RAG."""
        try:
            # Load dataset embeddings
            if not self.load_dataset_embeddings(dataset_id):
                return "Sorry, I couldn't find the dataset embeddings. Please make sure the dataset is properly processed."
            
            # Search for relevant context
            relevant_docs = self.vector_store.search(query, top_k=5)
            
            if not relevant_docs:
                return "I couldn't find relevant information in the dataset to answer your question."
            
            # Prepare context
            context = "\n\n".join([doc['text'] for doc in relevant_docs])
            
            # Create prompt
            prompt_template = """
            You are a data analyst assistant. Based on the dataset information provided below, answer the user's question accurately and concisely.
            
            Dataset Information:
            {context}
            
            User Question: {query}
            
            Previous Conversation:
            {conversation_history}
            
            Please provide a clear, data-driven answer based on the dataset information. If you cannot answer the question based on the provided data, please say so.
            
            Answer:
            """
            
            # Format conversation history
            history_text = ""
            if conversation_history:
                for msg in conversation_history[-5:]:  # Last 5 messages
                    history_text += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "query", "conversation_history"]
            )
            
            # Create chain
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Get response
            response = chain.run(
                context=context,
                query=query,
                conversation_history=history_text
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in chat_with_data: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def generate_insights(self, dataset_id: int, analysis_type: str = "general") -> str:
        """Generate insights from dataset."""
        try:
            # Load dataset embeddings
            if not self.load_dataset_embeddings(dataset_id):
                return "Could not load dataset for analysis."
            
            # Create insight query based on analysis type
            if analysis_type == "general":
                query = "What are the key patterns, trends, and insights from this dataset?"
            elif analysis_type == "statistical":
                query = "What are the statistical insights and distributions in this dataset?"
            elif analysis_type == "anomalies":
                query = "What anomalies or outliers are present in this dataset?"
            elif analysis_type == "correlations":
                query = "What are the correlations and relationships between variables?"
            else:
                query = f"Provide {analysis_type} insights from this dataset"
            
            # Search for relevant context
            relevant_docs = self.vector_store.search(query, top_k=10)
            
            if not relevant_docs:
                return "Could not find sufficient data for analysis."
            
            # Prepare context
            context = "\n\n".join([doc['text'] for doc in relevant_docs])
            
            # Create insight prompt
            prompt_template = """
            You are a senior data analyst. Based on the dataset information provided below, generate comprehensive insights and recommendations.
            
            Dataset Information:
            {context}
            
            Analysis Type: {analysis_type}
            
            Please provide:
            1. Key insights and patterns
            2. Statistical observations
            3. Potential business implications
            4. Recommendations for further analysis
            5. Any data quality issues or limitations
            
            Generate a comprehensive analysis report:
            """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "analysis_type"]
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            response = chain.run(
                context=context,
                analysis_type=analysis_type
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return f"Error generating insights: {str(e)}"
    
    def generate_report(self, dataset_id: int, report_type: str = "comprehensive") -> str:
        """Generate automated report from dataset."""
        try:
            # Load dataset embeddings
            if not self.load_dataset_embeddings(dataset_id):
                return "Could not load dataset for report generation."
            
            # Search for all relevant information
            relevant_docs = self.vector_store.search("dataset summary statistics overview", top_k=15)
            
            if not relevant_docs:
                return "Could not find sufficient data for report generation."
            
            # Prepare context
            context = "\n\n".join([doc['text'] for doc in relevant_docs])
            
            # Create report prompt
            prompt_template = """
            You are a professional data analyst creating a comprehensive data report. Based on the dataset information provided below, create a detailed report.
            
            Dataset Information:
            {context}
            
            Report Type: {report_type}
            
            Please create a well-structured report with the following sections:
            
            # Dataset Analysis Report
            
            ## Executive Summary
            [Brief overview of key findings]
            
            ## Dataset Overview
            [Dataset description, size, columns, data types]
            
            ## Key Statistics
            [Statistical summary of important metrics]
            
            ## Data Quality Assessment
            [Missing values, data types, potential issues]
            
            ## Key Insights
            [Main patterns, trends, and observations]
            
            ## Recommendations
            [Actionable recommendations based on the data]
            
            ## Limitations
            [Any limitations or caveats in the analysis]
            
            Generate the report:
            """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "report_type"]
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            response = chain.run(
                context=context,
                report_type=report_type
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def suggest_analyses(self, dataset_id: int) -> List[str]:
        """Suggest potential analyses for a dataset."""
        try:
            # Load dataset embeddings
            if not self.load_dataset_embeddings(dataset_id):
                return ["Could not load dataset for analysis suggestions."]
            
            # Search for dataset structure information
            relevant_docs = self.vector_store.search("columns data types statistics", top_k=10)
            
            if not relevant_docs:
                return ["Could not analyze dataset structure."]
            
            # Prepare context
            context = "\n\n".join([doc['text'] for doc in relevant_docs])
            
            # Create suggestion prompt
            prompt_template = """
            You are a data science consultant. Based on the dataset information provided below, suggest 5-10 specific analyses that would be valuable for this dataset.
            
            Dataset Information:
            {context}
            
            Please suggest analyses in the following format:
            - [Analysis Name]: [Brief description of what this analysis would reveal]
            
            Focus on practical, actionable analyses that match the data structure and business context.
            
            Suggested Analyses:
            """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context"]
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            response = chain.run(context=context)
            
            # Parse suggestions into a list
            suggestions = []
            for line in response.strip().split('\n'):
                if line.strip().startswith('-'):
                    suggestions.append(line.strip()[1:].strip())
            
            return suggestions if suggestions else ["No specific suggestions available."]
            
        except Exception as e:
            logger.error(f"Error suggesting analyses: {str(e)}")
            return [f"Error suggesting analyses: {str(e)}"]
    
    def detect_insights(self, dataset_id: int, columns_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect proactive insights from dataset metadata."""
        try:
            insights = []
            
            # Load dataset embeddings
            if not self.load_dataset_embeddings(dataset_id):
                return [{"title": "Dataset Not Available", "description": "Could not load dataset for analysis.", "type": "error", "severity": "high"}]
            
            # Analyze the dataset structure for insights
            if "basic_info" in columns_info:
                basic_info = columns_info["basic_info"]
                
                # Check for missing data issues
                total_cells = basic_info.get("shape", [0, 0])[0] * basic_info.get("shape", [0, 0])[1]
                null_counts = basic_info.get("null_counts", {})
                total_nulls = sum(null_counts.values())
                
                if total_nulls > total_cells * 0.1:  # More than 10% missing
                    insights.append({
                        "title": "High Missing Data Detected",
                        "description": f"Dataset has {total_nulls:,} missing values ({(total_nulls/total_cells)*100:.1f}% of all data)",
                        "type": "data_quality",
                        "severity": "high" if total_nulls > total_cells * 0.2 else "medium"
                    })
                
                # Check for duplicate rows
                duplicate_rows = basic_info.get("duplicate_rows", 0)
                total_rows = basic_info.get("shape", [0, 0])[0]
                if duplicate_rows > 0:
                    insights.append({
                        "title": "Duplicate Records Found",
                        "description": f"Dataset contains {duplicate_rows:,} duplicate rows ({(duplicate_rows/total_rows)*100:.1f}% of data)",
                        "type": "data_quality",
                        "severity": "medium" if duplicate_rows > total_rows * 0.05 else "low"
                    })
            
            # Analyze correlations
            if "correlations" in columns_info and "strong_correlations" in columns_info["correlations"]:
                strong_corr = columns_info["correlations"]["strong_correlations"]
                if strong_corr:
                    insights.append({
                        "title": "Strong Variable Correlations",
                        "description": f"Found {len(strong_corr)} pairs of strongly correlated variables that could be used for predictive modeling",
                        "type": "correlation",
                        "severity": "low"
                    })
            
            # Analyze data quality score
            if "data_quality" in columns_info:
                quality_score = columns_info["data_quality"].get("quality_score", 100)
                if quality_score < 80:
                    insights.append({
                        "title": "Data Quality Issues",
                        "description": f"Dataset quality score is {quality_score:.1f}/100. Consider data cleaning before analysis.",
                        "type": "data_quality",
                        "severity": "high" if quality_score < 60 else "medium"
                    })
            
            # Analyze column details for outliers
            if "column_details" in columns_info:
                outlier_columns = []
                for col, details in columns_info["column_details"].items():
                    if details.get("has_outliers", False):
                        outlier_count = details.get("outliers", {}).get("count", 0)
                        if outlier_count > 0:
                            outlier_columns.append(f"{col} ({outlier_count} outliers)")
                
                if outlier_columns:
                    insights.append({
                        "title": "Outliers Detected",
                        "description": f"Found outliers in columns: {', '.join(outlier_columns[:3])}{'...' if len(outlier_columns) > 3 else ''}",
                        "type": "anomaly",
                        "severity": "medium"
                    })
            
            # Business insights from patterns
            if "patterns" in columns_info and "business_insights" in columns_info["patterns"]:
                business_insights = columns_info["patterns"]["business_insights"]
                if business_insights:
                    insights.append({
                        "title": "Business Metrics Available",
                        "description": f"Dataset contains business-relevant metrics: {', '.join(business_insights[:2])}",
                        "type": "business",
                        "severity": "low"
                    })
            
            # If no insights found, add a default one
            if not insights:
                insights.append({
                    "title": "Dataset Ready for Analysis",
                    "description": "Dataset appears to be in good condition and ready for analysis",
                    "type": "general",
                    "severity": "low"
                })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error detecting insights: {str(e)}")
            return [{
                "title": "Insight Detection Error",
                "description": f"Error analyzing dataset: {str(e)}",
                "type": "error",
                "severity": "high"
            }]
