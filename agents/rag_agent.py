"""
RAG Agent
Retrieval-Augmented Generation agent with coding capabilities.
Retrieves relevant code snippets from the knowledge base to assist with
visualization and data manipulation tasks.
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, BASE_DIR


class RAGAgent:
    """RAG Agent with code retrieval and memory capabilities"""
    
    def __init__(self):
        """Initialize the RAG agent with vector store and LLM"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'}
        )
        
        # Load pre-built vector store
        self.vectorstore = self._load_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Initialize LLM
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=OLLAMA_TEMPERATURE
        )
        
        # Memory for intermediate data and conversation
        self.memory: Dict[str, Any] = {
            "intermediate_data": {},  # Store transformed DataFrames
            "conversation_history": [],  # Store chat messages
            "code_history": []  # Store generated code
        }
        
        # Create the RAG chain
        self.chain = self._create_chain()
    
    def _load_vectorstore(self) -> FAISS:
        """Load the pre-built FAISS vector store"""
        vectorstore_path = BASE_DIR / "data" / "knowledge_database"
        
        if not (vectorstore_path / "index.faiss").exists():
            raise FileNotFoundError(
                f"Vector store not found at {vectorstore_path}. "
                "Please ensure index.faiss and index.pkl exist."
            )
        
        return FAISS.load_local(
            str(vectorstore_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
    
    def _create_chain(self):
        """Create the RAG chain with retrieval and generation"""
        
        template = """You are an expert data visualization and Python coding assistant. 
Your role is to help generate code for data analysis and visualization using matplotlib and seaborn.

Use the following retrieved code examples as reference for generating your response:

RETRIEVED CODE EXAMPLES:
{context}

CONVERSATION HISTORY:
{chat_history}

AVAILABLE INTERMEDIATE DATA:
{intermediate_data}

CURRENT DATA CONTEXT:
{data_context}

USER REQUEST:
{question}

INSTRUCTIONS:
1. Use the retrieved code examples as reference for syntax and best practices
2. Generate clean, executable Python code that works with pandas DataFrames
3. Use matplotlib and/or seaborn for visualizations
4. If intermediate data is available, reference it in your code
5. Do NOT include imports for pandas, numpy, matplotlib, or seaborn — they are already imported as pd, np, plt, sns
6. Add comments explaining key steps
7. Return the code in a markdown code block
8. CRITICAL: The dataset is ALREADY loaded as a variable called 'df'. NEVER use pd.read_csv(), pd.read_excel(), or any file-loading function. Always use the 'df' variable directly. The DataState singleton is available as 'state' if you need to save figures or update the active dataframe.

RESPONSE:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n---\n\n".join([
                f"Example {i+1}:\n```python\n{doc.page_content}\n```" 
                for i, doc in enumerate(docs)
            ])
        
        def format_history(memory):
            history = self.memory.get("conversation_history", [])
            if not history:
                return "No previous conversation."
            return "\n".join([
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:200]}..."
                if len(m.content) > 200 else 
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in history[-6:]  # Last 6 messages
            ])
        
        def format_intermediate_data(memory):
            data = self.memory.get("intermediate_data", {})
            if not data:
                return "No intermediate data stored."
            info = []
            for name, df_info in data.items():
                info.append(f"- {name}: {df_info['shape'][0]} rows × {df_info['shape'][1]} cols, columns: {', '.join(df_info['columns'][:5])}...")
            return "\n".join(info)
        
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: format_history(self.memory),
                "intermediate_data": lambda x: format_intermediate_data(self.memory),
                "data_context": lambda x: self._get_data_context()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def _get_data_context(self) -> str:
        """Get current data context from DataState"""
        try:
            from utils.data_state import DataState
            state = DataState()
            
            if not state.is_data_loaded():
                return "No data currently loaded."
            
            df = state.get_dataframe()
            file_info = state.get_file_info()
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            return f"""
- File: {file_info['file_name']}
- Shape: {file_info['rows']:,} rows × {file_info['columns']} columns
- Numeric columns: {', '.join(numeric_cols[:10])}
- Categorical columns: {', '.join(cat_cols[:10])}
- Sample data types: {dict(df.dtypes.head())}
"""
        except Exception as e:
            return f"Error getting data context: {str(e)}"
    
    def invoke(self, query: str) -> Dict[str, Any]:
        """
        Process a query and return code/response.
        
        Args:
            query: User's natural language query
            
        Returns:
            Dict with 'response', 'retrieved_docs', and 'code' keys
        """
        try:
            # Add to conversation history
            self.memory["conversation_history"].append(HumanMessage(content=query))
            
            # Retrieve relevant documents
            retrieved_docs = self.retriever.invoke(query)
            
            # Generate response
            response = self.chain.invoke(query)
            
            # Add response to history
            self.memory["conversation_history"].append(AIMessage(content=response))
            
            # Extract code if present
            code = self._extract_code(response)
            if code:
                self.memory["code_history"].append({
                    "query": query,
                    "code": code
                })
            
            return {
                "response": response,
                "retrieved_docs": [doc.page_content[:200] + "..." for doc in retrieved_docs],
                "code": code
            }
        except Exception as e:
            return {
                "response": f"Error processing query: {str(e)}",
                "retrieved_docs": [],
                "code": None
            }
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from markdown code blocks"""
        import re
        pattern = r"```python\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        return matches[0].strip() if matches else None
    
    def store_intermediate_data(self, name: str, df) -> str:
        """
        Store intermediate DataFrame for later use.
        
        Args:
            name: Name to store the data under
            df: pandas DataFrame to store
            
        Returns:
            Confirmation message
        """
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            return "Error: Can only store pandas DataFrames"
        
        self.memory["intermediate_data"][name] = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": dict(df.dtypes.astype(str)),
            "df": df  # Store actual DataFrame
        }
        
        return f"Stored '{name}' ({df.shape[0]} rows × {df.shape[1]} cols)"
    
    def get_intermediate_data(self, name: str):
        """Retrieve stored intermediate DataFrame"""
        data = self.memory["intermediate_data"].get(name)
        if data:
            return data.get("df")
        return None
    
    def list_intermediate_data(self) -> str:
        """List all stored intermediate data"""
        data = self.memory["intermediate_data"]
        if not data:
            return "No intermediate data stored."
        
        result = "**Stored Intermediate Data:**\n"
        for name, info in data.items():
            result += f"- `{name}`: {info['shape'][0]} rows × {info['shape'][1]} cols\n"
            result += f"  Columns: {', '.join(info['columns'][:5])}{'...' if len(info['columns']) > 5 else ''}\n"
        
        return result
    
    def clear_memory(self, what: str = "all") -> str:
        """
        Clear memory components.
        
        Args:
            what: 'all', 'conversation', 'intermediate', or 'code'
        """
        if what == "all":
            self.memory = {
                "intermediate_data": {},
                "conversation_history": [],
                "code_history": []
            }
            return "All memory cleared."
        elif what == "conversation":
            self.memory["conversation_history"] = []
            return "Conversation history cleared."
        elif what == "intermediate":
            self.memory["intermediate_data"] = {}
            return "Intermediate data cleared."
        elif what == "code":
            self.memory["code_history"] = []
            return "Code history cleared."
        else:
            return f"Unknown memory type: {what}"
    
    def get_code_history(self) -> List[Dict]:
        """Get all generated code from this session"""
        return self.memory.get("code_history", [])
