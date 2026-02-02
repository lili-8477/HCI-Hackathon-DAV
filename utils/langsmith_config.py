"""
LangSmith Configuration
Configuration for LangSmith tracing and monitoring.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_langsmith(project_name: str = "HCI-Hackathon-DAV"):
    """
    Configure LangSmith tracing.
    
    To use LangSmith:
    1. Sign up at https://smith.langchain.com/
    2. Get your API key from Settings
    3. Set environment variables:
       - LANGCHAIN_TRACING_V2=true
       - LANGCHAIN_API_KEY=your_api_key
       - LANGCHAIN_PROJECT=your_project_name
    
    Args:
        project_name: Name of the LangSmith project
    """
    # Check if LangSmith is configured
    api_key = os.getenv("LANGCHAIN_API_KEY")
    
    if api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = project_name
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        print(f"✅ LangSmith tracing enabled for project: {project_name}")
        return True
    else:
        print("⚠️ LangSmith not configured. Set LANGCHAIN_API_KEY to enable tracing.")
        return False


def disable_langsmith():
    """Disable LangSmith tracing"""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("LangSmith tracing disabled.")


def get_langsmith_status() -> dict:
    """Get current LangSmith configuration status"""
    return {
        "enabled": os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        "api_key_set": bool(os.getenv("LANGCHAIN_API_KEY")),
        "project": os.getenv("LANGCHAIN_PROJECT", "Not set"),
        "endpoint": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    }


# Auto-configure on import if environment variables are set
if os.getenv("LANGCHAIN_API_KEY"):
    setup_langsmith()
