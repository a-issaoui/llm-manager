#!/usr/bin/env python3
"""
Example 04: LangChain Integration

Demonstrates:
- Using llm_manager server with LangChain
- Chains and agents
- RAG (Retrieval Augmented Generation)
- Memory and conversation history

Prerequisites:
    pip install langchain langchain-openai

Usage:
    # Start server
    llm-manager --port 8000
    
    # Run example
    python examples/04_agents_langchain.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_manager import get_config


def get_first_model():
    """Get first available model from registry."""
    models_file = get_config().models.get_registry_path()
    if models_file.exists():
        with open(models_file) as f:
            models = json.load(f)
        if models:
            return list(models.keys())[0]
    return None


def example_basic_langchain():
    """Basic LangChain usage."""
    print("=" * 60)
    print("LangChain: Basic Usage")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # Create LLM pointing to local server
        llm = ChatOpenAI(
            model=model_name,
            openai_api_base="http://localhost:8000/v1",
            openai_api_key="not-needed",
            temperature=0.7
        )
        
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the capital of France?")
        ]
        
        print("Sending request...")
        response = llm.invoke(messages)
        print(f"Response: {response.content}\n")
        
    except ImportError:
        print("Install LangChain: pip install langchain langchain-openai\n")
    except Exception as e:
        print(f"Note: {e}")
        print("Make sure server is running\n")


def example_chain():
    """Simple chain example."""
    print("=" * 60)
    print("LangChain: Simple Chain")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ChatOpenAI(
            model=model_name,
            openai_api_base="http://localhost:8000/v1",
            openai_api_key="not-needed"
        )
        
        template = """
        You are a helpful coding assistant.
        Write a {language} function to {task}.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        # Use modern pipe operator for chaining
        chain = prompt | llm
        
        result = chain.invoke({
            "language": "Python",
            "task": "reverse a string"
        })
        
        print(f"Result: {result.content}\n")
        
    except Exception as e:
        print(f"Note: {e}\n")


def example_conversation_memory():
    """Conversation with memory."""
    print("=" * 60)
    print("LangChain: Conversation with Memory")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, AIMessage
        
        llm = ChatOpenAI(
            model=model_name,
            openai_api_base="http://localhost:8000/v1",
            openai_api_key="not-needed"
        )
        
        # Simple conversation with manual message management
        messages = [HumanMessage(content="Hi, I'm Alice!")]
        
        print("Turn 1:")
        r1 = llm.invoke(messages)
        print(f"AI: {r1.content}\n")
        messages.append(AIMessage(content=r1.content))
        
        messages.append(HumanMessage(content="What's my name?"))
        print("Turn 2:")
        r2 = llm.invoke(messages)
        print(f"AI: {r2.content}\n")
        
    except Exception as e:
        print(f"Note: {e}\n")


def example_rag():
    """RAG example."""
    print("=" * 60)
    print("LangChain: RAG (Retrieval Augmented Generation)")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    example_code = f'''
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load documents
loader = TextLoader("documents.txt")
documents = loader.load()

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",  # Or local embedding model
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="not-needed"
)
vectorstore = Chroma.from_documents(texts, embeddings)

# Create RAG chain
llm = ChatOpenAI(
    model="{model_name}",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="not-needed"
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Query
text = "What information is in the documents about..."
print(qa.run(text))
'''
    print(example_code)


def example_agent_with_tools():
    """Agent with tools."""
    print("=" * 60)
    print("LangChain: Agent with Tools")
    print("=" * 60)
    
    model_name = get_first_model()
    if not model_name:
        print("No models found in registry")
        return
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        
        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"
        
        @tool
        def calculate(expression: str) -> str:
            """Calculate mathematical expression."""
            try:
                return str(eval(expression))
            except:
                return "Error in calculation"

        llm = ChatOpenAI(
            model=model_name,
            openai_api_base="http://localhost:8000/v1",
            openai_api_key="not-needed",
            temperature=0
        )
        
        tools = [search, calculate]
        
        # Simple tool usage without complex agent
        print("Using tools directly...")
        print(f"Search: {search.invoke('Python')}")
        print(f"Calculate: {calculate.invoke('25 * 4')}")
        print()
        
    except Exception as e:
        print(f"Note: {e}\n")


def main():
    """Run LangChain examples."""
    print("\n" + "=" * 60)
    print("LLM Manager - LangChain Integration Examples")
    print("=" * 60)
    print("\nPrerequisites:")
    print("  pip install langchain langchain-openai")
    print("  llm-manager --port 8000\n")
    
    example_basic_langchain()
    example_chain()
    example_conversation_memory()
    example_rag()
    example_agent_with_tools()


if __name__ == "__main__":
    main()
