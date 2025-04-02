from openai import OpenAI
from typing import List, Dict, Any, Tuple
from freeOpenAITools import (
    load_csv_tool_definition,
    load_pdf_tool_definition,
    extract_pdf_names_tool_definition,
    extract_urls_tool_definition,
    duckduckgo_search_tool_definition,
    download_file_tool_definition,
    html_to_clean_markdown_tool_definition,
    scrape_meta_tags_tool_definition,
    rag_tool_definition,
    process_tool_calls
)

def get_agent_config(agent_type: str) -> Tuple[str, List[Dict]]:
    """
    Get the configuration for a specific type of agent.

    Args:
        agent_type (str): The type of agent to configure ("Financial Analyst", "Librarian", or "Internet Researcher")

    Returns:
        tuple: A tuple containing (system_message, tools)
    """
    if agent_type == "Financial Analyst":
        system_message = """You are a skilled Financial Analyst AI assistant focused on data analysis and financial insights.
        
Your primary responsibilities are:
1. Analyze financial data from various sources
2. Provide market insights and trends
3. Generate comprehensive reports

When analyzing data:
- Use load_csv for CSV data (automatically handles large files with RAG)
- Use load_pdf for PDF documents (automatically handles large files with RAG)
- Use duckduckgo_search for current market information
- Combine data from multiple sources for comprehensive analysis

Use the available tools to gather comprehensive data and provide well-reasoned analysis."""

        tools = [
            load_csv_tool_definition(),
            load_pdf_tool_definition(),
            extract_urls_tool_definition(),
            duckduckgo_search_tool_definition(),
            download_file_tool_definition(),
            rag_tool_definition()
        ]
        
    elif agent_type == "Librarian":
        system_message = """You are a skilled Librarian AI assistant focused on document management and analysis.
        
You have access to several tools to help with document management and analysis:
- Use load_pdf to read PDF files (automatically handles large files with RAG)
- Use load_csv for CSV data (automatically handles large files with RAG)
- Use extract_pdf_names to find PDF file names mentioned in the user's message
- Use extract_urls to find URLs in text
- Use duckduckgo_search for additional context

Follow this strict workflow for handling document requests:

1. Initial Document Location:
   - First, use extract_pdf_names to check if the user explicitly mentioned any PDF files
   - If no exact file is mentioned, use duckduckgo_search to find relevant information
   - NEVER say "please hold" or "please wait" without showing results

2. Document Processing:
   - After identifying the document, use the appropriate tool:
     * For PDFs: use load_pdf (handles RAG automatically)
     * For CSVs: use load_csv (handles RAG automatically)
   - Then present a complete analysis of ALL the data

Remember:
- ALWAYS provide comprehensive analysis
- Use RAG automatically for large documents
- Combine information from multiple sources when relevant"""

        tools = [
            load_csv_tool_definition(),
            load_pdf_tool_definition(),
            extract_pdf_names_tool_definition(),
            extract_urls_tool_definition(),
            duckduckgo_search_tool_definition(),
            rag_tool_definition()
        ]
        
    elif agent_type == "Internet Researcher":
        tools = [
            download_file_tool_definition(),
            scrape_meta_tags_tool_definition(),
            duckduckgo_search_tool_definition(),
            extract_urls_tool_definition(),
            html_to_clean_markdown_tool_definition(),
            rag_tool_definition()
        ]
        
        system_message = """You are a skilled Internet Researcher AI assistant focused on gathering and analyzing online information.
        
Your primary responsibilities are:
1. Search for relevant information online
2. Download and analyze web content
3. Extract key insights from online sources
4. Present findings in a clear, organized manner

Always use your tools when appropriate, especially:
- Use duckduckgo_search to find relevant information online
- Use extract_urls to find URLs in text
- Use download_file to save important documents
- Use scrape_meta_tags to get webpage metadata
- Use html_to_clean_markdown to clean up web content
- Use rag for processing large documents

When conducting research:
1. Start with reliable sources
2. Verify information across multiple sources
3. Present findings with proper citations
4. Use RAG for processing large documents automatically"""

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    return system_message, tools

def create_financial_analyst(model_name: str = "gpt-4-1106-preview") -> OpenAI:
    """
    Creates a financial analyst agent with relevant tools and expertise.
    
    Args:
        model_name (str): The OpenAI model to use. Defaults to GPT-4 Turbo.
        
    Returns:
        OpenAI: Configured OpenAI client for financial analysis
    """
    system_message, tools = get_agent_config("Financial Analyst")
    
    client = OpenAI()
    
    # Initialize the client with the system prompt
    messages = [{"role": "system", "content": system_message}]
    
    # Create the completion with tools
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools
    )
    
    return completion

def create_librarian(model_name: str = "gpt-4-1106-preview") -> OpenAI:
    """
    Creates a librarian agent with tools for document analysis.
    
    Args:
        model_name (str): The OpenAI model to use. Defaults to GPT-4 Turbo.
        
    Returns:
        OpenAI: Configured OpenAI client for document analysis
    """
    system_message, tools = get_agent_config("Librarian")
    
    client = OpenAI()
    
    # Initialize the client with the system prompt
    messages = [{"role": "system", "content": system_message}]
    
    # Create the completion with tools
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools
    )
    
    return completion

def create_internet_researcher(model_name: str = "gpt-4-1106-preview") -> OpenAI:
    """
    Creates an internet researcher agent with tools for online research.
    
    Args:
        model_name (str): The OpenAI model to use. Defaults to GPT-4 Turbo.
        
    Returns:
        OpenAI: Configured OpenAI client for online research
    """
    system_message, tools = get_agent_config("Internet Researcher")
    
    client = OpenAI()
    
    # Initialize the client with the system prompt
    messages = [{"role": "system", "content": system_message}]
    
    # Create the completion with tools
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools
    )
    
    return completion

def process_agent_response(agent_response: Any, conversation: Dict) -> Dict:
    """
    Process the agent's response and execute any tool calls.
    
    Args:
        agent_response: The response from the OpenAI API
        conversation: The conversation context with tool calls
        
    Returns:
        Dict: Results of processing the tool calls
    """
    if not agent_response.tool_calls:
        return {"type": "message", "content": agent_response.content}
    
    # Process tool calls
    results = process_tool_calls(conversation)
    
    return {
        "type": "tool_calls",
        "results": results,
        "message": agent_response.content
    }

def test_agent_config():
    """
    Test the agent configuration function.
    """
    system_message, tools = get_agent_config("Librarian")
    print(system_message)
    print(tools)
