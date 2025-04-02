import re
import requests
import csv
import json
import fitz  # PyMuPDF
import os
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore', message='.*Examining the path of torch.classes.*')
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path
import pickle
import tiktoken
from bs4 import BeautifulSoup
import yfinance as yf
import html2text

def define_tool(name, description, parameters):
    """
    Generates a JSON definition for an OpenAI tool.

    Args:
        name (str): Name of the tool.
        description (str): Description of what the tool does.
        parameters (dict): A dictionary of parameters, including their types and descriptions.

    Returns:
        dict: A tool definition compatible with OpenAI's API.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
        }
    }


def download_file(url):
    """
    Downloads any file type from a given URL and saves it locally in a downloads folder.
    The file name is automatically determined from the URL.

    Args:
        url (str): The URL of the file to download.

    Returns:
        str: Path to the saved file.
    """
    # Create downloads directory if it doesn't exist
    downloads_dir = os.path.join(os.getcwd(), "knowledge_files")
    os.makedirs(downloads_dir, exist_ok=True)
    
    # Extract the file name from the URL
    file_name = os.path.basename(url.split("?")[0])  # Remove query parameters if present
    
    if not file_name:  # Fallback name if URL doesn't contain a file name
        file_name = "downloaded_file"

    # Create full file path
    file_path = os.path.join(downloads_dir, file_name)
    
    # Download the file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return file_path
    else:
        raise Exception(f"Failed to download file: Status code {response.status_code}")


def download_file_tool_definition():
    """
    Returns the OpenAI tool definition for the download_file function.
    """
    return define_tool(
        name="download_file",
        description="Given a URL, this function downloads a file from the internet and saves it locally to be accessed later.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the file to download."
                }
            },
            "required": ["url"]
        }
    )


def extract_urls(text):
    """
    Extracts URLs from the given text using a regex pattern.

    Args:
        text (str): The input text to search for URLs.

    Returns:
        list: A list of extracted URLs.
    """
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    urls = re.findall(url_pattern, text)
    return urls


def extract_pdf_names(text):
    """
    Extracts PDF filenames (e.g., 'recipes.pdf') from the given text.

    Args:
        text (str): The input text to search for PDF filenames.

    Returns:
        list: A list of extracted PDF filenames.
    """
    pdf_pattern = r'\b[\w\-.]+\.pdf\b'
    pdfs = re.findall(pdf_pattern, text)
    return pdfs


def scrape_meta_tags(url):
    """
    Extracts meta tags (title, description, keywords) from a webpage.

    Args:
        url (str): The URL of the webpage.

    Returns:
        dict: A dictionary containing title, description, and keywords.
    """
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch webpage: Status code {response.status_code}")
        
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract metadata
        title = soup.title.string if soup.title else "No title found"
        description = soup.find("meta", attrs={"name": "description"})
        keywords = soup.find("meta", attrs={"name": "keywords"})

        return {
            "title": title,
            "description": description["content"] if description else "No description found",
            "keywords": keywords["content"] if keywords else "No keywords found"
        }
    except Exception as e:
        raise Exception(f"Error scraping meta tags: {str(e)}")


def scrape_meta_tags_tool_definition():
    """
    Returns the OpenAI tool definition for the scrape_meta_tags function.
    """
    return define_tool(
        name="scrape_meta_tags",
        description="Extracts metadata (title, description, keywords) from a webpage. Good for getting a quick summary of a url",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the webpage to scrape for meta tags."
                }
            },
            "required": ["url"]
        }
    )


def _convert_csv_to_json(csv_file_path: str, chunk_size: int = 100) -> dict:
    """
    Internal helper function to convert CSV to JSON format.
    """
    try:
        # Handle both absolute and relative paths
        if os.path.isabs(csv_file_path):
            full_path = csv_file_path
        else:
            # Try current directory first
            full_path = os.path.join(os.getcwd(), csv_file_path)
            if not os.path.exists(full_path):
                # Try knowledge_files directory
                knowledge_dir = os.path.join(os.getcwd(), "knowledge_files")
                full_path = os.path.join(knowledge_dir, csv_file_path)
        
        if not os.path.exists(full_path):
            raise Exception(f"CSV file not found at: {full_path}")

        with open(full_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            all_rows = list(reader)
            total_rows = len(all_rows)
            
            # Split into chunks
            chunks = []
            for i in range(0, total_rows, chunk_size):
                chunk = all_rows[i:i + chunk_size]
                chunks.append(chunk)
            
            return {
                "chunks": chunks,
                "total_rows": total_rows,
                "chunk_size": chunk_size,
                "num_chunks": len(chunks),
                "current_chunk": 0,
                "columns": reader.fieldnames
            }

    except Exception as e:
        return {"error": f"Error processing CSV: {str(e)}"}


def load_csv(file_path: str, query: str = "") -> dict:
    """
    Load and process a CSV file, automatically using RAG for large files.
    
    Args:
        file_path (str): Path to the CSV file
        query (str, optional): Specific query to search within the CSV data
        
    Returns:
        dict: A dictionary containing either:
            - For small files: The complete CSV data in JSON format
            - For large files: Relevant chunks from RAG processing
            Also includes metadata about processing method and file info
    """
    try:
        # First convert to JSON to get a consistent format
        json_data = _convert_csv_to_json(file_path)
        if "error" in json_data:
            return json_data
            
        # Convert to string to estimate tokens
        data_str = json.dumps(json_data["chunks"])
        
        # Check if we should use RAG
        if _should_use_rag(data_str):
            # Use RAG for large files
            rag_results = rag_tool(query if query else "", [file_path])
            return {
                "data": rag_results["relevant_chunks"],
                "total_rows": json_data["total_rows"],
                "columns": json_data["columns"],
                "using_rag": True,
                "file_path": file_path,
                "query": query
            }
        else:
            # Return full data for small files
            return {
                "data": json_data["chunks"],
                "total_rows": json_data["total_rows"],
                "columns": json_data["columns"],
                "using_rag": False,
                "file_path": file_path
            }
            
    except Exception as e:
        return {"error": f"Error loading CSV: {str(e)}"}


def load_csv_tool_definition():
    """
    Returns the OpenAI tool definition for the load_csv function.
    """
    return define_tool(
        name="load_csv",
        description="Load and process a CSV file, automatically handling large files with RAG",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the CSV file to load"
                },
                "query": {
                    "type": "string",
                    "description": "Optional query to search within the CSV data"
                }
            },
            "required": ["file_path"]
        }
    )


def _pdf_to_clean_markdown(pdf_path: str) -> str:
    """
    Internal helper function to convert PDF to markdown format.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        # Clean and format the text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Replace multiple newlines with double newline
        
        # Add markdown formatting
        text = f"# {os.path.basename(pdf_path)}\n\n{text}"
        
        return text
        
    except Exception as e:
        return f"Error processing PDF: {str(e)}"


def load_pdf(file_path: str, query: str = "") -> dict:
    """
    Load and process a PDF file, automatically using RAG for large files.
    
    Args:
        file_path (str): Path to the PDF file
        query (str, optional): Specific query to search within the PDF content
        
    Returns:
        dict: A dictionary containing either:
            - For small files: The complete PDF content in markdown format
            - For large files: Relevant chunks from RAG processing
            Also includes metadata about processing method and file info
    """
    try:
        # Handle both absolute and relative paths
        if os.path.isabs(file_path):
            full_path = file_path
        else:
            # Try current directory first
            full_path = os.path.join(os.getcwd(), file_path)
            if not os.path.exists(full_path):
                # Try knowledge_files directory
                knowledge_dir = os.path.join(os.getcwd(), "knowledge_files")
                full_path = os.path.join(knowledge_dir, file_path)
        
        if not os.path.exists(full_path):
            return {"error": f"PDF file not found at: {full_path}"}

        # First convert to markdown to get a consistent format
        content = _pdf_to_clean_markdown(full_path)
        if content.startswith("Error"):
            return {"error": content}
            
        # Check if we should use RAG
        if _should_use_rag(content):
            # Use RAG for large files
            rag_results = rag_tool(query if query else "", [full_path])
            return {
                "content": rag_results["relevant_chunks"],
                "using_rag": True,
                "file_path": full_path,
                "query": query
            }
        else:
            # Return full content for small files
            return {
                "content": content,
                "using_rag": False,
                "file_path": full_path
            }
            
    except Exception as e:
        return {"error": f"Error loading PDF: {str(e)}"}


def load_pdf_tool_definition():
    """
    Returns the OpenAI tool definition for the load_pdf function.
    """
    return define_tool(
        name="load_pdf",
        description="Load and process a PDF file, automatically handling large files with RAG",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file to load"
                },
                "query": {
                    "type": "string",
                    "description": "Optional query to search within the PDF content"
                }
            },
            "required": ["file_path"]
        }
    )


def html_to_clean_markdown(html_content):
    """
    Cleans HTML content and converts it into Markdown format.

    Args:
        html_content (str): The raw HTML content as a string.

    Returns:
        str: Cleaned and formatted Markdown text.
    """
    try:
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Get clean text
        cleaned_text = soup.get_text(separator=" ", strip=True)

        # Convert cleaned text to Markdown
        h = html2text.HTML2Text()
        h.body_width = 0  # No wrapping
        markdown_text = h.handle(html_content)

        return markdown_text.strip()
    except Exception as e:
        raise Exception(f"Error converting HTML to Markdown: {str(e)}")
    
def html_to_clean_markdown_tool_definition():
    """
    Returns the OpenAI tool definition for the html_to_clean_markdown function.
    """
    return define_tool(
        name="html_to_clean_markdown",
        description="Cleans raw HTML content and converts it into formatted Markdown text. Good for getting in depth longer review of URL",
        parameters={
            "type": "object",
            "properties": {
                "html_content": {
                    "type": "string",
                    "description": "The raw HTML content to clean and convert to Markdown."
                }
            },
            "required": ["html_content"]
        }
    )

def get_stock_cur_info(ticker):
    """
    Fetches the current stock price for a given ticker symbol.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL" for Apple).

    Returns:
        dict: A dictionary containing the ticker and its current price.
    """
    print(f"ticker:{ticker}")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        print("NO ERROR FOR STOCK INFO")
        return {
            "ticker": ticker,
            "all_info": info
        }
    except Exception as e:
        print(f"Error fetching stock price: {str(e)}")
        raise Exception(f"Error fetching stock price: {str(e)}")
    
def get_stock_cur_info_tool_definition():
    """
    Returns the OpenAI tool definition for the get_stock_cur_info function.
    """
    return define_tool(
        name="get_stock_cur_info",
        description="Get the current stock information for a given ticker symbol",
        parameters={
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., 'AAPL' for Apple)"
                }
            },
            "required": ["ticker"]
        }
    )

def extract_stock_tickers(text):
    """
    Extracts potential stock tickers (1–5 uppercase letters) from input text.

    Args:
        text (str): The input text.

    Returns:
        list: A list of extracted stock ticker symbols.
    """
    ticker_pattern = r'\b[A-Z]{1,5}(?:\.[A-Z]{1,5})?\b'
    tickers = re.findall(ticker_pattern, text)
    return tickers

def extract_stock_tickers_tool_definition():
    """
    Returns the OpenAI tool definition for the extract_stock_tickers function.
    """
    return define_tool(
        name="extract_stock_tickers",
        description="Extract potential stock tickers (1-5 uppercase letters) from text",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to extract stock tickers from"
                }
            },
            "required": ["text"]
        }
    )

def fetch_latest_news(query):
    """
    Fetches the latest news articles for a given search query using Google News.

    Args:
        query (str): The search keyword or topic.

    Returns:
        list: A list of dictionaries containing article titles, URLs, and published dates.
    """
    try:
        gn = GoogleNews(lang='en')
        search_results = gn.search(query)
        
        # Extract and format the first entry
        articles = []
        
        if isinstance(search_results, dict) and 'entries' in search_results:
            entries = list(search_results['entries'])[:1]
            for entry in entries:
                articles.append({
                    'title': entry.get('title', ''),
                    'url': entry.get('link', ''),
                    'published': entry.get('published', '')
                })
        
        return articles
    except Exception as e:
        raise Exception(f"Error fetching news: {str(e)}")
    
def fetch_latest_news_tool_definition():
    """
    Returns the OpenAI tool definition for the fetch_latest_news function.
    """
    return define_tool(
        name="fetch_latest_news",
        description="Fetch latest news articles for a given search query",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query or topic"
                }
            },
            "required": ["query"]
        }
    )


def extract_urls_tool_definition():
    """
    Returns the OpenAI tool definition for the extract_urls function.
    """
    return define_tool(
        name="extract_urls",
        description="Extracts URLs from the given text using a regex pattern.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The input text to search for URLs."
                }
            },
            "required": ["text"]
        }
    )

def extract_pdf_names_tool_definition():
    """
    Returns the OpenAI tool definition for the extract_pdf_names function.
    """
    return define_tool(
        name="extract_pdf_names",
        description="Extracts PDF filenames (e.g., 'recipes.pdf') from the given text.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The input text to search for PDF filenames."
                }
            },
            "required": ["text"]
        }
    )


def duckduckgo_search(query):
    """
    Performs a search using DuckDuckGo and returns relevant results.

    Args:
        query (str): The search term or question to look up.

    Returns:
        dict: A dictionary containing search results with titles, snippets, and URLs.
    """
    try:
        # Use DuckDuckGo's HTML search endpoint
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        url = f"https://html.duckduckgo.com/html/?q={query}"
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Use BeautifulSoup for better HTML parsing
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        # Find all result containers
        for result in soup.find_all('div', class_='result'):
            # Extract title and URL
            title_elem = result.find('a', class_='result__a')
            if title_elem:
                title = title_elem.get_text(strip=True)
                url = title_elem.get('href', '')
                
                # Extract snippet
                snippet_elem = result.find('a', class_='result__snippet')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                
                results.append({
                    'title': title,
                    'url': url,
                    'snippet': snippet
                })
                
                # Limit to top 5 results
                if len(results) >= 5:
                    break
        
        return {
            "results": results,
            "query": query,
            "total_results": len(results)
        }
        
    except Exception as e:
        print(f"Error fetching search results: {str(e)}")
        raise Exception(f"Error fetching search results: {str(e)}")


def duckduckgo_search_tool_definition():
    """
    Returns the OpenAI tool definition for the duckduckgo_search function.
    """
    return define_tool(
        name="duckduckgo_search",
        description="Searches the web using DuckDuckGo and returns relevant results including titles, snippets, and URLs. Use this to find current information about topics.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    )


def get_all_tool_definitions():
    """
    Returns a list of all available tool definitions for use with OpenAI.
    """
    return [
        get_stock_cur_info_tool_definition(),
        extract_stock_tickers_tool_definition(),
        fetch_latest_news_tool_definition(),
        get_historical_stock_data_tool_definition(),
        extract_urls_tool_definition(),
        extract_pdf_names_tool_definition(),
        scrape_meta_tags_tool_definition(),
        load_csv_tool_definition(),
        get_csv_chunk_tool_definition(),  
        load_pdf_tool_definition(),
        html_to_clean_markdown_tool_definition(),
        duckduckgo_search_tool_definition(),
        download_file_tool_definition(),
        suggest_file_name_tool_definition(),
        load_benchmark_data_tool_definition(),
        compare_historical_performance_tool_definition(),
        evaluate_math_expression_tool_definition(),
        rag_tool_definition()
    ]


def load_benchmark_data(tickers, start_date=None, end_date=None):
    """
    Loads historical data for multiple benchmark tickers using yfinance.

    Args:
        tickers (list): List of ticker symbols (e.g., ["SPY", "TLT", "VNQ"]).
        start_date (str, optional): Start date in 'YYYY-MM-DD' format.
        end_date (str, optional): End date in 'YYYY-MM-DD' format.

    Returns:
        dict: Dictionary containing historical data and basic statistics for each ticker.
    """
    try:
        # Download data for all tickers
        data = yf.download(tickers, start=start_date, end=end_date)
        adj_close = data['Adj Close']
        
        # Calculate basic statistics for each ticker
        results = {}
        for ticker in tickers:
            ticker_data = adj_close[ticker]
            daily_returns = ticker_data.pct_change().dropna()
            
            results[ticker] = {
                'historical_data': ticker_data.to_dict(),
                'stats': {
                    'total_return': ((ticker_data[-1] / ticker_data[0]) - 1) * 100,
                    'annualized_return': (((ticker_data[-1] / ticker_data[0]) ** (252/len(ticker_data))) - 1) * 100,
                    'volatility': daily_returns.std() * (252 ** 0.5) * 100,
                    'max_drawdown': ((ticker_data - ticker_data.expanding().max()) / ticker_data.expanding().max()).min() * 100
                }
            }
        
        return results
    except Exception as e:
        return {"error": str(e)}

def load_benchmark_data_tool_definition():
    """
    Returns the OpenAI tool definition for the load_benchmark_data function.
    """
    return define_tool(
        name="load_benchmark_data",
        description="Load historical data and calculate statistics for multiple benchmark tickers",
        parameters={
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols (e.g., ['SPY', 'TLT', 'VNQ'])"
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format"
                }
            },
            "required": ["tickers"]
        }
    )

def compare_historical_performance(tickers, start_date=None, end_date=None, rolling_window=252):
    """
    Compares historical performance metrics between multiple assets.

    Args:
        tickers (list): List of ticker symbols to compare.
        start_date (str, optional): Start date in 'YYYY-MM-DD' format.
        end_date (str, optional): End date in 'YYYY-MM-DD' format.
        rolling_window (int, optional): Window size for rolling calculations (default: 252 trading days).

    Returns:
        dict: Dictionary containing comparative analysis results.
    """
    try:
        # Download data
        data = yf.download(tickers, start=start_date, end=end_date)
        prices = data['Adj Close']
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Calculate rolling metrics
        rolling_vol = returns.rolling(window=rolling_window).std() * np.sqrt(252) * 100
        rolling_sharpe = returns.rolling(window=rolling_window).mean() / returns.rolling(window=rolling_window).std() * np.sqrt(252)
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr().to_dict()
        
        # Prepare results
        results = {
            'correlation_matrix': correlation_matrix,
            'assets': {}
        }
        
        for ticker in tickers:
            price_data = prices[ticker]
            return_data = returns[ticker]
            
            results['assets'][ticker] = {
                'total_return': ((price_data[-1] / price_data[0]) - 1) * 100,
                'annualized_return': (((price_data[-1] / price_data[0]) ** (252/len(price_data))) - 1) * 100,
                'volatility': return_data.std() * np.sqrt(252) * 100,
                'sharpe_ratio': return_data.mean() / return_data.std() * np.sqrt(252),
                'max_drawdown': ((price_data - price_data.expanding().max()) / price_data.expanding().max()).min() * 100,
                'rolling_metrics': {
                    'volatility': rolling_vol[ticker].dropna().to_dict(),
                    'sharpe_ratio': rolling_sharpe[ticker].dropna().to_dict()
                }
            }
        
        return results
    except Exception as e:
        return {"error": str(e)}

def compare_historical_performance_tool_definition():
    """
    Returns the OpenAI tool definition for the compare_historical_performance function.
    """
    return define_tool(
        name="compare_historical_performance",
        description="Compare historical performance metrics between multiple assets",
        parameters={
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols to compare"
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format"
                },
                "rolling_window": {
                    "type": "integer",
                    "description": "Window size for rolling calculations (default: 252 trading days)"
                }
            },
            "required": ["tickers"]
        }
    )

def evaluate_math_expression(expression):
    """
    Safely evaluates a mathematical expression string using Python's eval().
    Only allows basic mathematical operations and numbers.

    Args:
        expression (str): A string containing a mathematical expression (e.g., "10 * 4 + 2")

    Returns:
        dict: Dictionary containing the original expression, result, and any error message
    """
    # List of allowed characters
    allowed_chars = set("0123456789.+-*/() ")
    
    try:
        # Check if expression contains only allowed characters
        if not all(c in allowed_chars for c in expression):
            return {
                "expression": expression,
                "error": "Invalid characters in expression. Only numbers and basic operators (+,-,*,/) are allowed."
            }
        
        # Additional security check: expression should only contain basic math
        if any(keyword in expression.lower() for keyword in ["import", "eval", "exec", "__"]):
            return {
                "expression": expression,
                "error": "Invalid expression: Contains forbidden keywords."
            }
            
        # Evaluate the expression
        result = eval(expression)
        
        return {
            "expression": expression,
            "result": result,
            "error": None
        }
        
    except Exception as e:
        return {
            "expression": expression,
            "error": f"Error evaluating expression: {str(e)}"
        }

def evaluate_math_expression_tool_definition():
    """
    Returns the OpenAI tool definition for the evaluate_math_expression function.
    """
    return define_tool(
        name="evaluate_math_expression",
        description="Evaluates a basic mathematical expression string (e.g., '10 * 4 + 2') and returns the result",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '10 * 4 + 2')"
                }
            },
            "required": ["expression"]
        }
    )

def save_webpage_as_markdown(url: str) -> dict:
    """
    Downloads a webpage and saves it as a markdown file in the knowledge_files directory. 

    Args:
        url (str): The URL of the webpage to download and convert.

    Returns:
        dict: A dictionary containing the status and path of the saved file.
    """
    try:
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        # Make the request to get the webpage content
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html_content = response.text

        # Convert HTML to markdown
        markdown_content = html_to_clean_markdown(html_content)

        # Create knowledge_files directory if it doesn't exist
        os.makedirs(os.path.join(os.getcwd(), "knowledge_files"), exist_ok=True)

        # Generate a safe filename from URL
        # Extract domain and last path component
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.split('.')[0]  # Get first part of domain
        path = parsed_url.path.strip('/').split('/')[-1]  # Get last part of path
        
        # If no path, use domain only
        if not path:
            filename = f"{domain}_article"
        else:
            # Combine domain and path, limit length
            filename = f"{domain}_{path[:30]}"  # Limit path part to 30 chars
        
        # Clean filename and ensure it ends with .md
        filename = re.sub(r'[^\w\-_.]', '_', filename)[:50]  # Limit total length to 50 chars
        if not filename.endswith('.md'):
            filename += '.md'

        # Save to knowledge_files directory
        file_path = os.path.join(os.getcwd(), "knowledge_files", filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        return {
            "status": "success",
            "file_path": file_path,
            "message": f"Webpage saved as markdown to {file_path}"
        }

    except Exception as e:
        print(f"Error saving webpage as markdown: {str(e)}")
        raise Exception(f"Error saving webpage as markdown: {str(e)}")

def save_webpage_as_markdown_tool_definition():
    """
    Returns the OpenAI tool definition for the save_webpage_as_markdown function.
    """
    return define_tool(
        name="save_webpage_as_markdown",
        description="Downloads a webpage and saves it as a markdown file in the knowledge_files directory for later reference. Use this when you want to save a webpage's content for future analysis. Use this when the user asks you to save a webpage.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the webpage to download and convert to markdown"
                }
            },
            "required": ["url"]
        }
    )

def suggest_file_name(query: str) -> dict:
    """
    Suggest file names from knowledge_files based on a search query using regex matching.
    Returns detailed information about potential file matches sorted by relevance.

    Args:
        query (str): Search query to find matching file names

    Returns:
        dict: A dictionary containing:
            - matches: List of matched files with their details (name, path, size, type)
            - total_matches: Number of matches found
            - search_query: Original search query
    """
    knowledge_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_files")
    if not os.path.exists(knowledge_dir):
        return {
            "matches": [],
            "total_matches": 0,
            "search_query": query,
            "error": "knowledge_files directory not found"
        }

    # Convert query terms to regex patterns
    query_terms = query.lower().split()
    patterns = [re.compile(rf'.*{re.escape(term)}.*', re.IGNORECASE) for term in query_terms]
    
    matches = []
    for file_name in os.listdir(knowledge_dir):
        score = 0
        file_lower = file_name.lower()
        file_path = os.path.join(knowledge_dir, file_name)
        
        # Score each file based on how many query terms it matches
        for pattern in patterns:
            if pattern.match(file_lower):
                score += 1
        
        if score > 0:
            # Get file details
            file_stat = os.stat(file_path)
            file_type = os.path.splitext(file_name)[1].lower()
            
            matches.append({
                'file_name': file_name,
                'full_path': file_path,
                'size': file_stat.st_size,
                'modified': file_stat.st_mtime,
                'file_type': file_type,
                'score': score
            })
    
    # Sort by score (highest first)
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        "matches": matches,
        "total_matches": len(matches),
        "search_query": query
    }

def suggest_file_name_tool_definition():
    """
    Returns the OpenAI tool definition for the suggest_file_name function.
    """
    return define_tool(
        name="suggest_file_name",
        description="Suggests file names from knowledge_files based on a search query, returning detailed information about matching files including their paths, sizes, and types.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find matching file names."
                }
            },
            "required": ["query"]
        }
    )

def get_historical_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data for a given ticker between specified dates.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL" for Apple).
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        dict: A dictionary containing historical data including:
            - opening and closing prices
            - high and low prices
            - volume
            - key statistics like average price, price change, and percentage change
    """
    try:
        # Get the stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            return {"error": f"No data found for {ticker} between {start_date} and {end_date}"}
        
        # Calculate key statistics
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        price_change = end_price - start_price
        percent_change = (price_change / start_price) * 100
        avg_price = hist['Close'].mean()
        max_price = hist['High'].max()
        min_price = hist['Low'].min()
        avg_volume = hist['Volume'].mean()
        
        # Format the response
        response = {
            "ticker": ticker,
            "period": {
                "start": start_date,
                "end": end_date
            },
            "summary_statistics": {
                "start_price": round(start_price, 2),
                "end_price": round(end_price, 2),
                "price_change": round(price_change, 2),
                "percent_change": round(percent_change, 2),
                "average_price": round(avg_price, 2),
                "highest_price": round(max_price, 2),
                "lowest_price": round(min_price, 2),
                "average_daily_volume": int(avg_volume)
            },
            "price_milestones": {
                "highest": {
                    "price": round(max_price, 2),
                    "date": hist['High'].idxmax().strftime('%Y-%m-%d')
                },
                "lowest": {
                    "price": round(min_price, 2),
                    "date": hist['Low'].idxmin().strftime('%Y-%m-%d')
                }
            }
        }
        
        return response
        
    except Exception as e:
        return {"error": f"Error fetching data for {ticker}: {str(e)}"}

def get_historical_stock_data_tool_definition():
    """
    Returns the OpenAI tool definition for the get_historical_stock_data function.
    """
    return define_tool(
        name="get_historical_stock_data",
        description="Fetches historical stock data between specified dates, providing comprehensive price analysis including opening/closing prices, highs/lows, volume, and key statistics.",
        parameters={
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol (e.g., 'AAPL' for Apple, 'MSFT' for Microsoft)"
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format"
                }
            },
            "required": ["ticker", "start_date", "end_date"]
        }
    )

def get_csv_chunk(csv_file_path: str, chunk_number: int) -> dict:
    """
    Gets a specific chunk of data from a CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file
        chunk_number (int): The chunk number to retrieve (0-based index)
        
    Returns:
        dict: A dictionary containing:
            - chunk: The requested chunk of data
            - chunk_number: The chunk number retrieved
            - total_chunks: Total number of chunks available
            - error: Error message if any
    """
    try:
        # First get all the data (this will be cached by the convert_csv_to_json function)
        data = _convert_csv_to_json(csv_file_path)
        
        if "error" in data:
            return data
            
        if chunk_number < 0 or chunk_number >= data["num_chunks"]:
            return {
                "error": f"Invalid chunk number. Must be between 0 and {data['num_chunks'] - 1}"
            }
            
        return {
            "chunk": data["chunks"][chunk_number],
            "chunk_number": chunk_number,
            "total_chunks": data["num_chunks"],
            "columns": data["columns"]
        }
        
    except Exception as e:
        return {"error": f"Error retrieving chunk: {str(e)}"}

def get_csv_chunk_tool_definition():
    """
    Returns the OpenAI tool definition for the get_csv_chunk function.
    """
    return define_tool(
        name="get_csv_chunk",
        description="Retrieves a specific chunk of data from a chunked CSV file. Use this to get subsequent chunks after using convert_csv_to_json.",
        parameters={
            "type": "object",
            "properties": {
                "csv_file_path": {
                    "type": "string",
                    "description": "The file path to the CSV file"
                },
                "chunk_number": {
                    "type": "integer",
                    "description": "The chunk number to retrieve (0-based index)"
                }
            },
            "required": ["csv_file_path", "chunk_number"]
        }
    )

def _get_embedding_model():
    """
    Get or initialize the sentence transformer model.
    Uses a singleton pattern to avoid reloading the model.
    """
    if not hasattr(_get_embedding_model, "model"):
        _get_embedding_model.model = SentenceTransformer('all-MiniLM-L6-v2')
    return _get_embedding_model.model

def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks of roughly equal size.
    
    Args:
        text (str): Text to split into chunks
        chunk_size (int): Target size of each chunk in characters
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    # Split text into sentences (roughly)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        if current_length + sentence_length > chunk_size and current_chunk:
            # Store current chunk
            chunks.append(" ".join(current_chunk))
            # Keep last sentence if overlap is desired
            if overlap > 0:
                current_chunk = current_chunk[-1:]
                current_length = len(current_chunk[0])
            else:
                current_chunk = []
                current_length = 0
                
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def _convert_to_markdown(file_path: str) -> str:
    """
    Convert various file types to markdown format for consistent processing.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Markdown formatted text
    """
    ext = Path(file_path).suffix.lower()
    
    if ext == '.csv':
        df = pd.read_csv(file_path)
        # Convert DataFrame to markdown table
        return df.to_markdown(index=False)
    elif ext == '.pdf':
        return _pdf_to_clean_markdown(file_path)
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.md':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def _create_or_load_index(index_path: str, dimension: int = 384) -> Tuple[faiss.Index, dict]:
    """
    Create a new FAISS index or load existing one.
    
    Args:
        index_path (str): Path to save/load the index
        dimension (int): Dimension of the embeddings
        
    Returns:
        Tuple[faiss.Index, dict]: FAISS index and metadata dictionary
    """
    index_file = Path(index_path)
    metadata_file = index_file.with_suffix('.metadata')
    
    if index_file.exists() and metadata_file.exists():
        # Load existing index and metadata
        index = faiss.read_index(str(index_file))
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
    else:
        # Create new index and metadata
        index = faiss.IndexFlatL2(dimension)
        metadata = {
            'texts': [],
            'files': [],
            'positions': []
        }
    
    return index, metadata

def _save_index(index: faiss.Index, metadata: dict, index_path: str):
    """
    Save FAISS index and metadata to disk.
    
    Args:
        index (faiss.Index): FAISS index to save
        metadata (dict): Metadata dictionary to save
        index_path (str): Path to save the index
    """
    index_file = Path(index_path)
    metadata_file = index_file.with_suffix('.metadata')
    
    # Save index
    faiss.write_index(index, str(index_file))
    
    # Save metadata
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

def _update_index(index: faiss.Index, metadata: dict, texts: List[str], file_path: str):
    """
    Update FAISS index with new text chunks.
    
    Args:
        index (faiss.Index): FAISS index to update
        metadata (dict): Metadata dictionary to update
        texts (List[str]): List of text chunks to add
        file_path (str): Source file path for the chunks
    """
    model = _get_embedding_model()
    
    # Create embeddings for new chunks
    embeddings = model.encode(texts, convert_to_tensor=False)
    
    # Add to index
    index.add(np.array(embeddings))
    
    # Update metadata
    start_pos = len(metadata['texts'])
    metadata['texts'].extend(texts)
    metadata['files'].extend([file_path] * len(texts))
    metadata['positions'].extend(range(start_pos, start_pos + len(texts)))

def _search_similar_chunks(query: str, index: faiss.Index, metadata: dict, k: int = 5) -> List[dict]:
    """
    Search for chunks similar to the query.
    
    Args:
        query (str): Search query
        index (faiss.Index): FAISS index to search
        metadata (dict): Metadata dictionary
        k (int): Number of results to return
        
    Returns:
        List[dict]: List of similar chunks with metadata
    """
    model = _get_embedding_model()
    
    # Create query embedding
    query_embedding = model.encode([query], convert_to_tensor=False)
    
    # Search index
    distances, indices = index.search(query_embedding, k)
    
    # Format results
    results = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0:  # FAISS returns -1 for empty slots
            continue
        results.append({
            'text': metadata['texts'][idx],
            'file': metadata['files'][idx],
            'position': metadata['positions'][idx],
            'similarity': 1 / (1 + distance)  # Convert distance to similarity score
        })
    
    return results

def _estimate_tokens(text: str, model: str = "gpt-4-turbo") -> int:
    """
    Estimate the number of tokens in a text string.
    
    Args:
        text (str): Text to estimate tokens for
        model (str): Model name to use for token estimation
        
    Returns:
        int: Estimated number of tokens
    """
    try:
        tokenizer = tiktoken.encoding_for_model(model)
        return len(tokenizer.encode(text))
    except Exception as e:
        print(f"Error estimating tokens: {str(e)}")
        # Fallback to rough estimation (1 token ≈ 4 characters)
        return len(text) // 4

def _should_use_rag(text: str, token_limit: int = 10000) -> bool:
    """
    Determine if RAG should be used based on token count.
    
    Args:
        text (str): Text to check
        token_limit (int): Token limit before using RAG
        
    Returns:
        bool: True if RAG should be used
    """
    return _estimate_tokens(text) > token_limit

def rag_tool(query: str, knowledge_files: Optional[List[str]] = None) -> dict:
    """
    Retrieve relevant information from knowledge files using semantic search.
    
    Args:
        query (str): The search query
        knowledge_files (List[str], optional): List of specific files to search within.
            If None, searches all files in the knowledge_files directory.
    
    Returns:
        dict: A dictionary containing:
            - relevant_chunks: List of relevant text chunks with metadata
            - files_processed: List of files that were processed
            - query: Original search query
    """
    try:
        # Set up paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        knowledge_dir = os.path.join(base_dir, "knowledge_files")
        index_dir = os.path.join(base_dir, ".index")
        os.makedirs(index_dir, exist_ok=True)
        index_path = os.path.join(index_dir, "knowledge.index")
        
        # Get list of files to process
        if knowledge_files is None:
            knowledge_files = [f for f in os.listdir(knowledge_dir)
                             if os.path.isfile(os.path.join(knowledge_dir, f))]
        files_to_process = [os.path.join(knowledge_dir, f) for f in knowledge_files]
        
        # Create or load index
        index, metadata = _create_or_load_index(index_path)
        
        # Check if we need to update the index
        indexed_files = set(metadata['files'])
        new_files = [f for f in files_to_process if f not in indexed_files]
        
        # Process new files
        for file_path in new_files:
            try:
                # Convert file to markdown
                text = _convert_to_markdown(file_path)
                
                # Chunk the text
                chunks = _chunk_text(text)
                
                # Update index with new chunks
                _update_index(index, metadata, chunks, file_path)
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
        
        # Save updated index if new files were processed
        if new_files:
            _save_index(index, metadata, index_path)
        
        # Search for similar chunks
        results = _search_similar_chunks(query, index, metadata)
        
        return {
            "relevant_chunks": results,
            "files_processed": list(set(metadata['files'])),
            "query": query
        }
        
    except Exception as e:
        return {
            "error": f"Error in RAG pipeline: {str(e)}",
            "query": query
        }

def rag_tool_definition():
    """
    Returns the OpenAI tool definition for the rag_tool function.
    """
    return define_tool(
        name="rag_tool",
        description="Retrieves relevant information from your knowledge base using semantic search",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or query to search for"
                },
                "knowledge_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific files to search within"
                }
            },
            "required": ["query"]
        }
    )

def process_tool_calls(conversation):
    """
    Process tool calls from an OpenAI conversation.
    
    Args:
        conversation: The OpenAI conversation message containing tool calls
        
    Returns:
        dict: Results from executing the tool calls
    """
    print("\nEntering process_tool_calls")
    results = {}
    
    try:
        if not hasattr(conversation, 'tool_calls'):
            print("No tool_calls attribute found")
            return results
        
        if not conversation.tool_calls:
            print("tool_calls is empty")
            return results
        
        print(f"Found {len(conversation.tool_calls)} tool calls")
        
        for tool_call in conversation.tool_calls:
            try:
                tool_call_id = tool_call.id
                print(f"\nProcessing tool call ID: {tool_call_id}")
                
                # Extract function name and arguments
                function_name = tool_call.function.name
                tool_arguments = json.loads(tool_call.function.arguments)
                
                print(f"Function: {function_name}")
                print(f"Arguments: {tool_arguments}")
                
                # Execute the tool function
                tool_function = globals().get(function_name)
                if tool_function:
                    print(f"Found function: {tool_function.__name__}")
                    result = tool_function(**tool_arguments)
                    print(f"Function result: {result}")
                    results[tool_call_id] = {"status": "success", "output": result}
                else:
                    print(f"Function not found: {function_name}")
                    results[tool_call_id] = {
                        "status": "error",
                        "error": f"Tool '{function_name}' not found"
                    }
            except Exception as e:
                import traceback
                print(f"Error processing tool call: {str(e)}")
                print("Traceback:")
                print(traceback.format_exc())
                if 'tool_call_id' in locals():
                    results[tool_call_id] = {
                        "status": "error",
                        "error": str(e)
                    }
    except Exception as e:
        import traceback
        print(f"Error in process_tool_calls: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return {}
    
    print(f"\nFinal results: {results}")
    return results

# TOOL REGISTRY - Moved to end of file after all functions are defined
AVAILABLE_TOOLS = {
    'get_stock_cur_info': get_stock_cur_info_tool_definition()['function']['name'],
    'extract_stock_tickers': extract_stock_tickers_tool_definition()['function']['name'],
    'fetch_latest_news': fetch_latest_news_tool_definition()['function']['name'],
    'load_benchmark_data': load_benchmark_data_tool_definition()['function']['name'],
    'compare_historical_performance': compare_historical_performance_tool_definition()['function']['name'],
    'evaluate_math_expression': evaluate_math_expression_tool_definition()['function']['name'],
    'download_file': download_file_tool_definition()['function']['name'],
    'scrape_meta_tags': scrape_meta_tags_tool_definition()['function']['name'],
    'load_csv': load_csv_tool_definition()['function']['name'],
    'load_pdf': load_pdf_tool_definition()['function']['name'],
    'html_to_clean_markdown': html_to_clean_markdown_tool_definition()['function']['name'],
    'extract_urls': extract_urls_tool_definition()['function']['name'],
    'extract_pdf_names': extract_pdf_names_tool_definition()['function']['name'],
    'duckduckgo_search': duckduckgo_search_tool_definition()['function']['name'],
    'save_webpage_as_markdown': save_webpage_as_markdown_tool_definition()['function']['name'],
    'suggest_file_name': suggest_file_name_tool_definition()['function']['name'],
    'get_historical_stock_data': get_historical_stock_data_tool_definition()['function']['name'],
    'get_csv_chunk': get_csv_chunk_tool_definition()['function']['name'],
    'rag_tool': rag_tool_definition()['function']['name']
}

# Update get_all_tool_definitions
def get_all_tool_definitions():
    """
    Returns a list of all available tool definitions for use with OpenAI.
    """
    return [
        get_stock_cur_info_tool_definition(),
        extract_stock_tickers_tool_definition(),
        fetch_latest_news_tool_definition(),
        get_historical_stock_data_tool_definition(),
        extract_urls_tool_definition(),
        extract_pdf_names_tool_definition(),
        scrape_meta_tags_tool_definition(),
        load_csv_tool_definition(),
        get_csv_chunk_tool_definition(),  
        load_pdf_tool_definition(),
        html_to_clean_markdown_tool_definition(),
        duckduckgo_search_tool_definition(),
        download_file_tool_definition(),
        suggest_file_name_tool_definition(),
        load_benchmark_data_tool_definition(),
        compare_historical_performance_tool_definition(),
        evaluate_math_expression_tool_definition(),
        rag_tool_definition()
    ]