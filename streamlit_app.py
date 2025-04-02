import streamlit as st
import os
from openai import OpenAI
from agents import get_agent_config
from freeOpenAITools import (
    AVAILABLE_TOOLS,
    get_all_tool_definitions,
    load_pdf,
    load_csv,
    process_tool_calls
)
import json
import pandas as pd

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_tools' not in st.session_state:
    st.session_state.selected_tools = []
if 'uploaded_file_content' not in st.session_state:
    st.session_state.uploaded_file_content = None
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["api_keys"]["openai"])
if 'current_agent' not in st.session_state:
    st.session_state.current_agent = "Librarian"
if 'current_tools' not in st.session_state:
    st.session_state.current_tools = []
if 'last_file_update' not in st.session_state:
    st.session_state.last_file_update = 0

# Create knowledge_files directory if it doesn't exist
KNOWLEDGE_FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_files")
os.makedirs(KNOWLEDGE_FILES_DIR, exist_ok=True)

# Function to update file list
def update_knowledge_files():
    """Update the list of files in knowledge_files directory"""
    if os.path.exists(KNOWLEDGE_FILES_DIR):
        files = []
        for file in os.listdir(KNOWLEDGE_FILES_DIR):
            file_path = os.path.join(KNOWLEDGE_FILES_DIR, file)
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)  # Convert to MB
            mod_time = os.path.getmtime(file_path)
            files.append({
                'name': file,
                'size': file_size_mb,
                'mod_time': mod_time,
                'path': file_path
            })
        # Sort by modification time, newest first
        return sorted(files, key=lambda x: x['mod_time'], reverse=True)
    return []

def delete_file(file_path: str) -> bool:
    """Delete a file from the knowledge_files directory"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting file: {str(e)}")
        return False

def read_file_content(file_path: str) -> str:
    """Read file content based on file type"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            return load_pdf(file_path)
            
        elif file_extension == '.csv':
            df = pd.read_csv(file_path)
            return df.to_markdown()
            
        elif file_extension == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return f"```json\n{json.dumps(data, indent=2)}\n```"
            
        else:  # .txt, .md, etc.
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
    except UnicodeDecodeError:
        # If UTF-8 fails, try with a different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Error reading file with alternative encoding: {str(e)}")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

def get_mime_type(filename: str) -> str:
    """Get the MIME type for a file based on its extension"""
    extension = os.path.splitext(filename)[1].lower()
    mime_types = {
        '.txt': 'text/plain',
        '.pdf': 'application/pdf',
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.md': 'text/markdown',
        '.html': 'text/html',
        '.htm': 'text/html',
        '.xml': 'application/xml',
        '.yaml': 'application/x-yaml',
        '.yml': 'application/x-yaml'
    }
    return mime_types.get(extension, 'application/octet-stream')

# Page config
st.set_page_config(page_title="AI Agent Interface", layout="wide")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    model = st.selectbox(
        "Select Model",
        ["gpt-4-1106-preview", "gpt-4", "gpt-3.5-turbo"],
        index=0
    )
    
    # Agent type selection
    agent_type = st.selectbox(
        "Select Agent Type",
        ["Financial Analyst", "Librarian", "Internet Researcher", "Custom Agent"],
        key="agent_selector"
    )
    
    # Update current agent in session state
    if agent_type != st.session_state.current_agent:
        st.session_state.current_agent = agent_type
        st.session_state.current_tools = []  # Reset tool selection for new agent
    
    # Handle tool selection based on agent type
    if agent_type == "Custom Agent":
        all_tools = get_all_tool_definitions()
        tool_names = [tool['function']['name'] for tool in all_tools]
        selected_tools = st.multiselect(
            "Select Tools",
            tool_names,
            default=tool_names
        )
        
        system_message = """You are a helpful AI assistant with access to various tools. 
        Use the tools when appropriate to provide accurate information."""
        
        tools = [tool for tool in all_tools if tool['function']['name'] in selected_tools]
    else:
        system_message, tools = get_agent_config(agent_type)
    
    # Update current tools in session state
    st.session_state.current_tools = tools

    # File uploader
    uploaded_file = st.file_uploader("Upload a file for analysis", type=['txt', 'pdf', 'csv', 'json'])
    if uploaded_file:
        # Save the uploaded file to knowledge_files directory
        file_path = os.path.join(KNOWLEDGE_FILES_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved to knowledge_files: {uploaded_file.name}")
        
        # Store file content in session state
        st.session_state.uploaded_file_content = {
            "name": uploaded_file.name,
            "path": file_path
        }

    # Display files in knowledge_files directory
    st.header("Knowledge Files")
    
    # Add refresh button
    if st.button("🔄 Refresh Files"):
        st.session_state.last_file_update = os.path.getmtime(KNOWLEDGE_FILES_DIR)
    
    # Get and display files
    files = update_knowledge_files()
    if files:
        for file in files:
            col1, col2, col3 = st.columns([4, 1, 1])
            
            with col1:
                st.text(f"{file['name']} ({file['size']:.2f} MB)")
                
            
            with col2:
                # Download/Open button
                try:
                    with open(file['path'], 'rb') as f:
                        file_content = f.read()
                        st.download_button(
                            label="💾",
                            data=file_content,
                            file_name=file['name'],
                            mime=get_mime_type(file['name']),
                            key=f"download_{file['name']}"
                        )
                except Exception as e:
                    st.error(f"Error preparing file for download: {str(e)}")
            
            with col3:
                # Delete button with confirmation
                if st.button("🗑️", key=f"delete_{file['name']}", type="secondary"):
                    if delete_file(file['path']):
                        st.success(f"Deleted {file['name']}")
                        st.rerun()  # Refresh the page to update the file list
                    else:
                        st.error(f"Failed to delete {file['name']}")

# Main title and dynamic assistant header
st.title("AI Agent Interface")
st.markdown(f"### You are currently chatting with: :blue[{st.session_state.current_agent}]")

# Display available tools
if st.session_state.current_tools:
    tool_names = [tool['function']['name'] for tool in st.session_state.current_tools]
    st.markdown("#### Available Tools:")
    cols = st.columns(3)
    for i, tool in enumerate(tool_names):
        cols[i % 3].markdown(f"- :gear: {tool}")

st.divider()

# Main chat interface
st.header("Chat Interface")

# Chat interface
if st.session_state.openai_client:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # For custom agent, filter tools based on selection
        if agent_type == "Custom Agent":
            all_tools = get_all_tool_definitions()
            tools = [tool for tool in all_tools if tool['function']['name'] in selected_tools]
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Prepare messages
                    messages = [{"role": "system", "content": system_message}]
                    
                    # Add file content if available
                    if st.session_state.uploaded_file_content:
                        file_info = f"\nAnalyzing file: {st.session_state.uploaded_file_content['name']}\n"
                        messages.append({"role": "system", "content": file_info})
                    
                    # Add chat history
                    messages.extend(st.session_state.messages)
                    
                    # Get initial response from OpenAI
                    print("\nGetting initial response from OpenAI...")
                    response = st.session_state.openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=st.session_state.current_tools
                    )
                    print("Got response from OpenAI")
                    
                    # Process tool calls if present
                    assistant_message = response.choices[0].message
                    print(f"\nAssistant message: {assistant_message}")
                    
                    if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                        print("\nProcessing tool calls...")
                        # Process tool calls and get results
                        tool_results = process_tool_calls(assistant_message)
                        if tool_results is None:
                            print("Warning: tool_results is None")
                            tool_results = {}
                        print("\nTool Results:", tool_results)
                        
                        # Add assistant's message and tool results to messages
                        tool_calls_list = []
                        for tc in assistant_message.tool_calls:
                            try:
                                tool_calls_list.append({
                                    "id": tc.id,
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                                    "type": "function"
                                })
                            except Exception as e:
                                print(f"Error processing tool call: {e}")
                                continue
                        
                        messages.append({
                            "role": "assistant",
                            "content": assistant_message.content or "",
                            "tool_calls": tool_calls_list
                        })
                        
                        # Add tool results to messages
                        for tool_call in assistant_message.tool_calls:
                            try:
                                result = tool_results.get(tool_call.id, {})
                                if isinstance(result, dict):
                                    result = result.get("output", "Error executing tool")
                                print(f"\nAdding tool result to messages:")
                                print(f"Tool: {tool_call.function.name}")
                                print(f"Result: {result}")
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": str(result)
                                })
                            except Exception as e:
                                print(f"Error adding tool result to messages: {e}")
                                continue
                        
                        print("\nFinal messages before getting response:")
                        for msg in messages:
                            try:
                                content = msg.get('content', '')
                                if content:
                                    print(f"{msg['role']}: {content[:100]}...")
                            except Exception as e:
                                print(f"Error printing message: {e}")
                        
                        # Get final response from OpenAI
                        print("\nGetting final response from OpenAI...")
                        final_response = st.session_state.openai_client.chat.completions.create(
                            model=model,
                            messages=messages
                        )
                        
                        # Display final response
                        full_response = final_response.choices[0].message.content
                        st.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Error getting response: {str(e)}")
