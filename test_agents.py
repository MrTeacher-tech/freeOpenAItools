import unittest
from unittest.mock import patch, MagicMock
from agents import (
    create_financial_analyst,
    create_research_assistant,
    process_agent_response
)

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_completion = MagicMock()
        self.mock_message = MagicMock()
        
        # Set up basic mock response
        self.mock_message.content = "Test response"
        self.mock_message.tool_calls = []
        self.mock_completion.choices = [MagicMock(message=self.mock_message)]
        
    @patch('agents.OpenAI')
    def test_create_financial_analyst(self, mock_openai):
        # Setup
        mock_openai.return_value = self.mock_client
        self.mock_client.chat.completions.create.return_value = self.mock_completion
        
        # Execute
        result = create_financial_analyst()
        
        # Verify
        mock_openai.assert_called_once()
        self.mock_client.chat.completions.create.assert_called_once()
        
        # Verify tools were included
        call_args = self.mock_client.chat.completions.create.call_args[1]
        self.assertIn('tools', call_args)
        tools = call_args['tools']
        self.assertEqual(len(tools), 3)  # Should have 3 tools
        
        # Verify system prompt
        messages = call_args['messages']
        self.assertEqual(messages[0]['role'], 'system')
        self.assertIn('financial analyst', messages[0]['content'].lower())
    
    @patch('agents.OpenAI')
    def test_create_research_assistant(self, mock_openai):
        # Setup
        mock_openai.return_value = self.mock_client
        self.mock_client.chat.completions.create.return_value = self.mock_completion
        
        # Execute
        result = create_research_assistant()
        
        # Verify
        mock_openai.assert_called_once()
        self.mock_client.chat.completions.create.assert_called_once()
        
        # Verify tools were included
        call_args = self.mock_client.chat.completions.create.call_args[1]
        self.assertIn('tools', call_args)
        tools = call_args['tools']
        self.assertEqual(len(tools), 3)  # Should have 3 tools
        
        # Verify system prompt
        messages = call_args['messages']
        self.assertEqual(messages[0]['role'], 'system')
        self.assertIn('research assistant', messages[0]['content'].lower())
    
    @patch('agents.process_tool_calls')
    def test_process_agent_response_with_message(self, mock_process_tool_calls):
        # Test regular message response
        response = MagicMock()
        response.content = "Test message"
        response.tool_calls = None
        
        result = process_agent_response(response, {})
        self.assertEqual(result['type'], 'message')
        self.assertEqual(result['content'], "Test message")
        mock_process_tool_calls.assert_not_called()
    
    @patch('agents.process_tool_calls')
    def test_process_agent_response_with_tool_calls(self, mock_process_tool_calls):
        # Test tool call response
        response = MagicMock()
        response.content = "Using tools"
        response.tool_calls = [MagicMock()]  # Mock tool calls
        
        mock_process_tool_calls.return_value = {"result": "tool output"}
        
        result = process_agent_response(response, {})
        self.assertEqual(result['type'], 'tool_calls')
        self.assertEqual(result['message'], "Using tools")
        self.assertEqual(result['results'], {"result": "tool output"})
        mock_process_tool_calls.assert_called_once()

if __name__ == '__main__':
    unittest.main()
