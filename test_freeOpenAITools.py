import unittest
from unittest.mock import patch, MagicMock
import json
import os
from freeOpenAITools import (
    download_file,
    extract_urls,
    extract_pdf_names,
    scrape_meta_tags,
    convert_csv_to_json,
    pdf_to_clean_markdown,
    html_to_clean_markdown,
    get_stock_cur_info,
    extract_stock_tickers,
    fetch_latest_news,
    process_tool_calls,
    get_all_tool_definitions
)

class TestFreeOpenAITools(unittest.TestCase):
    def setUp(self):
        # Create any test files or setup needed
        self.test_csv_content = "name,age\nJohn,30\nJane,25"
        with open('test.csv', 'w') as f:
            f.write(self.test_csv_content)
            
    def tearDown(self):
        # Clean up any test files
        if os.path.exists('test.csv'):
            os.remove('test.csv')
        if os.path.exists('test_download.txt'):
            os.remove('test_download.txt')

    @patch('requests.get')
    def test_download_file(self, mock_get):
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b'test content']
        mock_get.return_value = mock_response

        result = download_file('http://example.com/test.txt')
        self.assertTrue(os.path.exists(result))
        with open(result, 'r') as f:
            content = f.read()
        self.assertEqual(content, 'test content')

    def test_extract_urls(self):
        test_text = "Check out https://example.com and www.test.com"
        urls = extract_urls(test_text)
        self.assertEqual(len(urls), 2)
        self.assertIn('https://example.com', urls)
        self.assertIn('www.test.com', urls)

    def test_extract_pdf_names(self):
        test_text = "Here are some PDFs: document.pdf and report-2023.pdf"
        pdfs = extract_pdf_names(test_text)
        self.assertEqual(len(pdfs), 2)
        self.assertIn('document.pdf', pdfs)
        self.assertIn('report-2023.pdf', pdfs)

    @patch('requests.get')
    def test_scrape_meta_tags(self, mock_get):
        mock_html = '''
        <html>
            <head>
                <title>Test Page</title>
                <meta name="description" content="Test description">
                <meta name="keywords" content="test, keywords">
            </head>
        </html>
        '''
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = mock_html
        mock_get.return_value = mock_response

        result = scrape_meta_tags('http://example.com')
        self.assertEqual(result['title'], 'Test Page')
        self.assertEqual(result['description'], 'Test description')
        self.assertEqual(result['keywords'], 'test, keywords')

    def test_convert_csv_to_json(self):
        result = convert_csv_to_json('test.csv')
        json_data = json.loads(result)
        self.assertEqual(len(json_data), 2)
        self.assertEqual(json_data[0]['name'], 'John')
        self.assertEqual(json_data[0]['age'], '30')

    @patch('fitz.open')
    def test_pdf_to_clean_markdown(self, mock_fitz_open):
        # Mock PDF content
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Test PDF Content\nWith multiple lines"
        mock_doc.__iter__.return_value = [mock_page]  # Make doc iterable
        mock_doc.__len__.return_value = 1
        mock_fitz_open.return_value = mock_doc

        result = pdf_to_clean_markdown('test.pdf')
        self.assertIn('Test PDF Content', result)

    def test_html_to_clean_markdown(self):
        test_html = "<h1>Title</h1><p>This is a test</p>"
        result = html_to_clean_markdown(test_html)
        self.assertIn('# Title', result)
        self.assertIn('This is a test', result)

    @patch('yfinance.Ticker')
    def test_get_stock_cur_info(self, mock_ticker):
        # Mock the ticker instance
        mock_instance = mock_ticker.return_value
        mock_instance.info = {'regularMarketPrice': 150.0}
        
        # Test the function
        result = get_stock_cur_info('AAPL')
        
        # Verify the result
        self.assertEqual(result['ticker'], 'AAPL')
        self.assertEqual(result['all_info']['regularMarketPrice'], 150.0)

    def test_extract_stock_tickers(self):
        test_text = "Buy AAPL and GOOGL stocks"
        tickers = extract_stock_tickers(test_text)
        self.assertEqual(len(tickers), 2)
        self.assertIn('AAPL', tickers)
        self.assertIn('GOOGL', tickers)

    @patch('requests.get')  # We'll patch the underlying request instead
    def test_fetch_latest_news(self, mock_get):
        # Create mock data
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '''
            <feed>
                <entry>
                    <title>Test News</title>
                    <link href="http://example.com/news"/>
                    <published>2023-12-17</published>
                </entry>
            </feed>
        '''
        mock_get.return_value = mock_response

        # Call the function
        result = fetch_latest_news('test query')
        
        # Verify the results
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['title'], 'Test News')
        self.assertEqual(result[0]['url'], 'http://example.com/news')

    def test_process_tool_calls(self):
        # Test with extract_urls since it doesn't require external services
        conversation = {
            "tool_calls": [
                {
                    "id": "test1",
                    "function": {
                        "name": "extract_urls",
                        "arguments": json.dumps({"text": "Visit https://example.com"})
                    }
                }
            ]
        }
        
        result = process_tool_calls(conversation)
        self.assertEqual(result["test1"]["status"], "success")
        self.assertIn("https://example.com", result["test1"]["output"])

    def test_get_all_tool_definitions(self):
        definitions = get_all_tool_definitions()
        self.assertTrue(isinstance(definitions, list))
        self.assertTrue(len(definitions) > 0)
        
        # Check that each definition has required fields
        for definition in definitions:
            self.assertIn('name', definition)
            self.assertIn('description', definition)
            self.assertIn('parameters', definition)

if __name__ == '__main__':
    unittest.main()
