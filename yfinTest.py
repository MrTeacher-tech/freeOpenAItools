import yfinance as yf

# Create a Ticker object for the stock (e.g., Apple)
ticker = yf.Ticker("AAPL")

print(ticker.info.get("regularMarketPrice"))

# Fetch historical market data
historical_data = ticker.history(period="1mo", interval="1d")

#print(historical_data)