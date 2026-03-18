
import yfinance as yf

data = yf.download(["AAPL","MSFT"], start="2020-01-01", end="2023-01-01")
print(data.head())

