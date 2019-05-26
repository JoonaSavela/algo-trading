import requests
import matplotlib.pyplot as plt
import json

def main():
    request = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=MSFT&interval=1min&apikey=CV8VMYTJCQBBSKXP")
    a = json.loads(request._content)
    b = a["Time Series (1min)"]
    times = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    for key, item in b.items():
        times.append(key)
        opens.append(item["1. open"])
        highs.append(item["2. high"])
        lows.append(item["3. low"])
        closes.append(item["4. close"])
        volumes.append(item["5. volume"])

    plt.plot(times, opens)
    plt.show()

if __name__ == "__main__":
    main()
