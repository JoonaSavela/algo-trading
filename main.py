import matplotlib.pyplot as plt
from utils import get_data

def main():
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=MSFT&interval=1min&apikey=CV8VMYTJCQBBSKXP&outputsize=full"
    times, opens, highs, lows, closes, volumes = get_data(url)

    # print(len(times))
    plt.plot(times, closes / opens[0])
    plt.show()

if __name__ == "__main__":
    main()
