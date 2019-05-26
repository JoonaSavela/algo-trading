import matplotlib.pyplot as plt
from utils import get_data, normalize_prices, times_min_max, is_valid_interval

def main():
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=MSFT&interval=1min&apikey=CV8VMYTJCQBBSKXP&outputsize=full"
    times, opens, highs, lows, closes, volumes = get_data(url)
    opens, highs, lows, closes = normalize_prices(opens, highs, lows, closes)

    time_min, time_max = times_min_max(times)

    # print(is_valid_interval(times[0], 0, 5, time_min, time_max, 1))
    # print(is_valid_interval(times[0], 0, 5, time_min, time_max, 5))
    # print(is_valid_interval(times[0], 0, 10000, time_min, time_max, 5))
    # print(is_valid_interval(times[100], 100, 100, time_min, time_max))
    # print(is_valid_interval(times[0], 1, 100, time_min, time_max))

    # print(len(times))
    # plt.plot(times, closes)
    # plt.show()

if __name__ == "__main__":
    main()
