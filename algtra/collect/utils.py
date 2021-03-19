if __name__ == "__main__":
    raise ValueError("algtra/collect/utils.py should not be run as main.")

import numpy as np

# import utils
import glob
import json

# TODO: is max (the output) the best?
# TODO: add comments
def get_average_spread_old(data_dir, coin, m, total_balances, m_bear=None, step=0.0001):
    bull_folder = get_symbol(coin, m, bear=False)
    foldernames_bull = [data_dir + "/" + f"orderbooks/{coin}/{bull_folder}/"]

    if (m_bear is not None) and (m_bear != 0):
        bear_folder = get_symbol(coin, m_bear, bear=True)

        foldernames_bear = [data_dir + "/" + f"orderbooks/{coin}/{bear_folder}/"]
    else:
        foldernames_bear = []

    foldernames = foldernames_bull + foldernames_bear

    average_spreads = [[] for total_balance in total_balances]

    for folder in foldernames:
        filenames = glob.glob(folder + "*.json")

        distribution_ask = []
        distribution_bid = []

        for filename in filenames:
            with open(filename, "r") as fp:
                orderbook = json.load(fp)

            asks = np.array(orderbook["asks"])
            bids = np.array(orderbook["bids"])
            price = (asks[0, 0] + bids[0, 0]) / 2

            usd_value_asks = np.prod(asks, axis=1)
            usd_value_bids = np.prod(bids, axis=1)
            percentage_ask = asks[:, 0] / price - 1
            percentage_bid = bids[:, 0] / price - 1

            while len(distribution_ask) * step < np.max(percentage_ask) + step:
                distribution_ask.append(0.0)

            idx_ask = np.ceil(percentage_ask / step).astype(int)

            distribution_ask = np.array(distribution_ask)
            distribution_ask[idx_ask] += usd_value_asks / len(filenames)
            distribution_ask = list(distribution_ask)

            while -len(distribution_bid) * step > np.min(percentage_bid) - step:
                distribution_bid.append(0.0)

            idx_bid = np.ceil(np.abs(percentage_bid) / step).astype(int)

            distribution_bid = np.array(distribution_bid)
            distribution_bid[idx_bid] += usd_value_bids / len(filenames)
            distribution_bid = list(distribution_bid)

        distribution_ask = np.array(distribution_ask)
        percentage_ask = np.linspace(
            0, len(distribution_ask) * step, len(distribution_ask)
        )

        distribution_bid = np.array(distribution_bid)
        percentage_bid = np.linspace(
            0, -len(distribution_bid) * step, len(distribution_bid)
        )

        for i, total_balance in enumerate(total_balances):
            li_asks = np.cumsum(distribution_ask) < total_balance
            weights_ask = distribution_ask[li_asks] / total_balance
            if len(weights_ask) < len(percentage_ask):
                weights_ask = np.concatenate([weights_ask, [1 - np.sum(weights_ask)]])
            else:
                weights_ask /= np.sum(weights_ask)
            average_spread_ask = np.sum(
                percentage_ask[: len(weights_ask)] * weights_ask
            )
            average_spreads[i].append(np.abs(average_spread_ask))

            li_bids = np.cumsum(distribution_bid) < total_balance
            weights_bid = distribution_bid[li_bids] / total_balance
            if len(weights_bid) < len(percentage_bid):
                weights_bid = np.concatenate([weights_bid, [1 - np.sum(weights_bid)]])
            else:
                weights_bid /= np.sum(weights_bid)
            average_spread_bid = np.sum(
                percentage_bid[: len(weights_bid)] * weights_bid
            )
            average_spreads[i].append(np.abs(average_spread_bid))

    # return np.array(average_spreads)
    return np.max(average_spreads, axis=1)


def get_average_spread(data_dir, coin, m, total_balances, m_bear=None, step=0.0001):
    bull_folder = get_symbol(coin, m, bear=False)
    foldernames_bull = [data_dir + "/" + f"orderbooks/{coin}/{bull_folder}/"]

    if (m_bear is not None) and (m_bear != 0):
        bear_folder = get_symbol(coin, m_bear, bear=True)

        foldernames_bear = [data_dir + "/" + f"orderbooks/{coin}/{bear_folder}/"]
    else:
        foldernames_bear = []

    foldernames = foldernames_bull + foldernames_bear

    average_spreads = [[] for total_balance in total_balances]

    for folder in foldernames:
        filenames = glob.glob(folder + "*.json")

        distribution_ask = []
        distribution_bid = []

        for filename in filenames:
            with open(filename, "r") as fp:
                orderbook = json.load(fp)

            asks = np.array(orderbook["asks"])
            bids = np.array(orderbook["bids"])
            price = (asks[0, 0] + bids[0, 0]) / 2

            usd_value_asks = np.prod(asks, axis=1)
            usd_value_bids = np.prod(bids, axis=1)
            percentage_ask = asks[:, 0] / price - 1
            percentage_bid = bids[:, 0] / price - 1

            while len(distribution_ask) * step < np.max(percentage_ask) + step:
                distribution_ask.append(0.0)

            idx_ask = np.ceil(percentage_ask / step).astype(int)

            distribution_ask = np.array(distribution_ask)
            # print(len(distribution_ask))
            distribution_ask[idx_ask] += usd_value_asks / len(filenames)
            distribution_ask = list(distribution_ask)

            while -len(distribution_bid) * step > np.min(percentage_bid) - step:
                distribution_bid.append(0.0)

            idx_bid = np.ceil(np.abs(percentage_bid) / step).astype(int)

            distribution_bid = np.array(distribution_bid)
            distribution_bid[idx_bid] += usd_value_bids / len(filenames)
            distribution_bid = list(distribution_bid)

        distribution_ask = np.array(distribution_ask)
        percentage_ask = np.linspace(
            0, len(distribution_ask) * step, len(distribution_ask)
        )

        distribution_bid = np.array(distribution_bid)
        percentage_bid = np.linspace(
            0, -len(distribution_bid) * step, len(distribution_bid)
        )

        for i, total_balance in enumerate(total_balances):
            li_asks = np.cumsum(distribution_ask) < total_balance
            weights_ask = distribution_ask[li_asks] / total_balance
            if len(weights_ask) < len(percentage_ask):
                weights_ask = np.concatenate([weights_ask, [1 - np.sum(weights_ask)]])
            else:
                weights_ask /= np.sum(weights_ask)
            average_spread_ask = np.sum(
                percentage_ask[: len(weights_ask)] * weights_ask
            )
            average_spreads[i].append(np.abs(average_spread_ask))

            li_bids = np.cumsum(distribution_bid) < total_balance
            weights_bid = distribution_bid[li_bids] / total_balance
            if len(weights_bid) < len(percentage_bid):
                weights_bid = np.concatenate([weights_bid, [1 - np.sum(weights_bid)]])
            else:
                weights_bid /= np.sum(weights_bid)
            average_spread_bid = np.sum(
                percentage_bid[: len(weights_bid)] * weights_bid
            )
            average_spreads[i].append(np.abs(average_spread_bid))

    # return np.array(average_spreads)
    return np.max(average_spreads, axis=1)
