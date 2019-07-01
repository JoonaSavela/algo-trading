import matplotlib.pyplot as plt
import numpy as np
import json
import glob

def main():
    times = []
    capitals = []
    bnbs = []

    for file in glob.glob('trading_logs/*.json'):
        with open(file, 'r') as f:
            obj = json.load(f)
            times.append(int(obj['initial_time']))
            times.append(int(obj['final_time']))
            capitals.append(float(obj['initial_capital']))
            capitals.append(float(obj['final_capital']))
            bnbs.append(float(obj['initial_bnb']))
            bnbs.append(float(obj['final_bnb']))

    times = np.array(times)
    capitals = np.array(capitals)
    bnbs = np.array(bnbs)

    i = np.argsort(times)
    times = times[i]
    capitals = capitals[i]
    bnbs = bnbs[i]

    times = (times - times[0]) / 60
    capitals = capitals / capitals[0]
    bnbs = bnbs / bnbs[0]

    plt.plot(times, capitals)
    plt.plot(times, bnbs)
    plt.show()


if __name__ == "__main__":
    main()
