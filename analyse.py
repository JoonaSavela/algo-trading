import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def analyse_search_results(filename):
    results = pd.read_csv(filename, index_col = 0)

    best = results.sort_values(by = 'profit', ascending = False)[:50]
    print(best.head())

    for col in best.columns:
        plt.hist(best[col])
        plt.title(col)
        plt.show()



if __name__ == '__main__':
    filename = 'search_results/random_search_Main_Strategy_True_True.csv'
    analyse_search_results(filename)
