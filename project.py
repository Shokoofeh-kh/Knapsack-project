import pandas as pd

from knapsack import generate_sample_dataset


def main():
    sample_dataset = generate_sample_dataset(10, 5, 10, save_dataset=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(sample_dataset)


if __name__ == "__main__":
    main()
