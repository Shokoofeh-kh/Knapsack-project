import pandas as pd

from knapsack import generate_sample_dataset
# from multiagent_llm.Agent.LLM_Agent import LLM_Agent
# from multiagent_llm.model_loaders import Casual_LLM_Loader


def main():
    sample_dataset = generate_sample_dataset(10, 5, 10, save_dataset=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(sample_dataset)

    # model_loader = Casual_LLM_Loader()

    # agent = LLM_Agent(

    # )


if __name__ == "__main__":
    main()
