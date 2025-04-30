import pandas as pd
import torch

from knapsack import generate_sample_dataset
from multiagent_llm.Agent.LLM_Agent import LLM_Agent
from multiagent_llm.model_loaders import Casual_LLM_Loader


def main():
    # sample_dataset = generate_sample_dataset(10, 5, 10, save_dataset=True)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # print(sample_dataset)

    model_loader = Casual_LLM_Loader(
        "Qwen/Qwen2.5-Math-7B-Instruct",
        max_memory="8000MB",
    )

    agent = LLM_Agent(
        model_loader,
        keep_memory=True
    )

    while True:
        prompt = input("prompt: ")
        print(agent.act(prompt))


if __name__ == "__main__":
    print("running on", "cuda" if torch.cuda.is_available() else "cpu")
    main()
