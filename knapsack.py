import random

import pulp
import pandas as pd


def generate_sample_dataset(total_num: int, num_items_low: int, num_items_high: int, save_dataset=False) -> pd.DataFrame:
    assert num_items_low <= num_items_high
    assert total_num >= num_items_high - num_items_low + 1, "higher total_num is required."
    out = pd.DataFrame(
        columns=[
            'capacity', 'num_items', 'total_value', 'items_list (weight, value)',
            'solution_value', 'solution_list (weight, value)'
        ]
    )
    per_num_item_instances = total_num // (num_items_high - num_items_low)
    print(per_num_item_instances)
    for num_items in range(num_items_low, num_items_high):
        for idx in range(per_num_item_instances):
            instance = generate_instance(num_items)
            solution_list, solution_value = instance.solve_lin_prog()

            next_line = [
                instance.capacity, num_items, instance.get_total_value(),
                [(i.weight, i.value) for i in instance.available_items],
                solution_value,
                [(i.weight, i.value) for i in solution_list]
            ]
            out.loc[len(out)] = next_line

    if save_dataset:
        out.to_csv('knapsack_dataset.csv', index=True)
    return out


def generate_instance(num_items=10, max_weight=10, max_value=100, capacity_factor=0.5):
    """
    Generate a random knapsack instance with given constraints.
    :param num_items: Number of items to generate.
    :param max_weight: Maximum weight of each item.
    :param max_value: Maximum value of each item.
    :param capacity_factor: Factor to determine knapsack capacity based on total item weight.
    :return: Knapsack instance.
    """
    items = [Item(random.randint(1, max_weight), random.randint(1, max_value)) for _ in range(num_items)]

    total_weight = sum(item.weight for item in items)
    capacity = int(total_weight * capacity_factor)

    return Knapsack(capacity, items)


class Item:
    def __init__(self, weight: float | int, value: float | int):
        self.weight = weight
        self.value = value


class Knapsack:
    def __init__(self, capacity: float | int, available_items: list[Item]):
        self.capacity = capacity
        self.available_items = available_items

    def get_total_value(self):
        return sum([i.value for i in self.available_items])

    def check_validity(self, item_set: list[Item]) -> bool:
        return sum([i.weight for i in item_set]) < self.capacity

    def solve_lin_prog(self) -> tuple[list[Item], float]:
        problem = pulp.LpProblem("Knapsack Problem", pulp.LpMaximize)

        item_vars = [pulp.LpVariable(f'item_{i}', cat='Binary') for i in range(len(self.available_items))]

        problem += pulp.lpSum([item.value * item_vars[i] for i, item in enumerate(self.available_items)])

        problem += pulp.lpSum(
            [item.weight * item_vars[i] for i, item in enumerate(self.available_items)]) <= self.capacity

        problem.solve()

        selected_items = [self.available_items[i] for i in range(len(item_vars)) if item_vars[i].varValue == 1]

        # total_value = sum(item.value for item in selected_items)
        total_weight = sum(item.weight for item in selected_items)

        return selected_items, total_weight
