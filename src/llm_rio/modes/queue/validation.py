def memory_budgets(utilization: float) -> list[float]:
    initial = utilization
    deltas = (0.02, 0.04, 0.06, 0.08, 0.10)
    return [initial, *(round(initial - delta, 4) for delta in deltas if initial - delta >= 0.40)]
