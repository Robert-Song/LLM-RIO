def memory_budgets(utilization: float, *, explicit: bool = False) -> list[float]:
    initial = utilization if explicit else 0.80
    deltas = (0.10, 0.20)
    return [initial, *(round(initial - delta, 4) for delta in deltas if initial - delta >= 0.40)]
