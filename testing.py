import numpy as np

max_events = 15
events = [2, 4, 5, 2, 2, 2, 4, 8, 6, 7, 7, 6]
machine_context = np.full((len(events), max_events), -1, dtype=int)
for i in range(max_events):
    machine_context[i+1:, max_events-i-1] = np.where(False, -1, events[:-i-1])

print(machine_context)
