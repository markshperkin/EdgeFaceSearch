import pandas as pd

df = pd.read_csv('second_search_results2.csv')

fitness_sorted = df.sort_values(by='fitness')
fitness_sorted.to_csv('fitnessSecond.csv', index=False)

print("Sorting complete.")
