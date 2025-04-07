import pandas as pd

# Load the CSV file
df = pd.read_csv('second_search_results2.csv')  # Replace with your actual filename

# Sort by fitness (ascending, since lower is often better)
fitness_sorted = df.sort_values(by='fitness')
fitness_sorted.to_csv('fitnessSecond.csv', index=False)

print("Sorting complete. Files saved: fitness.csv, val_loss.csv, latency.csv")
