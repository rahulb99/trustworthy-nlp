import pandas as pd

# download the dataset from Hugging Face
splits = {'harmful': 'data/harmful-behaviors.csv', 'benign': 'data/benign-behaviors.csv'}
df = pd.read_csv("hf://datasets/JailbreakBench/JBB-Behaviors/" + splits["harmful"])

print(df.head())
print(df['Category'].value_counts())

# select 2 rows of each category
df = df.groupby('Category').head(2).drop(columns=['Index', 'Source'])
df.to_csv("data/harmful-behaviors.csv", index=False)
