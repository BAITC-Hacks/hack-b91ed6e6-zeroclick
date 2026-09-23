import pandas as pd

edges = pd.read_parquet("data/edges.parquet")
nodes = pd.read_parquet("data/nodes.parquet")
transactions = pd.read_parquet("data/transactions.parquet")

print("EDGES:")
print(edges.head())

print("\nNODES:")
print(nodes.head())

print("\nTRANSACTIONS:")
print(transactions.head())