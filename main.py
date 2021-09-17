import pandas as pd
from pprint import pprint
from decision_tree_without_package import calculate_max_accuracy

df1 = pd.read_csv("data/parkinsons.data")
df1 = df1.rename(columns={"status": "label"})
df = df1[['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
          'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ',
          'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE', 'label']]

# For Pruning pass the parameter: min_samples & max_depth , in calculate_max_accuracy()
# sample - calculate_max_accuracy(df, min_samples=<Value>, max_depth=<Value>)
# Default value used : min_samples=2, max_depth=10

tree, max_accuracy, max_accuracy_depth = calculate_max_accuracy(df)
print("Max Accuracy:", round(max_accuracy * 100, 4), "%")
print("Tree Depth:", max_accuracy_depth)
print("Decision Tree:")
pprint(tree)