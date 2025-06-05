import pandas as pd

def find_s_compact(file_path):
    data = pd.read_csv(file_path)
    attributes = data.columns[:-1]
    class_label = data.columns[-1]
    print(data)

    # Initialize hypothesis with the first positive example's attributes if available, else all '?'
    # This is a common Find-S initialization strategy for the "most specific" hypothesis.
    positive_examples = data[data[class_label] == 'Yes']
    if not positive_examples.empty:
        hypothesis = positive_examples.iloc[0][attributes].tolist()
    else:
        return ["No positive examples found, cannot form a hypothesis."]

    # Iterate through subsequent positive examples to generalize the hypothesis
    for _, row in positive_examples.iloc[1:].iterrows():
        for i, value in enumerate(row[attributes]):
            if hypothesis[i] != value:
                hypothesis[i] = '?'
    return hypothesis



file_path = 'training.csv'
final_hypothesis = find_s_compact(file_path)
print("The final hypothesis is:", final_hypothesis)