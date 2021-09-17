import numpy as np
import random

global impurity_analysis_choice, depth
depth = {}


def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))
    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df


def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)
    if len(unique_classes) == 1:
        return True
    return False


def classify_data(data):
    label_column = data[:, -1]
    unique_class, count_unique_class = np.unique(label_column, return_counts=True)
    index = count_unique_class.argmax()
    classification = unique_class[index]
    return classification


def get_potential_splits(data):
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)

        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2

                potential_splits[column_index].append(potential_split)
    return potential_splits


def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]
    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values > split_value]
    return data_below, data_above


def calculate_gini(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)
    probabilities = counts / counts.sum()
    gini = 1 - sum(probabilities * probabilities)
    return gini


def calculate_overall_gini(data_below, data_above):
    n_data_points = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n_data_points
    p_data_above = len(data_above) / n_data_points
    overall_gini = p_data_below * calculate_gini(data_below) + p_data_above * calculate_gini(data_above)
    return overall_gini


def calculate_entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy


def calculate_overall_entropy(data_below, data_above):
    n_data_points = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n_data_points
    p_data_above = len(data_above) / n_data_points
    overall_entropy = p_data_below * calculate_entropy(data_below) + p_data_above * calculate_entropy(data_above)
    return overall_entropy


def determine_best_split(data, potential_splits):
    if impurity_analysis_choice == '1':
        overall_gini = 1
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                data_below, data_above = split_data(data, split_column=column_index, split_value=value)
                current_overall_gini = calculate_overall_gini(data_below, data_above)
                if current_overall_gini < overall_gini:
                    overall_gini = current_overall_gini
                    best_split_column = column_index
                    best_split_value = value
    elif impurity_analysis_choice == '2':
        overall_gain = 999
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                data_below, data_above = split_data(data, split_column=column_index, split_value=value)
                current_overall_gain = calculate_overall_entropy(data_below, data_above)
                if current_overall_gain < overall_gain:
                    overall_gain = current_overall_gain
                    best_split_column = column_index
                    best_split_value = value
    else:
        print("Invalid Input")
        exit()
    return best_split_column, best_split_value


def decision_tree_algorithm(df, min_samples, max_depth, counter=0):
    if counter == 0:
        global column_headers
        column_headers = df.columns
        data = df.values
    else:
        data = df
    if check_purity(data) or len(data) < min_samples or counter == max_depth:
        classification = classify_data(data)
        return classification
    else:
        counter += 1
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        feature_name = column_headers[split_column]
        question = "{} <= {}".format(feature_name, split_value)
        sub_tree = {question: []}
        yes_answer = decision_tree_algorithm(data_below, min_samples, max_depth, counter)
        no_answer = decision_tree_algorithm(data_above, min_samples, max_depth, counter)
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        if sample in depth.keys():
            depth[sample].append(counter)
        else:
            depth[sample] = [counter]
        return sub_tree


def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split()
    if example[feature_name] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]
    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)


def calculate_accuracy(df, tree):
    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))
    df["classification_correct"] = df["classification"] == df["label"]
    accuracy = df["classification_correct"].mean()
    return accuracy


def calculate_max_accuracy(df, min_samples=2, max_depth=10):
    global impurity_analysis_choice

    impurity_analysis_choice = input("Enter 1 to use Gini Impurity for Best Split, "
                                     "2 to use Information Gain for Best Split: ")
    max_accuracy = 0
    max_accuracy_depth = 0
    tree = {}
    for i in range(1, 11):
        global sample
        sample = i
        maxim_depth = 0
        print(f"Calculating Accuracy and Depth of Random Sample {sample}")
        random.seed(sample)
        train_df, test_df = train_test_split(df, test_size=0.2)
        curr_tree = decision_tree_algorithm(train_df, min_samples, max_depth)
        curr_accuracy = calculate_accuracy(test_df, curr_tree)
        print("Accuracy calculated:", round(curr_accuracy * 100, 4), "%")
        for values in depth[sample]:
            if maxim_depth < values:
                maxim_depth = values
        print("Depth Calculated:", maxim_depth)
        print("")
        if max_accuracy < curr_accuracy:
            max_accuracy = curr_accuracy
            tree = curr_tree
            max_accuracy_depth = maxim_depth
    return tree, max_accuracy, max_accuracy_depth
