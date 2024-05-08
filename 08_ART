import numpy as np

def initialize_weights(input_dim):
    # Initialize weights randomly and normalize them
    weights = np.random.uniform(size=(input_dim,))
    weights /= np.sum(weights)
    return weights

def calculate_similarity(input_pattern, weights):
    # Calculate the similarity between the input pattern and the weight vector
    return np.minimum(input_pattern, weights).sum()

def update_weights(input_pattern, weights, vigilance):
    # Update weights based on the input pattern and the vigilance parameter
    while True:
        activation = calculate_similarity(input_pattern, weights)
        if activation >= vigilance:
            return weights
        else:
            # Increase the weight of the most active input element and normalize
            weights[np.argmax(input_pattern)] += 1
            weights /= np.sum(weights)

def ART_neural_network(input_patterns, vigilance):
    categories = []

    for pattern in input_patterns:
        matched_category = None
        # Check if the pattern matches any existing category
        for category in categories:
            if calculate_similarity(pattern, category["weights"]) >= vigilance:
                matched_category = category
                break

        if matched_category is None:
            # Create a new category if no match is found
            weights = initialize_weights(len(pattern))
            matched_category = {"weights": weights, "patterns": []}
            categories.append(matched_category)

        # Add the pattern to the matched category and update weights
        matched_category["patterns"].append(pattern)
        matched_category["weights"] = update_weights(pattern, matched_category["weights"], vigilance)

    return categories

# Example usage
input_patterns = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0]])
vigilance = 0.5

categories = ART_neural_network(input_patterns, vigilance)

# Print the learned categories
for i, category in enumerate(categories):
    print(f"Category {i+1}:")
    print("Patterns:")
    [print(pattern) for pattern in category["patterns"]]
    print("Weights:")
    print(category["weights"])
    print()
