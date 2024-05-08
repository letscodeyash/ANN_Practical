import numpy as np
while True :
  # Receive user input for j
  j = int(input("Enter a Number (0-9): "))

  # Define the step function
  step_function = lambda x: 1 if x >= 0 else 0

  # Define the training data
  # representing 48 - 57 in binary form (Ascii values of 0-9)
  training_data = [
      {'input': [1, 1, 0, 0, 0, 0], 'label': 1},
      {'input': [1, 1, 0, 0, 0, 1], 'label': 0},
      {'input': [1, 1, 0, 0, 1, 0], 'label': 1},
      {'input': [1, 1, 0, 1, 1, 1], 'label': 0},
      {'input': [1, 1, 0, 1, 0, 0], 'label': 1},
      {'input': [1, 1, 0, 1, 0, 1], 'label': 0},
      {'input': [1, 1, 0, 1, 1, 0], 'label': 1},
      {'input': [1, 1, 0, 1, 1, 1], 'label': 0},
      {'input': [1, 1, 1, 0, 0, 0], 'label': 1},
      {'input': [1, 1, 1, 0, 0, 1], 'label': 0},
  ]

  # Initialize weights = 1 in binary 
  weights = np.array([0, 0, 0, 0, 0, 1])

  # Training the perceptron
  for data in training_data:
      input_vector = np.array(data['input'])
      label = data['label']
      output = step_function(np.dot(input_vector, weights))
      error = label - output
      weights += input_vector * error

  # Prepare input vector for user input
  user_input = np.array([int(x) for x in list('{0:06b}'.format(j))])

  # Predict if the user input is odd or even
  output = "odd" if step_function(np.dot(user_input, weights)) == 0 else "even"
  print(j, " is ", output)
