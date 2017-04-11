# machine_learning_3

AI assignment on machine learning.

### Perceptron Learning (perceptron_learning.py)

Takes in `perceptron_learning_input.csv` file and prints variable weights and bias after each iteration to show how how they're adjusted as algorithm progresses. 

Upon convergence, final weights and bias are printed in the output csv file to define the decision boundary that PLA computed for the given dataset.

##### usage: 
`python3 perceptron_learning.py perceptron_learning_input.csv <desired_output_filename.csv>`

### Linear Regression (linear_regression.py)

Takes in `linear_regression_input.csv` file and implements gradient descent with learning rates (α) 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, and 10 at 100 at 100 iterations to find a regression model. It then uses a learning rate and iterations number of choice.

The ouput csv file shows alpha, number of iterations, b_0, b_1, and b_2 for each model that gradient descent computed on the given dataset.

##### usage: 
`python3 linear_regression.py linear_regression_input.csv <desired_output_filename.csv>`

### Classification (classification.py)

Takes in `classification_input.csv` file and uses SVMs with different kernels, logistic regression, k-nearest neighbors, decision trees, and random forest to build classifiers.

The output csv file shows the method name, best score, and actual test score for each.

Data set was split into 60/40 and used stratified sampling with cross validation.

##### usage: 
`python3 classification_learning.py classification_input.csv <desired_output_filename.csv>`
