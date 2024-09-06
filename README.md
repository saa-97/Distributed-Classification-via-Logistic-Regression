# Distributed-Classification-via-Logistic-Regression
The objective is to apply a Gradient Tracking algorithm to achieve consensus optimization across multiple agents. This will demostrate how distributed systems can collaborate to determine a nonlinear classifier for data points in a feature space.

## Task 1.1: Gradient Tracking Algorithm 
1. Open file task_1.1_1.2
2. Set the variable Task1_1 to 1
3. You can set the following parameters as you wish from the file directly:
	- NN (number of agents)
	- alpha (learning rate)
	- MAXITERS (Maximum Iterations for the update steps)
	- graph_type (you can choose which type of graph you want among the agents from the following: "cycle", "path", "star", "complete")
4. Now run the python file

## Tasl 1.2: Centralized Gradient Method using the logistic regression cost function
1. Open file task_1.1_1.2
2. Set the variable Task_1_2 to 1
3. You can set the following parameters as you wish from the file directly:
	- mm_training (defines the size of the training dataset)
	- alpha_w (learning rate for weights)
	- alpha_b (learning rate for bias)
	- MAXITERS (Maximum Iterations for the update steps)
	- Now run the python file
3. You can also choose the decision type of decision boundary in the dataset from the following:
	- ellipse (This is an elliptical boundary, set it to 1 and curve to 0 to choose 	this)
	- curve (This is a non-linear curvy boundary, set it to 1 and ellipse to 0 to 	choose this)
