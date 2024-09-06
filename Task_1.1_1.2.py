import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.lines import Line2D
import functions as fn
import plots as plot
import scipy.sparse as sp

###### CHOOSE TASKS TO RUN ######
Task1_1 = 0
Task1_2 = 1
##################################

##### PARAMETERS #####
np.random.seed(42)
NN = 10                     # Number of agents
alpha = 0.01                # Step size
MAXITERS = 10000              # Maximum number of iterations
I_NN = np.eye(NN)       
dd = 2                      # Dimensionality of the data
stopping_criteria = 1e-6    # Stopping criteria for the gradient tracking algorithm
graph_type = "path"         # "cycle", "path", "star", "complete"


################################## 
# TASK 1.1 
##################################

'''
    The goal of Task 1.1 is to implement a gradient tracking algorithm for a network of agents.
'''
if Task1_1 == True:

    print("\033[1;34mSTARTING TASK 1.1...\033[0m")

    # Generate the graph and calculate the weights using Metropolis-Hastings weights
    Adj = fn.generate_random_graph(NN, graph_type)
    AA = fn.calculate_weights(NN, Adj)

    print("Selected graph type: ", graph_type)

    # Define the QQ and RR matrices for the cost function
    QQ = np.ones((NN, dd, dd))
    for ii in range(NN):
        QQ[ii] = np.eye(dd) * np.random.uniform(low=0.1, high=1)
    # RR = np.ones((NN, dd))
    RR = np.zeros((NN, dd))

    # Calculate the optimal solution ZZ_opt
    ZZ_opt = - np.linalg.inv(np.sum(QQ, axis = 0)) @ np.sum(RR, axis = 0)

    # Initialize the ZZ and SS tracking term
    ZZ_gt = np.zeros((MAXITERS, NN, dd))
    SS_gt = np.zeros((MAXITERS, NN, dd))

    # Initialize ZZ_gt to ones with some perturbation
    for ii in range(NN):
        ZZ_gt[0,ii,:] = np.ones((dd)) + np.random.uniform(low=-0.3, high=0.3, size=(dd))

    # Initialize the cost and norm terms
    cost_gt = np.zeros((MAXITERS))
    norm_gt = np.zeros(MAXITERS)

    # initialize the SS_gt to the gradient of the cost at the initial ZZ_gt
    for ii in range(NN):
        _, SS_gt[0, ii, :] = fn.quadratic_fn(ZZ_gt[0, ii], QQ[ii,:,:], RR[ii, :])

    # Run the gradient tracking algorithm for MAXITERS iterations
    for kk in range(MAXITERS - 1):
        # print(f"iter {kk}")

        # gradient tracking algorithm
        grad_norm_gt = np.zeros(dd)

        # Update the ZZ_gt and SS_gt terms for each agent
        for ii in range(NN):
            N_ii = np.nonzero(Adj[ii])[0]                               # get the neighbors of agent i

            ZZ_gt[kk + 1, ii, :] += AA[ii, ii] * ZZ_gt[kk, ii, :]       # weighing self loops
            SS_gt[kk + 1, ii, :] += AA[ii, ii] * SS_gt[kk, ii, :]
            for jj in N_ii:
                ZZ_gt[kk + 1, ii, :] += AA[ii, jj] * ZZ_gt[kk, jj, :]   # weighing the neighbors
                SS_gt[kk + 1, ii, :] += AA[ii, jj] * SS_gt[kk, jj, :]

            ZZ_gt[kk + 1, ii, :] -= alpha * SS_gt[kk, ii, :]            # compute ZZ_gt[k+1] update


            # Compute gradients at iteration k and k+1 for SS_gt innovation term
            _, grad_ell_ii_new = fn.quadratic_fn(ZZ_gt[kk + 1, ii, :], QQ[ii,:,:], RR[ii, :])
            _, grad_ell_ii_old = fn.quadratic_fn(ZZ_gt[kk, ii, :], QQ[ii,:,:], RR[ii, :])
            SS_gt[kk + 1, ii, :] += grad_ell_ii_new - grad_ell_ii_old

            ell_ii_gt, _ = fn.quadratic_fn(ZZ_gt[kk, ii, :], QQ[ii,:,:], RR[ii, :])
            cost_gt[kk] += ell_ii_gt
            grad_norm_gt += grad_ell_ii_old
        norm_gt[kk] = np.linalg.norm(grad_norm_gt)

        # stop the algorithm if the stopping criteria is achieved
        if norm_gt[kk] < stopping_criteria:
            iters = kk+1
            print("Gradient tracking converged in ", iters, " iterations, with graph type ", graph_type)
            break
        iters = kk+1

    cost_gt = cost_gt[:-1]
    norm_gt = norm_gt[:-1]

    # ZZ_opt = -np.sum(RR) / np.sum(QQ)
    # opt_cost = 0.5 * np.sum(QQ) * ZZ_opt**2 + np.sum(RR) * ZZ_opt

    # plot everything
    if 1:
        plot.plot_cost_and_norm(cost_gt, norm_gt, iters, 'Gradient Tracking')
        plot.convergence_of_z(ZZ_gt, iters, NN, graph_type)
        plot.error_of_z(ZZ_gt, kk, NN, ZZ_opt)

##### TASK 1.2 #####

if Task1_2 == True:
    print("\033[1;34mSTARTING TASK 1.2...\033[0m")

    # Generate dataset
    mm_training = 5000                      # Number of points to generate
    test_data_size = int(0.25*mm_training)
    dd = 2                                  # Dimensionality of the data
    alpha_w = 0.01                          # the stepsize for w
    alpha_b = 0.01                          # the stepsize for b
    MAXITERS = 10000                        # Maximum number of iterations
    '''
        We want our decision boundary to either be an ellipse or a nonlinear curve.
    '''
    ellipse = 0
    curve = 1

    # These are the parameters for the shape and location of the ellipse
    AA_ellipse = 1
    BB = 0
    CC = 2
    DD = -5
    EE = 2

    if ellipse:
        w_true = np.array((AA_ellipse, BB, CC, DD, EE))
        b_true = -(1)
        type = 'Ellipse'
    if curve: 
        w_true = np.array([0.5, -0.5, 1.0, -1.0, 0.5])
        b_true = -(1)
        type = 'Curve'

    print ("type of curve: ", type)

    # Ensure we have two classes by retrying if necessary
    while True:
        # generate a dataset
        dataset = fn.generate_dataset(mm_training, dd)

        # Classify each data point
        labels = np.zeros(mm_training)
        for i in range(mm_training):
            labels[i], _ = fn.binary_classifier(fn.phi(dataset[i],type), w_true, b_true)

        # Check if we have at least one of each class
        if len(np.unique(labels)) == 2:
            break


    # Plot the dataset with labels
    plot.plot_dataset(dataset, labels, w_true, b_true, 'Generated Dataset', type)

    print("True weight vector:", w_true)
    print("True bias term:", b_true)

    print("Starting gradient descent...")

    # Initialize parameters for the algorithm
    ww = np.zeros(len(w_true))
    bb = 0
 

    # initialze empty lists to store the cost and gradient norm
    cost_history = []
    grad_norm_history = []

    # initialize the plot arrays to plot the decision boundary evolution
    ww_plot = np.zeros((3, len(w_true)))
    bb_plot = np.zeros(3)

    # Select the timesteps to plot the decision boundary evolution
    t1=50
    t2=200
    t3=1000

    # Centralized Gradient algorithm 
    for k in range(MAXITERS):
        if k % 100 == 0:
            print(f"Iteration {k}")
        dw = np.zeros(len(w_true))
        db = 0
        cost = 0

        # the loop runs through all the data points
        for i in range(mm_training):
            x_transformed = fn.phi(dataset[i], type)                # transform the data point to the feature space
            z = np.dot(ww, x_transformed) + bb                      
            y = labels[i]
            gradient_w, gradient_b = fn.grad(y, z, x_transformed)   # compute the gradients
            dw += gradient_w
            db += gradient_b
            cost += fn.cost_func(y, z)

        # store the weights and bias at the selected timesteps for plotting
        if k == t1 or k == t2 or k == t3:
                if k == t1:
                    index = 0
                elif k == t2:
                    index = 1
                elif k == t3:
                    index = 2
                ww_plot[index] = ww
                bb_plot[index] = bb
        
        ww -= alpha_w * dw / mm_training                               # update the weights and bias
        bb -= alpha_b * db / mm_training
        cost_history.append(cost / mm_training)
        grad_norm_history.append(np.linalg.norm(dw))

    print("Final weight vector:", ww)
    print("Final bias term:", bb)

    # Training set evaluation
    train_predictions = fn.predict(dataset, ww, bb, type)
    train_accuracy = np.mean(train_predictions == labels)
    print("Training accuracy:", train_accuracy*100)

    # Test set evaluation
    test_dataset = fn.generate_dataset(test_data_size, dd)
    test_labels = np.array([fn.binary_classifier(fn.phi(test_dataset[i],type), w_true, b_true)[0] for i in range(test_data_size)])
    test_predictions = fn.predict(test_dataset, ww, bb, type)
    test_accuracy = np.mean(test_predictions == test_labels)
    
    print("Test accuracy:", test_accuracy*100)

    if 1:
        plot.plot_cost_and_norm(cost_history, grad_norm_history, MAXITERS, 'Gradient Tracking')
        plot.plot_decision_boundary_evolution(ww_plot, bb_plot, w_true, b_true,type, type,t1,t2,t3)
        plot.plot_test_dataset(dataset, labels, train_predictions, ww, bb, w_true, b_true, mm_training, 'Generated Dataset with Actual and Predicted Elliptical Boundaries' ,type)
        plot.plot_test_dataset(test_dataset, test_labels, test_predictions, ww, bb, w_true, b_true, test_data_size, 'Test Dataset with Actual and Predicted Elliptical Decision Boundaries' ,type)

