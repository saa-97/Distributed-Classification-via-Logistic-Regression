import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import functions as fn
import plots as plot
import scipy.sparse as sp


NN = 10
alpha = 0.03
MAXITERS = 10000
dataset_size = 5000
m_test = int(0.2*dataset_size)
d = 2
graph_type = "star"
np.random.seed(7)

# Here we can choose between an ellipse or a nonlinear curve for decision boundary
ellipse = 1
curve = 0

##### TASK 1.3 #####

'''
    The goal of task 1.3 is to implement a gradient tracking algorithm for a network of agents
    where each agent has its own dataset.
'''

print("\033[1;34mSTARTING TASK 1.3...\033[0m")
print("Selected graph type: ", graph_type)

# Generate the graph and calculate the weights using Metropolis-Hastings weights
Adj = fn.generate_random_graph(NN, graph_type)
AA = fn.calculate_weights(NN, Adj)



# Parameters for the shape of the ellipse
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

print ("Shape of decision boundary: ", type)

dataset = fn.generate_dataset(dataset_size, d)

labels = np.zeros(dataset_size)
for i in range(dataset_size):
    labels[i], _ = fn.binary_classifier(fn.phi(dataset[i], type), w_true, b_true)

# We sort the dataset and then divide among the agents
N = NN
sorted_indices = np.argsort(dataset[:, 0])
sorted_dataset = dataset[sorted_indices]
sorted_labels = labels[sorted_indices]
data_split = np.array_split(sorted_dataset, N)
labels_split = np.array_split(sorted_labels, N)

plot.plot_sorted_dataset(data_split, NN, w_true, b_true, type)


# Initial parameters
ww = np.zeros(5)
bb = 0

# Define the dimensions
dim_w = len(ww)
dim_b = 1
dd = dim_b + dim_w

# Initialize gradient tracking variables
ZZ_gt = np.zeros((MAXITERS, NN, dim_w + dim_b)) 
# for ii in range(NN):
#     ZZ_gt[0,ii,:] = np.ones((dd))

SS_gt = np.zeros((MAXITERS, NN, dim_w + dim_b))

for ii in range(NN):
    dw = np.zeros(5)
    db = 0
    for jj in range(len(data_split[ii])):
        x_transformed = fn.phi(data_split[ii][jj], type)
        z = np.dot(ww, x_transformed) + bb
        y = labels_split[ii][jj]
        gradient_w, gradient_b = fn.grad(y, z, x_transformed)
        dw += gradient_w
        db += gradient_b

    dw = dw / len(data_split[ii])
    db = db / len(data_split[ii])
    gradient = np.concatenate((dw, [db]))

    # Initiliaze the SS_gt term to the gradient of ZZ_gt at iteration 0
    SS_gt[0, ii] = gradient

# Initlization of cost and gradient norm
cost_gt = np.zeros(MAXITERS)
norm_gt = np.zeros(MAXITERS)
grad_gt = np.zeros((MAXITERS, NN, dim_w + dim_b))

# Here we set th timestamps to plot the decision boundary evolution
t1=50
t2=200
t3=1000
iterations = [t1, t2, t3]
ww_plot = np.zeros((NN, len(iterations), dim_w))
bb_plot = np.zeros((NN, len(iterations)))
error = np.zeros((MAXITERS, NN))

# Centralized Gradient Tracking algorithm
for kk in range(MAXITERS - 1):
    if kk%100 == 0:
        print(f"Iteration: #{kk}")

    # This loop iterates over all the agents in the network at iteration kk
    for ii in range(NN):
        N_ii = np.nonzero(Adj[ii])[0]                           # Neighbors of agent ii

        ZZ_gt[kk + 1, ii] = AA[ii, ii] * ZZ_gt[kk, ii]          # self loop weights
        SS_gt[kk + 1, ii] = AA[ii, ii] * SS_gt[kk, ii]
        for jj in N_ii:                                                
            ZZ_gt[kk + 1, ii] += AA[ii, jj] * ZZ_gt[kk, jj]     # weighing the neighbors
            SS_gt[kk + 1, ii] += AA[ii, jj] * SS_gt[kk, jj]

        ZZ_gt[kk + 1, ii] -= alpha * SS_gt[kk, ii]
  
        dw = np.zeros(5)
        db = 0
        ell_ii_gt = 0

        # this loop iterates over all the datapoints of agent ii, for iteration kk
        for ll in range(len(data_split[ii])):
            x_transformed = fn.phi(data_split[ii][ll], type)        # get the transformed feature space
            w = ZZ_gt[kk, ii][:-1]
            b = ZZ_gt[kk, ii][-1]
            z = np.dot(w, x_transformed) + b
            y = labels_split[ii][ll]
            gradient_w, gradient_b = fn.grad(y, z, x_transformed)
            dw += gradient_w
            db += gradient_b
            ell_ii_gt += fn.cost_func(y, z)
        dw = dw / len(data_split[ii])
        db = db / len(data_split[ii])
        grad_ell_ii_old = np.concatenate((dw, [db]))

        # Calculate the error in the SS_gt term for plotting
        diff = grad_ell_ii_old - SS_gt[kk, ii]
        diff = np.linalg.norm(diff)
        error[kk, ii] = diff

        dw = np.zeros(5)
        db = 0
        grad_norm_gt = np.zeros(6)

        # this loop iterates over all the datapoints of agent ii, for iteration kk+1
        for ll in range(len(data_split[ii])):
            x_transformed = fn.phi(data_split[ii][ll], type)
            w = ZZ_gt[kk + 1, ii][:-1]
            b = ZZ_gt[kk + 1, ii][-1]
            z = np.dot(w, x_transformed) + b
            y = labels_split[ii][ll]
            gradient_w, gradient_b = fn.grad(y, z, x_transformed)
            dw += gradient_w
            db += gradient_b
        
        dw = dw / len(data_split[ii])
        db = db / len(data_split[ii])
        grad_ell_ii_new = np.concatenate((dw, [db]))

        grad_gt[kk,ii] += grad_ell_ii_old

        # Compute the SS_gt innovation term
        SS_gt[kk + 1, ii] += grad_ell_ii_new - grad_ell_ii_old

        cost_gt[kk] += ell_ii_gt / len(data_split[ii])

        # Store the values of the weights and biases at the selected timestamps for plotting
        if kk == t1 or kk == t2 or kk == t3:
            if kk == t1:
                index = 0
            elif kk == t2:
                index = 1
            elif kk == t3:
                index = 2
            ww_plot[ii, index,:] = ZZ_gt[kk + 1, ii][:-1]
            bb_plot[ii, index] = ZZ_gt[kk + 1, ii][-1]

    norm_gt[kk] = np.linalg.norm(grad_gt[kk])

    if norm_gt[kk] < 1e-6:
        print("Converged")
        break

cost_gt = cost_gt[:-1]

norm_gt = norm_gt[:-1]

if 1:
    plot.plot_cost_T3(cost_gt, norm_gt, MAXITERS, 'Gradient Tracking')
    plot.plot_error_evolution(error, NN, MAXITERS)
    plot.plot_decision_boundaries_agents_T3(ww_plot, bb_plot, w_true, b_true, type, NN, iterations)

WW = ZZ_gt[:, :, :-1]
BB = ZZ_gt[:, :, -1]

# the final weights and biases of the agents
final_gt_ww = ZZ_gt[-1, :, :-1]
final_gt_bb = ZZ_gt[-1, :, -1]

count_gt = 0
# compute the number of misclassified points in the training data and the accuracy
for jj in range(NN):
    for ii in range(len(data_split[jj])):
        prediction_gt,_ = fn.binary_classifier(fn.phi(data_split[jj][ii], type), final_gt_ww[jj], final_gt_bb[jj])
        if prediction_gt != labels_split[jj][ii]:
            count_gt += 1
    accuracy_gt = (1 - count_gt / dataset_size)*100

print("Accuracy of Gradient Tracking:", accuracy_gt)
print("Misclassified points in training data:", count_gt)
print("True weights:", w_true)
print("True bias:", b_true)
print("Final weights:", final_gt_ww)
print("Final bias:", final_gt_bb)

if 1:
    plot.plot_parameter_convergence(WW, BB, w_true, b_true, dim_w, NN, MAXITERS)
    plot.plot_predictions(data_split, labels_split, final_gt_ww, final_gt_bb, 'Training Data Predictions', NN, w_true, b_true, type)


##### Testing on test data #####

# Generate the test data of size 0.25*dataset_size, and generate the labels
test_dataset = fn.generate_dataset(m_test, d)
test_labels = np.array([fn.binary_classifier(fn.phi(test_dataset[i], type), w_true, b_true)[0] for i in range(m_test)])

# sort and split the test data among the agents
N = NN
sorted_test_indices = np.argsort(test_dataset[:, 0])
sorted_test_dataset = test_dataset[sorted_test_indices]
sorted_test_labels = test_labels[sorted_test_indices]
test_data_split = np.array_split(sorted_test_dataset, N)
test_labels_split = np.array_split(sorted_test_labels, N)

# Compute number of misclassified points in the test data and the accuracy
count_gt_test = 0
predictions = []
for jj in range(NN):
    for ii in range(len(test_data_split[jj])):
        prediction_gt_test,_ = fn.binary_classifier(fn.phi(test_data_split[jj][ii], type), final_gt_ww[jj], final_gt_bb[jj])
        # prediction_dg_test,_ = binary_classifier(phi(test_data_split[jj][ii], type), final_dg_ww[jj], final_dg_bb[jj])
        predictions.append(prediction_gt_test)
        if prediction_gt_test != test_labels_split[jj][ii]:
            count_gt_test += 1
accuracy_gt_test = (1 - count_gt_test / m_test)*100

print("Accuracy of Gradient Tracking on Test Data:", accuracy_gt_test)
print("Misclassified points in test data:", count_gt_test)

if 1:
    plot.plot_sorted_dataset(test_data_split, NN, w_true, b_true, type)
    plot.plot_predictions(test_data_split, test_labels_split, final_gt_ww, final_gt_bb, 'Test Data Predictions', NN, w_true, b_true, type)