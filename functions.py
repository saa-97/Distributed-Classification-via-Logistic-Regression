import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

def generate_random_graph(NN, graph_type):
    if graph_type == "cycle":
        G = nx.cycle_graph(NN)
    elif graph_type == "path":
        G = nx.path_graph(NN)
    elif graph_type == "star":
        G = nx.star_graph(NN-1)
    elif graph_type == "complete":
        G = nx.complete_graph(NN)
    else:
        raise ValueError("Invalid graph type")
    
    # nx.draw(G, with_labels=True)
    # plt.suptitle(graph_type + ' graph')
    # plt.show()
    
    Adj = nx.adjacency_matrix(G).toarray()
    return Adj

def quadratic_fn(z, q, r):
    quad_func = 0.5 * z.T @ q @ z + r.T @ z
    grad = q @ z + r
    return quad_func, grad

def generate_dataset(m, d):
    """
    Generates a dataset of m points in Rd.
    
    Args:
        m: The number of points to generate.
        d: The dimensionality of the data.
        
    Returns:
        A numpy array of shape (m, d) containing the dataset.
    """
    d1 = np.random.uniform(low = -10, high = 10, size=(m, 1))
    d2 = np.random.uniform(low = -5, high = 5, size=(m, 1))

    D = np.concatenate((d1, d2), axis=1)

    # return np.random.uniform(low=-10, high=10, size=(m, d))
    return D

def phi(D, type):
    """
    Nonlinear transformation function to capture elliptical 
    or non linear boundary shape.
    
    Args:
        D: A numpy array of shape (d,).
        
    Returns:
        A transformed numpy array.
    """

    if type == 'Ellipse': 
        output = np.array([D[0]**2, D[0]*D[1], D[1]**2, D[0], D[1]])
    elif type == 'Curve':
        output = np.array([D[0], D[1], D[0]**2, D[1]**2, D[0]*D[1]])

    return output

def binary_classifier(x, w, b):
    """
    Classifies a point using a logistic function.
    
    Args:
        x: A numpy array of shape (d,) representing a data point.
        w: A numpy array of shape (d,) representing the weight vector.
        b: A float representing the bias term.
        
    Returns:
        A tuple (class_label, linear_combination).
        class_label: 1 if x is classified as positive, -1 otherwise.
        linear_combination: The result of the linear combination of weights and input features.
    """
    z = np.dot(w, x) + b
    # probability = 1 / (1 + np.exp(-z))
    # class_label = 1 if probability >= 0.5 else -1
    class_label = 1 if z >= 0 else -1
    return class_label, z

def cost_func(y, z):
    """
    Logistic loss function for labels 1 and -1.
    """
    return np.log(1 + np.exp(-y * z))

def grad(y, z, x):
    """
    Calculates the gradient of the logistic cost function.
    
    Args:
        y: True label.
        z: Linear combination of weights and features.
        x: Feature vector.
        
    Returns:
        Gradients w.r.t. weights and bias.
    """
    gradient = -y / (1 + np.exp(y * z))
    return gradient * x, gradient


def predict(X, w, b, type):
    predictions = []
    for x in X:
        x_transformed = phi(x,type)
        pred = binary_classifier(x_transformed, w, b)[0]
        predictions.append(pred)
    return np.array(predictions)

def calculate_weights(NN, Adj):
    I_NN = np.eye(NN)
    AA = np.zeros(shape=(NN, NN))
    for ii in range(NN):
        N_ii = np.nonzero(Adj[ii])[0]
        deg_ii = len(N_ii)
        for jj in N_ii:
            deg_jj = len(np.nonzero(Adj[jj])[0])
            AA[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))
    AA += I_NN - np.diag(np.sum(AA, axis=0))
    return AA