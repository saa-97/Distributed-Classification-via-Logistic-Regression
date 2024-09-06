import matplotlib.pyplot as plt
import numpy as np
import functions as fn
from matplotlib.lines import Line2D
import scipy.sparse as sp


def plot_cost_and_norm(cost, norm, iters, title):
    plt.figure('Cost and Norm of ' + title)
    plt.subplot(1, 2, 1)
    plt.semilogy(np.arange(iters),cost[:iters])
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Cost', fontsize=14)
    plt.title('Cost of ' + title,  fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.subplot(1, 2, 2)
    plt.semilogy(np.arange(iters), norm[:iters])
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Norm', fontsize=14)
    plt.title('Norm of ' + title, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

def convergence_of_z(ZZ_gt, iters, NN, graph_type):
    plt.figure('Convergence of Z')
    plt.suptitle('Convergence with ' + graph_type + ' graph')

    plt.subplot(1, 2, 1)
    for ii in range(NN):
        plt.semilogx(np.arange(iters), ZZ_gt[:iters, ii, 0], label='Node ' + str(ii))
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Z', fontsize=14)
    plt.title('Convergence of Z_x', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)

    plt.subplot(1, 2, 2)
    for ii in range(NN):
        plt.semilogx(np.arange(iters), ZZ_gt[:iters, ii, 1], label='Node ' + str(ii))
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Z', fontsize=14)
    plt.title('Convergence of Z_y', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.show()

def error_of_z(ZZ_gt, iters, NN, ZZ_opt):

    error = np.zeros((iters,NN,2))

    for kk in range(iters):
        for ii in range(NN):
            error[kk,ii,:] = ZZ_gt[kk,ii]-ZZ_opt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ii in range(NN):
        ax1.semilogx(np.arange(iters-1), error[:iters-1, ii, 0], label='Node ' + str(ii))
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Consensus error - Z[0]')
    ax1.set_title('Consensus Error Evolution - Z[0]')
    ax1.grid(True)
    ax1.legend()

    for ii in range(NN):
        ax2.semilogx(np.arange(iters-1), error[:iters-1, ii, 1], label='Node ' + str(ii))
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Consensus error - Z[1]')
    ax2.set_title('Consensus Error Evolution - Z[1]')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_dataset(dataset, labels, ww, bb, title, type):
    for ii in range(dataset.shape[0]):
        if labels[ii] == 1:
            plt.plot(dataset[ii, 0], dataset[ii, 1], 'bo', label='Class 1' if ii == 0 else "")
        else:
            plt.plot(dataset[ii, 0], dataset[ii, 1], 'ro', label='Class -1' if ii == 0 else "")

    resolution = 1000 
    xx = np.linspace(min(dataset[:, 0]), max(dataset[:, 0]), resolution)
    yy = np.linspace(min(dataset[:, 1]), max(dataset[:, 1]), resolution)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = np.zeros(XX.shape)

    for ii in range(XX.shape[0]):
        for jj in range(XX.shape[1]):
            ZZ[ii, jj] = fn.binary_classifier(fn.phi(np.array([XX[ii, jj], YY[ii, jj]]), type), ww, bb)[0]

    plt.contour(XX, YY, ZZ, levels=[0], colors='k')
    plt.grid()
    plt.title(title)
    plt.show()

def plot_test_dataset(test_dataset, test_labels, test_predictions, ww, bb, w_true, b_true, test_data_size,title, type):

    plt.figure(figsize=(10, 6))
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = np.dot(w_true, fn.phi([X[i, j], Y[i, j]], type)) + b_true

    plt.contour(X, Y, Z, levels=[0], colors='black')

    Z_pred = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z_pred[i, j] = np.dot(ww, fn.phi([X[i, j], Y[i, j]], type)) + bb

    plt.contour(X, Y, Z_pred, levels=[0], colors='green', linestyles='dashed')

    for i in range(test_data_size-1):

        if test_predictions[i] == 1 and test_labels[i] == 1:
            plt.scatter(test_dataset[i, 0], test_dataset[i, 1], color='blue', marker='o')
        elif test_predictions[i] == -1 and test_labels[i] == -1:
            plt.scatter(test_dataset[i, 0], test_dataset[i, 1], color='red', marker='o')
        else:
            plt.scatter(test_dataset[i, 0], test_dataset[i, 1], color='green', marker='x')

    plt.title(title)
    plt.xlabel("D_1")
    plt.ylabel("D_2")
    plt.grid(True)

    scatter_correct_class1 = plt.scatter([], [], color='blue', marker='o', label='Correct Class 1')
    scatter_correct_class_neg1 = plt.scatter([], [], color='red', marker='o', label='Correct Class -1')
    scatter_misclassified = plt.scatter([], [], color='green', marker='x', label='Misclassified')

    plt.legend(handles=[scatter_correct_class1, scatter_correct_class_neg1, scatter_misclassified], loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

def plot_decision_boundary_evolution(ww_plot, bb_plot, w_true, b_true, type, title,t1,t2,t3):
    plt.figure(figsize=(10, 6))
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = np.dot(w_true, fn.phi([X[i, j], Y[i, j]], type)) + b_true

    actual_contour = plt.contour(X, Y, Z, levels=[0], colors='black')

    colors = ['green', 'blue', 'red']
    labels = ['Iteration ' + str(t1), 'Iteration ' + str(t2), 'Iteration ' + str(t3)]
    linestyles = ['dashed', 'dashed', 'dashed']

    for idx in range(3):
        ww = ww_plot[idx]
        bb = bb_plot[idx]
        Z_pred = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z_pred[i, j] = np.dot(ww, fn.phi([X[i, j], Y[i, j]], type)) + bb

        plt.contour(X, Y, Z_pred, levels=[0], colors=colors[idx], linestyles=linestyles[idx])

    plt.title(title)
    plt.xlabel("D_1")
    plt.ylabel("D_2")
    plt.grid(True)

    custom_lines = [
        Line2D([0], [0], color='black', lw=2, label='Actual Decision Boundary'),
        Line2D([0], [0], color='green', lw=2, linestyle='dashed', label=labels[0]),
        Line2D([0], [0], color='blue', lw=2, linestyle='dashed', label=labels[1]),
        Line2D([0], [0], color='red', lw=2, linestyle='dashed', label=labels[2])
    ]
    plt.legend(handles=custom_lines)
    plt.show()

def plot_decision_boundaries_agents_T3(ww_plot_agents, bb_plot_agents, w_true, b_true, type, NN, iterations):
    fig, axs = plt.subplots(5, 2, figsize=(20, 10))  # 2 rows, 5 columns for 10 agents
    axs = axs.flatten()
    
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = np.dot(w_true, fn.phi([X[i, j], Y[i, j]], type)) + b_true

    for agent in range(NN):
        ax = axs[agent]
        ax.contour(X, Y, Z, levels=[0], colors='black')

        colors = ['green', 'blue', 'red']
        labels = [f'Iteration {t}' for t in iterations]

        for idx in range(len(iterations)):
            ww = ww_plot_agents[agent, idx]
            bb = bb_plot_agents[agent, idx]
            Z_pred = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z_pred[i, j] = np.dot(ww, fn.phi([X[i, j], Y[i, j]], type)) + bb

            ax.contour(X, Y, Z_pred, levels=[0], colors=colors[idx], linestyles='dashed')
        
        ax.set_title(f'Agent {agent + 1}')
        ax.set_xlabel("D_1")
        ax.set_ylabel("D_2")
        ax.grid(True)

    custom_lines = [
        Line2D([0], [0], color='black', lw=2, label='Actual Decision Boundary'),
        Line2D([0], [0], color='green', lw=2, linestyle='dashed', label=labels[0]),
        Line2D([0], [0], color='blue', lw=2, linestyle='dashed', label=labels[1]),
        Line2D([0], [0], color='red', lw=2, linestyle='dashed', label=labels[2])
    ]
    fig.legend(handles=custom_lines, loc='upper right', ncol=2 )
    plt.suptitle('Decision Boundaries of Agents at Different Iterations')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_error_evolution(error, NN, MAXITERS):
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))  
    axs = axs.flatten()

    for agent in range(NN):
        ax = axs[agent]
        ax.semilogy(range(MAXITERS-1), error[:-1, agent], label=f'Agent {agent + 1}')
        
        ax.set_title(f'Agent {agent + 1}')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Error $||\\nabla \\ell_i - s_i||$')
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Error Evolution of Agents')
    plt.show()

def plot_predictions(dataset, labels, ww, bb, title, NN, w_true, b_true, type):
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = np.dot(w_true, fn.phi([X[i, j], Y[i, j]], type)) + b_true

    fig, axs = plt.subplots(5, 2, figsize=(20, 25))  
    fig.suptitle(title, fontsize=16)

    for jj in range(NN):
        for ii in range(len(dataset[jj])):
            prediction, _ = fn.binary_classifier(fn.phi(dataset[jj][ii], type), ww[jj], bb[jj])
            if prediction == 1 and labels[jj][ii] == 1:
                axs[jj // 2, jj % 2].scatter(dataset[jj][ii][0], dataset[jj][ii][1], color='blue', marker='o', s=10)
            elif prediction == -1 and labels[jj][ii] == -1:
                axs[jj // 2, jj % 2].scatter(dataset[jj][ii][0], dataset[jj][ii][1], color='red', marker='o', s=10)
            else:
                axs[jj // 2, jj % 2].scatter(dataset[jj][ii][0], dataset[jj][ii][1], color='green', marker='x', s=25)

        Z_pred = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z_pred[i, j] = np.dot(ww[jj], fn.phi([X[i, j], Y[i, j]], type)) + bb[jj]
        axs[jj // 2, jj % 2].contour(X, Y, Z_pred, levels=[0], colors='green', linestyles='dashed')
        axs[jj // 2, jj % 2].contour(X, Y, Z, levels=[0], colors='black')
        axs[jj // 2, jj % 2].set_title(f'Agent {jj+1}')

    scatter_correct_class1 = plt.scatter([], [], color='blue', marker='o', s=25, label='Correct Class 1')
    scatter_correct_class_neg1 = plt.scatter([], [], color='red', marker='o', s=25, label='Correct Class -1')
    scatter_misclassified = plt.scatter([], [], color='green', marker='x', s=25, label='Misclassified')

    fig.legend(handles=[scatter_correct_class1, scatter_correct_class_neg1, scatter_misclassified], loc='upper left')

    plt.tight_layout(rect=[0.1, 0.1, 1, 0.95])  
    plt.show()

def plot_sorted_dataset(dataset, N, w_true, b_true, type):
    plt.figure(figsize=(10, 6))
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = np.dot(w_true, fn.phi([X[i, j], Y[i, j]], type)) + b_true

    colors = plt.cm.jet(np.linspace(0, 1, N))  

    for i, split in enumerate(dataset):
        plt.scatter(split[:, 0], split[:, 1], color=colors[i], label=f'Agent {i+1}')

    plt.title('Dataset Splits Among Agents')
    plt.xlabel('D[0]')
    plt.ylabel('D[1]')
    plt.legend()
    plt.show()

def plot_cost_T3(cost_gt, norm_gt, MAXITERS, title):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.semilogy(np.arange(MAXITERS - 1), cost_gt, label="Cost of " + title)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost vs Iteration')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.semilogy(np.arange(MAXITERS - 1), norm_gt, label="Norm of gradient of " + title)
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm vs Iteration')
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_parameter_convergence(WW, BB, w_true, b_true, dim_w, NN, MAXITERS):
    for dim in range(dim_w):
        plt.figure(figsize=(12, 6))
        for ii in range(NN):
            plt.semilogx(np.arange(MAXITERS), WW[:, ii, dim], label=f'Node {ii+1}')
        plt.axhline(y=w_true[dim], color='r', linestyle='--', label='True Weight' if ii == 0 else "")
        plt.xlabel('Iteration')
        plt.ylabel(f'Weight {dim+1} Value')
        plt.title(f'Convergence of Weight {dim+1}')
        plt.legend()
        plt.grid()
        plt.show()

    plt.figure(figsize=(12, 6))
    for ii in range(NN):
        plt.semilogx(np.arange(MAXITERS), BB[:, ii], label=f'Node {ii+1}')
    plt.axhline(y=b_true, color='r', linestyle='--', label='True Bias')
    plt.xlabel('Iteration')
    plt.ylabel('Bias Value')
    plt.title('Convergence of Bias')
    plt.legend()
    plt.grid()
    plt.show()