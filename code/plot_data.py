

import matplotlib.pyplot as plt
import numpy as np

def get_model_vals(exp_res, index, model):
        row = np.array(exp_res[index])
        vals = row[row[:,0]==model][:,1].astype(float)
        return vals

def calc_stats(vals):
        return np.mean(vals), np.std(vals)

def get_experiment_statistics(exp_2d_res):

    pls_mse_mean, pls_mse_std, pls_corr_mean, pls_corr_std, nn_mse_mean, nn_mse_std, nn_corr_mean, nn_corr_std = [],[],[],[],[],[],[],[]

    for i in range(len(exp_2d_res)):
        # Get model values
        mse_vals_pls = get_model_vals(exp_2d_res[i], 2, 'pls')
        mse_vals_nn = get_model_vals(exp_2d_res[i], 2, 'NN')
        corr_vals_pls = get_model_vals(exp_2d_res[i], 1, 'pls')
        corr_vals_nn = get_model_vals(exp_2d_res[i], 1, 'NN')

        # Compute and print stats
        mse_mean_pls, mse_std_pls = calc_stats(mse_vals_pls)
        mse_mean_nn, mse_std_nn = calc_stats(mse_vals_nn)
        corr_mean_pls, corr_std_pls = calc_stats(corr_vals_pls)
        corr_mean_nn, corr_std_nn = calc_stats(corr_vals_nn)

        # Store stats
        pls_mse_mean.append(mse_mean_pls)
        pls_mse_std.append(mse_std_pls)
        pls_corr_mean.append(corr_mean_pls)
        pls_corr_std.append(corr_std_pls)
        nn_mse_mean.append(mse_mean_nn)
        nn_mse_std.append(mse_std_nn)
        nn_corr_mean.append(corr_mean_nn)
        nn_corr_std.append(corr_std_nn)

    return pls_mse_mean, pls_mse_std, pls_corr_mean, pls_corr_std, nn_mse_mean, nn_mse_std, nn_corr_mean, nn_corr_std




def plot_metrics(exp_data, tick_array):
    
    nn_mse_data = [np.array(np.array(res)[2])[np.array(np.array(res)[2])[:,0]=='NN'][:,1].astype(float) for res in exp_data]
    nn_corr_data = [np.array(np.array(res)[1])[np.array(np.array(res)[2])[:,0]=='NN'][:,1].astype(float) for res in exp_data]
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs = axs.flatten()



    # Plot boxplots for NN MSE
    axs[0].boxplot(nn_mse_data)
    axs[0].set_title('MSE', fontsize=20)
    axs[0].set_xticklabels(tick_array, fontsize=12)
    axs[0].set_xlabel('% used features', fontsize=20)
    axs[0].set_ylabel('Metric value', fontsize=20)


    # Plot boxplots for NN correlation
    axs[1].boxplot(nn_corr_data)
    axs[1].set_title('$R^2$', fontsize=20)
    axs[1].set_xticklabels(tick_array, fontsize=12)
    axs[1].set_xlabel('% used features', fontsize=20)
    axs[1].set_ylabel('Metric value', fontsize=20)

    fig.tight_layout()
    plt.show()

def plot_boxplots(exp_res, tick_array):
    # Prepare data for boxplots
    mse_data = [res[2] for res in exp_res]
    corr_data = [res[1] for res in exp_res]

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs = axs.flatten()

    # Plot boxplots for MSE
    axs[0].boxplot(mse_data)
    axs[0].set_title('MSE', fontsize=20)
    axs[0].set_xticklabels(tick_array, fontsize=12)
    axs[0].set_xlabel('% used features', fontsize=20)
    axs[0].set_ylabel('Metric value', fontsize=20)

    # Plot boxplots for correlation
    axs[1].boxplot(corr_data)
    axs[1].set_title('$R^2$', fontsize=20)
    axs[1].set_xticklabels(tick_array, fontsize=12)
    axs[1].set_xlabel('% used features', fontsize=20)
    axs[1].set_ylabel('Metric value', fontsize=20)

    fig.tight_layout()
    plt.show()


