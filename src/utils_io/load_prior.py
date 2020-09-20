import pandas as pd

def load_prior(filepaths, num_nodes):

    #=== Mean ===#
    df_prior_mean = pd.read_csv(filepaths.prior_mean + '.csv')
    prior_mean = df_prior_mean.to_numpy()

    #=== Covariance ===#
    df_prior_covariance = pd.read_csv(filepaths.prior_covariance + '.csv')
    prior_covariance = df_prior_covariance.to_numpy()
    prior_covariance = prior_covariance.reshape((num_nodes, num_nodes))

    #=== Covariance Cholesky ===#
    df_prior_covariance_cholesky =\
            pd.read_csv(filepaths.prior_covariance_cholesky + '.csv')
    prior_covariance_cholesky = df_prior_covariance_cholesky.to_numpy()
    prior_covariance_cholesky = prior_covariance_cholesky.reshape((num_nodes, num_nodes))

    #=== Covariance Cholesky Inverse ===#
    df_covariance_cholesky_inverse =\
            pd.read_csv(filepaths.prior_covariance_cholesky_inverse + '.csv')
    prior_covariance_cholesky_inverse = df_covariance_cholesky_inverse.to_numpy()
    prior_covariance_cholesky_inverse =\
            prior_covariance_cholesky_inverse.reshape((num_nodes, num_nodes))

    return prior_mean, prior_covariance,\
            prior_covariance_cholesky, prior_covariance_cholesky_inverse
