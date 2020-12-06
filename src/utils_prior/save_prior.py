import os
import pandas as pd

def save_prior(filepaths,
               mean_vec, cov,
               inv_L, L):
    if not os.path.exists(filepaths.directory_dataset):
        os.makedirs(filepaths.directory_dataset)

    df_prior_mean = pd.DataFrame({'prior_mean': mean_vec.flatten()})
    df_prior_mean.to_csv(filepaths.prior_mean + '.csv', index=False)

    df_prior_covariance = pd.DataFrame({'prior_covariance': cov.flatten()})
    df_prior_covariance.to_csv(filepaths.prior_covariance + '.csv', index=False)

    df_prior_covariance_cholesky = pd.DataFrame({'prior_covariance_cholesky': L.flatten()})
    df_prior_covariance_cholesky.to_csv(
            filepaths.prior_covariance_cholesky + '.csv', index=False)

    df_prior_covariance_cholesky_inverse = pd.DataFrame(
            {'prior_covariance_inverse': inv_L.flatten()})
    df_prior_covariance_cholesky_inverse.to_csv(
            filepaths.prior_covariance_cholesky_inverse + '.csv',
            index=False)

    print('Prior constructed and saved')
