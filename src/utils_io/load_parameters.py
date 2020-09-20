import pandas as pd

def load_parameters(filepaths, num_nodes, num_data):

    df_parameters = pd.read_csv(filepaths.parameter + '.csv')
    parameters = df_parameters.to_numpy()
    parameters = parameters.reshape((num_data, num_nodes))

    print('parameters loaded')

    return parameters
