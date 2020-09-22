import pandas as pd

def load_dataset(filepath, num_nodes, num_data):

    df_dataset = pd.read_csv(filepath + '.csv')
    dataset = df_dataset.to_numpy()
    dataset = dataset.reshape((num_data, num_nodes))

    print('data loaded')

    return dataset
