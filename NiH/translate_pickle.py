import pandas as pd


def pickle_five_to_four(path, new_path):
    df: pd.DataFrame = pd.read_pickle(path)

    df.to_pickle(new_path, protocol=4)


pickle_five_to_four('./data/test_data_extended.pkl', './data/test_data_extended_4.pkl')
pickle_five_to_four('./data/training_data_extended.pkl', './data/training_data_extended_4.pkl')
pickle_five_to_four('./data/validation_data_extended.pkl', './data/validation_data_extended_4.pkl')
