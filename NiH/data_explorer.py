import pandas as pd

df = pd.read_pickle('./data/validation_data_extended_4.pkl')
print(df['No Finding'].value_counts())