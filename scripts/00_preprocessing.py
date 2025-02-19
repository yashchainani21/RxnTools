import pandas as pd

brenda_filepath = '../data/raw/enzymemap_v2_brenda2023.csv'

df = pd.read_csv(brenda_filepath)
print(df)