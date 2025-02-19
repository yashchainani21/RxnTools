import pandas as pd
from rxntools import reaction

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


with open('../data/raw/all_cofactors.csv') as f:
    cofactors_df = pd.read_csv(f, sep=',')

brenda_filepath = '../data/raw/enzymemap_v2_brenda2023.csv'

df = pd.read_csv(brenda_filepath)
for rxn_str in list(df['unmapped'])[-10:-1]:
    rxn = reaction.unmapped_reaction(rxn_str)
    print(rxn.get_JN_rxn_descriptor(cofactors_df, consider_stereo = False))