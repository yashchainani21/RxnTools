import pandas as pd
from rxntools import reaction

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


with open('../data/raw/all_cofactors.csv') as f:
    cofactors_df = pd.read_csv(f, sep=',')

brenda_filepath = '../data/raw/enzymemap_v2_brenda2023.csv'

brenda_df = pd.read_csv(brenda_filepath)

#
for i, rxn_str in enumerate(list(brenda_df['unmapped'])):
    reactants, products = rxn_str.split('>>')
    reactants_list = reactants.split('.')
    products_list = products.split('.')

    if len(reactants_list) == 1 and len(products_list) == 2:

        for product_smiles in products_list:
            if product_smiles == 'O=C=O':
                keep_idx.append(i)

query_df = BRENDA_df.iloc[keep_idx, :]
query_df = query_df[query_df['source'] == 'direct']
query_rxns_frm_BRENDA = set(query_df['mapped'])
