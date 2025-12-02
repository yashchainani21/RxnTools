import json
import pandas as pd
from rxntools import reaction, utils
from typing import List

with open('../data/raw/cofactors.json') as f:
    cofactors_dict = json.load(f)

all_cofactor_codes: List[str] = list(cofactors_dict.keys())
cofactors_list: List[str] = [cofactors_dict[key] for key in cofactors_dict.keys()]
cofactors_df = pd.read_csv('../data/raw/all_cofactors.csv')

# load in interim mapped reactions data
input_rxns_w_JN_mappings = '../data/interim/enzymemap_KEGG_JN_mapped.parquet'
input_rxns_w_JN_mappings_df = pd.read_parquet(input_rxns_w_JN_mappings)
all_unmapped_rxns_list = input_rxns_w_JN_mappings_df['unmapped'].to_list()

# remove the 'top_mapped_operator' column if it exists
if 'top_mapped_operator' in input_rxns_w_JN_mappings_df.columns:
    input_rxns_w_JN_mappings_df = input_rxns_w_JN_mappings_df.drop(columns=['top_mapped_operator'])

rxns_skipped_count = 0

for i, rxn_SMILES in enumerate(all_unmapped_rxns_list):
    unmapped_rxn = reaction.unmapped_reaction(rxn_SMILES)
    
    try:
        # extract substrates, products, and cofactors
        substrates_list = unmapped_rxn.get_substrates(cofactors_list = cofactors_list, consider_stereo=False)
        products_list = unmapped_rxn.get_products(cofactors_list = cofactors_list, consider_stereo=False)
        lhs_cofactors_list = unmapped_rxn.get_lhs_cofactors(cofactors_list = cofactors_list, consider_stereo=False)
        rhs_cofactors_list = unmapped_rxn.get_rhs_cofactors(cofactors_list = cofactors_list, consider_stereo=False)
    
        # from cofactor SMILES extracted, get their cofactor codes
        # extract cofactor codes (leave out H+)
        lhs_cofactor_codes = [utils.get_cofactor_CoF_code(cofactor_smiles, cofactors_df) for cofactor_smiles in lhs_cofactors_list]
        rhs_cofactor_codes = [utils.get_cofactor_CoF_code(cofactor_smiles, cofactors_df) for cofactor_smiles in rhs_cofactors_list]

    except Exception as e:
        print(f"Error processing reaction {i}: {e}")
        rxns_skipped_count += 1
        continue

print(rxns_skipped_count, "reactions were skipped due to errors.")