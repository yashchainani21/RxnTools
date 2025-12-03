import json
import pandas as pd
from rxntools import reaction, utils
from typing import List

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# load in cofactors data and JN generalized reaction rules
with open('/Users/yashchainani/Desktop/PythonProjects/RxnTools/data/raw/cofactors.json') as f:
    cofactors_dict = json.load(f)

all_cofactor_codes: List[str] = list(cofactors_dict.keys())
cofactors_list: List[str] = [cofactors_dict[key] for key in cofactors_dict.keys()]
cofactors_df = pd.read_csv('/Users/yashchainani/Desktop/PythonProjects/RxnTools/data/raw/all_cofactors.csv')

JN_rules_df = pd.read_csv('/Users/yashchainani/Desktop/PythonProjects/RxnTools/data/raw/JN1224MIN_rules.tsv', delimiter='\t')

def get_top_operator(op_list):
    """
    Given a list like ['rule0002', 'rule0754'], return the one with
    the smallest integer value (e.g. 'rule0002').
    """
    if not op_list:
        return None  # or np.nan if you prefer

    # extract integer part: "rule0034" → 34
    nums = [int(op.replace("rule", "")) for op in op_list]

    # lowest rule number
    min_num = min(nums)

    # convert back to rule format
    return f"rule{min_num:04d}"

# load in interim mapped reactions data
input_rxns_w_JN_mappings = '/Users/yashchainani/Desktop/PythonProjects/RxnTools/data/interim/enzymemap_KEGG_JN_mapped.parquet'
input_rxns_w_JN_mappings_df = pd.read_parquet(input_rxns_w_JN_mappings)                
print(f"\nTotal reactions to re-process: {input_rxns_w_JN_mappings_df.shape[0]}\n")       

all_unmapped_rxns_list = input_rxns_w_JN_mappings_df['unmapped'].to_list()

# remove the 'top_mapped_operator' column if it exists
if 'top_mapped_operator' in input_rxns_w_JN_mappings_df.columns:
    input_rxns_w_JN_mappings_df = input_rxns_w_JN_mappings_df.drop(columns=['top_mapped_operator'])

# initialize a list to store the best revised JN rule mapping for each reaction
all_top_mapped_operators: List[str] = []

# initialize a list to store the indices of successfully processed reactions
keep_idx = []
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
        lhs_cofactor_codes = [code for code in lhs_cofactor_codes if code!='H+']
        
        rhs_cofactor_codes = [utils.get_cofactor_CoF_code(cofactor_smiles, cofactors_df) for cofactor_smiles in rhs_cofactors_list]
        rhs_cofactor_codes = [code for code in rhs_cofactor_codes if code!='H+']

        # get all possible JN rule mappings for this reaction
        all_rule_mappings = input_rxns_w_JN_mappings_df.iloc[i, :]['all_mapped_operators']

        # initialize an empty list to store the best matching JN rule for this given reaction
        best_mapped_rules: List[str] = []
        
        # identify which JN rule mapping has the best cofactor/ substrate-product pairs match
        for rule in all_rule_mappings:
            JN_reactants = JN_rules_df[JN_rules_df['Name']==rule]['Reactants'].to_list()[0].split(';')
            JN_products = JN_rules_df[JN_rules_df['Name']==rule]['Products'].to_list()[0].split(';')
            
            JN_lhs_cofactors = [x for x in JN_reactants if x!='Any']
            JN_rhs_cofactors = [x for x in JN_products if x!='Any']
            JN_substrates = [x for x in JN_reactants if x=='Any']
            JN_products = [x for x in JN_products if x=='Any']

            # check first that the number of substrates and products match that specified by the JN rule
            if len(substrates_list) == len(JN_substrates) and len(products_list) == len(JN_products):

                # then, check if the cofactors match that specified by the JN rule
                if set(lhs_cofactor_codes) == set(JN_lhs_cofactors) and set(rhs_cofactor_codes) == set(JN_rhs_cofactors):
                    best_mapped_rules.append(rule)

        # of the all the best mapped rules, pick the first one (since the JN rules are ordered by frequency of occurrence) 
        all_top_mapped_operators.append(get_top_operator(best_mapped_rules))

        # if everything went well, keep this index
        keep_idx.append(i)

    except Exception as e:
        print(f"Error processing reaction {i}: {e}")
        rxns_skipped_count += 1
        continue

final_df = input_rxns_w_JN_mappings_df.iloc[keep_idx, :].copy()
final_df['top_mapped_operator'] = all_top_mapped_operators

print(rxns_skipped_count, "reactions were skipped due to errors.")
print(f"Final processed reactions count: {final_df.shape[0]}")