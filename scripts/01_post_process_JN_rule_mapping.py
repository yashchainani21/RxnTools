import json
import pandas as pd
from rxntools import reaction, utils
from typing import List
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# load in cofactors data and JN generalized reaction rules
with open('../data/raw/cofactors.json') as f:
    cofactors_dict = json.load(f)

all_cofactor_codes: List[str] = list(cofactors_dict.keys())
cofactors_list: List[str] = [cofactors_dict[key] for key in cofactors_dict.keys()]

cofactors_df = pd.read_csv('../data/raw/all_cofactors.csv')
JN_rules_df = pd.read_csv('../data/raw/JN1224MIN_rules.tsv', delimiter='\t')

def parse_operator_list(x):
    # already a list → return as-is
    if isinstance(x, list):
        return x

    # bytes → decode
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8")

    # missing / empty
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []

    if isinstance(x, str):
        x = x.strip()

        if x in ("", "None", "nan"):
            return []

        try:
            # JSON-style list
            return json.loads(x)
        except json.JSONDecodeError:
            pass

    # fallback: return empty list or raise
    return []

def get_top_operator(op_list: List[str]) -> str:
    """
    Given a list like ['rule0002', 'rule0754'], return the one with
    the smallest integer value (e.g. 'rule0002').
    """
    if not op_list:
        return None  

    # extract integer part: "rule0034" → 34
    nums = [int(op.replace("rule", "")) for op in op_list]

    # lowest rule number
    min_num = min(nums)

    # convert back to rule format
    return f"rule{min_num:04d}"

###### ------- loading in KEGG and MetaCyc reactions data ------- ######
#output_filepath = '../data/processed/enzymemap_MetaCyc_JN_mapped_non_unique.parquet'
#input_rxns_w_JN_mappings = '../data/interim/enzymemap_MetaCyc_JN_mapped.parquet'
#input_rxns_w_JN_mappings_df = pd.read_parquet(input_rxns_w_JN_mappings)
###### ---------------------------------------------------------- ######                

###### ------- loading in BRENDA reactions data in batches ------- ######
output_filepath = '../data/processed/enzymemap_BRENDA_JN_mapped_non_unique.parquet'
input_rxns_df0 = pd.read_parquet('../data/interim/enzymemap_v2_brenda2023_JN_mapped_unique_rxns_batch0.parquet')
input_rxns_df1 = pd.read_parquet('../data/interim/enzymemap_v2_brenda2023_JN_mapped_unique_rxns_batch1.parquet')
input_rxns_df2 = pd.read_parquet('../data/interim/enzymemap_v2_brenda2023_JN_mapped_unique_rxns_batch2.parquet')
input_rxns_df3 = pd.read_parquet('../data/interim/enzymemap_v2_brenda2023_JN_mapped_unique_rxns_batch3.parquet')
input_rxns_df4 = pd.read_parquet('../data/interim/enzymemap_v2_brenda2023_JN_mapped_unique_rxns_batch4.parquet')
input_rxns_df5 = pd.read_parquet('../data/interim/enzymemap_v2_brenda2023_JN_mapped_unique_rxns_batch5.parquet')
input_rxns_df6 = pd.read_parquet('../data/interim/enzymemap_v2_brenda2023_JN_mapped_unique_rxns_batch6.parquet')
input_rxns_df7 = pd.read_parquet('../data/interim/enzymemap_v2_brenda2023_JN_mapped_unique_rxns_batch7.parquet')
input_rxns_df8 = pd.read_parquet('../data/interim/enzymemap_v2_brenda2023_JN_mapped_unique_rxns_batch8.parquet')
input_rxns_df9 = pd.read_parquet('../data/interim/enzymemap_v2_brenda2023_JN_mapped_unique_rxns_batch9.parquet')
input_rxns_df10 = pd.read_parquet('../data/interim/enzymemap_v2_brenda2023_JN_mapped_unique_rxns_batch10.parquet')
input_rxns_df11 = pd.read_parquet('../data/interim/enzymemap_v2_brenda2023_JN_mapped_unique_rxns_batch11.parquet')
input_rxns_df12 = pd.read_parquet('../data/interim/enzymemap_v2_brenda2023_JN_mapped_unique_rxns_batch12.parquet')
input_rxns_df13 = pd.read_parquet('../data/interim/enzymemap_v2_brenda2023_JN_mapped_unique_rxns_batch13.parquet')

input_rxns_w_JN_mappings_df = pd.concat([input_rxns_df0,
                                         input_rxns_df1,
                                         input_rxns_df2,
                                         input_rxns_df3,
                                         input_rxns_df4,
                                         input_rxns_df5,
                                         input_rxns_df6,
                                         input_rxns_df7,
                                         input_rxns_df8,
                                         input_rxns_df9,
                                         input_rxns_df10,
                                         input_rxns_df11,
                                         input_rxns_df12,
                                         input_rxns_df13], ignore_index=True)

input_rxns_w_JN_mappings_df.reset_index(drop=True, inplace=True)

# retain unique reactions only based on the 'unmapped' column
input_rxns_w_JN_mappings_df = input_rxns_w_JN_mappings_df.drop_duplicates(subset=['unmapped']).reset_index(drop=True)

input_rxns_w_JN_mappings_df["all_mapped_operators"] = (
    input_rxns_w_JN_mappings_df["all_mapped_operators"]
    .apply(parse_operator_list))

###### ------------------------------------------------------------- ######

print(f"\nTotal reactions to re-process: {input_rxns_w_JN_mappings_df.shape[0]}\n")       

all_unmapped_rxns_list = input_rxns_w_JN_mappings_df['unmapped'].to_list()

# remove the 'top_mapped_operator' column if it exists
if 'top_mapped_operator' in input_rxns_w_JN_mappings_df.columns:
    input_rxns_w_JN_mappings_df = input_rxns_w_JN_mappings_df.drop(columns=['top_mapped_operator'])

# initialize a list to store the best revised JN rule mapping for each reaction
all_top_mapped_operators: List[str] = []

# initialize a list to store the indices of successfully processed reactions
keep_idx: List[int] = []
all_substrates: List[List[str]] = []
all_products: List[List[str]] = []
all_LHS_cofactors: List[List[str]] = []
all_RHS_cofactors: List[List[str]] = []
all_LHS_cofactor_codes: List[List[str]] = []
all_RHS_cofactor_codes: List[List[str]] = []

rxns_skipped_count = 0

for i, rxn_SMILES in enumerate(all_unmapped_rxns_list):
    unmapped_rxn = reaction.unmapped_reaction(rxn_SMILES)
    
    try:
        # extract substrates, products, and cofactors (list of SMILES strings)
        substrates_list = unmapped_rxn.get_substrates(cofactors_list = cofactors_list, consider_stereo=False)
        products_list = unmapped_rxn.get_products(cofactors_list = cofactors_list, consider_stereo=False)
        lhs_cofactors_list = unmapped_rxn.get_lhs_cofactors(cofactors_list = cofactors_list, consider_stereo=False)
        lhs_cofactors_list = [x for x in lhs_cofactors_list if x!='[H+]']  # remove H+ from cofactors
        rhs_cofactors_list = unmapped_rxn.get_rhs_cofactors(cofactors_list = cofactors_list, consider_stereo=False)
        rhs_cofactors_list = [x for x in rhs_cofactors_list if x!='[H+]']  # remove H+ from cofactors
    
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
        single_best_mapped_rule = get_top_operator(best_mapped_rules)
        all_top_mapped_operators.append(single_best_mapped_rule)

        # store substrates, products, and cofactors
        all_substrates.append(substrates_list)
        all_products.append(products_list)
        all_LHS_cofactors.append(lhs_cofactors_list)
        all_RHS_cofactors.append(rhs_cofactors_list)
        all_LHS_cofactor_codes.append(lhs_cofactor_codes)
        all_RHS_cofactor_codes.append(rhs_cofactor_codes)
        
        # if everything went well, keep this index
        keep_idx.append(i)

    except Exception as e:
        print(f"Error processing reaction {i}: {e}\n")
        rxns_skipped_count += 1
        continue

# slice the dataframe to keep only successfully processed reactions
final_df = input_rxns_w_JN_mappings_df.iloc[keep_idx, :].copy()

# add new columns to the final dataframe
final_df['substrates'] = all_substrates
final_df['products'] = all_products
final_df['LHS_cofactors'] = all_LHS_cofactors
final_df['RHS_cofactors'] = all_RHS_cofactors
final_df['LHS_cofactor_codes'] = all_LHS_cofactor_codes
final_df['RHS_cofactor_codes'] = all_RHS_cofactor_codes
final_df['top_mapped_operator'] = all_top_mapped_operators

# save the final processed dataframe
final_df.to_parquet(output_filepath, index=False)
print(f"\nProcessed reactions saved to {output_filepath}\n")

print(rxns_skipped_count, "reactions were skipped due to errors.")
print(f"Final processed reactions count: {final_df.shape[0]}")