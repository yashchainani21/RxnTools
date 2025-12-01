import json
import pandas as pd
from typing import List
from rxntools import reaction, utils
from collections import Counter

reported_rxns_df = pd.read_parquet("../data/interim/enzymemap_MetaCyc_JN_mapped.parquet")
JN_rules_df = pd.read_csv('../data/raw/JN1224MIN_rules.tsv', delimiter='\t')

with open('../data/raw/cofactors.json') as f:
    cofactors_dict = json.load(f)

all_cofactor_codes: List[str] = list(cofactors_dict.keys())
cofactors_list: List[str] = [cofactors_dict[key] for key in cofactors_dict.keys()]
cofactors_df = pd.read_csv('../data/raw/all_cofactors.csv')
