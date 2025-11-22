from tqdm import tqdm
import pandas as pd
from ergochemics import mapping
from typing import List
import multiprocessing
import os
multiprocessing.set_start_method("fork", force=True)

rxns_df_input_filepath = "../data/raw/MetaCyc_proccessed.csv"
rxns_df_output_filepath = 

def make_rule_id(n: int, prefix: str = "rule", width: int = 4) -> str:
    """
    Convert an integer into a zero-padded rule ID of the form 'rule0001'.

    Args:
        n (int): The integer to convert.
        prefix (str): Optional prefix before the number. Defaults to "rule".
        width (int): Zero-padding width. Defaults to 4.

    Returns:
        str: Formatted rule ID (e.g., 'rule0004').
    """
    if n < 1:
        raise ValueError("Input must be >= 1.")
    return f"{prefix}{n:0{width}d}"

# extract and create a list of all minimal operators' SMARTS strings
gen_rxn_operators_df = pd.read_csv("../data/raw/JN1224MIN_rules.tsv", delimiter='\t')
gen_rxn_operators_list: List[str] = gen_rxn_operators_df["SMARTS"].to_list()

# extract and create a list of all unmapped MetaCyc reactions
enzymatic_rxns_df = pd.read_csv(rxns_df_filepath)
EnzymeMap_MetaCyc_rxns_unmapped: List[str] = enzymatic_rxns_df["unmapped"].to_list()

# remove all hydrogen ions from rxn strings so that they can be mapped by Stefan's ergochemics
EnzymeMap_MetaCyc_rxns_cleaned: List[str] = []

for rxn in EnzymeMap_MetaCyc_rxns_unmapped:
    rxn = rxn.replace(".[H+]","").replace("[H+].","")
    EnzymeMap_MetaCyc_rxns_cleaned.append(rxn)