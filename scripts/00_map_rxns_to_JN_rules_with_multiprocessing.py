from tqdm import tqdm
import pandas as pd
from ergochemics import mapping
from typing import List
import multiprocessing
import os
multiprocessing.set_start_method("fork", force=True)

rxns_df_input_filepath = "../data/raw/enzymemap_v2_brenda2023.csv"
rxns_df_output_filepath = "../data/interim/enzymemap_v2_brenda2023_JN_mapped_unique_rxns.parquet"

# enable batching to map large numbers of reactions without running out of memory
batch_size = 10000  # number of reactions to process in each batch
batch_num = 0  # current batch number
start_idx = 0  # starting index for the current batch


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

# extract and create a list of all unmapped reactions
enzymatic_rxns_df = pd.read_csv(rxns_df_input_filepath)
enzymatic_rxns_df = enzymatic_rxns_df[~enzymatic_rxns_df['mapped'].duplicated()]
unmapped_rxns_list: List[str] = enzymatic_rxns_df["unmapped"].to_list()

# remove all hydrogen ions from rxn strings so that they can be mapped by Stefan's ergochemics
cleaned_rxns_list: List[str] = []

for rxn in unmapped_rxns_list:
    rxn = rxn.replace(".[H+]","").replace("[H+].","")
    cleaned_rxns_list.append(rxn)

def map_single_reaction(args):
    """(index, rxn, operator_list) → (index, mapped_ops)"""
    idx, rxn, gen_rxn_operators_list = args

    mapped_ops = []
    try:
        for i, operator in enumerate(gen_rxn_operators_list):
            try:
                mapped_rxn = mapping.operator_map_reaction(rxn=rxn, operator=operator)
                if mapped_rxn.did_map:
                    mapped_ops.append(make_rule_id(i+1))
            except Exception:
                pass
    except Exception as e:
        return idx, f"__WORKER_FAILED__: {repr(e)}"

    return idx, mapped_ops


# ---- MAIN ----
rxns = cleaned_rxns_list
tasks = [(i, rxn, gen_rxn_operators_list) for i, rxn in enumerate(rxns)]

results = [None] * len(tasks)

with multiprocessing.Pool(os.cpu_count()) as p:
    for idx, mapped_ops in tqdm(
        p.imap_unordered(map_single_reaction, tasks),
        total=len(tasks),
        desc="Mapping reactions",
    ):
        results[idx] = mapped_ops

all_mapped_operators = results

enzymatic_rxns_df["all_mapped_operators"] = all_mapped_operators

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


enzymatic_rxns_df["top_mapped_operator"] = (
    enzymatic_rxns_df["all_mapped_operators"]
    .apply(get_top_operator)
)

enzymatic_rxns_df.to_parquet(rxns_df_output_filepath)