from tqdm import tqdm
import pandas as pd
from ergochemics import mapping
from typing import List
import os

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------
rxns_df_input_filepath = "/Users/yashchainani/Desktop/PythonProjects/RxnTools/data/raw/enzymemap_v2_brenda2023.csv"
rxns_df_output_filepath = "/Users/yashchainani/Desktop/PythonProjects/RxnTools/data/interim/enzymemap_v2_brenda2023_JN_mapped_unique_rxns.parquet"

# ----------------------------------------------------------
# Helper — make rule ID
# ----------------------------------------------------------
def make_rule_id(n: int, prefix: str = "rule", width: int = 4) -> str:
    if n < 1:
        raise ValueError("Input must be >= 1.")
    return f"{prefix}{n:0{width}d}"

# ----------------------------------------------------------
# Load SMARTS rules
# ----------------------------------------------------------
gen_rxn_operators_df = pd.read_csv("/Users/yashchainani/Desktop/PythonProjects/RxnTools/data/raw/JN1224MIN_rules.tsv", delimiter="\t")
gen_rxn_operators_list: List[str] = gen_rxn_operators_df["SMARTS"].to_list()

# ----------------------------------------------------------
# Load reaction dataset
# ----------------------------------------------------------
enzymatic_rxns_df = pd.read_csv(rxns_df_input_filepath)

# keep only unique unmapped reactions
enzymatic_rxns_df = enzymatic_rxns_df[~enzymatic_rxns_df["mapped"].duplicated()]
unmapped_rxns_list: List[str] = enzymatic_rxns_df["unmapped"].to_list()

# ----------------------------------------------------------
# Clean RXN strings
# ----------------------------------------------------------
cleaned_rxns_list = [
    rxn.replace(".[H+]", "").replace("[H+].", "")
    for rxn in unmapped_rxns_list
]

# ----------------------------------------------------------
# Single-core mapping function
# ----------------------------------------------------------
def map_single_reaction(idx: int, rxn: str, operators: List[str]):
    mapped_ops = []
    try:
        for i, operator in enumerate(operators):
            try:
                mapped_rxn = mapping.operator_map_reaction(rxn=rxn, operator=operator)
                if mapped_rxn.did_map:
                    mapped_ops.append(make_rule_id(i + 1))
            except Exception:
                # ignore operator-level failures
                pass
    except Exception as e:
        return idx, f"__WORKER_FAILED__: {repr(e)}"

    return idx, mapped_ops

# ----------------------------------------------------------
# MAIN — Serial loop (replaces multiprocessing entirely)
# ----------------------------------------------------------
all_mapped_operators = [None] * len(cleaned_rxns_list)

for i, rxn in tqdm(
    enumerate(cleaned_rxns_list),
    total=len(cleaned_rxns_list),
    desc="Mapping reactions (single core)"
):
    _, ops = map_single_reaction(i, rxn, gen_rxn_operators_list)
    all_mapped_operators[i] = ops

enzymatic_rxns_df["all_mapped_operators"] = all_mapped_operators

# ----------------------------------------------------------
# Compute top_mapped_operator
# ----------------------------------------------------------
def get_top_operator(op_list):
    if not op_list:
        return None
    nums = [int(op.replace("rule", "")) for op in op_list]
    min_num = min(nums)
    return f"rule{min_num:04d}"

enzymatic_rxns_df["top_mapped_operator"] = (
    enzymatic_rxns_df["all_mapped_operators"].apply(get_top_operator)
)

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------
enzymatic_rxns_df.to_parquet(rxns_df_output_filepath)

print("Completed single-core mapping and saved output →", rxns_df_output_filepath)
