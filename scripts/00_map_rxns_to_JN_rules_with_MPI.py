from mpi4py import MPI
from tqdm import tqdm
import pandas as pd
from ergochemics import mapping
from typing import List, Tuple, Any
import numpy as np
import os


# ------------------ Helper functions ------------------ #

def make_rule_id(n: int, prefix: str = "rule", width: int = 4) -> str:
    if n < 1:
        raise ValueError("Input must be >= 1.")
    return f"{prefix}{n:0{width}d}"


def map_single_reaction(idx: int, rxn: str, gen_rxn_operators_list: List[str]) -> Tuple[int, Any]:
    mapped_ops = []
    try:
        for i, operator in enumerate(gen_rxn_operators_list):
            try:
                mapped_rxn = mapping.operator_map_reaction(rxn=rxn, operator=operator)
                if mapped_rxn.did_map:
                    mapped_ops.append(make_rule_id(i + 1))
            except Exception:
                pass
    except Exception as e:
        return idx, f"__WORKER_FAILED__: {repr(e)}"
    return idx, mapped_ops


def get_top_operator(op_list):
    if not isinstance(op_list, list) or not op_list:
        return None
    nums = [int(op.replace("rule", "")) for op in op_list]
    return f"rule{min(nums):04d}"


def chunk_list(lst, n):
    k, m = divmod(len(lst), n)
    chunks = []
    start = 0
    for i in range(n):
        end = start + k + (1 if i < m else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks


# ------------------ MPI Setup ------------------ #

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ------------------ File Paths ------------------ #

rxns_df_input_filepath = "../data/raw/enzymemap_v2_brenda2023.csv"
rxns_df_output_filepath = "../data/interim/enzymemap_v2_brenda2023_JN_mapped.parquet"
rules_filepath = "../data/raw/JN1224MIN_rules.tsv"


# ------------------ Rank 0: Load and Deduplicate Reactions ------------------ #

if rank == 0:
    print(f"[rank 0] Reading operator rules...")
    gen_rxn_operators_df = pd.read_csv(rules_filepath, delimiter="\t")
    gen_rxn_operators_list = gen_rxn_operators_df["SMARTS"].to_list()

    print(f"[rank 0] Reading reactions...")
    df = pd.read_csv(rxns_df_input_filepath)

    raw_rxns = df["unmapped"].astype(str).tolist()

    # Clean hydrogens
    cleaned_rxns = [
        r.replace(".[H+]", "").replace("[H+].", "") for r in raw_rxns
    ]

    df["cleaned_unmapped"] = cleaned_rxns

    # Create list of unique reactions
    print(f"[rank 0] Finding unique reaction strings...")
    unique_rxns = sorted(list(set(cleaned_rxns)))
    print(f"[rank 0] Unique reactions: {len(unique_rxns)} (from {len(cleaned_rxns)})")

    # Build mapping cleaned_rxn → unique_index
    rxn_to_uid = {rxn: i for i, rxn in enumerate(unique_rxns)}

    # Build tasks: (unique_index, unique_rxn)
    unique_tasks = [(i, rxn) for i, rxn in enumerate(unique_rxns)]
    n_unique_tasks = len(unique_tasks)

    print(f"[rank 0] MPI will map {n_unique_tasks} unique reactions.")

    task_chunks = chunk_list(unique_tasks, size)

else:
    gen_rxn_operators_list = None
    df = None
    unique_rxns = None
    rxn_to_uid = None
    unique_tasks = None
    n_unique_tasks = None
    task_chunks = None


# ------------------ Broadcast Shared Data ------------------ #

gen_rxn_operators_list = comm.bcast(gen_rxn_operators_list, root=0)
unique_rxns = comm.bcast(unique_rxns, root=0)
rxn_to_uid = comm.bcast(rxn_to_uid, root=0)
n_unique_tasks = comm.bcast(n_unique_tasks, root=0)

# Each rank receives its own list of unique tasks
local_tasks = comm.scatter(task_chunks, root=0)


# ------------------ Parallel Mapping on Each Rank ------------------ #

local_results = []

for uid, rxn in tqdm(local_tasks, desc=f"[Rank {rank}]", disable=False):
    local_results.append(map_single_reaction(uid, rxn, gen_rxn_operators_list))


# ------------------ Gather Unique Reaction Results ------------------ #

all_results = comm.gather(local_results, root=0)


# ------------------ Rank 0: Expand Back to Full DataFrame ------------------ #

if rank == 0:
    print("[rank 0] Assembling unique mapping results...")

    # Flatten
    unique_results = [None] * n_unique_tasks
    for rank_results in all_results:
        for uid, mapped_ops in rank_results:
            unique_results[uid] = mapped_ops

    # Expand back to original rows
    print("[rank 0] Expanding unique results to full dataframe...")
    all_mapped = []
    for cleaned in df["cleaned_unmapped"]:
        uid = rxn_to_uid[cleaned]
        all_mapped.append(unique_results[uid])

    df["all_mapped_operators"] = all_mapped
    df["top_mapped_operator"] = df["all_mapped_operators"].apply(get_top_operator)

    print(f"[rank 0] Saving to {rxns_df_output_filepath}")
    df.to_parquet(rxns_df_output_filepath, index=False)

    print("[rank 0] Done.")
