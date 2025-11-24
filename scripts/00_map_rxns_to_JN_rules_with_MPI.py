from mpi4py import MPI
from tqdm import tqdm
import pandas as pd
from ergochemics import mapping
from typing import List, Tuple, Any
import numpy as np
import os

# ------------------ Helper functions ------------------ #

def make_rule_id(n: int, prefix: str = "rule", width: int = 4) -> str:
    """
    Convert an integer into a zero-padded rule ID of the form 'rule0001'.
    """
    if n < 1:
        raise ValueError("Input must be >= 1.")
    return f"{prefix}{n:0{width}d}"


def map_single_reaction(idx: int, rxn: str, gen_rxn_operators_list: List[str]) -> Tuple[int, Any]:
    """
    Map a single reaction against all operators.
    Returns (index, list_of_rule_ids) or (index, '__WORKER_FAILED__...').
    """
    mapped_ops = []
    try:
        for i, operator in enumerate(gen_rxn_operators_list):
            try:
                mapped_rxn = mapping.operator_map_reaction(rxn=rxn, operator=operator)
                if mapped_rxn.did_map:
                    mapped_ops.append(make_rule_id(i + 1))
            except Exception:
                # ignore per-operator failures
                pass
    except Exception as e:
        return idx, f"__WORKER_FAILED__: {repr(e)}"

    return idx, mapped_ops


def get_top_operator(op_list):
    """
    Given a list like ['rule0002', 'rule0754'], return the one with
    the smallest integer value (e.g. 'rule0002').

    Handles None, empty lists, or failure strings by returning None.
    """
    if not isinstance(op_list, list) or not op_list:
        return None

    nums = [int(op.replace("rule", "")) for op in op_list]
    min_num = min(nums)
    return f"rule{min_num:04d}"


def chunk_list(lst, n):
    """Split a list into n (almost) equal-sized chunks."""
    k, m = divmod(len(lst), n)
    chunks = []
    start = 0
    for i in range(n):
        end = start + k + (1 if i < m else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks


# ------------------ MPI setup ------------------ #

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ------------------ File paths ------------------ #

rxns_df_input_filepath = "../data/raw/enzymemap_v2_brenda2023.csv"
rxns_df_output_filepath = "../data/interim/enzymemap_v2_brenda2023_JN_mapped.parquet"
rules_filepath = "../data/raw/JN1224MIN_rules.tsv"

# ------------------ Rank 0: load data & prepare tasks ------------------ #

if rank == 0:
    print(f"[rank 0] Reading operators from: {rules_filepath}")
    gen_rxn_operators_df = pd.read_csv(rules_filepath, delimiter="\t")
    gen_rxn_operators_list: List[str] = gen_rxn_operators_df["SMARTS"].to_list()

    print(f"[rank 0] Reading reactions from: {rxns_df_input_filepath}")
    enzymatic_rxns_df = pd.read_csv(rxns_df_input_filepath)

    unmapped_rxns_list: List[str] = enzymatic_rxns_df["unmapped"].astype(str).to_list()

    # Clean hydrogens
    cleaned_rxns_list: List[str] = []
    for rxn in unmapped_rxns_list:
        rxn = rxn.replace(".[H+]", "").replace("[H+].", "")
        cleaned_rxns_list.append(rxn)

    # Build tasks: (index, rxn)
    tasks = [(i, rxn) for i, rxn in enumerate(cleaned_rxns_list)]
    n_tasks = len(tasks)
    print(f"[rank 0] Total reactions to map: {n_tasks}")

    # Split tasks into chunks for each rank
    task_chunks = chunk_list(tasks, size)

else:
    gen_rxn_operators_list = None
    enzymatic_rxns_df = None
    tasks = None
    n_tasks = None
    task_chunks = None

# ------------------ Broadcast operators and total task count ------------------ #

# Broadcast the operator list and the total number of tasks to all ranks
gen_rxn_operators_list = comm.bcast(gen_rxn_operators_list if rank == 0 else None, root=0)
n_tasks = comm.bcast(n_tasks if rank == 0 else None, root=0)

# Scatter the chunks of tasks
local_tasks = comm.scatter(task_chunks if rank == 0 else None, root=0)

# ------------------ Each rank: process its local tasks ------------------ #

local_results: List[Tuple[int, Any]] = []

# tqdm per rank; you could disable for rank>0 if you want less noise
for idx, rxn in tqdm(local_tasks, desc=f"Rank {rank} mapping", disable=False):
    local_results.append(map_single_reaction(idx, rxn, gen_rxn_operators_list))

# ------------------ Gather all results on rank 0 ------------------ #

all_results = comm.gather(local_results, root=0)

# ------------------ Rank 0: assemble results, compute top operator, save ------------------ #

if rank == 0:
    # Flatten and put back in original order
    results_flat: List[Any] = [None] * n_tasks
    for rank_results in all_results:
        for idx, mapped_ops in rank_results:
            results_flat[idx] = mapped_ops

    # Attach column with full list of mapped operators
    enzymatic_rxns_df["all_mapped_operators"] = results_flat

    # Compute top operator column
    enzymatic_rxns_df["top_mapped_operator"] = enzymatic_rxns_df["all_mapped_operators"].apply(
        get_top_operator
    )

    # Save to parquet
    print(f"[rank 0] Writing results to: {rxns_df_output_filepath}")
    enzymatic_rxns_df.to_parquet(rxns_df_output_filepath, index=False)
    print("[rank 0] Done.")
