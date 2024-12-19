from rdkit import Chem

class unmapped_reaction:
    """
    Parameters
    ----------
    rxn_str : str
        Unmapped reaction string of the form "A + B = C + D" or "A.B >> C.D"
    """
    def __init__(self, rxn_str: str):
        self.rxn_str = rxn_str

class mapped_reaction:
    """
    Parameters
    ----------
    rxn_smarts : str
        Mapped reaction smarts with complete atom mapping of all species in reaction
    """
    def __init__(self, rxn_smarts: str):
        self.rxn_smarts = rxn_smarts

    def get_mapped_bonds(self, mol):
        pass