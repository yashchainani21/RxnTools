from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D, rdDepictor, MolDraw2DSVG
from IPython.display import display, SVG
from typing import Tuple
from copy import deepcopy

def highlight_substructures_in_notebook(substrate_smarts: str,
                                        substructure_smarts: str,
                                        size: Tuple[int, int] = (400, 200),
                                        consider_stereo: bool = False,
                                        allow_multiple_matches: bool = False,
                                        atoms_to_include: any = None) -> SVG:

    """
    Highlights substructures in a jupyter notebook.
    This function was modified from: https://stackoverflow.com/questions/69735586/how-to-highlight-the-substructure-of-a-molecule-with-thick-red-lines-in-rdkit-as

    Parameters
    ----------
    substrate_smarts: str
        Fully atom-mapped SMARTS string of the substrate molecule.
        This SMARTS representation may or may not include stereochemistry.

    substructure_smarts: str
        Fully atom-mapped SMARTS string of the query substructure.
        This query substructure will be highlighted within the substrate if there is a match.
        This SMARTS representation may or may not include stereochemistry.

    size: Tuple[int, int]
        Resolution size of output image of the substrate & any highlighted substructures.

    consider_stereo: bool
        Whether stereochemistry should be considered in finding substructure matches.

    allow_multiple_matches: bool
        Whether all matching substructures should be shown or only the first matching substructure.

    Returns
    -------
    IPython.core.display.SVG
    The SVG image of the substrate with highlighted substructures.

    """

    # convert the input, atom-mapped substrate SMARTS string to its corresponding SMILES string
    substrate_smiles = Chem.MolToSmiles(Chem.MolFromSmarts(substrate_smarts))
    substrate_mol = Chem.MolFromSmiles(substrate_smiles)

    # for generating an RDKit mol object from the query substructure, however, use Chem.MolFromSMarts
    substructure_mol = Chem.MolFromSmarts(substructure_smarts)

    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])

    # we add up substructure matches because later on, highlightAtoms expects only one tuple
    # consequently, the tuples of tuples needs to be merged into a single tuple
    if consider_stereo:
        if allow_multiple_matches:
            matches = sum(substrate_mol.GetSubstructMatches(substructure_mol,
                                                            useChirality = True), ())
        else:
            matches = substrate_mol.GetSubstructMatches(substructure_mol, useChirality = True)[0]
    else:
        if allow_multiple_matches:
            matches = sum(substrate_mol.GetSubstructMatches(substructure_mol,
                                                            useChirality = False), ())
        else:
            matches = substrate_mol.GetSubstructMatches(substructure_mol, useChirality = False)[0]

    drawer.DrawMolecule(substrate_mol, highlightAtoms = matches)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')

    return SVG(svg)