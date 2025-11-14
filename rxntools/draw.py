from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import display, SVG
from typing import Tuple, Optional, Union

def draw_molecule(
    mol_string: str,
    outfile: str | None = None,
    width: int = 800,
    height: int = 600,
    line_width: float = 1.5,
    kekulize: bool = True,
    show_in_notebook: bool = True,
):
    """
    Draws a molecule from a SMILES or SMARTS string (no stereochemistry labels).

    Args:
        mol_string (str): SMILES or SMARTS representation of the molecule.
        outfile (str, optional): Path to save the SVG file (if desired).
        width (int): Width of the rendered SVG.
        height (int): Height of the rendered SVG.
        line_width (float): Line thickness for bonds.
        kekulize (bool): Whether to draw in Kekulé form.
        show_in_notebook (bool): Whether to display automatically if running in a notebook.

    Returns:
        str: SVG string representation of the molecule.
    """
    # --- Parse molecule as SMILES or SMARTS ---
    mol = Chem.MolFromSmiles(mol_string)
    if mol is None:
        mol = Chem.MolFromSmarts(mol_string)
        if mol is None:
            raise ValueError("❌ Invalid molecule string: not valid SMILES or SMARTS.")
        mol.UpdatePropertyCache()
        mol.GetRingInfo()

    # --- Set up drawing ---
    drawer = Draw.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.bondLineWidth = line_width
    opts.dotsPerAngstrom = 200
    opts.addAtomIndices = False
    opts.addStereoAnnotation = False   # 🚫 Disable E/Z and R/S labels

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, kekulize=kekulize)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # --- Save or show ---
    if outfile:
        with open(outfile, "w") as f:
            f.write(svg)
        print(f"✅ Saved molecule drawing to: {outfile}")
    elif show_in_notebook:
        display(SVG(svg))

    return svg

def highlight_substructures(
    substrate_smarts: str,
    substructure_smarts: str,
    size: Tuple[int, int] = (400, 300),
    consider_stereo: bool = False,
    allow_multiple_matches: bool = False,
    highlight_color: Tuple[float, float, float] = (1.0, 0.3, 0.3),
    line_width: float = 1.5,
    outfile: Optional[str] = None,
) -> Union[SVG, str]:
    """
    Unified function to highlight substructures either in a Jupyter notebook
    or save the highlighted molecule as an SVG.

    Parameters
    ----------
    substrate_smarts : str
        SMARTS or SMILES for the full substrate molecule.

    substructure_smarts : str
        SMARTS for the query substructure to highlight.

    size : Tuple[int, int]
        Output image size in pixels.

    consider_stereo : bool
        Whether stereochemistry should be considered in substructure matching.

    allow_multiple_matches : bool
        Whether to highlight all matches or only the first one.

    highlight_color : Tuple[float, float, float]
        RGB triple for highlight color (0–1).

    line_width : float
        Bond thickness.

    outfile : str or None
        If provided, the highlighted molecule will be saved to this SVG path.
        Otherwise, the SVG is displayed in the notebook.

    Returns
    -------
    SVG or str
        SVG object for notebook display OR filepath to saved SVG file.
    """

    # Convert substrate SMARTS → SMILES → Mol (avoids atom mapping issues)
    substrate_smiles = Chem.MolToSmiles(Chem.MolFromSmarts(substrate_smarts))
    mol = Chem.MolFromSmiles(substrate_smiles)
    submol = Chem.MolFromSmarts(substructure_smarts)

    if mol is None or submol is None:
        raise ValueError("Invalid SMARTS/SMILES. Could not parse molecule(s).")

    # Get matches
    matches = mol.GetSubstructMatches(submol, useChirality=consider_stereo)

    if not matches:
        print("⚠️ No substructure match found.")
        return None

    # Merge multiple matches if requested
    if allow_multiple_matches:
        highlight_atoms = [a for match in matches for a in match]
    else:
        highlight_atoms = list(matches[0])

    # Highlight bonds associated with atoms
    highlight_bonds = [
        b.GetIdx()
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in highlight_atoms and b.GetEndAtomIdx() in highlight_atoms
    ]

    # Prepare drawer
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.bondLineWidth = line_width
    opts.dotsPerAngstrom = 200  # high resolution

    # Color dictionaries
    atom_colors = {a: highlight_color for a in highlight_atoms}
    bond_colors = {b: highlight_color for b in highlight_bonds}

    # Draw
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')

    # Save OR return SVG for notebook display
    if outfile:
        with open(outfile, "w") as f:
            f.write(svg)
        print(f"✅ Saved highlighted SVG to: {outfile}")
        return outfile
    else:
        return SVG(svg)
