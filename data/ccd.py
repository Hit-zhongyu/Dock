import argparse
import functools
import logging
import multiprocessing
import pickle
import subprocess as sp
from pathlib import Path
from typing import Optional, Union

import gemmi
import numpy as np
import rdkit
import tqdm
from biotite.structure.io import pdbx
from pdbeccdutils.core import ccd_reader

bond_order_map = {
    "SING": 1,
    "DOUB": 2,
    "TRIP": 3
}
aromatic_map = {
    'N': 0,
    'Y': 1
}

@functools.lru_cache
def gemmi_load_ccd_cif(ccd_cif: Union[Path, str]) -> gemmi.cif.Document:
    """
    Load CCD components file by gemmi

    ccd_cif (Union[Path, str]): The path to the CCD CIF file.

    Returns:
        Document: gemmi ccd components file
    """
    return gemmi.cif.read(str(ccd_cif))


def get_smile_from_ccd_block(ccd_block):
    loop = ccd_block.find(
        "_pdbx_chem_comp_descriptor.", ["comp_id", "type", "program", "program_version", "descriptor"]
    )

    smile = None
    for row in loop:
        if row[1] == 'SMILES_CANONICAL' and row[2] == 'CACTVS':
            smile = row[4]
        
    if smile is None:
        return None
    else:
        return smile.strip('"')


def get_bonds_from_ccd_block(ccd_block):
    atom_loop = ccd_block.find(
        "_chem_comp_atom.", ["atom_id", "pdbx_ordinal"])
    if atom_loop is None:
        return []
    id_to_index = {}
    for row in atom_loop:
        atom = row[0]
        idx = int(row[1])
        id_to_index[atom] = idx-1

    bond_loop = ccd_block.find(
        "_chem_comp_bond.", ["comp_id", "atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag",
                             "pdbx_stereo_config", "pdbx_ordinal"])
    if bond_loop is None:
        return []
    bonds = []
    for row in bond_loop:
        atom1 = row[1]
        atom2 = row[2]
        bond_type = row[3]
        aromatic = row[4]
        if atom1.upper().startswith('H') or atom2.upper().startswith('H'):
            continue
        # bonds.append((atom1, atom2, bond_type, aromatic))
        bonds.append((id_to_index[atom1], id_to_index[atom2], bond_order_map[bond_type],
                      aromatic_map[aromatic]))
    return bonds


def get_component_processing(
    ccd_code: str,
    ccd_cif_file: tuple[str, Path]
) -> Optional[rdkit.Chem.Mol]:
    """
    Get rdkit mol by PDBeCCDUtils
    https://github.com/PDBeurope/ccdutils

    Args:
        ccd_code (str): ccd code
        ccd_cif_file (Path): The path to the CCD CIF file.

    Returns
        rdkit.Chem.Mol: rdkit mol with ref coord
    """
    # ccd_code, ccd_cif_file = ccd_code_and_cif_file
    ccd_cif = gemmi_load_ccd_cif(ccd_cif_file)
    try:
        # ccd_block = ccd_cif[ccd_code]
        ccd_block = ccd_cif.find_block(ccd_code)
    except KeyError:
        return None
    
    smile = get_smile_from_ccd_block(ccd_block)
    bonds = get_bonds_from_ccd_block(ccd_block)
    
    
    return smile, bonds


if __name__ == "__main__":
    get_component_processing(ccd_code='QFC', ccd_cif_file='/mnt/rna01/lzy/Dock/components.v20240608.cif')


