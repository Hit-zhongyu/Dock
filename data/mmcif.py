from pathlib import Path
from typing import Any, Optional, Union
import logging
import os
import numpy as np
from .filter import Filter
from .parser import DistillationMMCIFParser, MMCIFParser
from rdkit import Chem
from .tokenizer import AtomArrayTokenizer
from .ccd import get_component_processing
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from protenix.utils.file_io import load_gzip_pickle

def extract_ccd_from_sample(sample_indices_list):
    ccd_list = set()
    for item in sample_indices_list:
        c1 = item.get('cluster_1_id')
        c2 = item.get('cluster_2_id')
        if c1 and c1 != 'NotInClusterTxt':
            ccd_list.add(c1)
        if c2 and c2 != 'NotInClusterTxt':
            ccd_list.add(c2)
    return list(ccd_list)

def process_single_ccd(ccd):
    try:
        smile, bonds = get_component_processing(ccd, ccd_cif_file='/mnt/rna01/lzy/Dock/components.v20240608.cif')
        return ccd, {"smile": smile, "bonds": bonds}
    except Exception as e:
        return ccd, None

def get_data_from_mmcif(
        mmcif: Union[str, Path],
        pdb_cluster_file: Union[str, Path, None] = None,
        dataset: str = "WeightedPDB",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Get raw data from mmcif with tokenizer and a list of chains and interfaces for sampling.

        Args:
            mmcif (Union[str, Path]): The raw mmcif file.
            pdb_cluster_file (Union[str, Path, None], optional): Cluster info txt file. Defaults to None.
            dataset (str, optional): The dataset type, either "WeightedPDB" or "Distillation". Defaults to "WeightedPDB".

        Returns:
            tuple[list[dict[str, Any]], dict[str, Any]]:
                sample_indices_list (list[dict[str, Any]]): The sample indices list (each one is a chain or an interface).
                bioassembly_dict (dict[str, Any]): The bioassembly dict with sequence, atom_array, and token_array.
        """
        import traceback
        try:
            if dataset == "WeightedPDB":
                parser = MMCIFParser(mmcif_file=mmcif)
                bioassembly_dict = parser.get_bioassembly()
            elif dataset == "Distillation":
                parser = DistillationMMCIFParser(mmcif_file=mmcif)
                bioassembly_dict = parser.get_structure_dict()
            else:
                raise NotImplementedError(
                    'Unsupported "dataset", please input either "WeightedPDB" or "Distillation".'
                )

            sample_indices_list = parser.make_indices(
                bioassembly_dict=bioassembly_dict, pdb_cluster_file=pdb_cluster_file
            )
            if len(sample_indices_list) == 0:
                # empty indices and AtomArray
                return [], bioassembly_dict

            atom_array = bioassembly_dict["atom_array"]
            atom_array.set_annotation(
                "resolution", [parser.resolution] * len(atom_array)
            )
            atom_array = Filter.remove_water(atom_array)
            atom_array = Filter.remove_hydrogens(atom_array)
            atom_array = parser.mse_to_met(atom_array)
            atom_array = Filter.remove_element_X(atom_array)

            if any(["DIFFRACTION" in m for m in parser.methods]):
                atom_array = Filter.remove_crystallization_aids(
                    atom_array, parser.entity_poly_type
                )
            ccd_list = extract_ccd_from_sample(sample_indices_list)
            ligand_dict = {}
            tasks = len(ccd_list)
            with ThreadPoolExecutor(max_workers=tasks) as executor:
                futures = {executor.submit(process_single_ccd, ccd): ccd for ccd in ccd_list}
                # for future in tqdm(as_completed(futures), total=len(futures), smoothing=0):
                for future in as_completed(futures):
                    result = future.result()
                    if result is None:
                        continue
                    ccd, data = result
                    ligand_dict[ccd] = data

            tokenizer = AtomArrayTokenizer(atom_array)
            token_array = tokenizer.get_token_array()
            bioassembly_dict["msa_features"] = None
            bioassembly_dict["template_features"] = None
            bioassembly_dict["ligand_features"] = ligand_dict
            bioassembly_dict["token_array"] = token_array
            return sample_indices_list, bioassembly_dict
            # return bioassembly_dict

        except Exception as e:
            logging.warning("Gen data failed for %s due to %s", mmcif, e)
            traceback.print_exc()
            return [], {}

def get_data_bioassembly(
        bioassembly_dict_fpath: Union[str, Path],
    ) -> dict[str, Any]:
        """
        Get the bioassembly dict.

        Args:
            bioassembly_dict_fpath (Union[str, Path]): The path to the bioassembly dictionary file.

        Returns:
            dict[str, Any]: The bioassembly dict with sequence, atom_array and token_array.

        Raises:
            AssertionError: If the bioassembly dictionary file does not exist.
        """
        assert os.path.exists(
            bioassembly_dict_fpath
        ), f"File not exists {bioassembly_dict_fpath}"
        bioassembly_dict = load_gzip_pickle(bioassembly_dict_fpath)

        return bioassembly_dict


if __name__ == "__main__":
    sample_indices_list, bioassembly_dict = get_data_from_mmcif(
        mmcif="/mnt/rna01/lzy/Dock/data/8d7z.cif"
    )
    print(sample_indices_list)
    print(bioassembly_dict)