import yaml
import tempfile
import os
import shutil
import numpy as np
import pandas as pd
import torch
from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    LigandInput,
    Modification,
    ProteinInput,
    DNAInput,
    RNAInput,
    StructurePredictionInput,
)
from esm.utils.msa import MSA
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from .StrucTools import calculate_ipSAE, determine_binding_interface, extract_atom_array
from .mmseqs2 import generate_msa

_model_cache: dict = {}


def load_model(model_name: str = "ESMFold2"):
    """Load the ESMFold2 model variants from HuggingFace, caching after first load."""
    global _model_cache
    if model_name not in ['ESMFold2', 'ESMFold2-Fast']:
        raise ValueError(f"Model name {model_name} not available on HuggingFace.")
    if model_name not in _model_cache:
        _model_cache[model_name] = (ESMFold2Model.from_pretrained(f"biohub/{model_name}").cuda().eval())
    return _model_cache[model_name]

def run_esmfold2_prediction(model_name: str, seq_list: list, msa_options: list, path_msa_folder: str, entity_type: list = [], ligand: list = [], 
                            num_diffusion_samples: int = 5, num_loops: int = 10, num_sampling_steps: int = 150, seed = 0):
    """ 
    Predicts structure for provided seqs and/or ligand using ESMFold2 model.

    Args:
        design_name (str): Name of the design.
        model_name (str): Name of the ESMFold2 model. (Either ESMFold2 or ESMFold2-Fast)
        seq_list (list): List of sequences to predict. (Single = Apo, Multiple = Holo)
        msa_options (list): List ('empty' = single sequence, '' = generate MSA via mmseqs)
        path_msa_folder (str): Str mapping to path within volume_save_path to where the unpaired MSAs should be saved
        entity_type (list): list of entity types for each sequence (Protein, DNA, RNA). Default assume protein.
        ligand (str): Ligand SMILES string. Default is empty.
        num_diffusion_samples (int): Number of diffusion samples to generate. Default is 5.
        num_loops (int): Number of loops to run. (Loops is somewhat analogous to recycles in AlphaFold2). Default is 10.
        num_sampling_steps (int): Number of sampling steps to run. Default is 150.
        seed (int): Seed for random number generator. Default is 0.

    Returns:
        pred_structs: ESMFold2 object containing each structure and its associated metrics

    """
    # Setup intial inputs for ESMFold2
    # 1. Define chain IDs
    chains = [chr(ord('A') + i) for i in range(len(seq_list))]
    print("Chains: ", chains)

    # 2. Define entity_types:
    if len(entity_type) == 0:
        entity_type = ['protein'] * len(seq_list)
    else:
        if len(entity_type) != len(seq_list):
            raise ValueError("Length of entity_type must match length of seq_list.")
        for i in range(len(entity_type)):
            entity_lower_case = entity_type[i].lower()
            if entity_lower_case not in ['protein', 'dna', 'rna']:
                raise ValueError("Entity type must be one of ['protein', 'dna', 'rna'].")
            entity_type[i] = entity_lower_case
    print("Entity Types: ", entity_type)

    # 2.5: Define MSA Options:
    if len(msa_options) == 0:
        msa_options = ['empty'] * len(seq_list)

    # 3. Validate MSAs are only used if model is ESMFold2 and not ESMFold2-Fast
    if model_name == 'ESMFold2-Fast' and ('empty' not in msa_options):
        raise ValueError("MSA generation is not supported for ESMFold2-Fast. Switch model_type to ESMFold2")

    # 3 Define list of input sequences
    esm_seqs = []
    # 3.5 Initialize yaml to record inputs to ESMFold2 Model:
    yaml_inputs = {"sequences" : []}
    for index in range(len(seq_list)):
        # Check whether user specified MSA generation ---------------------------------------
        if msa_options[index] == '':
            # Check whether user specified RNA or Protein as those only allow for MSA input
            if entity_type[index] not in ['protein', 'rna']:
                raise ValueError(f"MSA generation is not supported for entity type: {entity_type[index]}.")
            # Generate MSA
            chain_id = chains[index]
            msa_path = generate_msa(chain_id = chain_id, sequence = seq_list[index], msa_dir = path_msa_folder)
        elif msa_options[index] != 'empty':
            # Assumed that the user provided a custom msa path for the sequence with msa path being an a3m file
            msa_path = msa_options[index]
        else:
            # No MSA generation
            msa_path = None
        msa = MSA.from_a3m(path = msa_path, remove_insertions = True, max_sequences = 1000) if msa_path else None
        # Create correct input for ESMFold2: --------------------------------------------------
        if entity_type[index] == 'protein':
            input = ProteinInput(id = chains[index], sequence = seq_list[index], msa = msa)
        elif entity_type[index] == 'rna':
            input = RNAInput(id = chains[index], sequence = seq_list[index], msa = msa)
        elif entity_type[index] == 'dna':
            input = DNAInput(id = chains[index], sequence = seq_list[index])
        esm_seqs.append(input)
        # Create entity dictionary for saving in yaml_inputs
        entity_dict = {
            entity_type[index] : {
                "id" : chains[index],
                "sequence" : seq_list[index],
                "msa" : msa_path
            }
        }
        # Add entity dictionary to yaml_inputs
        yaml_inputs["sequences"].append(entity_dict)

    print("Non-Ligand Sequences: ", esm_seqs)

    # 4. Define ligands
    if len(ligand) != 0:
        for index, lig in enumerate(ligand):
            chain_id = chr(ord('A') + len(seq_list) + index)
            ligand_esm2 = LigandInput(id = chain_id, smiles = lig)
            entity_dict = {"ligand" : {"id" : chain_id, "smiles" : lig}}
            esm_seqs.append(ligand_esm2)
            yaml_inputs["sequences"].append(entity_dict)
    print("Ligand: ", ligand)
    print("Final ESMFold2 Inputs: ", esm_seqs)

    # 5. Define StructurePredictionInput
    spi = StructurePredictionInput(sequences = esm_seqs)

    # 6. Generate Apo or Holo Structres
    model_esmfold2 = load_model(model_name)
    pred_strucs = ESMFold2InputBuilder().fold(model_esmfold2, spi, num_loops = num_loops,
                                               num_sampling_steps = num_sampling_steps,
                                               num_diffusion_samples = num_diffusion_samples,
                                               seed = seed)
    return pred_strucs, yaml_inputs

def analyze_structure(pred_struc, volume_save_path: str, design_name: str, model_id: int, desired_epitope_residues: list, num_targets: int = 1):
    """
    Analyzes the predicted structure and saves the results to a dictionary of metrics associated with the design_name and model_id 
    Args:
        pred_struc (StructurePrediction): Predicted structure from ESMFold2
        volume_save_path (str): Path to folder in volume to save structure
        design_name (str): Name of the design
        model_id (int): ID of the model
        desired_epitope_residues (list): List of desired epitope residues
        num_targets (int): Number of target chains in complex to analyze
    Returns:
        metrics: Dictionary of metrics associated with the design_name and model_id

    """
    metrics = {"design_id" : f"{design_name}_{model_id}", "design_name" : design_name, "model_id" : model_id}

    # Save the predicted structure to a CIF file in the volume_save_path
    # Volume_save_path points to overarching folder, each design_name gets subfolder, and each model_id is saved within respective design_name folder
    # 1. Save predicted structure & associated paths
    predicted_structure = pred_struc[model_id]
    path_predicted_structure = os.path.join(volume_save_path, design_name, f"{design_name}_model_{model_id}.cif")
    path_predictions = os.path.dirname(path_predicted_structure)
    with open(path_predicted_structure, "w") as f:
        f.write(predicted_structure.complex.to_mmcif())
    
    # 1.5 Save predicted structure's pae matrix: Workaround required to address writing to volume isssue
    path_predicted_structure_pae = os.path.join(volume_save_path, design_name, f"{design_name}_model_{model_id}_pae.npz")
    pae_matrix = predicted_structure.pae
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_npz_stem = os.path.join(tmpdir, 'pae')
        np.savez(tmp_npz_stem, pae=pae_matrix)  # writes pae.npz to local disk, with key: pae to match Boltz model parsing done in ipsae.py script
        shutil.copy2(tmp_npz_stem + '.npz', path_predicted_structure_pae)
    
    # 2. Save confidence metrics
    if num_targets == 0:
        # If apo:
        metrics.update({"ptm" : predicted_structure.ptm, "plddt" : predicted_structure.plddt.mean().item()})
    else:
        # If holo:
        # 1. Update complex-specific metrics
        metrics.update({"iptm" : predicted_structure.iptm, "complex_plddt" : predicted_structure.plddt.mean().item(), "ptm" : predicted_structure.ptm})
        
        # 2. Conduct contact check
        target_chain_id = ",".join(chr(ord("B") + i) for i in range(num_targets))
        contact_information = determine_binding_interface(pdb_file_path = path_predicted_structure,
                                                          desired_epitope_residues = desired_epitope_residues,
                                                          binder_chain_id = "A",
                                                          target_chain_id = target_chain_id)
        metrics.update(contact_information)
        # 3. Calculate ipSAE
        ipsae_dict = calculate_ipSAE(pae_file = path_predicted_structure_pae,
                                     binder_chain = "A",
                                     target_chains = target_chain_id,
                                     path_input_structure = path_predicted_structure)
        ipsae_values = [value for key, value in ipsae_dict.items() if key.startswith("ipSAE_")]
        if ipsae_values:
            ipsae_dict["ipsae_min"] = min(ipsae_values)
            ipsae_dict["ipsae_max"] = max(ipsae_values)
        metrics.update(ipsae_dict)
    
    # 3. Add paths to structure, predictions, and pae path
    metrics.update({"path_structure" : path_predicted_structure, "path_predictions" : path_predictions, "path_pae" : path_predicted_structure_pae})
    return metrics

def esmfold2_predict_analyze(design_name: str, model_name: str, volume_save_path: str, seq_list: list, msa_options: list = [],
                             entity_type: list = [], desired_epitope_residues: list = [], num_models: int = 5, ligand: list = [],
                             num_loops: int = 10, num_sampling_steps: int = 150, seed: int = 0):
    """
    Function to predict apo or holo structures using ESMFold2, save predicted structures and pae matrics, analyze predicted structures, and save metrics to a pandas dataframe
    
    Args:
        - design_name (str): Name of the design
        - model_name (str): Name of the model to use
        - volume_save_path (str): Path to overarching folder where subfolders denoted by design_name will contain predicted structures and pae matrices
        - seq_list (list): List of sequences to predict
        - msa_options (list): List of MSA options to use
        - entity_type (list): List of entity types to use
        - desired_epitope_residues (list): List of desired epitope residues to use
        - num_models (int): Number of models to predict for specific design
        - ligand (list): List of ligands to use
        - num_loops (int): Number of loops to use (somewhat analogous to number of recycles used in AlphaFold)
        - num_sampling_steps (int): Number of sampling steps to use
        - seed (int): Seed to use
    
    Returns:
        - metrics (dict): Dictionary of metrics for each of the predicted design's number of models
    
    """
    num_targets = len(seq_list) - 1
    # 0. Create MSA Directory if not present
    path_msa_folder = os.path.join(volume_save_path, "msa")
    if not os.path.exists(path_msa_folder):
      os.makedirs(path_msa_folder)
      print(f"Created MSA Subfolder as not present in volume_save_path: {volume_save_path}")
    # 1. Predict structure via ESMFold2 and generate yaml documenting inputs
    pred_struc, input_yaml = run_esmfold2_prediction(model_name = model_name, seq_list = seq_list, msa_options = msa_options, path_msa_folder= path_msa_folder, 
                                                     entity_type = entity_type, ligand = ligand, 
                                                     num_diffusion_samples = num_models, num_loops = num_loops, num_sampling_steps = num_sampling_steps, seed = seed)
    
    # 2. Create subdirectory under design_name to save predicted structures and pae matrices
    path_design_structures_folder = os.path.join(volume_save_path, design_name)
    if not os.path.exists(path_design_structures_folder):
        os.makedirs(os.path.join(volume_save_path, design_name))

    # 2.5 Save the input yaml file to path_design_structure_folder
    path_yaml = os.path.join(path_design_structures_folder, "input.yaml")
    with open(path_yaml, "w") as file:
        yaml.dump(input_yaml, file)
    
    # 3. Analyze each diffusion sample structure
    metrics_list = []
    for index in range(len(pred_struc)):
        metrics = analyze_structure(pred_struc = pred_struc, volume_save_path = volume_save_path, design_name = design_name, model_id = index,
                                    desired_epitope_residues = desired_epitope_residues, num_targets = num_targets)
        metrics_list.append(metrics)
    
    # 4. Create pandas dataframe of metrics
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(os.path.join(path_design_structures_folder, "all_models_metrics.csv"), index = False)

    return df_metrics
