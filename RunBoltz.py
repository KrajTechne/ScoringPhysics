import yaml
import tempfile
import shutil
import os
import json
import py2Dmol
import subprocess
import pandas as pd
from StrucTools import *

def create_boltz_yaml(design_name: str, seq_list: list, msa_options: list, template_paths: list, entity_type: list = []):
    """ 
    Create YAML File for running structure prediction with Boltz 2
        Args:
            - design_name (str): Unique ID indicating what protein or set of proteins are being predicted structures
            - seq_list (list): List of sequences whose structures are to be predicted (Preferred: Binder -> Target in that order)
            - msa_options (list): List of MSA options to be used for each sequence
                Specify MSA option for each seq in order of seq_list (empty = single_sequence, '' = generate MSA via mmseqs)
            - template_paths (list): List of template paths for each sequence
            - entity_type (list): List of entity types for each sequence (Protein, DNA, nucleic acid) assumption typically proteins
        Returns:
            - yaml_file (str): Path to the YAML file
    """
    # Setup initial yaml file inputs
    chains = [chr(ord('A') + i) for i in range(len(seq_list))]
    print("Chains: ", chains)

    if entity_type == []:
        entity_type = ['protein'] * len(seq_list)

    # 2. Create a dictionary with the Boltz2 modelling options for each seq. Templates & Constraints can be added as another key-list pair
    yaml_data = {"version" : 1, "sequences" : [], "templates" : []}

    for index in range(len(seq_list)):
        # Convert sequence to associated entity type
        entity_dict = {
            entity_type[index] : {
                "id" : chains[index],
                "sequence" : seq_list[index],
                "msa" : msa_options[index],
            }
        }

        yaml_data["sequences"].append(entity_dict)

        if template_paths[index] != "":
            template_dict = {
                "cif" : template_paths[index],
                "chain_id" : chains[index],
            }
        
            yaml_data["templates"].append(template_dict)
    
    print("Yaml Data: --------------")
    print(yaml_data)
    print("--------------------------")

    # 3. Have to define savepath and create overarching design folder first, prior to saving/creating yaml file
    temp_save_dir = f"/tmp/{design_name}"
    if os.path.exists(temp_save_dir):
        shutil.rmtree(temp_save_dir)
    os.makedirs(temp_save_dir)
    yaml_save_path = f"{temp_save_dir}/{design_name}.yaml"
    with open(yaml_save_path, "w") as file:
        documents = yaml.dump(yaml_data, file)
    return temp_save_dir, yaml_save_path

def run_boltz_prediction(design_name: str, temp_save_dir: str, yaml_save_path: str, num_models = 5,
                         volume_save_path: str = "/Volumes/sandbox/denovotrial/structure_prediction_boltz2"):
    """ 
    Run Boltz2 to generate structures, save them to temporary directory and then move to volume
        Args:
            - temp_save_dir (str): Path to the temporary save directory
            - yaml_save_path (str): Path to the YAML file
        Returns:
            - volume_save_path (str): Path to the volume save directory
    """
    # 1. Run Boltz Structure Prediction
    # Define your command as a list of strings
    command = [
        "boltz", "predict", str(yaml_save_path),
        "--diffusion_samples", str(num_models),
        "--out_dir", str(temp_save_dir),
        "--write_full_pae",
        "--use_msa_server",
        "--use_potentials"
    ]

    # Run the command
    print("Running Boltz prediction...")
    subprocess.run(command, check=True)
    
    # 2. Move YAML file into new folder with predicted structures and metrics
    shutil.move(yaml_save_path, f"{temp_save_dir}/boltz_results_{design_name}")

    # 3. Use shutil.copytree instead of dbutils
    # dirs_exist_ok=True allows it to overwrite/merge if the folder already exists
    shutil.copytree(temp_save_dir, volume_save_path, dirs_exist_ok=True)

def analyze_structure(volume_save_path: str, design_name: str, model_id: int, len_binder: int,  desired_epitope_residues: list = []):
    """
        Analyze the structure of a given design
        Args:
            volume_save_path (str): Path to the volume save directory
            design_name (str): Name of the design
            model_id (int): ID of the model
        Returns:
            metrics: Dictionary of metrics for given design's model_id structure
    """
    metrics = {"design_id" : f"{design_name}_{model_id}", "design_name": design_name, "model_id": model_id}
    
    # 1. Load the structure & path to Boltz2 structure confidence metrics along with pae_matrix path for ipsae calculations
    structure_path = f"{volume_save_path}/boltz_results_{design_name}/predictions/{design_name}/{design_name}_model_{model_id}.cif"
    predictions_path = "/".join(structure_path.split('/')[:-1])
    confidence_path = predictions_path + f"/confidence_{design_name}_model_{model_id}.json"
    pae_path = predictions_path + f"/pae_{design_name}_model_{model_id}.npz"

    # 2. Load the Boltz2 structure confidence metrics
    with open(confidence_path, "r") as f:
        confidence_metrics = json.load(f)
    metrics.update(confidence_metrics)

    # 3 Determine Binding Interface Metrics
    contact_information_a_b = determine_binding_interface(pdb_file_path= structure_path,
                                                      desired_epitope_residues= desired_epitope_residues,
                                                      binder_chain_id= "A", target_chain_id= "B")
    
    metrics.update(contact_information_a_b)

    # 3. Calculate ipsae metrics from PAE matrix
    pae_matrix = np.load(pae_path)['pae']
    
    # 4. Calculate ipsae
    ipsae_min, ipsae_max = calculate_ipsae_complex(pae_matrix=pae_matrix, len_binder=len_binder, pae_cutoff=15)
    ipsae_dict = {"ipsae_min": ipsae_min, "ipsae_max": ipsae_max}
    metrics.update(ipsae_dict)

    # 5. Add paths to structure, predictions, confidence, pae matrices
    metrics.update({"path_structure": structure_path, "path_predictions": predictions_path, "path_confidence": confidence_path, 
                    "path_pae": pae_path})
    
    return metrics

def boltz_predict_analyze(design_name: str, volume_save_path: str, seq_list: list, msa_options: list = [],
                          template_paths: list = [], entity_type: list = [], desired_epitope_residues: list = [], num_models: int = 5):
    """
    Args:
            - design_name (str): Unique ID indicating what protein or set of proteins are being predicted structures
            - seq_list (list): List of sequences whose structures are to be predicted (Preferred: Binder -> Target in that order)
            - msa_options (list): List of MSA options to be used for each sequence
                Specify MSA option for each seq in order of seq_list (empty = single_sequence, '' = generate MSA via mmseqs)
            - template_paths (list): List of template paths for each sequence
            - entity_type (list): List of entity types for each sequence (Protein, DNA, nucleic acid) assumption typically proteins
            - num_models (int): Number of models to predict for each sequence
        Returns:
            - metrics_design (pd.DataFrame): Pandas DataFrame of metrics for all predicted models for given design
    """
    binder_seq = seq_list[0] # Always binder seq first and then target
    len_binder = len(binder_seq)
    temp_save_dir, yaml_save_path = create_boltz_yaml(design_name=design_name, seq_list=seq_list, msa_options=msa_options,
                                                         template_paths=template_paths, entity_type=entity_type)
    run_boltz_prediction(design_name=design_name, yaml_save_path=yaml_save_path, temp_save_dir=temp_save_dir,
                         num_models = num_models, volume_save_path=volume_save_path)
    
    # For each of the "num_models" predicted, analyze the structure
    metrics_design = []
    for model_id in range(num_models):
        metrics = analyze_structure(volume_save_path=volume_save_path, design_name=design_name, model_id=model_id,
                                    desired_epitope_residues=desired_epitope_residues, len_binder=len_binder)
        metrics_design.append(metrics)
    
    # Convert to DataFrame and save as csv
    df_design_metrics = pd.DataFrame(metrics_design)
    df_design_metrics.to_csv(f"{volume_save_path}/boltz_results_{design_name}/predictions/{design_name}/all_models_metrics.csv", index=False)
    
    return df_design_metrics