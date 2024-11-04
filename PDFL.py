#Summation of Statistics across all filtration values for Betti-0 and Betti-1

import math
import numpy as np
import os
import csv
from scipy.spatial.distance import cdist
import networkx as nx
import argparse
from Extraction import Extraction
import sys
from PersistentLaplacians import PersistentDirectedFlagLaplacian
import logging
from bond_extraction import Atom, get_bonded_atoms, en_vals 


# Constants and parameters as previously defined
protein_atoms = ['C', 'N', 'O', 'S']
ligand_atoms = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
radii = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98}
en_vals = {'C': 2.55, 'N': 3.04, 'O': 3.5, 'S': 2.58, 'P': 2.19, 'F': 3.98, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66}
param_combinations = [['lor', 2, 3, 12]]
cutoff_distance = 12
bond_limits = {
    ("C", "C"): 1.64, ("C", "S"): 1.92, ("C", "N"): 1.57,
    ("C", "O"): 1.57, ("N", "O"): 1.5, ("N", "N"): 1.55,
    ("N", "S"): 2.06, ("O", "S"): 1.52, ("O", "O"): 1.58,
    ("S", "S"): 2.17}
betti_numbers = ['0', '1']
intervals = ['(0, 0.8)', '(0.8, 0.85)', '(0.85, 0.9)', '(0.9, 0.95)', '(0.95, 1)']
stats = ['sum', 'min', 'max', 'mean', 'median', 'std_dev', 'variance', 'count', 'sum_squares']

class DigraphGenerator:
    def __init__(self, param_combinations, cutoff_distance=12):
        self.cutoff_distance = cutoff_distance
        self.param_combinations = param_combinations
        self.G = None

    def create_digraph(self, extraction_instance, pro_key, lig_key, param):
        if len(param) != 4:
            logger.error(f"Invalid param: {param}")
            return None
        kernel, tau, power, cutoff_distance = param

        # Get coordinates for the current protein and ligand atom keys
        protein_coords = extraction_instance.protein_atoms_dict[pro_key]
        ligand_coords = extraction_instance.ligand_atoms_dict[lig_key]
        
        # Exit the function if no coordinates are present
        if protein_coords.size == 0 or ligand_coords.size == 0:
            return None

        # Convert coordinates to Atom objects and bond them
        pro_atoms = [Atom(coord, pro_key) for coord in protein_coords]
        lig_atoms = [Atom(coord, lig_key) for coord in ligand_coords]

        pro_atoms = get_bonded_atoms(pro_atoms, bond_limits)
        lig_atoms = get_bonded_atoms(lig_atoms, bond_limits)

        # Calculate distances and apply cutoff
        dist = cdist(protein_coords, ligand_coords)
        within_cutoff_pro = np.any(dist <= cutoff_distance, axis=1)
        pro_atoms = [pro_atoms[i] for i in range(len(pro_atoms)) if within_cutoff_pro[i]]

        dist = cdist(protein_coords[within_cutoff_pro], ligand_coords)
        eta_scale = tau * (radii[pro_key] + radii[lig_key])
        dist_matrix_values = np.zeros_like(dist)

        if kernel == 'exp':
            dist_matrix_values = 1 - np.exp(-(dist / eta_scale) ** power)
        elif kernel == 'lor':
            dist_matrix_values = 1 - (1 / (1 + (dist / eta_scale) ** power))
        
        
        self.G = nx.DiGraph()
        num_pro_atoms = len(pro_atoms)
        num_lig_atoms = len(lig_atoms)
        self.G.add_nodes_from(range(num_pro_atoms + num_lig_atoms), weight=0)

        # Connect nodes based on their electronegativity and distances
        for i, pro_atom in enumerate(pro_atoms):
            for j, lig_atom in enumerate(lig_atoms):
                weight = dist_matrix_values[i, j]  # Assuming dist indices align with pro_atoms and lig_atoms indices
                if pro_key == lig_key:
                    self.handle_identical_keys(i, num_pro_atoms + j, weight, pro_atom, lig_atom)
                else:
                    pro_en = en_vals.get(pro_key, 0)
                    lig_en = en_vals.get(lig_key, 0)
                    if pro_en < lig_en:
                        self.G.add_edge(i, num_pro_atoms + j, weight=weight)                       
                    elif pro_en > lig_en:
                        self.G.add_edge(num_pro_atoms + j, i, weight=weight)                       
       
        return self.G

    def handle_identical_keys(self, i, j, weight, pro_atom, lig_atom):
        pro_sum_en = sum(en_vals[atom.etype] for atom in pro_atom.bonds)
        lig_sum_en = sum(en_vals[atom.etype] for atom in lig_atom.bonds)
        if pro_sum_en < lig_sum_en:
            self.G.add_edge(i, j, weight=weight)           
        elif pro_sum_en > lig_sum_en:
            self.G.add_edge(j, i, weight=weight)                 
        #else:
            #print(f"No edge added for identical keys due to equal or zero electronegativity sums: protein {i}, ligand {j}")


def format_number(num, tolerance):
    if abs(num) < tolerance:
        return "0"
    elif num < 0:
        return f"{num:.7e}"
    else:
        formatted = f"{num:.7f}"
        if '.' in formatted:
            return formatted.rstrip('0').rstrip('.')
        return formatted

def parse_line(line, tolerance=0.001):
    return ' '.join([format_number(float(v), tolerance) for v in line.strip().split()])


def run_flagser_with_temp_file(digraph, subfolder, pro_key, lig_key, error_log_path, betti_numbers, intervals, stats):
    flag_file_name = f"{subfolder}_{pro_key}_{lig_key}"
    flag_file_path = f"/mnt/home/ziamusha/Documents/Flagser-Laplacian/v2007/L_t2_v3/{flag_file_name}.flag"
    empty_data = ([0] * len(stats) * len(intervals), [0] * len(intervals), [], False)

    if digraph.number_of_nodes() == 0 or digraph.number_of_edges() == 0:
        return empty_data 


    # Construct the flag file content
    node_data = " ".join(str(data.get('filtration', 0)) for _, data in digraph.nodes(data=True))
    edge_data = "\n".join(f"{u} {v} {data.get('weight', np.inf):.3f}" 
                          for u, v, data in digraph.edges(data=True) if data.get('weight', np.inf) != np.inf)
    input_str = f"dim 0:\n{node_data}\n"
    input_str += f"dim 1:\n{edge_data}" if edge_data else "dim 1:\n"

    # Attempt to write the flag file with permissions set at creation
    try:
        fd = os.open(flag_file_path, os.O_WRONLY | os.O_CREAT, 0o644)
        with os.fdopen(fd, 'w') as flag_file:
            flag_file.write(input_str)
    except Exception as e:
        #print(f"Failed to write flag file: {e}")
        return empty_data
    
    flag_file_created = os.path.exists(flag_file_path)
    
    try:
        pdfl = PersistentDirectedFlagLaplacian(flag_file_path, 2)
        has_dim1 = "dim 1:\n" in input_str and len(input_str.split("dim 1:\n")[1].strip()) > 0
        step_size = 0.01
        max_filtration_value = 1.0

        if has_dim1:
            spectra_files = []
            summary_content = ["a\tb\tbetti_0\tbetti_1\tlambda_0\tlambda_1"]
            for dim in range(2):
                spectra_list = []
                a = 0
                while a <= max_filtration_value:
                    b = min(a + step_size, max_filtration_value)
                    spectra = pdfl.spectra(dim, a, b)
                    formatted_spectra = parse_line(" ".join(map(str, spectra)), tolerance)
                    spectra_list.append(formatted_spectra)
                    if a == max_filtration_value:
                        break
                    a = b
                spectra_filename = f"{flag_file_name}_spectra_{dim}.txt"
                with open(spectra_filename, 'w') as file:
                    file.write("\n".join(spectra_list))
                spectra_files.append(spectra_filename)
                #print(f"Spectra for dimension {dim} stored successfully at {spectra_filename}")

            # Construct and write summary data 
            for i, lines in enumerate(zip(*[open(f).readlines() for f in spectra_files])):
                parts = [line.strip().split() for line in lines]
                a = round(i * step_size, 3)
                b = round((i + 1) * step_size, 3) if (i + 1) * step_size <= max_filtration_value else max_filtration_value
                betti_0 = parts[0].count('0')
                betti_1 = parts[1].count('0') if len(parts) > 1 else 0
                lambda_0 = min([float(v) for v in parts[0] if float(v) > 0], default=0)
                lambda_1 = min([float(v) for v in parts[1] if float(v) > 0], default=0) if len(parts) > 1 else 0
                summary_content.append(f"{a}\t{b}\t{betti_0}\t{betti_1}\t{format_number(lambda_0, tolerance)}\t{format_number(lambda_1, tolerance)}")


            summary_filename = flag_file_path.replace('.flag', '_spectra_summary.txt')
            with open(summary_filename, 'w') as file:
                file.write("\n".join(summary_content))
        else:
            spectra_filename = f"{flag_file_name}_spectra_0.txt"
            node_count_line = " ".join('0' for _ in digraph.nodes())
            with open(spectra_filename, 'w') as file:
                file.write(node_count_line)
            summary_content = ["a\tb\tbetti_0\tlambda_0"]
            summary_content.append(f"0\t0\t{len(digraph.nodes())}\t0")
            summary_filename = flag_file_path.replace('.flag', '_spectra_summary.txt')
            with open(summary_filename, 'w') as file:
                file.write("\n".join(summary_content))
    except Exception as e:
        return empty_data

    # Collect stats for both dimensions
    default_stats = [0] * 9
    default_zero_count = 0
    intervals = [(0, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1)]
    all_stats = []
    all_zero_counts = []
    spectra_file_paths = []

    for dim in range(2):
        spectra_file_path = f"/mnt/home/ziamusha/Documents/Flagser-Laplacian/v2007/L_t2_v3/{flag_file_name}_spectra_{dim}.txt"
        if os.path.exists(spectra_file_path):
            filtration_values = get_filtration_values(summary_filename)
            spectra_lines = [line.strip() if line.strip() else '0' for line in open(spectra_file_path)]
            if len(filtration_values) != len(spectra_lines):
                return [], [], [], False
            zero_counts = get_zero_counts(summary_filename)
            aggregated_stats, interval_zero_counts = process_spectra_file(spectra_file_path, filtration_values, zero_counts, dim)
            for interval in intervals:
                for stat in aggregated_stats.get(interval, default_stats):
                    all_stats.append(stat)
                all_zero_counts.append(interval_zero_counts.get(interval, default_zero_count))
            spectra_file_paths.append(spectra_file_path)

    spectra_file_paths.append(summary_filename)
    
    if os.path.exists(flag_file_path):
        os.remove(flag_file_path)

    return all_stats, all_zero_counts, spectra_file_paths, flag_file_created

def get_filtration_values(summary_file_path):
    try:
        filtration_values = []
        with open(summary_file_path, 'r') as file:
            for line in file:
                if line.startswith('a') or line.startswith('b') or not line.strip():
                    continue
                parts = line.split()
                try:
                    filtration_value = float(parts[1])
                    filtration_values.append(filtration_value)
                except ValueError as ve:
                    #print(f"Error converting {parts[1]} to float: {ve}")
                    continue
        return filtration_values
    except FileNotFoundError:
        #print(f"File not found: {summary_file_path}")
        return []
    except Exception as e:
        #print(f"Error reading file {summary_file_path}: {e}")
        return []

    
def calculate_eigenvalue_statistics(eigenvalues, tolerance):
    """
    Calculate statistics from eigenvalues, including a specific focus on the minimum positive eigenvalue.
    """
    zero_count = sum(1 for val in eigenvalues if abs(val) < tolerance)
    significant_eigenvalues = [val for val in eigenvalues if abs(val) >= tolerance]

    # Initialize statistics with placeholders
    stats = [0] * 9  # Adjust size based on needed statistics
    if not significant_eigenvalues:
        return stats, zero_count

    # Ensure to consider only positive eigenvalues for certain statistics
    positive_eigenvalues = [val for val in significant_eigenvalues if val > 0]

    if positive_eigenvalues:
        # Compute statistics
        stats[0] = sum(positive_eigenvalues)  # Total sum of positive eigenvalues
        stats[1] = min(positive_eigenvalues)  # Minimum of positive eigenvalues
        stats[2] = max(positive_eigenvalues)  # Maximum of positive eigenvalues
        stats[3] = np.mean(positive_eigenvalues)  # Mean of positive eigenvalues
        stats[4] = np.median(positive_eigenvalues)  # Median of positive eigenvalues
        stats[5] = np.std(positive_eigenvalues)  # Standard deviation of positive eigenvalues
        stats[6] = np.var(positive_eigenvalues)  # Variance of positive eigenvalues
        stats[7] = len(positive_eigenvalues)  # Count of positive eigenvalues
        stats[8] = np.sum(np.power(positive_eigenvalues, 2))  # Sum of squares of positive eigenvalues
    else:
        # Fallback if no positive significant eigenvalues are found
        stats[1] = 0  # Min eigenvalue as 0 if no positive values are found

    return stats, zero_count    
    

def get_zero_counts(summary_file_path):
    zero_counts = {'betti_0': [], 'betti_1': []}
    try:
        with open(summary_file_path, 'r') as file:
            next(file)
            for line in file:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    filtration_value = float(parts[1])
                    zero_counts['betti_0'].append((filtration_value, int(parts[2])))
                    zero_counts['betti_1'].append((filtration_value, int(parts[3])))
                except ValueError as ve:
                    #print(f"Error converting {parts} to appropriate types: {ve}")
                    continue
        return zero_counts
    except Exception as e:
        #print(f"Error reading zero counts from {summary_file_path}: {e}")
        return zero_counts

    
def process_spectra_file(spectra_file_path, filtration_values, zero_counts, betti_number):
    TOL = 0.001
    #logger.info(f"Processing spectra file: {spectra_file_path}")
    intervals = [(0, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1)]
    interval_stats = {interval: [] for interval in intervals}
    interval_zero_counts = {interval: 0 for interval in intervals}

    for value, count in zero_counts[f'betti_{betti_number}']:
        for interval in intervals:
            if interval[0] <= value < interval[1]:
                interval_zero_counts[interval] += count

    if os.path.exists(spectra_file_path):
        with open(spectra_file_path, 'r') as file:
            lines = [line.strip() if line.strip() else '0' for line in file]
        if len(lines) != len(filtration_values):
            return {}, {}

        for line, f_value in zip(lines, filtration_values):
            eigenvalues = [0 if abs(float(val)) < TOL else float(val) for val in line.split()]
            for interval in intervals:
                if interval[0] <= f_value < interval[1]:
                    stats, _ = calculate_eigenvalue_statistics(eigenvalues, tolerance)
                    interval_stats[interval].append(stats)

    aggregated_stats = {interval: aggregate_statistics(stats) if stats else [0]*6 for interval, stats in interval_stats.items()}
    return aggregated_stats, interval_zero_counts


def aggregate_statistics(stats_per_filtration):
    if not stats_per_filtration:
        return [0]*9
    aggregated_stats = np.sum(stats_per_filtration, axis=0)
    if isinstance(aggregated_stats, np.float64):
        return [aggregated_stats]
    return aggregated_stats.tolist()


def process_subfolder(subfolder, main_folder, csv_file_path, error_log_path):
    pdb_file_path = os.path.join(main_folder, subfolder, f"{subfolder}_protein.pdb")
    mol2_file_path = os.path.join(main_folder, subfolder, f"{subfolder}_ligand.mol2")
    
    row_data = [subfolder]
    spectra_files_to_delete = []

    if os.path.exists(pdb_file_path) and os.path.exists(mol2_file_path):
        extraction_instance = Extraction(pdb_file_path, mol2_file_path, protein_atoms, ligand_atoms)
        for pro_key in protein_atoms:
            for lig_key in ligand_atoms:
                dg = DigraphGenerator(param_combinations, cutoff_distance)
                digraph = dg.create_digraph(extraction_instance, pro_key, lig_key, param_combinations[0])
                if digraph is not None:
                    all_stats, all_zero_counts, spectra_file_paths, flag_file_created = run_flagser_with_temp_file(digraph, subfolder, pro_key, lig_key, error_log_path, betti_numbers, intervals, stats)
                    if flag_file_created:
                        row_data.extend(all_stats)
                        row_data.extend(all_zero_counts)
                    else:
                        dummy_data = [0] * 10 * 5
                        row_data.extend(dummy_data * 2)
                    spectra_files_to_delete.extend(spectra_file_paths)
                else:
                    dummy_data = [0] * 10 * 5
                    row_data.extend(dummy_data * 2)

        with open(csv_file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(row_data)
            #logger.info(f"Written data for subfolder {subfolder}.")
            
        for file_path in spectra_files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
                
    else:
        with open(error_log_path, 'a') as error_file:
            error_file.write(f"Required files for subfolder {subfolder} are not found.\n")

            
def main():
    parser = argparse.ArgumentParser(description='Process specific subfolders for Barcode calculation.')
    parser.add_argument('job_id', type=int, help='Unique identifier for this job')
    parser.add_argument('subfolders', nargs='+', help='List of subfolders to process')
    args = parser.parse_args()
    job_id = args.job_id
    subfolders_to_process = args.subfolders

    main_folder = "/mnt/scratch/ziamusha/v2007"
    csv_output_folder = "/mnt/home/ziamusha/Documents/Flagser-Laplacian/v2007/L_t2_v3/CSVlor_t2_v3"
    csv_file_path = os.path.join(csv_output_folder, f'pdfl_betti_bar_sums_{job_id}.csv')
    error_log_path = f"/mnt/home/ziamusha/Documents/Flagser-Laplacian/v2007/L_t2_v3/LOG/error_log_{job_id}.txt"

    for subfolder in subfolders_to_process:
        process_subfolder(subfolder, main_folder, csv_file_path, error_log_path)

if __name__ == "__main__":
    main()
