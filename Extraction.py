import numpy as np
import os
import csv
from Bio.PDB import PDBParser
import pandas as pd

class Extraction():
    def __init__(self, pdb_file_path, mol2_file_path, protein_atoms=('C', 'N', 'O', 'S'), ligand_atoms=('H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I')):
        self.protein_atoms = protein_atoms
        self.ligand_atoms = ligand_atoms
        self.protein_atoms_dict = self.extractProtein(pdb_file_path)
        self.ligand_atoms_dict = self.extractLigand(mol2_file_path)
        
    def extractProtein(self, pdb_file):
        print("Processing file:", pdb_file)
        # Extract the data for proteins using BioPython
        p = PDBParser(PERMISSIVE=True, QUIET=True)
        structure = p.get_structure('', pdb_file)
        self.protein_atoms_dict = {p : [] for p in self.protein_atoms}
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atom_id = atom.get_id()
                        for atom_type in self.protein_atoms:
                            if atom_type in atom_id:
                                xp, yp, zp = atom.get_coord()
                                self.protein_atoms_dict[atom_type].append((xp, yp, zp))
                                break
                                                     
        for atom_type in self.protein_atoms:
            #if there is only one coordinate tuple in the list for the given atom type
            if(len(self.protein_atoms_dict[atom_type])) == 1:
                #Reshapes into a 2DNumPyarray with 1 row and as many columns as there are needed in tuple 
                self.protein_atoms_dict[atom_type] = np.array(self.protein_atoms_dict[atom_type]).reshape(1, -1)
            else:
                #Converts the list of coordinate tuples into a 2D NumPy array.
                self.protein_atoms_dict[atom_type] = np.array(self.protein_atoms_dict[atom_type])
                
        # Create a reverse mapping from coordinates to atom type
        self.coord_to_atom_type = {}
        for atom_type in self.protein_atoms:
            for coord in self.protein_atoms_dict[atom_type]:
                # Use the coordinate tuple as a key to map back to atom type
                self.coord_to_atom_type[tuple(coord)] = atom_type
        return self.protein_atoms_dict
    
    def get_protein_atom_type(self, coord):
        # Use the coordinate tuple to retrieve the atom type
        return self.coord_to_atom_type.get(tuple(coord), None)
    
    def get_ligand_atom_type(self, coord):
        # Use the coordinate tuple to retrieve the atom type for ligands
        return self.ligand_coord_to_atom_type.get(tuple(coord), None)
    
    def extractLigand(self, mol2_file):
        print("Processing file:", mol2_file)
        self.ligand_atoms_dict = {l : [] for l in self.ligand_atoms}
        with open(mol2_file, 'r') as f:
            #Flag to indicate if the line containing atom coordinates has started
            atom_coords_started = False
            for line in f:
                #Skip lines before the @<TRIPOS>ATOM line
                if not atom_coords_started and not line.startswith("@<TRIPOS>ATOM"):
                    continue
                else:
                    atom_coords_started = True
                    #Check if the line marks the end of the atom coordinates section
                    if line.startswith("@<TRIPOS>BOND"):
                        break
                    #Split the line into its component fields
                    data = line.split()
                    count = 0
                    for lig_type in self.ligand_atoms:
                        if len(data) >= 6:
                            if lig_type in data[5]:
                                if 'BR' in lig_type:
                                    lig_type = lig_type.replace('BR', 'Br')
                                elif 'CL' in lig_type:
                                    lig_type = lig_type.replace('CL', 'Cl')
                                if lig_type in data[5]:
                                    lig_id = data[1]
                                    xl = float(data[2])
                                    yl = float(data[3])
                                    zl = float(data[4])
                                    self.ligand_atoms_dict[lig_type].append(np.array([xl, yl, zl]))
                                    count +=1
                                    #print(lig_type)
                                    #break
                                    
        for lig_type in self.ligand_atoms:
            if(len(self.ligand_atoms_dict[lig_type])) == 1:
                self.ligand_atoms_dict[lig_type] = np.array(self.ligand_atoms_dict[lig_type]).reshape(1, -1)
            else:
                self.ligand_atoms_dict[lig_type] = np.array(self.ligand_atoms_dict[lig_type])
        # Create a reverse mapping from coordinates to atom type       
        self.ligand_coord_to_atom_type = {}
        for lig_type in self.ligand_atoms:
            for coord in self.ligand_atoms_dict[lig_type]:
                self.ligand_coord_to_atom_type[tuple(coord)] = lig_type

        return self.ligand_atoms_dict
    