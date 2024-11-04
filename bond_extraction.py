from Bio import PDB
import numpy as np
import pandas
import math
import sys
import time
import os
import argparse
import pickle
from scipy.spatial import cKDTree

# Elements in ligand to be considered
el_l = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
el_p = ["C", "N", "O", "S"] 
# Electronegativity values for the elements
en_vals = {'C': 2.55, 'N': 3.04, 'O': 3.5, 'S': 2.58, 'P': 2.19, 'F': 3.98, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66}

def Gauss_linking_integral(line1, line2):
    """
    a: tuple; elements are head and tail
    of line segment, each is a (3,) array representing the xyz coordinate.
    """
    a = [line1.startpoint, line1.endpoint]
    b = [line2.startpoint, line2.endpoint]

    R = np.empty((2, 2), dtype=tuple)
    for i in range(2):
        for j in range(2):
            R[i, j] = a[i] - b[j]

    n = []
    cprod = []

    cprod.append(np.cross(R[0, 0], R[0, 1]))
    cprod.append(np.cross(R[0, 1], R[1, 1]))
    cprod.append(np.cross(R[1, 1], R[1, 0]))
    cprod.append(np.cross(R[1, 0], R[0, 0]))

    for c in cprod:
        n.append(c / (np.linalg.norm(c) + 1e-6))

    area1 = np.arcsin(np.dot(n[0], n[1]))
    area2 = np.arcsin(np.dot(n[1], n[2]))
    area3 = np.arcsin(np.dot(n[2], n[3]))
    area4 = np.arcsin(np.dot(n[3], n[0]))

    sign = np.sign(np.cross(a[1] - a[0], b[1] - b[0]).dot(a[0] - b[0]))
    Area = area1 + area2 + area3 + area4

    return sign * Area

class Line:
    def __init__(self, startpoint, endpoint, start_type, end_type):
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.start_type = start_type
        self.end_type = end_type

# Stores data about an atom and manages its bonding relationships
class Atom:
    def __init__(self, data, etype, eid=None, serial_id=None):
        self.data = data  # 3D coordinates of the atom
        self.etype = etype  # Element type
        self.eid = eid  # Full identifier
        self.serial_id = serial_id  # Serial ID for identification
        self.bonds = []  # List of other Atoms this one is bonded to

    def add_bond(self, other_atom):
        if other_atom not in self.bonds:
            self.bonds.append(other_atom)

    def sum_bonded_electronegativities(self):
        return sum(en_vals[bond.etype] for bond in self.bonds)


class Line:
    """Class to represent a bond as a line between two points."""
    def __init__(self, startpoint, endpoint, start_type, end_type):
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.start_type = start_type
        self.end_type = end_type

class bonded_Atom:
    def __init__(self, atom, lines=None):
        if lines is None:
            lines = []  # Set default empty list if not provided
        self.atom = atom  # The actual Atom object    
        self.lines = lines  
        self.bonds = atom.bonds  # This ensures bonds are correctly referenced

    def add_line(self, line):
        self.lines.append(line)

    def sum_bonded_electronegativities(self):
        return self.atom.sum_bonded_electronegativities()

def get_bonded_atoms(atoms, bond_limits):
    positions = np.array([atom.data for atom in atoms])
    tree = cKDTree(positions)
    max_distance = max(bond_limits.values())
    pairs = tree.query_pairs(max_distance)

    for atom in atoms:
        atom.bonds = []

    for i, j in pairs:
        atom1, atom2 = atoms[i], atoms[j]
        dist = np.linalg.norm(atom1.data - atom2.data)
        pair_key = tuple(sorted([atom1.etype, atom2.etype]))
        if dist < bond_limits.get(pair_key, float('inf')):
            atom1.add_bond(atom2)
            atom2.add_bond(atom1)

    return [bonded_Atom(atom) for atom in atoms]

# Define bond limits in a dictionary for quick lookup
bond_limits = {
    ("C", "C"): 1.64, ("C", "S"): 1.92, ("C", "N"): 1.57,
    ("C", "O"): 1.57, ("N", "O"): 1.5, ("N", "N"): 1.55,
    ("N", "S"): 2.06, ("O", "S"): 1.52, ("O", "O"): 1.58,
    ("S", "S"): 2.17
}

import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Suppress PDB construction warnings
warnings.simplefilter('ignore', PDBConstructionWarning)


def get_protein_struct_from_pdb(pdbid, filepath):
    parser = PDB.PDBParser()
    struct = parser.get_structure(pdbid, filepath)
    total_struct = []
    for model in struct:
        for chain in model:
            cur_struc = None
            pre_struc = None
            for residue in chain:
                atoms = []
                for elm in residue:
                    if elm.get_full_id()[3][0].strip() == "":
                        atom_type = "ATOM"
                    else:
                        atom_type = "HETATM"
                    if atom_type == "ATOM":
                        atom = Atom(
                            elm.get_coord(),
                            elm.element,
                            elm.fullname,
                            elm.get_serial_number(),
                        )
                        atoms.append(atom)

                # find all bonds inside a residue
                cur_struc = get_bonded_atoms(atoms)

                # add bond between adjecant residue
                if pre_struc != None:
                    for cur_bonded_atom in cur_struc:
                        if cur_bonded_atom.atom.eid == " N  ":
                            for pre_bonded_atom in pre_struc:
                                if pre_bonded_atom.atom.eid == " C  ":
                                    startpoint = pre_bonded_atom.atom.data
                                    endpoint = cur_bonded_atom.atom.data
                                    start_type = pre_bonded_atom.atom.etype
                                    end_type = cur_bonded_atom.atom.etype
                                    midpoint = (startpoint + endpoint) / 2

                                    line1 = Line(
                                        startpoint, midpoint, start_type, end_type
                                    )
                                    # line2 = Line(endpoint, midpoint,
                                    #              start_type, end_type)
                                    line2 = Line(
                                        endpoint, midpoint, end_type, start_type
                                    )

                                    pre_bonded_atom.add_line(line1)
                                    cur_bonded_atom.add_line(line2)

                total_struct += cur_struc
                pre_struc = cur_struc

    return total_struct


def get_atom_and_bond_list_from_mol2(mol2file):
    print("Processing Mol2 file:", mol2file)
    with open(mol2file, 'r') as file:
        lines = file.readlines()

    atom_start = lines.index('@<TRIPOS>ATOM\n') + 1
    bond_start = lines.index('@<TRIPOS>BOND\n') + 1

    atoms = []
    for line in lines[atom_start:bond_start-1]:
        parts = line.split()
        atom_id = int(parts[0])
        x, y, z = map(float, parts[2:5])
        etype = parts[5].split('.')[0]  # Remove hybridization info
        atoms.append(Atom(data=np.array([x, y, z]), etype=etype, eid=parts[1], serial_id=atom_id))

    bondlist = []
    for line in lines[bond_start:]:
        if line.startswith('@<TRIPOS>'):
            break
        parts = line.split()
        bondlist.append([int(parts[1])-1, int(parts[2])-1])  # Convert to 0-based indices

    print(f"Processed {len(atoms)} atoms and {len(bondlist)} bonds.")
    return atoms, bondlist

def get_ligand_struct_from_mol2(mol2file):
    atoms, bondlist = get_atom_and_bond_list_from_mol2(mol2file)
    print("# of atoms: {}, # of bonds: {}".format(len(atoms), len(bondlist)))

    # Link atoms with their bonds
    for bond in bondlist:
        atom1 = atoms[bond[0]]
        atom2 = atoms[bond[1]]
        atom1.add_bond(atom2)
        atom2.add_bond(atom1)  # Ensure bidirectional linking

    bonded_atoms = [bonded_Atom(atom) for atom in atoms]  
    return bonded_atoms
