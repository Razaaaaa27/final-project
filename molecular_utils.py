# molecular_utils.py
# Helper functions untuk operasi pada molekul SMILES.
# DIPERBAIKI: featurisasi atom sekarang KONSISTEN PERSIS dengan
# konstruksi-graf-new.ipynb (skema V2) yang dipakai saat training.

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


# ============================================================
# Definisi fitur node 43 dimensi (skema V2 - sama dgn training)
# ============================================================

ATOM_TYPES = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I',
              'Si', 'Se', 'B', 'As', 'other']                  # 14 dim
NUM_H_LIST = [0, 1, 2, 3, 4]                                   # 5 dim
FORMAL_CHARGE_LIST = [-2, -1, 0, 1, 2, 'other']                # 6 dim
CHIRAL_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    'other'
]                                                              # 4 dim
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    'other'
]                                                              # 5 dim
DEGREE_LIST = [0, 1, 2, 3, 4, 5, 6]                            # 7 dim
# is_aromatic: 1 dim binary + 1 dim padding = 2 dim
# Total: 14+5+6+4+5+2+7 = 43

SUPPORTED_ATOMS = set(ATOM_TYPES) - {'other'}


# ============================================================
# Definisi fitur edge 12 dimensi (skema V2 - sama dgn training)
# ============================================================

BOND_TYPE_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]                                                              # 4 dim
STEREO_LIST = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
    Chem.rdchem.BondStereo.STEREOANY,
]                                                              # 6 dim
# is_conjugated: 1 dim, is_in_ring: 1 dim
# Total: 4+6+1+1 = 12


# ============================================================
# One-hot helper
# ============================================================

def one_hot(value, choices):
    # KONSISTEN dengan training V2: kalau value tidak di choices,
    # one-hot di slot TERAKHIR (yang biasanya 'other').
    enc = [0] * len(choices)
    if value in choices:
        enc[choices.index(value)] = 1
    else:
        enc[-1] = 1
    return enc


# ============================================================
# Featurisasi atom dan bond (PERSIS sama dengan training V2)
# ============================================================

def get_atom_features(atom):
    # PERSIS sama dengan konstruksi-graf-new.ipynb.
    # JANGAN diubah skemanya - harus match training.
    atom_type = atom.GetSymbol() if atom.GetSymbol() in ATOM_TYPES else 'other'
    num_h = min(atom.GetTotalNumHs(), 4)
    charge = atom.GetFormalCharge() if atom.GetFormalCharge() in [-2, -1, 0, 1, 2] else 'other'
    chiral = atom.GetChiralTag() if atom.GetChiralTag() in CHIRAL_LIST[:-1] else 'other'
    hybrid = atom.GetHybridization() if atom.GetHybridization() in HYBRIDIZATION_LIST[:-1] else 'other'
    degree = min(atom.GetDegree(), 6)

    feats = (
        one_hot(atom_type, ATOM_TYPES) +
        one_hot(num_h, NUM_H_LIST) +
        one_hot(charge, FORMAL_CHARGE_LIST) +
        one_hot(chiral, CHIRAL_LIST) +
        one_hot(hybrid, HYBRIDIZATION_LIST) +
        [int(atom.GetIsAromatic())] +
        [0] +
        one_hot(degree, DEGREE_LIST)
    )
    return feats


def get_bond_features(bond):
    # PERSIS sama dengan konstruksi-graf-new.ipynb.
    feats = (
        one_hot(bond.GetBondType(), BOND_TYPE_LIST) +
        one_hot(bond.GetStereo(), STEREO_LIST) +
        [int(bond.GetIsConjugated())] +
        [int(bond.IsInRing())]
    )
    return feats


# ============================================================
# Validasi dan preprocessing SMILES (tidak berubah)
# ============================================================

def validate_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return False, "SMILES kosong"
    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "SMILES tidak valid (tidak dapat di-parse RDKit)"
    if mol.GetNumHeavyAtoms() == 0:
        return False, "SMILES tidak memiliki atom heavy"
    if mol.GetNumBonds() == 0:
        return False, "SMILES tidak memiliki ikatan kimia"
    return True, "OK"


def salt_removal(smiles):
    if '.' not in smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        return Chem.MolToSmiles(mol), mol

    fragments = smiles.split('.')
    best_mol, best_count = None, -1
    for frag in fragments:
        mol = Chem.MolFromSmiles(frag)
        if mol is not None and mol.GetNumHeavyAtoms() > best_count:
            best_count = mol.GetNumHeavyAtoms()
            best_mol = mol

    if best_mol is None:
        return None, None
    return Chem.MolToSmiles(best_mol), best_mol


def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        pass
    return None


# ============================================================
# Konversi SMILES ke graf PyG
# ============================================================

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor(
        [get_atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float
    )

    edge_indices, edge_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_features += [bond_feat, bond_feat]

    if len(edge_indices) == 0:
        return None

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    return data


# ============================================================
# Pipeline lengkap (tidak berubah)
# ============================================================

def preprocess_smiles(raw_smiles):
    is_valid, msg = validate_smiles(raw_smiles)
    if not is_valid:
        return {'success': False, 'graph': None, 'canonical_smiles': None, 'error': msg}

    cleaned_smiles, mol = salt_removal(raw_smiles.strip())
    if cleaned_smiles is None:
        return {'success': False, 'graph': None, 'canonical_smiles': None,
                'error': "Salt removal gagal"}

    canonical = canonicalize_smiles(cleaned_smiles)
    if canonical is None:
        return {'success': False, 'graph': None, 'canonical_smiles': None,
                'error': "Kanonikalisasi gagal"}

    graph = smiles_to_graph(canonical)
    if graph is None:
        return {'success': False, 'graph': None, 'canonical_smiles': canonical,
                'error': "Konversi ke graf gagal"}

    return {'success': True, 'graph': graph, 'canonical_smiles': canonical, 'error': None}


# ============================================================
# Properti molekul (tidak berubah)
# ============================================================

def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        scaffold_smi = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
    except Exception:
        scaffold_smi = ""

    return {
        'MW':         Descriptors.MolWt(mol),
        'AlogP':      Descriptors.MolLogP(mol),
        'num_atoms':  mol.GetNumHeavyAtoms(),
        'num_bonds':  mol.GetNumBonds(),
        'num_rings':  mol.GetRingInfo().NumRings(),
        'scaffold':   scaffold_smi,
    }


# ============================================================
# Out-of-domain detection (tidak berubah)
# ============================================================

DOMAIN_MW_MIN = 100.0
DOMAIN_MW_MAX = 1000.0
DOMAIN_NATOMS_MIN = 5
DOMAIN_NATOMS_MAX = 100


def check_out_of_domain(mol_or_smiles):
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            return False, ["SMILES tidak valid"]
    else:
        mol = mol_or_smiles

    warnings = []

    mw = Descriptors.MolWt(mol)
    if mw < DOMAIN_MW_MIN:
        warnings.append(f"MW {mw:.1f} Da di bawah range training (minimal {DOMAIN_MW_MIN} Da)")
    elif mw > DOMAIN_MW_MAX:
        warnings.append(f"MW {mw:.1f} Da di atas range training (maksimal {DOMAIN_MW_MAX} Da)")

    n_atoms = mol.GetNumHeavyAtoms()
    if n_atoms < DOMAIN_NATOMS_MIN:
        warnings.append(f"Jumlah atom {n_atoms} di bawah range training (minimal {DOMAIN_NATOMS_MIN})")
    elif n_atoms > DOMAIN_NATOMS_MAX:
        warnings.append(f"Jumlah atom {n_atoms} di atas range training (maksimal sekitar 71)")

    atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    unsupported = atom_symbols - SUPPORTED_ATOMS
    if unsupported:
        warnings.append(f"Atom tidak didukung: {', '.join(sorted(unsupported))}")

    is_in_domain = len(warnings) == 0
    return is_in_domain, warnings