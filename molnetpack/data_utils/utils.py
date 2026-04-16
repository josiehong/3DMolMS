import numpy as np
import re
from decimal import *
from tqdm import tqdm

from rdkit import Chem

# ignore the warning
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem, rdDepictor


def ms_vec2dict(spec, resolution=1):
    x = []
    y = []
    for i, j in enumerate(spec):
        if j != 0:
            x.append(str(i * resolution))
            y.append(str(j))
    return {"m/z": ",".join(x), "intensity": ",".join(y)}


def generate_ms(x, y, precursor_mz, resolution=1, max_mz=1500, charge=1):
    """
    Input:  x   [float list denotes the x-coordinate of peaks]
                    y   [float list denotes the y-coordinate of peaks]
                    precursor_mz	[float denotes the parention]
                    resolution	[float denotes the resolution of spectra]
                    max_mz		[integer denotes the maxium m/z value of spectra]
                    charge		[float denotes the charge of spectra]
    Return: ms	[numpy array denotes the mass spectra]
    """
    # generate isotropic peaks (refers to Kaiyuan's codes:
    # https://github.com/lkytal/PredFull/blob/master/train_model.py)
    isotropic_peaks = []
    for delta in (0, 1, 2):
        tmp = precursor_mz + delta / charge
        isotropic_peaks.append(int(tmp // resolution))

    # prepare parameters
    precursor_mz = Decimal(str(precursor_mz))
    resolution = Decimal(str(resolution))
    max_mz = Decimal(str(max_mz))
    right_bound = int(
        precursor_mz // resolution
    )  # make precursor_mz as the right bound

    # init mass spectra vector: add "0" to y data
    ms = [0] * int(max_mz // resolution)

    # convert x, y to vector
    for idx, val in enumerate(x):
        val = int(round(Decimal(str(val)) // resolution))
        if val >= right_bound:  # remove precusor peak
            continue
        if val in isotropic_peaks:
            continue
        ms[val] += y[idx]

    # normalize to 0-1
    if np.max(ms) - np.min(ms) == 0:
        print(
            "The maximum intensity and minimum intensity of this spectrum are the same!"
        )
        print("right bound", right_bound)
        for i, j in zip(x, y):
            print(i, j)
        return False, np.array(ms)
    ms = (ms - np.min(ms)) / (np.max(ms) - np.min(ms))

    # smooth out large values
    ms = np.sqrt(np.array(ms))
    return True, ms


def ce2nce(ce, precursor_mz, charge):
    charge_factor = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}
    return ce * 500 * charge_factor[charge] / precursor_mz


def nce2ce(nce, precursor_mz, charge):
    charge_factor = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}
    return nce * precursor_mz / (500 * charge_factor[charge])


def parse_collision_energy(ce_str, precursor_mz, charge=1):
    # ratio constants for NCE
    charge_factor = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}

    ce = None
    nce = None

    # match collision energy (eV)
    matches_ev = {
        # NIST20
        r"^[\d]+[.]?[\d]*$": lambda x: float(x),
        r"^[\d]+[.]?[\d]*[ ]?eV$": lambda x: float(x.rstrip(" eV")),
        r"^[\d]+[.]?[\d]*[ ]?ev$": lambda x: float(x.rstrip(" ev")),
        r"^[\d]+[.]?[\d]*[ ]?v$": lambda x: float(x.rstrip(" v")),
        r"^[\d]+[.]?[\d]*[ ]?V$": lambda x: float(x.rstrip(" V")),
        r"^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$": lambda x: float(
            x.split()[1].rstrip("eV")
        ),
        r"^nce=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*ev$": lambda x: float(
            x.split()[1].rstrip("ev")
        ),
        # MassBank
        r"^[\d]+[.]?[\d]*[ ]?V$": lambda x: float(x.rstrip(" V")),
        r"^hcd[\d]+[.]?[\d]*$": lambda x: float(x.lstrip("hcd")),
        r"^[\d]+HCD$": lambda x: float(x.rstrip("HCD")),  # 35HCD
    }
    for k, v in matches_ev.items():
        if re.match(k, ce_str):
            ce = v(ce_str)
            break
    # match collision energy (NCE)
    matches_nce = {
        # MassBank
        r"^[\d]+[.]?[\d]*[ ]?[%]? \(nominal\)$": lambda x: float(
            x.rstrip("% (nominal)")
        ),
        r"^[\d]+[.]?[\d]*[ ]?nce$": lambda x: float(x.rstrip(" nce")),
        r"^[\d]+[.]?[\d]*[ ]?\(nce\)$": lambda x: float(x.rstrip(" (nce)")),
        r"^NCE=[\d]+\%$": lambda x: float(x.lstrip("NCE=").rstrip("%")),
        r"^[\d]+[.]?[\d]*\([Nn][Cc][Ee]\)$": lambda x: float(
            x.split("(")[0]
        ),  # 90(NCE)
        r"^HCD \(NCE [\d]+[.]?[\d]*%\)$": lambda x: float(
            x.split(" ")[-1].rstrip("%)")
        ),  # HCD (NCE 40%)
        # CASMI
        r"^[\d]+[.]?[\d]*[ ]?\(nominal\)$": lambda x: float(
            x.rstrip("(nominal)").rstrip(" ")
        ),
    }
    for k, v in matches_nce.items():
        if re.match(k, ce_str):
            nce = v(ce_str) * 0.01
            break

    # unknown collision energy
    if ce_str == "Unknown":
        ce = 40

    if nce == None and ce != None:
        nce = ce * 500 * charge_factor[charge] / precursor_mz
    elif ce == None and nce != None:
        ce = nce * precursor_mz / (500 * charge_factor[charge])
    else:
        # raise Exception('Collision energy parse error: {}'.format(ce_str))
        return None, None
    return ce, nce


def conformation_array(x, conf_type):
    # convert smiles to molecule
    if conf_type == "etkdg":
        mol = Chem.MolFromSmiles(x)
        mol_from_smiles = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_from_smiles)

    elif conf_type == "etkdgv3":
        mol = Chem.MolFromSmiles(x)
        mol_from_smiles = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 0xF00D
        AllChem.EmbedMolecule(mol_from_smiles, ps)

    elif conf_type == "2d":
        mol = Chem.MolFromSmiles(x)
        mol_from_smiles = Chem.AddHs(mol)
        rdDepictor.Compute2DCoords(mol_from_smiles)

    elif conf_type == "origin":
        mol_from_smiles = Chem.AddHs(x)

    elif conf_type == "omega":
        raise ValueError("OMEGA conformation will be supported soon. ")
    else:
        raise ValueError("Unsupported conformation type. {}".format(conf_type))

    # get the x,y,z-coordinates of atoms
    try:
        conf = mol_from_smiles.GetConformer()
    except:
        return False, None, None
    xyz_arr = conf.GetPositions()
    # center the x,y,z-coordinates
    centroid = np.mean(xyz_arr, axis=0)
    xyz_arr -= centroid

    # concatenate with atom attributes
    xyz_arr = xyz_arr.tolist()
    for i, atom in enumerate(mol_from_smiles.GetAtoms()):
        xyz_arr[i] += [atom.GetDegree()]
        xyz_arr[i] += [atom.GetExplicitValence()]
        xyz_arr[i] += [atom.GetMass() / 100]
        xyz_arr[i] += [atom.GetFormalCharge()]
        xyz_arr[i] += [atom.GetNumImplicitHs()]
        xyz_arr[i] += [int(atom.GetIsAromatic())]
        xyz_arr[i] += [int(atom.IsInRing())]
    xyz_arr = np.array(xyz_arr)

    # get the atom types of atoms
    atom_type = [atom.GetSymbol() for atom in mol_from_smiles.GetAtoms()]
    return True, xyz_arr, atom_type


def precursor_calculator(precursor_type, mass):
    if precursor_type == "[M+H]+":
        return mass + 1.007276
    elif precursor_type == "[M+Na]+":
        return mass + 22.989218
    elif precursor_type == "[2M+H]+":
        return 2 * mass + 1.007276
    elif precursor_type == "[M-H]-":
        return mass - 1.007276
    elif precursor_type == "[M+H-H2O]+":
        return mass - 17.0038370665
    elif precursor_type == "[M+2H]2+":
        return mass / 2 + 1.007276
    else:
        raise ValueError("Unsupported precursor type: {}".format(precursor_type))


if __name__ == "__main__":
    example_x = [
        41.0399,
        43.0192,
        43.0766,
        55.0301,
        57.0709,
        65.015,
        65.0366,
        67.0302,
        67.0517,
        82.0409,
        92.0251,
        94.0407,
        109.0512,
        119.0354,
        119.1288,
        121.0556,
        136.0619,
        136.1628,
        148.0621,
    ]
    example_y = [
        6.207207,
        49.711712,
        1.986987,
        2.316316,
        1.600601,
        3.496496,
        1.581582,
        6.727728,
        4.393393,
        2.598599,
        16.601602,
        8.375375,
        5.252252,
        100.0,
        4.228228,
        1.025025,
        41.038038,
        1.571572,
        2.447447,
    ]
    example_precursor_mz = 220.1193
    flag, example_ms = generate_ms(
        example_x,
        example_y,
        example_precursor_mz,
        resolution=0.2,
        max_mz=1500,
        charge=1,
    )
    example_spec = spec_convert(example_ms, resolution=0.2)
    print(example_spec)
