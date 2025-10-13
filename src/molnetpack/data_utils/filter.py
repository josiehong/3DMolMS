import re
import numpy as np
from tqdm import tqdm

from rdkit import Chem

# ignore the warning
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from molmass import Formula


def filter_spec(spectra, config, type2charge):
    clean_spectra = []
    smiles_list = []
    for idx, spectrum in enumerate(tqdm(spectra)):
        smiles = spectrum["params"]["smiles"]
        try:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        except:
            print("Invalid SMILES: {}".format(smiles))
            continue
        if mol == None:
            continue

        # Filter by collision energy
        if (
            spectrum["params"]["collision_energy"].startswith("ramp")
            or spectrum["params"]["collision_energy"].startswith("Ramp")
            or spectrum["params"]["collision_energy"].endswith("ramp")
        ):  # we can not process this type collision energy now
            continue

        # Filter by instrument type
        if "instrument_type" in config.keys():
            instrument_type = spectrum["params"]["instrument_type"]
            if instrument_type not in config["instrument_type"]:
                continue

        # Filter by instrument (MoNA contains too many intrument names to filter out)
        if "instrument" in config.keys():
            instrument = spectrum["params"]["source_instrument"]
            if instrument not in config["instrument"]:
                continue

        # Filter by mslevel
        if "ms_level" in config.keys():
            mslevel = spectrum["params"]["ms_level"]
            if mslevel != config["ms_level"]:
                continue

        # Filter by atom number and atom type
        if check_atom(mol, config, in_type="molh") < 0:
            continue

        # Filter by precursor type
        precursor_type = spectrum["params"]["precursor_type"]
        if precursor_type not in config["precursor_type"]:
            continue

        # Filt by peak number
        if len(spectrum["m/z array"]) < config["min_peak_num"]:
            continue

        # Filter by max m/z
        if (
            np.max(spectrum["m/z array"]) < config["min_mz"]
            or np.max(spectrum["m/z array"]) > config["max_mz"]
        ):
            continue

        # Filter by ppm (mass error)
        try:
            f = CalcMolFormula(mol)
            f = added_formula(f, precursor_type)
            f = Formula(f)
            theo_mz = f.isotope.mass
            ppm = (
                abs(theo_mz - float(spectrum["params"]["precursor_mz"]))
                / theo_mz
                * 10**6
            )
        except:  # invalud formula, unsupported precursor type, or invalid precursor m/z
            continue
        if ppm > config["ppm_tolerance"]:
            continue

        # add charge
        spectrum["params"]["charge"] = type2charge[precursor_type]
        clean_spectra.append(spectrum)
        smiles_list.append(smiles)
    return clean_spectra, smiles_list


def f_str2dict(f):
    f_dict = {}
    frags = re.findall(
        r"(He|Li|Be|Ne|Na|Mg|Al|Si|Cl|Ar|Ca|Sc|Ti|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br\
							|Kr|Rb|Sr|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|Xe|Cs|Ba|La|Hf|Ta|Re|Os|Ir\
							|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|Fr|Ra|Ac|Rf|Db|Sg|Bh|Hs|Mt|Ds|Rg|Cn|Nh|Fl|Mc|Lv\
							|Ts|Og|[A-Z])(\d\d|\d|)",
        f,
    )
    for frag in frags:
        # print(frag)
        atom_sym = frag[0]
        atom_num = frag[1]
        if atom_num == None:
            atom_num = 0
        elif atom_num == "":
            atom_num = 1

        f_dict[atom_sym] = int(atom_num)
    return f_dict


def f_dict2str(f_dict):
    f = ""
    for k, v in f_dict.items():
        if v > 1:
            f += k + str(v)
        elif v == 1:
            f += k
    return f


def added_formula(f, precursor_type):
    f_dict = f_str2dict(str(f))
    if precursor_type == "[M+H]+":
        f_dict["H"] += 1
    elif precursor_type == "[M+Na]+":
        if "Na" in f_dict.keys():
            f_dict["Na"] += 1
        else:
            f_dict["Na"] = 1
    elif precursor_type == "[M-H]-":
        f_dict["H"] -= 1
    elif precursor_type == "[M+H-H2O]+":
        f_dict["H"] -= 1
        f_dict["O"] -= 1
    elif precursor_type == "[M-H2O+H]+":
        f_dict["H"] -= 1
        f_dict["O"] -= 1
    elif precursor_type == "[M+2H]2+":
        f_dict["H"] += 2
    elif precursor_type == "[2M+H]+":
        f_dict = {k: int(v * 2) for k, v in f_dict.items()}
        f_dict["H"] += 1
    elif precursor_type == "[2M-H]-":
        f_dict = {k: int(v * 2) for k, v in f_dict.items()}
        f_dict["H"] -= 1
    else:
        raise ValueError("Unsupported precursor type: {}".format(precursor_type))

    f = f_dict2str(f_dict)
    return f


def filter_mol(suppl, config):
    clean_suppl = []
    smiles_list = []
    exceed_atom_num = 0
    unsupported_atom_type = 0
    for idx, mol in enumerate(tqdm(suppl)):
        if mol == None:
            continue
        mol = Chem.AddHs(mol)

        # Filter by atom number and atom type
        flag_atom = check_atom(mol, config, in_type="molh")
        if flag_atom < 0:
            if flag_atom == -1:
                exceed_atom_num += 1
            elif flag_atom == -2:
                unsupported_atom_type += 1
            continue

        clean_suppl.append(mol)
        smiles_list.append(Chem.MolToSmiles(mol))
    # print('# mol: Exceed atom number: {}'.format(exceed_atom_num))
    # print('# mol: Unsupported atom type: {}'.format(unsupported_atom_type))
    return clean_suppl, smiles_list


def check_atom(x, config, in_type="smiles"):
    assert in_type in ["smiles", "mol", "molh"]
    if in_type == "smiles":
        mol = Chem.AddHs(Chem.MolFromSmiles(x))
    elif in_type == "mol":
        mol = Chem.AddHs(x)
    else:
        mol = x

    if (
        len(mol.GetAtoms()) > config["max_atom_num"]
        or len(mol.GetAtoms()) < config["min_atom_num"]
    ):
        return -1

    is_compound_countain_rare_atom = False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in config["atom_type"]:
            is_compound_countain_rare_atom = True
            break
    if is_compound_countain_rare_atom:
        return -2
    return 1
