import numpy as np
from tqdm import tqdm
from pyteomics import mgf

from rdkit import Chem

# ignore the warning
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


"""mgf format
[{
	'params': {
		'title': prefix_<index>, 
		'precursor_type': <precursor_type (e.g. [M+NH4]+ and [M+H]+)>, 
		'precursor_mz': <precursor m/z>,
		'molmass': <isotopic mass>, 
		'ms_level': <ms_level>, 
		'ionmode': <POSITIVE|NEGATIVE>, 
		'source_instrument': <source_instrument>,
		'instrument_type': <instrument_type>, 
		'collision_energy': <collision energe>, 
		'smiles': <smiles>, 
		'inchi_key': <inchi_key>, 
	},
	'm/z array': mz_array,
	'intensity array': intensity_array
}, ...]
"""


def sdf2mgf(path, prefix):
    supp = Chem.SDMolSupplier(path)
    print("Read {} data from {}".format(len(supp), path))

    spectra = []
    for idx, mol in enumerate(tqdm(supp)):
        if (
            mol == None
            or not mol.HasProp("MASS SPECTRAL PEAKS")
            or not mol.HasProp("PRECURSOR TYPE")
            or not mol.HasProp("PRECURSOR M/Z")
            or not mol.HasProp("SPECTRUM TYPE")
            or not mol.HasProp("COLLISION ENERGY")
            or not mol.HasProp("ION MODE")
            or not mol.HasProp("EXACT MASS")
            or not mol.HasProp("INSTRUMENT TYPE")
        ):
            continue

        mz_array = []
        intensity_array = []
        raw_ms = mol.GetProp("MASS SPECTRAL PEAKS").split("\n")
        for line in raw_ms:
            mz_array.append(float(line.split()[0]))
            intensity_array.append(float(line.split()[1]))
        mz_array = np.array(mz_array)
        intensity_array = np.array(intensity_array)

        inchi_key = (
            "Unknown" if not mol.HasProp("INCHIKEY") else mol.GetProp("INCHIKEY")
        )
        instrument = (
            "Unknown" if not mol.HasProp("INSTRUMENT") else mol.GetProp("INSTRUMENT")
        )
        spectrum = {
            "params": {
                "title": prefix + "_" + str(idx),
                "precursor_type": mol.GetProp("PRECURSOR TYPE"),
                "precursor_mz": mol.GetProp("PRECURSOR M/Z"),
                "molmass": mol.GetProp("EXACT MASS"),
                "ms_level": mol.GetProp("SPECTRUM TYPE"),
                "ionmode": mol.GetProp("ION MODE"),
                "source_instrument": instrument,
                "instrument_type": mol.GetProp("INSTRUMENT TYPE"),
                "collision_energy": mol.GetProp("COLLISION ENERGY"),
                "smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
                "inchi_key": inchi_key,
            },
            "m/z array": mz_array,
            "intensity array": intensity_array,
        }
        spectra.append(spectrum)
    return spectra
