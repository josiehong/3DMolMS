import sys
import pickle
import numpy as np
from numpy.linalg import norm
from pyteomics import mgf
from decimal import Decimal
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt


def load_pickle_spectra(pickle_file):
    with open(pickle_file, "rb") as f:
        return pickle.load(f)


def load_mgf_spectra(mgf_file):
    return list(mgf.read(mgf_file))


def generate_ms(x, y, resolution=0.2, max_mz=1500):
    # Prepare parameters
    resolution = Decimal(str(resolution))
    max_mz = Decimal(str(max_mz))

    # Initialize mass spectra vector
    ms = [0] * int(max_mz // resolution)

    # Convert x, y to vector
    for idx, val in enumerate(x):
        val = int(round(Decimal(str(val)) // resolution))
        ms[val] += y[idx]

    if sum(ms) == 0:
        return False, ms

    return True, ms


def bin_mgf_spectrum(spectrum, resolution=0.2, max_mz=1500):
    mz_array = spectrum["m/z array"]
    intensity_array = spectrum["intensity array"]

    success, binned_spectrum = generate_ms(
        mz_array, intensity_array, resolution=resolution, max_mz=max_mz
    )
    return success, binned_spectrum


def cal_cosine_similarity(spec1, spec2):
    # spec1 is already binned (from pickle file)
    # spec2 needs to be binned (from MGF file)
    success, binned_spec2 = bin_mgf_spectrum(spec2)

    if not success:
        return None

    if len(spec1) != len(binned_spec2):
        return None

    cosine = np.dot(spec1, binned_spec2) / (norm(spec1) * norm(binned_spec2))
    return cosine


def main(pickle_file, mgf_file):
    # Load spectra from both files
    pkl_spectra = load_pickle_spectra(pickle_file)
    mgf_spectra = load_mgf_spectra(mgf_file)

    # Create dictionaries with titles as keys
    pkl_dict = {spec["title"]: spec["spec"] for spec in pkl_spectra}
    mgf_dict = {spec["params"]["title"]: spec for spec in mgf_spectra}

    # Calculate cosine similarities and extract precursor types
    result = []
    for title, _ in tqdm(pkl_dict.items()):
        if title in mgf_dict:
            similarity = cal_cosine_similarity(pkl_dict[title], mgf_dict[title])
            if similarity is not None:
                precursor_type = mgf_dict[title]["params"].get(
                    "precursor_type", "Unknown"
                )
                smiles = mgf_dict[title]["params"].get("smiles", "Unknown")
                collision_energy = mgf_dict[title]["params"].get(
                    "collision_energy", "Unknown"
                )
                result.append(
                    {
                        "title": title,
                        "smiles": smiles,
                        "collision_energy": collision_energy,
                        "precursor_type": precursor_type,
                        "similarity": similarity,
                    }
                )

    return result


if __name__ == "__main__":
    # Usage
    pickle_file = sys.argv[1]
    mgf_file = sys.argv[2]
    if len(sys.argv) == 4:
        plot_file = sys.argv[4]
    elif len(sys.argv) == 5:
        res_file = sys.argv[3]
        plot_file = sys.argv[4]
    else:
        raise ValueError("Invalid number of arguments")

    results = main(pickle_file, mgf_file)

    # Print results
    # for item in results:
    #     print(f"Title: {item['title']}, Cosine Similarity: {item['similarity']}")

    # Save results to file
    df_results = pd.DataFrame(results)
    print(df_results.head())
    df_results.to_csv(res_file, index=False)

    # Plot boxplot
    print("Mean cosine similarity by precursor type:")
    print(df_results.groupby("precursor_type")["similarity"].mean())

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(df_results["similarity"].tolist(), bins=50, edgecolor="black")
    plt.title("Histogram of Cosine Similarities")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_file)
    plt.close()
    print(f"Histogram of cosine similarities saved to {plot_file}")
