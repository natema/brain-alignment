import os
import pickle
import numpy as np


if __name__ == "__main__":
    animals = []
    with open("animal_list.txt") as handle:
        for line in handle:
            animals.append(line.strip())
    n = len(animals)

    M = np.empty((n, n))
    for (i, animal1) in enumerate(animals):
        for (j, animal2) in enumerate(animals):
            if j < i:
                continue
            with open(os.path.join("distance", f"{animal1}_vs_{animal2}.pickle"), "rb") as handle:
                distance = pickle.load(handle)
            M[i, j] = distance
            M[j, i] = distance

    os.makedirs("alignment-matrices", exist_ok=True)

    with open("dissimilarity-matrix.csv", "w") as fo:
        header = ",".join(animals)
        fo.write(f",{header}\n")
        for (i, animal1) in enumerate(animals):
            fo.write(animal1)
            for (j, animal2) in enumerate(animals):
                fo.write(f",{M[i, j]}")
            fo.write("\n")
