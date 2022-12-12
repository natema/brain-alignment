import os
import numpy as np

k = 5
(n, d) = (200, 3)

os.makedirs("brains-space", exist_ok=True)
os.makedirs("brains-matrix", exist_ok=True)
animals = []
for i in range(1, k+1):
    Si = np.random.normal(0, 1, size=(n, d))
    Mi = np.random.randint(0, 10, size=(n, n))
    animals.append(f"animal{i}")
    np.savetxt(os.path.join("brains-space", f"animal{i}.txt"), Si, fmt="%2.6e")
    np.savetxt(os.path.join("brains-matrix", f"animal{i}.txt"), Mi, fmt="%d")

with open("animal_list.txt", "w") as handle:
    for animal in animals:
        handle.write(f"{animal}\n")
