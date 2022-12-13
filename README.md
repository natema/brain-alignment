# Brain Alignment Methods

Implementation of methods to compute various brain similarity measures across animal species.

It requires `python3.7+` as well as `numpy, scipy`.

## Random Data Generation

Run `python3 utils/generate-random-data.py` to generate random data to play with the alignment.

The script will generate 5 random brains in `brains-space/`and `brains-matrix/`as well as a list of animals `animal_list.txt`.

## Brain-Alignment Computation

The alignment will consider as distance the linear combination $A \cdot Euclidean + B \cdot StrongestEdge$, for $A,B \in \mathbb{R^+}$. Execute the following command for all the options:

```python3 brain-alignment.py -h```

For example, to align the brains of “animal1” and “animal2” run:

```python3 brain-alignment.py -a1 animal1 -a2 animal2```

Note that the files `brains-space/animal1.txt` and `brains-matrix/animal1.txt` must exist for the script to run correctly.

The script requires the following flags:

* `-a1`, the name of animal1; 
* `-a2`, the name of animal2;

The script accepts the following optional flags:

* `-n`, the number of points used to represent the animals (default: 200);
* `-d`, the degrees of the canonical rotations (default: 90);
* `-r`, the number of small random rotations (default: 150);
* `-a`, the coefficient of Euclidean distance (default: 1); 
* `-b`, the coefficient of StrongestEdge distance (default: 1); 
* `-s`, use option to save the alignment to file (default: alignment not saved to file).

The output is saved in `distance/{animal1}_vs_{animal2}.pickle` (distance) and `alignment/{animal1}_vs_{animal2}.pickle` (alignment, optionally). The data is serialized using Pickle.

In order to generate the pairwise distance matrix, one must execute all pairwise distances. By running `python3 utils/generate-parallel-distance.py` a file `parallel_distance.txt` is created. The file contains the list of commands to be run to compute the pairwise alignments (all commands can be run in parallel). Once all distances have been computed, one can aggregate the data and create the pairwise distance matrix `distance_matrix.csv` by running `python3 utils/aggregate-distances.py`.
