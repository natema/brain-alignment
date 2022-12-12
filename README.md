# Brain Alignment Methods

Implementation of methods to compute various brain similarity measures across animal species.

It requires `python3.7+` as well as `numpy, scipy`.

## Random Data Generation

Run `python3 utils/generate-random-data.py` to generate random data to play with the alignment.

The script will generate 5 random brains in `brains-space/`and `brains-matrix/`as well as a list of animals `animal_list.txt`.

## Brain-Alignment Computation

To align the brains of “animal1” and “animal2” the corresponding files `brains-space/animal1.txt` and `brains-matrix/animal1.txt` must exist. The alignment will consider as distance the linear combination $a \cdot Euclidean + b \cdot StrongestEdge$. To align the brains, run for example:

```python3 brain-alignment.py -a1 animal1 -a2 animal2 -n 200 -d 90 -r 200 -a 1 -b 1 -s```

where: `-a1` is the name of animal1; `-a2` is the name of animal2; `-n` is the number of points used to represent the animals (200 in our experiments); `-d` is the degrees of the canonical rotations (90 in our experiments); `-r` is the number of small random rotations (200 in our experiments); `-a` is the coefficient of Euclidean distance; `-b` is the coefficient of StrongestEdge distance; `-s` is an optional flag to save the alignment to file.

The output is saved in `distance/{animal1}_vs_{animal2}.pickle` (distance) and `alignment/{animal1}_vs_{animal2}.pickle` (alignment). The data is serialized using Pickle.

In order to generate the pairwise distance matrix, one must execute all pairwise distances. By running `python3 utils/generate-parallel-distance.py` a file `parallel_distance.txt` is created/ The file contains the list of commands to be run to compute the pairwise alignments (all commands can be run in parallel). Once all distances have been computed, one can aggregate the data and create the pairwise distance matrix `distance_matrix.csv` by running `python3 utils/aggregate-distances.py`.
