# Brain Alignment Methods

Implementation of methods to compute various brain similarity measures across animal species.

It requires `python3.7+` as well as `numpy, scipy`.

## Random Data Generation

Run `python3 utils/generate-random-data.py` to generate random data to play with the alignment.

The script will generate 5 random brains in `brains-space/`and `brains-matrix/`as well as a list of animals `animal_list.txt`.

## Brain-Alignment Computation

The alignment will consider as dissimilarity the linear combination $A \cdot Euclidean + B \cdot StrongestEdge$, for $A,B \in \mathbb{R^+}$. Execute the following command for all the options:

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
* `-a`, the coefficient of Euclidean dissimilarity (default: 1); 
* `-b`, the coefficient of StrongestEdge dissimilarity (default: 1); 
* `-s`, use option to save the alignment to file (default: alignment not saved to file).

The output is saved in `dissimilarity/{animal1}_vs_{animal2}.pickle` (dissimilarity) and `alignment/{animal1}_vs_{animal2}.pickle` (alignment, optionally). The data is serialized using Pickle.

In order to generate the pairwise dissimilarity matrix, one must execute all pairwise dissimilarities. By running `python3 utils/generate-parallel-dissimilarity.py` a file `parallel_dissimilarity.txt` is created. The file contains the list of commands to be run to compute the pairwise alignments (all commands can be run in parallel). Once all dissimilarities have been computed, one can aggregate the data and create the pairwise dissimilarity matrix `alignment-matrices/dissimilarity-matrix.csv` by running `python3 utils/aggregate-dissimilarity.py`.

## Pruning alignment dissimilarity matrix

If `dissimilarity_matrix.csv` is a dissimilarity matrix obtained by the alignment process and is saved in the folder `alignment-matrices`, run:

```python3 prune.py -t dissimilarity_matrix```.

The output is saved in the folder `pruning-results` and consists in a new dissimilarity matrix `pruned-dissimilarity-matrix.csv`, which is the result of a pruning procedure, and plots `pruning-round-1.png`, `pruning-round-2.png`, etc., showing which brains are 'bad' in each of the pruning iteration.

Execute the following command for a summary of the script functionalities:

```python3 prune.py -h```

Code example:

```python3 prune.py -t y-raw```

## Merging species in pruned dissimilarity matrices

If `pruned-dissimilarity_matrix.csv` is a pruned dissimilarity matrix obtained by the pruning process and is saved in the folder `pruning-results`, run:

```python3 meerge-species.py -t pruned-dissimilarity_matrix```

The output is saved in the folder `merging-results` and consists in a new dissimilarity matrix `merged-pruned-dissimilarity-matrix.csv`, which is the result of a merging procedure: it merges different individuals of the same species into an 'average' individual representing the species.

Execute the following command for a summary of the script functionalities:

```python3 merge-species.py -h```

Code example (after having performed the pruning code example):

```python3 merge-species.py -t pruned-y-raw```

## Backward elimination procedure

After the merging procedure, the backward elimination procedure selects which of the explanatory matrices (stored in the folder `explanatory-matrices`) are significant for the regression. 

Execute the following command to list all options:

```python3 backward-procedure.py -h```

The script requires the following flags:

* `-t`, the target dissimilarity matrix without the extension, which must be saved in the folder `target-matrices`;
* `-e`, the explanatory dissimilarity matrix (possibly, more than  one) without the extension, saved in the folder `explanatory-matrices`.

Moreover, the script accepts the following optional flags:

* `-p`, number of permutations in the permutation test (default: 999);
* `-r`, p-to-remove value for the permutation test (default: 0.05).

Example:

```python3 backward-procedure.py -t y2 -e x1 x2 x3 x4 x5 -p 999 -r 0.05```

The output of the script is saved in `backward-procedure-results/output.txt`: it shows which explanatory variables are significant for the regression.

## Multivariate linear regression

Once identified the significant variables through the backward elimination procedure, the regression must be run. The target dissimilarity matrix `y.csv` must be saved in the folder `target-matrices`. Any explanatory dissimilarity matrix `x.csv` must be saved in the folder `explanatory-matrices`.

Execute the following command to list all options:

```python3 regression.py -h```

The script requires the following flags:

* `-t`, the target dissimilarity matrix without the extension, which must be saved in the folder `target-matrices`;
* `-e`, the explanatory dissimilarity matrix (possibly, more than  one) without the extension, saved in the folder `explanatory-matrices`.

Moreover, the script accepts the following optional flag:

* `-p`, number of permutations in the permutation test (default: 999).


Example:

```python3 regression.py -t y2 -e x1 x5 -p 999```

The output of the script is saved in `regression-results/regression-output.txt`: it shows the output of the multivariate linear regression. Furthermore, for each explanatory variable `x`, there will be a corresponding figure `regression-results/plt-x-y.png` which plots the target variable `y` against `x`.
