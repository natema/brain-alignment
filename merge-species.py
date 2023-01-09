# Correction of our distance matrix: merge of more individuals of the same species

import numpy as np
import sys 
import argparse
import os 


#%%

def addSuffix(args):
    suffix = ".csv"
    args.target = args.target + suffix
    return args

def getDissMat(path): # returns both matrix and unfolded matrix, standardized
    dissMat = np.genfromtxt(path, dtype=float, skip_header=1, delimiter=",")
    dissMat = dissMat[0:,1:]
    (n,m) = np.shape(dissMat)
    if n != m:
        sys.error("The target dissimilarity matrix is not a square matrix")
    return (dissMat, n)

def getAnimals(path): 
    animals = np.genfromtxt(path, dtype = str, usecols=0, skip_header=1, delimiter=",")
    return animals

def getAnimalMap(path):
    map = np.genfromtxt(path, dtype=str, delimiter=",", skip_header=1, usecols=(0,1))
    return map

def rmNumbers(mystr):
    newstr = ''
    for x in mystr:
        if not x.isdigit():
            newstr = newstr + x
    return newstr

def searchNextAnimal(animals, i, n):
    animal = rmNumbers(animals[i])
    while i+1<n:
        tmp = rmNumbers(animals[i+1])
        if tmp != animal:
            return i+1
        else:
            i = i+1
    return i+1

def mergeAnimalRaws(dissMat, n, firstIndex, nextIndex):
    newRaw = np.zeros(n)
    for i in range(n):
        newRaw[i] = np.mean(dissMat[range(firstIndex,nextIndex),i])
    return newRaw

def mergeAnimalColumns(dissMat, n, firstIndex, nextIndex):
    newCol = np.zeros(n)
    for i in range(n):
        newCol[i] = np.mean(dissMat[i,range(firstIndex,nextIndex)])
    return newCol

def mergeDissMat(dissMat, n, animals):
    i = 0
    speciesIndeces = [i]
    while i < n:
        i = searchNextAnimal(animals, i, n)
        speciesIndeces = speciesIndeces + [i]
    N = len(speciesIndeces) - 1
    mergedDissMatRaws = np.zeros((N,n))
    for i in range(N):
        mergedDissMatRaws[i,:] = mergeAnimalRaws(dissMat, n, speciesIndeces[i], speciesIndeces[i+1])
    mergedDissMatColumns = np.zeros((N,N))
    for i in range(N):
        mergedDissMatColumns[:,i] = mergeAnimalColumns(mergedDissMatRaws, N, speciesIndeces[i], speciesIndeces[i+1])
    for i in range(N):
        mergedDissMatColumns[i,i] = 0
    return (mergedDissMatColumns, N, speciesIndeces[0:N])

def deleteUnclassifiedSpecies(dissMat, n, animals, animalIndeces, animalMap):
    species = []
    speciesIndeces = []
    for i in range(n):
        tmp = rmNumbers(animals[animalIndeces[i]])
        speciesIndex = np.where(animalMap[:,0] == tmp)[0]
        if len(speciesIndex) == 0:
            continue
        else:
            species = species + [animalMap[speciesIndex[0],1]]
            speciesIndeces = speciesIndeces + [i]
    dissMat = dissMat[speciesIndeces, :]
    dissMat = dissMat[:, speciesIndeces]
    return (dissMat, len(species), species)

def saveDissMat(path, dissMat, animals, n, folder):
    mystr = f"{n}"
    for x in animals:
        mystr = mystr + f",{x}"
    mystr = mystr + "\n"
    for i in range(n):
        mystr = mystr + f"{animals[i]}"
        for j in range(n):
            mystr = mystr + f",{dissMat[i,j]}"
        mystr = mystr + "\n"
    with open(f"{folder}/{path}", "w") as f:
        f.write(mystr)
    print(f"Output saved in folder '{folder}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that merges different animals of the same species into a single 'individual' by averaging")
    parser.add_argument("-t", "--target", type=str, metavar="Y", required=True, help="Target dissimilarity matrix in csv format: name of the file without extension. The file must be in the folder 'pruning-results'")
    args = parser.parse_args()
    args = addSuffix(args)
    prefix = "pruning-results/"
    (dissMat, n) = getDissMat(prefix+args.target)
    animals = getAnimals(prefix+args.target)
    animalMap = getAnimalMap("./utils/map-of-animals.csv")
    (dissMat, n, animalIndeces) = mergeDissMat(dissMat, n, animals)
    (dissMat, n, species) = deleteUnclassifiedSpecies(dissMat, n, animals, animalIndeces, animalMap)
    folder = "merging-results"
    os.makedirs(folder, exist_ok=True)
    saveDissMat("merged-"+args.target, dissMat, species, n, folder)