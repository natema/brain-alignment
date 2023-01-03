#%%

import numpy as np
import sys 
import argparse
import matplotlib.pyplot as plt
import os 

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

def rmNumbers(mystr):
    newstr = ''
    for x in mystr:
        if not x.isdigit():
            newstr = newstr + x
    return newstr

def getSpecies(animals, n):
    species = []
    count = 0
    for x in animals:
        tmp = rmNumbers(x)
        if tmp not in species:
            species = species + [tmp]
            count = count + 1
    return (species, count)

def getGlobalAvg(dissMat, n):
    if n == 1:
        return 0
    return np.triu(dissMat,k=1).sum() / (n*(n-1)/2)

def countAnimals(animals, animal, j, n): # counting animals within a species
    count = 1
    for i in range(j+1, n):
        tmp = rmNumbers(animals[i])
        if tmp != animal:
            return (count, j+1)
        else:
            j = j + 1
            count = count + 1
    return (count, j+1)

def compareSpeciesAvg(dissMat, speciesDissMat, count, j, globalAvg, n): # it always leaves at least one animal per species
    withinSpeciesAvg = np.zeros(count)
    toKeep = []
    if count == 1:
        return (withinSpeciesAvg, [j])
    for i in range(count):
        withinSpeciesAvg[i] = np.mean(speciesDissMat[i,list(range(i)) + list(range(i+1,count))])
        if withinSpeciesAvg[i] <= globalAvg:
            toKeep = toKeep + [j+i]
    if len(toKeep) == 0:
        argmins = np.where(withinSpeciesAvg == np.min(withinSpeciesAvg))[0]
        numMins = len(argmins)
        if numMins == 1:
            toKeep = [argmins[0] + j] 
        else:
            speciesGlobalAvgs = np.zeros(numMins)
            for i in range(numMins):
                speciesGlobalAvgs[i] = np.mean(dissMat[argmins[i]+j,list(range(argmins[i]+j)) + list(range(argmins[i]+1+j,n))])
                toKeep = [argmins[np.argmin(speciesGlobalAvgs)] + j]
    return (withinSpeciesAvg, toKeep)

def pruneMat(dissMat, animals, n, species, N, folder):
    toPrune = True
    pruningRound = 1
    while toPrune:
        toKeep = []
        globalAvg = getGlobalAvg(dissMat, n)
        withinSpeciesAvg = np.zeros(n)
        j = 0
        for i in range(N):
            tmp = rmNumbers(animals[j])
            if tmp != species[i]:
                raise TypeError("Mixing different species")
            (count, newj) = countAnimals(animals, tmp, j, n)
            speciesDissMat = dissMat[j:newj, :]
            speciesDissMat = speciesDissMat[:,j:newj]
            (withinSpeciesAvg[j:newj], indeces) = compareSpeciesAvg(dissMat, speciesDissMat, count, j, globalAvg, n)
            toKeep = toKeep + indeces
            j = newj
            if j > n-1:
                break
        pltDissWithinSpecies(animals, withinSpeciesAvg, n, globalAvg, pruningRound, folder)
        dissMat = dissMat[toKeep, :]
        dissMat = dissMat[:, toKeep]
        animals = animals[toKeep]
        newn = len(toKeep)
        if newn == n:
            toPrune = False
        else:
            n = newn
        pruningRound = pruningRound + 1
    return (dissMat, animals, n) 

def pltDissWithinSpecies(animals, withinSpeciesAvg, n, globalAvg, pruningRound, folder): # plots only species with more than an individual
    toKeep = []
    for i in range(n):
        if withinSpeciesAvg[i] > 0:
            toKeep = toKeep + [i]
    animals = animals[toKeep]
    withinSpeciesAvg = withinSpeciesAvg[toKeep]
    n = len(toKeep)


    fig = plt.figure(1, figsize=(n/7,max(withinSpeciesAvg.tolist() + [globalAvg])/10))
    myplt = fig.add_subplot(111)
    myplt.plot()

    xaxis = 10*np.array(list(range(1,n+1)))
    myplt.plot(xaxis, globalAvg * np.ones(n), label = "global average")
    myplt.plot(xaxis, withinSpeciesAvg, label = "average within animal species")
    myplt.xaxis.set_ticks(xaxis)
    myplt.xaxis.set_ticklabels(animals)
    plt.xlabel("animal", labelpad=20)
    plt.xticks(rotation = 90)
    plt.margins(x=0)
    plt.rcParams["figure.autolayout"] = True
    plt.savefig(f"./{folder}/pruning-round-{pruningRound}.png")
    plt.close()

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
    parser = argparse.ArgumentParser(description="Script that prunes 'bad' animals from alignment matrices")
    parser.add_argument("-t", "--target", type=str, metavar="Y", required=True, help="Target alignment matrix in csv format: name of the file without extension. The file must be in the folder 'alignment-matrices'")
    args = parser.parse_args()

    args = addSuffix(args)
    prefix = "alignment-matrices/" 
    (dissMat, n) = getDissMat(prefix+args.target)
    animals = getAnimals(prefix+args.target)

    (species, N) = getSpecies(animals, n)

    folder = "pruning-results"
    os.makedirs(folder, exist_ok=True)

    (dissMat, animals, n) = pruneMat(dissMat, animals, n, species, N, folder)
    saveDissMat("pruned-"+args.target, dissMat, animals, n, folder)
