import numpy as np
import sys 
import argparse
import matplotlib.pyplot as plt
import os 

def myflatten(dissMat, n):
    N = n*(n-1)
    var = np.zeros(N, dtype=float)
    for i in range(n):
        for j in range(n):
            tmp = i*(n-1)
            var[tmp:tmp+n-1] = dissMat[i,list(range(i))+list(range(i+1,n))]
    return (var, N)

def centerVariable(var, avg):
    return var - avg

def normalizeVariable(var, c):
    return np.divide(var, c)

def getDissMat(path): # returns both matrix and unfolded matrix, standardized
    DissMat = np.genfromtxt(path, dtype=float, skip_header=1, delimiter=",")
    DissMat = DissMat[0:,1:]
    (n,m) = np.shape(DissMat)
    if n != m:
        raise TypeError("The target dissimilarity matrix is not a square matrix")
    (Var, N) = myflatten(DissMat, n)
    avg = np.mean(Var)
    std = np.std(Var)
    return (normalizeVariable(centerVariable(DissMat,avg),std), normalizeVariable(centerVariable(Var,avg),std), n, N)

def getExpRegMat(paths, numExpVars, N):
    expRegMat = np.zeros((N, numExpVars), dtype=float)
    for i in range(numExpVars):
        (expDissMat, expVar, n, N) = getDissMat(paths[i])
        expRegMat[:,i] = expVar
    return expRegMat

def computeRsquare(tgtVar, expRegMat, regCoeffs, N):
    tgtPred = expRegMat @ regCoeffs
    RSS = 0 # initializing sum of squared residuals
    TSS = 0 # initializing sum of total squares
    for i in range(N):
        RSS = RSS + (tgtPred[i] - tgtVar[i])**2
        TSS = TSS + tgtVar[i]**2 # the variable is standardized: it's avarage is 0
    return 1 - RSS/TSS

def permTest(tgtDissMat, expRegMat, coeffMat, numPerm, n, N):
    rsquareLst = np.zeros(numPerm)
    idMat = np.eye(n)
    tmp = int(np.sqrt(numPerm))
    print(f"Counting permutations modulo {tmp} = sqrt({numPerm})")
    for i in range(0, numPerm):
        if i%tmp==tmp-1:
            print(f"Permutation number {i+1}")
        permutation = idMat[np.random.permutation(range(0,n)),:]
        permTgtDissMat = permutation.T @ tgtDissMat @ permutation
        (tgtVar, N) = myflatten(permTgtDissMat, n) # already standardized
        regCoeffs = coeffMat @ tgtVar
        rsquareLst[i] = computeRsquare(tgtVar, expRegMat, regCoeffs, N)
    return rsquareLst

def pvalueComputation(rsquareLst, n):
    rsquare = rsquareLst[0]
    rsquareLst = np.sort(rsquaresLst)
    index = np.searchsorted(rsquareLst,rsquare)
    return 2*min((index+1)/n, 1-index/n)

def saveRegOutput(regCoeffs, rsquare, pvalue, numExpVars, explInputs, folder):
    mystr = "Regression results\ny ="
    for i in range(numExpVars):
        mystr = mystr + f" {regCoeffs[i]}*{explInputs[i][21:-4]} +"
    mystr = mystr[0:-2] + "\n"
    mystr = mystr + f"rsquare = {rsquare}\np-value = {pvalue}"
    print(mystr)
    with open(f"{folder}/regression-output.txt", "w") as f:
        f.write(mystr)
    print(f"Output saved in folder '{folder}'")
    return

def pltCorr(Y, Xmat, n, explInputs, folder):
    for i in range(n):
        X = Xmat[:,i]
        p = plt.scatter(X,Y)
        plt.xlabel(f"explanatory variable {explInputs[i][21:-4]}")
        plt.ylabel("target variable")
        plt.savefig(f"{folder}/plt-{explInputs[i][21:-4]}-y.png")
        plt.close()
    return

def addSuffix(args):
    suffix = ".csv"
    args.target = args.target + suffix
    args.explanatory = [x + suffix for x in args.explanatory]
    return args

def addPrefix(args):
    tgtPrefix = "target-matrices/"
    explPrefix = "explanatory-matrices/"
    args.target = tgtPrefix + args.target
    args.explanatory = [explPrefix + x for x in args.explanatory]
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multivariate Linear Regression")
    parser.add_argument("-t", "--target", type=str, metavar="Y", required=True, help="Target dissimilarity matrix in csv format: name of the file without extension. The file must be in the folder 'target-matrices'")
    parser.add_argument("-e", "--explanatory", metavar="X", type=str, nargs="+", help="Explanatory dissimilarity matrix in csv format: name of the file without extension. The file must be in the folder 'explanatory-matrices'")
    parser.add_argument("-p", "--permutations", metavar="P", type=int, default=999, help="Number of permutations for the permutation test" )
    args = parser.parse_args()

    print("Running regression with permutation test")

    args = addSuffix(args)
    args = addPrefix(args)

    (tgtDissMat, tgtVar, n, N) = getDissMat(args.target)

    numExpVars = len(args.explanatory)
    expRegMat = getExpRegMat(args.explanatory, numExpVars, N) # unfolds and standardizes the matrices

    coeffMat = np.linalg.inv(expRegMat.T @ expRegMat) @ expRegMat.T
    regCoeffs = coeffMat @ tgtVar

    rsquaresLst = np.zeros(args.permutations+1)
    rsquaresLst[0] = computeRsquare(tgtVar, expRegMat, regCoeffs, N)
    rsquaresLst[1:] = permTest(tgtDissMat, expRegMat, coeffMat, args.permutations, n, N)
    pvalue = pvalueComputation(rsquaresLst, args.permutations+1)

    folder = "regression-results"
    os.makedirs(folder, exist_ok=True)
    saveRegOutput(regCoeffs, rsquaresLst[0], pvalue, numExpVars, args.explanatory, folder)
    pltCorr(tgtVar, expRegMat, numExpVars, args.explanatory, folder)