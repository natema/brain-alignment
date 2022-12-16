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

def standardizeVariable(var, n):
    var = var - np.mean(var)
    return np.divide(var, np.std(var))

def getDissMat(path):
    tgtDissMat = np.genfromtxt(path, dtype=float, skip_header=1, delimiter=",")
    tgtDissMat = tgtDissMat[0:,1:]
    (n,m) = np.shape(tgtDissMat)
    if n != m:
        sys.error("The target dissimilarity matrix is not a square matrix")
    return (tgtDissMat, n)

def getTgtVar(tgtDissMat, n):    
    (tgtVar, N) = myflatten(tgtDissMat, n)
    avg = np.mean(tgtVar)
    tgtVar = tgtVar - avg
    std = np.std(tgtVar)
    ## we also centralize the dissimilarity matrix and divide by std: it is useful for the permutation test
    return (np.divide(tgtVar, std), np.divide(tgtDissMat - avg, std), N) 

def getExpRegMat(paths, numExpVars, N):
    expRegMat = np.zeros((N, numExpVars), dtype=float)
    for i in range(numExpVars):
        (expDissMat, n) = getDissMat(paths[i])
        (expVar, N) = myflatten(expDissMat, n)
        expRegMat[:,i] = standardizeVariable(expVar, N)
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
    #coeffLst = np.zeros((numPerm,numExpVars))
    rsquareLst = np.zeros(numPerm)
    idMat = np.eye(n)
    for i in range(0, numPerm):
        permutation = idMat[np.random.permutation(range(0,n)),:]
        permTgtDissMat = permutation.T @ np.copy(tgtDissMat) @ permutation

        (tgtVar, N) = myflatten(permTgtDissMat, n) # already standardized

        regCoeffs = coeffMat @ tgtVar
        #coeffLst[i,:] = regCoeffs
        rsquareLst[i] = computeRsquare(tgtVar, expRegMat, regCoeffs, N)
    return rsquareLst

def pvalueComputation(rsquareLst, n):
    rsquare = rsquareLst[0]
    rsquareLst = np.sort(rsquaresLst)
    index = np.searchsorted(rsquareLst,rsquare)
    return 2*min((index+1)/n, 1-index/n)

def saveRegOutput(regCoeffs, rsquare, pvalue, numExpVars):
    mystr = "Regression results\nY ="
    for i in range(numExpVars):
        mystr = mystr + f" {regCoeffs[i]}*X_{i} +"
    mystr = mystr[0:-2] + "\n"
    mystr = mystr + f"rsquare = {rsquare}\np-value = {pvalue}"
    with open("regression-result/regression-output.txt", "w") as f:
        f.write(mystr)
    return

def pltCorr(Y, Xmat, n):
    for i in range(n):
        X = Xmat[:,i]
        p = plt.scatter(X,Y)
        plt.xlabel(f"explanatory variable {i+1}")
        plt.ylabel("target variable")
        plt.savefig(f"regression-result/plt-x{i+1}-y.png")
        plt.close()
    return

def addSuffix(args):
    suffix = ".csv"
    args.target = args.target + suffix
    args.explanatory = [x + suffix for x in args.explanatory]
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multivariate Linear Regression")
    parser.add_argument("-t", "--target", type=str, metavar="Y", required=True, help="Target dissimilarity matrix in csv format: path without extension")
    parser.add_argument("-e", "--explanatory", metavar="X", type=str, nargs="+", help="Explanatory dissimilarity matrix in csv format: path without extension")
    parser.add_argument("-p", "--permutations", metavar="P", type=int, required=True, help="Number of permutations for the permutation test" )
    args = parser.parse_args()

    args = addSuffix(args)

    (tgtDissMat, n) = getDissMat(args.target)
    (tgtVar, tgtDissMat, N) = getTgtVar(tgtDissMat, n) # unfolds and standardizes the matrix

    numExpVars = len(args.explanatory)
    expRegMat = getExpRegMat(args.explanatory, numExpVars, N) # unfolds and standardizes the matrices

    coeffMat = np.linalg.inv(expRegMat.T @ expRegMat) @ expRegMat.T
    regCoeffs = coeffMat @ tgtVar

    rsquaresLst = np.zeros(args.permutations+1)
    rsquaresLst[0] = computeRsquare(tgtVar, expRegMat, regCoeffs, N)
    rsquaresLst[1:] = permTest(tgtDissMat, expRegMat, coeffMat, args.permutations, n, N)
    pvalue = pvalueComputation(rsquaresLst, args.permutations+1)

    print(pvalue)

    os.makedirs("regression-result", exist_ok=True)
    saveRegOutput(regCoeffs, rsquaresLst[0], pvalue, numExpVars)
    pltCorr(tgtVar, expRegMat, numExpVars)