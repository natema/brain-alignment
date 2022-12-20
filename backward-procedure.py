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
        sys.error("The target dissimilarity matrix is not a square matrix")
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

def rmNegExpVar(expRegMat, regCoeffs, numExpVars):
    indeces = []
    control = False
    for i in range(numExpVars):
        if regCoeffs[i] >= 0:
            indeces += [i]
        else:
            control = True
    return (expRegMat[:,indeces], indeces, control)

def computeRestrictedMats(expRegMat, numExpVars):
    restrictedCoeffMats = []
    restrcitedExpRegMats = []
    for i in range(0,numExpVars):
        cols = list(range(i)) + list(range(i+1,numExpVars))
        restrcitedExpRegMats = restrcitedExpRegMats + [expRegMat[:,cols]]
        restrictedCoeffMats = restrictedCoeffMats + [np.linalg.inv(restrcitedExpRegMats[i].T @ restrcitedExpRegMats[i]) @ restrcitedExpRegMats[i].T]
    return (restrictedCoeffMats, restrcitedExpRegMats)

def computeFtests(tgtVar, tgtPred, restrictedExpMats, restrictedCoeffMats, N, numExpVars):
    RSS = 0 # initializing sum of squared residuals
    for i in range(N):
        RSS = RSS + (tgtPred[i] - tgtVar[i])**2
    ftests = np.zeros(numExpVars)
    for i in range(numExpVars):
        regCoeffs = restrictedCoeffMats[i] @ tgtVar
        tgtPred = restrictedExpMats[i] @ regCoeffs
        restrictedRSS = 0
        for j in range(0,N):
            restrictedRSS = restrictedRSS + (tgtPred[j] - tgtVar[j])**2
        ftests[i] = (restrictedRSS - RSS)/(RSS/(N - numExpVars))
    return ftests

def permFtests(tgtDissMat, expRegMat, coeffMats, restrictedExpMats, restrictedCoeffMats, numPerm, n, N, numExpVars):
    #coeffLst = np.zeros((numPerm,numExpVars))
    ftestLst = np.zeros((numPerm,numExpVars))
    idMat = np.eye(n)
    tmp = int(np.sqrt(numPerm))
    print(f"Counting permutation modulo {tmp}")
    for i in range(0, numPerm):
        if i%tmp==tmp-1:
            print(f"Permutation number {i+1}")
        permutation = idMat[np.random.permutation(range(0,n)),:]
        permTgtDissMat = permutation.T @ tgtDissMat @ permutation
        (tgtVar, N) = myflatten(permTgtDissMat, n) # already standardized
        regCoeffs = coeffMats @ tgtVar
        tgtPred = expRegMat @ regCoeffs
        #coeffLst[i,:] = regCoeffs
        ftestLst[i,:] = computeFtests(tgtVar, tgtPred, restrictedExpMats, restrictedCoeffMats, N, numExpVars)
    return ftestLst

def pvalueComputation(ftestLst, ftests, numPerm, numExpVars):
    ftestLst[0,:] = ftests
    indeces = np.zeros(numExpVars, dtype=int)
    pvalues = np.zeros(numExpVars)
    for i in range(numExpVars):
        ftestLst[:,i] = np.sort(ftestLst[:,i])
        indeces[i] = np.searchsorted(ftestLst[:,i], ftests[i])
        pvalues[i] = 2*min((indeces[i]+1)/(numPerm+1), 1-indeces[i]/(numPerm+1))
    return pvalues

def pvaluesCheck(pvalues, numExpVars, pToRemove):
    end = True
    if numExpVars == 0:
        return ([], end)
    index = 0
    tmpMax = pvalues[index]
    for i in range(1,numExpVars):
        tmp = pvalues[i]
        if tmpMax < tmp:
            tmpMax = tmp
            index = i
    if tmpMax > pToRemove:
        end = False
        indeces = list(range(index)) + list(range(index + 1, numExpVars))
    else:
        indeces = list(range(numExpVars))
    return (indeces, end)

def saveBackwardProcOutput(mystr):
    with open("backward-procedure-results/output.txt", "w") as f:
        f.write(mystr)
    return

def addSuffix(args):
    suffix = ".csv"
    args.target = args.target + suffix
    args.explanatory = [x + suffix for x in args.explanatory]
    return args

def initOutput(numPerm, pToRemove):
    mystr = f"Backward elimination procedure results\nNumber of permutations {numPerm}\nP-to-remove {pToRemove}\n\n" # this string keeps track of the Backward El. procedure results and will be saved as the final output
    print(mystr)
    return mystr

def addCoeffOutput(regCoeffs, numExpVars, ftestRound, explInputs):
    mystr = f"Elimination round {ftestRound}: y ="
    for i in range(numExpVars):
        mystr = mystr + f" {regCoeffs[i]}*{explInputs[i][0:-4]} +"
    mystr = mystr[0:-2] + "\n"
    print(mystr)
    return mystr

def explRemOutput(regCoeffs, numExpVars, explInputs):
    mystr = "Removing negative correlations: y ="
    for i in range(numExpVars):
        mystr = mystr + f" {regCoeffs[i]}*{explInputs[i][0:-4]} +"
    mystr = mystr[0:-2] + "\n"
    print(mystr)
    return mystr

def addFtestsOutput(regCoeffs, ftests, pvalues, numExpVars, explInputs, indeces, end):
    mystr = "F-tests results:\n"
    for i in range(numExpVars):
        mystr = mystr + f"Explanatory variable {explInputs[i][0:-4]}\t Coefficient {regCoeffs[i]}\tF-test {ftests[i]}\tp-value {pvalues[i]}\n"
    if end:
        mystr = mystr + "Completed: all remaining variables to keep\n"
    else:
        mystr = mystr + "To keep: "
        for x in indeces:
            mystr = mystr + f"{explInputs[x][0:-4]} "
        mystr = mystr + "\n"
    mystr = mystr + "\n"
    print(mystr)
    return mystr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backward Elimination Procedure")
    parser.add_argument("-t", "--target", type=str, metavar="Y", required=True, help="Target dissimilarity matrix in csv format: path without extension")
    parser.add_argument("-e", "--explanatory", metavar="X", type=str, nargs="+", help="Explanatory dissimilarity matrix in csv format: path without extension")
    parser.add_argument("-p", "--permutations", metavar="P", type=int, required=True, help="Number of permutations for the permutation test" )
    parser.add_argument("-r", "--ptoremove", metavar="R", type=float, required=True, help="P-to-remove value for the backward elimination procedure" )
    
    args = parser.parse_args()

    mystr = initOutput(args.permutations, args.ptoremove)

    args = addSuffix(args)
    (tgtDissMat, tgtVar, n, N) = getDissMat(args.target)

    explInputs = args.explanatory

    numExpVars = len(explInputs)
    expRegMat = getExpRegMat(explInputs, numExpVars, N) # unfolds and standardizes the matrices
    
    end = False
    ftestRound = 1

    while not end:

        if numExpVars == 0:
            end = True
            mystr = mystr + "Completed: no significant explanatory variables left\n\n"
            continue


        coeffMat = np.linalg.inv(expRegMat.T @ expRegMat) @ expRegMat.T
        regCoeffs = coeffMat @ tgtVar
        mystr = mystr + addCoeffOutput(regCoeffs, numExpVars, ftestRound, explInputs)

        (expRegMat, indeces, control) = rmNegExpVar(expRegMat, regCoeffs, numExpVars)

        while control == True:
            explInputs = [explInputs[i] for i in indeces]
            coeffMat = np.linalg.inv(expRegMat.T @ expRegMat) @ expRegMat.T
            regCoeffs = coeffMat @ tgtVar
            numExpVars = len(regCoeffs)
            mystr = mystr + explRemOutput(regCoeffs, numExpVars, explInputs)
            (expRegMat, indeces, control) = rmNegExpVar(expRegMat, regCoeffs, numExpVars)
        
        if numExpVars == 0:
            end = True
            mystr = mystr + "All variables deleted because of negative correlations\n\n"
            continue

        tgtPred = expRegMat @ regCoeffs
        
        (restrictedCoeffMats, restrictedExpRegMats) = computeRestrictedMats(expRegMat, numExpVars)
        ftests = computeFtests(tgtVar, tgtPred, restrictedExpRegMats, restrictedCoeffMats, N, numExpVars)

        ftestLst = np.zeros((args.permutations+1, numExpVars))
        ftestLst[1:,:] = permFtests(tgtDissMat, expRegMat, coeffMat, restrictedExpRegMats, restrictedCoeffMats, args.permutations, n, N, numExpVars)
        pvalues = pvalueComputation(ftestLst, ftests, args.permutations, numExpVars)
        
        (indeces, end) = pvaluesCheck(pvalues, numExpVars, args.ptoremove)
        mystr = mystr + addFtestsOutput(regCoeffs, ftestLst[0, :], pvalues, numExpVars, explInputs, indeces, end)

        explInputs = [explInputs[i] for i in indeces]
        expRegMat = expRegMat[:, indeces]
        numExpVars = len(indeces)

        ftestRound = ftestRound + 1

    os.makedirs("backward-procedure-results", exist_ok=True)
    saveBackwardProcOutput(mystr)

