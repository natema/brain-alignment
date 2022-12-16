# R^2 test

#%%

import numpy as np 
import random as rand
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

def myflatten(distMat):
    (a,b) = np.shape(distMat)
    newmat = []
    for i in range(a):
        newline = []
        for j in range(b):
            if i != j:
                newline += [distMat[i,j]]
        newmat += [newline]
    newmat = np.array(newmat)
    return newmat.flatten()

#%% 

tgtvarinput = './NaiveMatchingCHAlignment_pruned_pruned_pruned_merged.csv'

tgtvar = np.genfromtxt(tgtvarinput, dtype = float, skip_header=1,delimiter=',')
tgtvar = np.copy(tgtvar[0:,1:])

n = (np.shape(tgtvar))[0]

whichones = [1]
numexp = len(whichones)

expvarinput = "./expvar"

newlength = int(n*(n-1))

expmat = np.zeros((newlength, numexp), dtype = float)

for i in range(0,numexp):
    
    expvar = np.genfromtxt(expvarinput+"_jumpchain{}.csv".format(whichones[i]), dtype = float, skip_header = 1,delimiter=',')
    expvar = np.copy(expvar[0:,1:])
    # if i == 10:
    #     for k in range(np.shape(expvar)[0]):
    #         for h in range(np.shape(expvar)[0]):
    #             expvar[k,h] = expvar[k,h]**(1/3)
    #expvar = np.divide(expvar, np.linalg.norm(expvar))
    expvarflat=np.copy(myflatten(expvar))
    expvarflat = expvarflat - expvarflat.mean()
    expvarflat = np.divide(expvarflat,np.std(expvarflat))
    expmat[:,i] = np.copy(expvarflat)




myprod = np.linalg.inv(expmat.T @ expmat)
#%%
tgtvarflat = np.copy(myflatten(tgtvar))
meantgtvarflat = tgtvarflat.mean()
tgtvarflat = tgtvarflat - meantgtvarflat

stdtgtvarflat = np.std(tgtvarflat)
tgtvarflat = np.divide(tgtvarflat,stdtgtvarflat)
#print(np.sort(tgtvarflat))
tgtvar = tgtvar - meantgtvarflat
tgtvar = np.divide(tgtvar, stdtgtvarflat)

#%%
# W = defaultdict(int)
# for (x, y) in zip(expmat[:,0], tgtvarflat):
#     W[(x, y)] += 1
# cols = [W[(x, y)] for (x, y) in zip(expmat[:,0], tgtvarflat)]

# mycmap = cm.get_cmap("coolwarm")
# plt.scatter(expmat[:,0], tgtvarflat, c=cols, cmap=mycmap)
# plt.colorbar()
# plt.savefig("./is_linear.png")
#%%

#print(tgtvarflat, expmat[:,0])

regcoeff = myprod @ expmat.T @ tgtvarflat

print("\nReg. coefficients are {}".format(regcoeff))

regpred = expmat @ regcoeff

newlength = len(tgtvarflat)

#print("\nnewlength = {}\n".format(newlength))


myaverage = 0
for j in range(0,newlength):
    myaverage += tgtvarflat[j]
myaverage /= newlength

RSS = 0
SStot = 0

for j in range(0,newlength):
    RSS += (regpred[j] - tgtvarflat[j])**2
    SStot += (tgtvarflat[j]-myaverage)**2


numperm = 10**3

Rsquares = np.zeros((numperm+1))
Rsquares[0] = 1 - RSS / SStot

print("R^2 = {}\naverage = {}\nRSS = {}\nSStot = {}\n".format(Rsquares[0],myaverage,RSS,SStot))

print("mantel test with first expvar yields {}".format(np.dot(tgtvarflat,expmat[:,0])/(np.linalg.norm(tgtvarflat)*np.linalg.norm(expmat[:,0]))))

# for i in range(0,newlength):
#     print(tgtvarflat[i])

permcoeffs = np.zeros((numperm+1,numexp))


myid = np.eye(n)

for count in range(0, numperm):
    # if np.mod(count,100) == 0:
    #         print("{}\n".format(count))
    permtgtvar = np.copy(tgtvar)
    #print("\npermutation number {}\n".format(count))
    myperm = myid[np.random.permutation(range(0,n)),:]
    permtgtvar = myperm.T @ permtgtvar @ myperm

    
    permtgtvarflat = np.copy(myflatten(permtgtvar))
    #print(np.sort(permtgtvarflat))
    #print(np.linalg.norm(permtgtvarflat))


    #permtgtvarflat = np.divide(permtgtvarflat, np.linalg.norm(permtgtvarflat))
    #print(np.sort(permtgtvarflat))


    permcoeff = myprod @ expmat.T @ permtgtvarflat


    permcoeffs[count+1,:] = np.copy(permcoeff)

    permpred = expmat @ permcoeff

    RSS = 0
    #SStot = 0
    #for j in range(0,newlength):
    #    myaverage += permtgtvarflat[j]
    #myaverage /= newlength
    for j in range(0,newlength):
        RSS += ( permpred[j] - permtgtvarflat[j])**2
        #SStot += (permtgtvarflat[j] - myaverage)**2

    #print(SStot)

    Rsquares[count+1] =1- RSS/SStot

Rsquare = Rsquares[0]
sortedRsquares = np.sort(Rsquares)

print("\nR^2 stat = {}; its p-value is {}\n".format(Rsquare,np.min([1 - (np.searchsorted(sortedRsquares,Rsquare))/numperm,(np.searchsorted(sortedRsquares,Rsquare))/numperm])+1/numperm))
# %%

# %%

# %%
