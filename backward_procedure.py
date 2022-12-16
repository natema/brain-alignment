# backward elimination procedure

#%%

import numpy as np 
import random as rand


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

for file in ["NaiveMatchingCHAlignment_pruned_pruned_pruned_merged.csv"]:
#for file in ['Euclid_BallNormAlign_Human.csv','StrEdge_plus_Euclid_BallNormAlign_Human.csv','2StrEdge_plus_Euclid_BallNormAlign_Human.csv','StrEdge_plus_2Euclid_BallNormAlign_Human.csv']:
    print(file)
    tgtvarinput = file

    tgtvar = np.genfromtxt(tgtvarinput, dtype = float, skip_header=1, delimiter=",")
    tgtvar = tgtvar[:,1:]
    #tgtvar = np.divide(tgtvar,np.linalg.norm(tgtvar))

    n = (np.shape(tgtvar))[0]
    # comment out if not needed
    #tgtvar = cubicRoot(tgtvar,n)

    
    toglie=True
    whichones = [1,2,3,5,9]
    while toglie==True:

        numexp = len(whichones)

        expvarinput = "./expvar_jumpchain"

        newlength = n**2 - n

        expmat = np.zeros((newlength, numexp), dtype = float)

        #expmat[:,0] = np.divide(np.ones(newlength),np.linalg.norm(np.ones(newlength)))


        for i in range(0,numexp):
            
            expvar = np.genfromtxt(expvarinput+"{}.csv".format(whichones[i]), dtype = float, skip_header = 1,delimiter=',')
            expvar = expvar[:,1:]
            #expvar = np.divide(expvar, np.linalg.norm(expvar))
            expvarflat = myflatten(expvar)
            expvarflat = expvarflat - expvarflat.mean()
            expvarflat = np.divide(expvarflat,np.std(expvarflat))
            expmat[:,i] = np.copy(expvarflat)



        myprod = np.linalg.inv(expmat.T @ expmat)

        ptoremove = 0.05/numexp
        
        


        tgtvarflat = myflatten(tgtvar)
        meantgtvarflat = tgtvarflat.mean()
        tgtvarflat = tgtvarflat - meantgtvarflat

        stdtgtvarflat = np.std(tgtvarflat)
        tgtvarflat = np.divide(tgtvarflat,stdtgtvarflat)
        #print(np.sort(tgtvarflat))
        tgtvar = tgtvar - meantgtvarflat
        tgtvar = np.divide(tgtvar, stdtgtvarflat)


        regcoeff = myprod @ expmat.T @ tgtvarflat

        regpred = expmat @ regcoeff

        if newlength != len(tgtvarflat):
            print("ERROR")

        RSS = 0

        for j in range(0,newlength):
            RSS += (tgtvarflat[j]-regpred[j])**2


        numperm = 10**3-1

        ftests = np.zeros((numperm+1,numexp+1))

        

        
        for i in range(0,numexp):
            cols = np.array(range(numexp))
            cols = np.delete(cols,i)

            newexpmat = expmat[:,cols]
            newprod = np.linalg.inv(newexpmat.T @ newexpmat)
            newregcoeff = newprod @ newexpmat.T @ tgtvarflat

            newregpred = newexpmat@newregcoeff

            rss = 0
            for j in range(0,newlength):
                rss += (tgtvarflat[j]-newregpred[j])**2
            ftests[0,i] = (rss - RSS)/(RSS / (n-numexp))

        


        myid = np.eye(n)

        
        for i in range(0,numexp):
            cols = np.array(range(0,numexp))
            cols = np.delete(cols,i)

            newexpmat = expmat[:,cols]
            newprod = np.linalg.inv(newexpmat.T @ newexpmat)

            for count in range(0, numperm):
                # if np.mod(count,100) == 0:
                #     print("{} - for expvar number {}\n".format(count,whichones[i]))
                permtgtvar = np.copy(tgtvar)
                #print("\npermutation number {}\n".format(count))
                myperm = myid[np.random.permutation(range(0,n)),:]
                permtgtvar = myperm @ permtgtvar @ myperm.T
                permtgtvarflat = myflatten(permtgtvar)
                #permtgtvarflat = np.divide(permtgtvarflat, np.linalg.norm(permtgtvarflat))

                permcoeff = myprod @ expmat.T @ permtgtvarflat
                permpred = expmat@permcoeff


                newpermcoeff = newprod @ newexpmat.T @ permtgtvarflat
                newpermpred = newexpmat@newpermcoeff


                rss = 0
                RSS = 0
                for j in range(0,newlength):
                    RSS += (permtgtvarflat[j] - permpred[j])**2
                    rss += (permtgtvarflat[j]-newpermpred[j])**2
                ftests[count+1,i] = (rss - RSS)/(RSS / (n-numexp))

        pvalues = np.zeros(numexp)

        # i = 0
        # ftest = ftests[0,i]
        # ftests[:,i] = np.sort(ftests[:,i])
        # myindex = np.searchsorted(ftests[:,i],ftest)
        # print("The F-test is {}".format(ftest))
        # print("\nintersect F-test p-value is {}\n".format(np.min([1-myindex/numperm,myindex/numperm])))

        numperm += 1
        ftests_pvalues = []
        for i in range(0, numexp):
            ftest = ftests[0,i]
            ftests[:,i] = np.sort(ftests[:,i])
            myindex = np.searchsorted(ftests[:,i],ftest)+1
            print("The F-test is {}".format(ftest))
            ftests_pvalues += [[whichones[i],2*np.min([1-(myindex-1)/numperm,myindex/numperm])]]
            print("\n{}-th F-test p-value is {}\n".format(whichones[i],ftests_pvalues[i][1]))


        print("\nFtests p-values are, in order: ", ftests_pvalues)
        print("\nreg. coeff.s are {}\n".format(regcoeff))
        toglie=False
        pvalues=[]
        numero=[]
        for k in range(0,len(ftests_pvalues)):
            pvalues.append(ftests_pvalues[k][1])
            numero.append(ftests_pvalues[k][0])
        if max(pvalues)>0.05:
            whichones.remove(numero[pvalues.index(max(pvalues))])
            toglie=True
        count = 0
        for x in regcoeff:
            if x < 0 and numero[count] in whichones:
                whichones.remove(numero[count])
                toglie=True
            count += 1



# %%

