import argparse
import os
import pickle
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment



def getSpatialInfo(animal, n):
    M = np.loadtxt(os.path.join("brains-space", "{}.txt".format(animal)))
    M -= np.asarray([[np.mean(M[:,0]), np.mean(M[:,1]), np.mean(M[:,2])] for _ in range(n)])  # centering
    M *= (0.75/np.pi)**(1/3) / np.linalg.norm(max(M, key=np.linalg.norm))  # normalization
    return M


def getConnectivityInfo(animal):
    return np.loadtxt(os.path.join("brains-matrix", "{}.txt".format(animal)))


def rotateMatrix(M, alpha, beta, gamma):
    sinAlpha, cosAlpha = np.sin(np.radians(alpha)), np.cos(np.radians(alpha))
    sinBeta, cosBeta = np.sin(np.radians(beta)), np.cos(np.radians(beta))
    sinGamma, cosGamma = np.sin(np.radians(gamma)), np.cos(np.radians(gamma))
    A = np.asarray([
        [cosAlpha, sinAlpha, 0],
        [-sinAlpha, cosAlpha, 0],
        [0, 0, 1],
    ], dtype=np.float64).transpose()
    B = np.asarray([
        [cosBeta, 0, sinBeta],
        [0, 1, 0],
        [-sinBeta, 0, cosBeta],
    ], dtype=np.float64).transpose()
    C = np.asarray([
        [1, 0, 0],
        [0, cosGamma, sinGamma],
        [0, -sinGamma, cosGamma],
    ], dtype=np.float64).transpose()
    return (M @ A @ B @ C)


def computeAlignmentDist(Xs, Ys, Xc, Yc, n, a, b):
    C = np.ndarray((n, n))
    for (i, u) in enumerate(Xs):
        for (j, v) in enumerate(Ys):
            if j < i:
                continue

            # euclidean distance
            euclidean = np.linalg.norm(u - v)

            # strongest edge
            i_prime, j_prime = np.argmax(Xc[i]), np.argmax(Yc[j])
            u_prime, v_prime = Xs[i_prime], Ys[j_prime]
            strongest_edge = np.linalg.norm(u_prime - v_prime)

            # linear combination
            distance_ij = a*euclidean + b*strongest_edge
            C[i, j] = distance_ij
            C[j, i] = distance_ij

    (row_ind, col_ind) = linear_sum_assignment(C)
    alignment = list(zip(row_ind, col_ind))
    distance = C[row_ind, col_ind].sum()

    return (alignment, distance)


def computeBestCanonicalRotation(Xs, Ys, Xc, Yc, args):
    (alignment, distance) = computeAlignmentDist(Xs, Ys, Xc, Yc, args.numberpoints, args.coeffA, args.coeffB)
    (alpha_star, beta_star, gamma_star) = (0, 0, 0)
    for alpha in range(0, 360, args.degree):
        for beta in range(0, 360, args.degree):
            for gamma in range(0, 360, args.degree):
                Ws = rotateMatrix(Ys, alpha, beta, gamma)
                (alignment_new, distance_new) = computeAlignmentDist(Xs, Ws, Xc, Yc, args.numberpoints, args.coeffA, args.coeffB)
                if distance_new < distance:
                    distance = distance_new
                    (alpha_star, beta_star, gamma_star) = (alpha, beta, gamma)
    Ys = rotateMatrix(Ys, alpha_star, beta_star, gamma_star)
    return Ys


def computeBestSmallRotation(Xs, Ys, Xc, Yc, args):
    (alignment, distance) = computeAlignmentDist(Xs, Ys, Xc, Yc, args.numberpoints, args.coeffA, args.coeffB)
    for _ in range(args.rotations):
        (alpha, beta, gamma) = (np.random.normal(0, 1), np.random.normal(0, 1), np.random.normal(0, 1))
        Ws = rotateMatrix(Ys, alpha, beta, gamma)
        (alignment_new, distance_new) = computeAlignmentDist(Xs, Ws, Xc, Yc, args.numberpoints, args.coeffA, args.coeffB)
        if distance_new < distance:
            (alignment, distance) = (alignment_new, distance_new)
            Ys = Ws
    return Ys


def computeBrainAlignment(Xs, Ys, Xc, Yc, args):
    Ys = computeBestCanonicalRotation(Xs, Ys, Xc, Yc, args)
    Ys = computeBestSmallRotation(Xs, Ys, Xc, Yc, args)
    (alignment, distance) = computeAlignmentDist(Xs, Ys, Xc, Yc, args.numberpoints, args.coeffA, args.coeffB)
    return (alignment, distance)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for Brain Alignment")
    parser.add_argument("-a1", "--animal1", type=str, metavar="", required=True, help="Input animal 1")
    parser.add_argument("-a2", "--animal2", type=str, metavar="", required=True, help="Input animal 2")
    parser.add_argument("-n", "--numberpoints", type=int, default=200, metavar="", help="Number of points for a brain")
    parser.add_argument("-d", "--degree", type=int, default=90, metavar="", help="Degrees of canonical rotations")
    parser.add_argument("-r", "--rotations", type=int, default=150, metavar="", help="Number of random rotations")
    parser.add_argument("-a", "--coeffA", type=float, default=1, metavar="", help="Coefficient A (Euclidean) in the dissimilarity computation")
    parser.add_argument("-b", "--coeffB", type=float, default=1, metavar="", help="Coefficient B (StrongestEdge) in the dissimilarity computation")
    parser.add_argument("-s", "--saveAlignment", action="store_true", help="Save alignment to file")
    args = parser.parse_args()

    # load data
    (Xs, Xc) = (getSpatialInfo(args.animal1, args.numberpoints), getConnectivityInfo(args.animal1))
    (Ys, Yc) = (getSpatialInfo(args.animal2, args.numberpoints), getConnectivityInfo(args.animal2))
    
    # compute alignment
    (alignment, distance) = computeBrainAlignment(Xs, Ys, Xc, Yc, args)
    
    # write results to file
    os.makedirs("dissimilarity", exist_ok=True)
    with open(os.path.join("dissimilarity", f"{args.animal1}_vs_{args.animal2}.pickle"), "wb") as handle:
        pickle.dump(distance, handle)
    if args.saveAlignment:
        os.makedirs("alignment", exist_ok=True)
        with open(os.path.join("alignment", f"{args.animal1}_vs_{args.animal2}.pickle"), "wb") as handle:
            pickle.dump(alignment, handle)
