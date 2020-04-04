from cvxopt import matrix, solvers
from math import exp
import matplotlib.pyplot as plt
from numpy import arange, array


def rbfKernel(v1, v2, sigma2=0.25):
    assert len(v1) == len(v2)
    assert sigma2 >= 0.0
    mag2 = sum(map(lambda x, y: (x - y) * (x - y), v1, v2))  ## Squared mag of diff.
    return exp(-mag2 / (2.0 * sigma2))


def makeLambdas(Xs, Ts, K=rbfKernel, C=1.0):
    "Solve constrained maximaization problem and return list of l's."
    P = makeP(Xs, Ts, K)  ## Build the P matrix.
    ## print(P)                   ## Debugging display of P matrix.
    n = len(Ts)
    q = matrix(-1.0, (n, 1))  ## This builds an n-element column
    ## vector of -1.0's (note the double-
    ## precision constant).
    h = matrix(0.0, (2 * n, 1))  ## 2n-element column vector of zeros.
    for i in range(n):  ## Set values from index n to end
        h[i + n] = C  ## equal to the softness parameter

    G = matrix(0.0, (2 * n, n))  ## These lines generate G, a
    # G[::(n+1)] = -1.0            ## 2n x n matrix with -1.0's on its
    for i in range(2 * n):  ## first diagonal and 1.0 on its second diagonal.
        for j in range(n):
            if i < n:
                if j == i:
                    G[i, j] = -1.0
            if i >= n:
                if j == (i - n):
                    G[i, j] = 1.0

    A = matrix(Ts, (1, n), tc='d')  ## A is an n-element row vector of
    ## training outputs.

    ##
    ## Now call "qp". Details of the parameters to the call can be
    ## found in the online cvxopt documentation.
    ##
    r = solvers.qp(P, q, G, h, A, matrix(0.0))  ## "qp" returns a dict, r.
    ##
    ## print(r)                    ## Dump entire result dictionary
    ##                             ## to terminal.
    ##
    ## Return results. Return a tuple, (Status,Ls).  First element is
    ## a string, which will be "optimal" if a solution has been found.
    ## The second element is a list of Lagrange multipliers for the problem,
    ## rounded to six decimal digits to remove algorithm noise.
    ##
    Ls = [round(l, 6) for l in list(r['x'])]  ## "L's" are under the 'x' key.
    return (r['status'], Ls)


def makeB(Xs, Ts, Ls=None, K=rbfKernel, C=1.0):
    "Generate the bias given Xs, Ts and (optionally) Ls and K"
    ## Note 0.0 bias parameter in call to setup (can't setup bias
    ## within bias setup routine, would lead to infinite regress.
    Ls, dummyB = setupMultipliersAndBias(Xs, Ts, Ls, 0.0, K, C)
    ## Calculate bias as average over all support vectors (non-zero
    ## Lagrange multipliers).
    sv_count = 0
    b_sum = 0.0
    for n in range(len(Ts)):
        if Ls[n] >= 1e-10:  ## 1e-10 for numerical stability.
            sv_count += 1
            b_sum += Ts[n]
            for i in range(len(Ts)):
                if Ls[i] >= 1e-10:
                    b_sum -= Ls[i] * Ts[i] * K(Xs[i], Xs[n])

    return b_sum / sv_count


def classify(x, Xs, Ts, Ls=None, b=None, K=rbfKernel, verbose=True, C=1.0):
    "Classify an input x into {-1,+1} given support vectors, outputs and L."
    Ls, b = setupMultipliersAndBias(Xs, Ts, Ls, b, K, C)
    ## Do classification.  y is the "activation level".
    y = activation(x, Xs, Ts, Ls, b, K)
    if verbose:
        print("{} {:8.5f}  -->".format(x, y), end=' ')
        if y > 0.0:
            print("+1")
        elif y < 0.0:
            print("-1")
        else:
            print("0  (ERROR)")
    if y > 0.0:
        return +1
    elif y < 0.0:
        return -1
    else:
        return 0


def testClassifier(Xs, Ts, test_x=None, test_t=None, do_Test=False, Ls=None, b=None, K=rbfKernel, verbose=True, C=1.0):
    "Test a classifier specifed by Lagrange mults, bias and kernel on all Xs/Ts pairs."
    assert len(Xs) == len(Ts)
    Ls, b = setupMultipliersAndBias(Xs, Ts, Ls, b, K, C)
    ## Do classification test.
    good = True
    train_MisClass = 0
    test_MisClass = 0
    for i in range(len(Xs)):
        c = classify(Xs[i], Xs, Ts, Ls, b, K, verbose, C)
        if c != Ts[i]:
            if verbose:
                print("Misclassification: input {}, output {:d}, "
                      "expected {:d}".format(Xs[i], c, Ts[i]))
            train_MisClass += 1
    if train_MisClass >= int(len(Ts) / 2.0):
        good = False

    print('Training status is: ' + str(good))

    if do_Test and good and test_x != None and test_t != None:
        for i in range(len(test_x)):
            test_C = classify(test_x[i], Xs, Ts, Ls, b, K, verbose, C)
            if test_C != test_t[i]:
                if verbose:
                    print("Misclassification: input {}, output {:d}, "
                          "expected {:d}".format(Xs[i], c, Ts[i]))
                test_MisClass += 1
        if test_MisClass >= int(len(test_t) / 2.0):
            good = False
    return good, train_MisClass, test_MisClass


def plotContours(Xs, Ts, Ls=None, b=None, K=rbfKernel, labelContours=False, labelPoints=False, minRange=-0.6,
                 maxRange=1.6, step=0.05, C=1.0, plotNewData=False, newX=None, newT=None):
    "Plot contours of activation function for a 2-d classifier, e.g. 2-input XOR."
    assert len(Xs) == len(Ts)
    assert len(Xs[0]) == 2  ## Only works with a 2-d classifier.
    Ls, b = setupMultipliersAndBias(Xs, Ts, Ls, b, K, C)
    ## Build activation level array.
    xs = arange(minRange, maxRange + step / 2.0, step)
    ys = arange(minRange, maxRange + step / 2.0, step)
    als = array([[activation([y, x], Xs, Ts, Ls, b, K) for y in ys] for x in xs])
    CS = plt.contour(xs, ys, als, levels=(-1.0, 0.0, 1.0), linewidths=(1, 2, 1), colors=('blue', '#40e040', 'red'))
    if ~plotNewData and newX is None and newT is None:
        for i, t in enumerate(Ts):
            if t < 0:
                col = 'blue'
            else:
                col = 'red'
            if labelPoints:
                ## print("Plotting %s (%d) as %s"%(Xs[i],t,col))
                plt.text(Xs[i][0] + 0.1, Xs[i][1], "%s: %d" % (Xs[i], t), color=col)
            plt.plot([Xs[i][0]], [Xs[i][1]], marker='o', color=col)
    elif plotNewData:
        for i, t in enumerate(newT):
            if t < 0:
                col = 'blue'
            else:
                col = 'red'
            if labelPoints:
                ## print("Plotting %s (%d) as %s"%(Xs[i],t,col))
                plt.text(newX[i][0] + 0.1, newX[i][1], "%s: %d" % (newX[i], t), color=col)
            plt.plot([newX[i][0]], [newX[i][1]], marker='o', color=col)

    ## Generate labels for contours if flag 'labelContours' is set to
    ## strings 'manual' or 'auto'.  Manual is manual labelling, auto is
    ## automatic labelling (which can mess up if hidden behind data
    ## points).
    if labelContours == 'manual':
        plt.clabel(CS, fontsize=9, manual=True)
    elif labelContours == 'auto':
        plt.clabel(CS, fontsize=9)
    plt.show()


def makeP(xs, ts, K):
    """Make the P matrix given the list of training vectors,
       desired outputs and kernel."""
    N = len(xs)
    assert N == len(ts)
    P = matrix(0.0, (N, N), tc='d')
    for i in range(N):
        for j in range(N):
            P[i, j] = ts[i] * ts[j] * K(xs[i], xs[j])
    return P


def activation(X, Xs, Ts, Ls, b, K):
    """Return activation level of a point X = [x1,x2,....] given
       training vectors, training (i.e., desired) outputs, Lagrange
       multipliers, bias and kernel."""
    y = b
    for i in range(len(Ts)):
        if Ls[i] >= 1e-10:
            y += Ls[i] * Ts[i] * K(Xs[i], X)
    return y


def setupMultipliersAndBias(Xs, Ts, Ls=None, b=None, K=rbfKernel, C=1.0):
    ## No Lagrange multipliers supplied, generate them.
    if Ls == None:
        status, Ls = makeLambdas(Xs, Ts, K, C)
        ## If Ls generation failed (non-seperable problem) throw exception
        if status != "optimal": raise Exception("Can't find Lambdas")
        print("Lagrange multipliers:", Ls)
    ## No bias supplied, generate it.
    if b == None:
        b = makeB(Xs, Ts, Ls, K, C)
        print("Bias:", b)
    return Ls, b


training_File = open('training-dataset-aut-2017.txt', 'r')
train = [line.split(',') for line in training_File.readlines()]
train_Vals, train_Xs, train_Ts = [[0]] * len(train), [[0, 0]] * len(train), [[0]] * len(train)

testing_File = open('testing-dataset-aut-2017.txt', 'r')
test = [line.split(',') for line in testing_File.readlines()]
test_Vals, test_Xs, test_Ts = [[0]] * len(test), [[0, 0]] * len(test), [[0]] * len(test)

for i in range(len(train)):
    train_Vals[i] = train[i][0].split()
    train_Xs[i] = train_Vals[i][0:2]
    train_Ts[i] = train_Vals[i][2]

for i in range(len(test)):
    test_Vals[i] = test[i][0].split()
    test_Xs[i] = test_Vals[i][0:2]
    test_Ts[i] = test_Vals[i][2]

for i in range(len(test)):
    test_Ts[i] = float(test_Ts[i])
    for j in range(2):
        test_Xs[i][j] = float(test_Xs[i][j])

for i in range(len(train)):
    train_Ts[i] = float(train_Ts[i])
    for j in range(2):
        train_Xs[i][j] = float(train_Xs[i][j])

""" Now the training and testing data have been saved as floats """

K = rbfKernel
title = ''
contours = True
minRange = -5
maxRange = 5
step = 0.05
C = 1.0

status, Ls = makeLambdas(train_Xs, train_Ts, K, C)
print("  Result status:", status)
print("  L vector:", Ls)

if status == "optimal":
    b = makeB(train_Xs, train_Ts, Ls, K, C)
    print("  bias:", b)
    trained, train_Misclass, test_Misclass = testClassifier(train_Xs, train_Ts, test_Xs, test_Ts, do_Test=True, Ls=Ls,
                                                            b=b, K=K, verbose=False, C=C)
    if trained:
        print("  Check PASSED")
        if contours:
            if title:
                t = title
            else:
                t = ""
            plt.figure(t, figsize=(6, 6))
            plotContours(train_Xs, train_Ts, Ls, b, K, False, False, minRange, maxRange, step, C)
            plotContours(train_Xs, train_Ts, Ls, b, K, False, False, minRange, maxRange, step, C, plotNewData=True,
                         newX=test_Xs,
                         newT=test_Ts)
    else:
        print("  Check FAILED: Classifier does not work correctly on inputs")
        if contours:
            if title:
                t = title
            else:
                t = ""
            plt.figure(t, figsize=(6, 6))
            plotContours(train_Xs, train_Ts, Ls, b, K, False, False, minRange, maxRange, step, C)
            plotContours(train_Xs, train_Ts, Ls, b, K, False, False, minRange, maxRange, step, C, plotNewData=True,
                         newX=test_Xs,
                         newT=test_Ts)
    print("\n\n")
    print("    Soft margin machine misclassified on training data " + str(train_Misclass) + " times\n")
    print("    Soft margin machine misclassified on testing data " + str(test_Misclass) + " times\n")
    print("\n\n")

C = 1000000.0

status, Ls = makeLambdas(train_Xs, train_Ts, K, C)
print("  Result status:", status)
print("  L vector:", Ls)

if status == "optimal":
    b = makeB(train_Xs, train_Ts, Ls, K, C)
    print("  bias:", b)
    trained, train_Misclass, test_Misclass = testClassifier(train_Xs, train_Ts, test_Xs, test_Ts, do_Test=True, Ls=Ls,
                                                            b=b, K=K, verbose=False, C=C)
    if trained:
        print("  Check PASSED")
        if contours:
            if title:
                t = title
            else:
                t = ""
            plt.figure(t, figsize=(6, 6))
            plotContours(train_Xs, train_Ts, Ls, b, K, False, False, minRange, maxRange, step, C)
            plotContours(train_Xs, train_Ts, Ls, b, K, False, False, minRange, maxRange, step, C, True, test_Xs,
                         test_Ts)
    else:
        print("  Check FAILED: Classifier does not work correctly on inputs")
        if contours:
            if title:
                t = title
            else:
                t = ""
            plt.figure(t, figsize=(6, 6))
            plotContours(train_Xs, train_Ts, Ls, b, K, False, False, minRange, maxRange, step, C)
            plotContours(train_Xs, train_Ts, Ls, b, K, False, False, minRange, maxRange, step, C, True, test_Xs,
                         test_Ts)
    print("\n\n")
    print("    Hard margin machine misclassified on training data " + str(train_Misclass) + " times\n")
    print("    Hard margin machine misclassified on testing data " + str(test_Misclass) + " times\n")
    print("\n\n")
