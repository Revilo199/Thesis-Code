import numpy as np
from typing import Iterator, List, Tuple, Any
import math
from os.path import isfile
import time
from numpy import *
from PIL import Image, ImageDraw, ImageFont

Point = Tuple[int, int]


def rref(mat,precision=0,GJ=False):
    m,n = mat.shape
    p,t = precision, 1e-1**precision
    A = around(mat.astype(float).copy(),decimals=p )
    if GJ:
        A = hstack((A,identity(n)))
    pcol = -1 #pivot colum
    for i in range(m):
        pcol += 1
        if pcol >= n : break
        #pivot index
        pid = argmax( abs(A[i:,pcol]) )
        #Row exchange
        A[i,:],A[pid+i,:] = A[pid+i,:].copy(),A[i,:].copy()
        #pivot with given precision
        while pcol < n and abs(A[i,pcol]) < t:
            pcol += 1
            if pcol >= n : break
            #pivot index
            pid = argmax( abs(A[i:,pcol]) )
            #Row exchange
            A[i,:],A[pid+i,:] = A[pid+i,:].copy(),A[i,:].copy()
        if pcol >= n : break
        pivot = float(A[i,pcol])
        for j in range(m):
            if j == i: continue
            mul = float(A[j,pcol])/pivot
            A[j,:] = around(A[j,:] - A[i,:]*mul,decimals=p)
        A[i,:] /= pivot
        A[i,:] = around(A[i,:],decimals=p)
        
    if GJ:
        return A[:,:n].copy(),A[:,n:].copy()
    else:
        return A


def rref2(matrix):
    A = np.array(matrix, dtype=np.float64)
    #print(A)
    i = 0 # row
    j = 0 # column
    while True:
        # find next nonzero column
        while all(A.T[j] == 0.0):
            j += 1
            # if reached the end, break
            if j == len(A[0]) - 1 : break
        # if a_ij == 0 find first row i_>=i with a 
        # nonzero entry in column j and swap rows i and i_
        if A[i][j] == 0:
            i_ = i
            while A[i_][j] == 0:
                i_ += 1
                # if reached the end, break
                if i_ == len(A) - 1 : break
            A[[i, i_]] = A[[i_, i]]
        # divide ith row a_ij to make it a_ij == 1
        A[i] = A[i] / A[i][j]
        # eliminate all other entries in the jth column by subtracting
        # multiples of of the ith row from the others
        for i_ in range(len(A)):
            if i_ != i:
                A[i_] = A[i_] - A[i] * A[i_][j] / A[i][j]
        # if reached the end, break
        if (i == len(A) - 1) or (j == len(A[0]) - 1): break
        # otherwise, we continue
        i += 1
        j += 1
    return(A)

def compair(a, b):
    for pointnumber in range(len(a)):
        if a[pointnumber][1] < b[pointnumber][1]:
            return True
        if a[pointnumber][1] > b[pointnumber][1]:
            return False
        if a[pointnumber][0] < b[pointnumber][0]:
            return True
        if a[pointnumber][0] > b[pointnumber][0]:
            return False
    return False


def location(element, sett):
    # print(element,sett)
    for spot in range(len(sett)):
        # print(spot,len(sett))
        # print(element,sett[spot])
        if compair(element, sett[spot]):
            newset = []
            for i in range(len(sett)):
                if i == spot:
                    newset.append(element)
                newset.append(sett[i])
            return newset
    sett.append(element)
    return sett


def orderset(set):
    output = []
    for i in range(len(set)):
        if output == []:
            output.append(set[i])
        else:
            output = location(set[i], output)
    return output


def printmatnice(mat):
    for i in range(len(mat)):
        line = []
        if rownumber:
            line.append(str(i))
            line.append(":")
            if i < 10:
                line.append("  [")
            else:
                line.append(" [")
        for j in range(len(mat[i])):
            # if mat[i][j]>=0:
            # line.append(" ")
            if mat[i][j] == 0:
                line.append(zerochar)
            elif mat[i][j] == 1:
                line.append(onechar)
            elif mat[i][j] == 2:
                line.append(twochar)
            elif mat[i][j] == -1 or mat[i][j]==255:
                line.append(negonechar)
            else:
                line.append(str(int(mat[i][j])))
            if j + 1 < len(mat[i]):
                line.append(" ")
        if rownumber:
            line.append("]")
        print("".join(line))


def modsizedetector(el):
    leftest = "none"
    rightest = "none"
    upest = "none"
    lowest = "none"
    for point in el:
        if leftest == "none":
            leftest = point[0]
            rightest = point[0]
            upest = point[1]
            lowest = point[1]
        else:
            if point[0] > rightest:
                rightest = point[0]
            if point[0] < leftest:
                leftest = point[0]
            if point[1] > upest:
                upest = point[1]
            if point[1] < lowest:
                lowest = point[1]
    width = rightest - leftest + 1
    hight = upest - lowest + 1
    return (width, hight, leftest, rightest, upest, lowest)


def printmatwithlines(mat, fin, var=False, rerereturn=False, printit=True):
    output = []
    if var:
        sizechange = []
        numberchange = []
        for i in range(len(fin) - 1):
            cors1 = modsizedetector(fin[i])
            cors2 = modsizedetector(fin[i + 1])
            if (
                cors1[2] != cors2[2]
                or cors1[3] != cors2[3]
                or cors1[4] != cors2[4]
                or cors1[5] != cors2[5]
            ):
                sizechange.append(i + 1)
    else:
        numberchange = []
        sizechange = []
        for i in range(len(fin) - 1):
            if len(fin[i]) != len(fin[i + 1]):
                numberchange.append(i + 1)
            cors1 = modsizedetector(fin[i])
            cors2 = modsizedetector(fin[i + 1])
            if cors1[0] != cors2[0] or cors1[1] != cors2[1]:
                sizechange.append(i + 1)
    for i in range(len(mat)):  # | or - is number change, * is size change, > is both
        if i in sizechange or i in numberchange:
            mlin = []
            for t in range(5):
                mlin.append(" ")
            mlin.append("[")
            for j in range(len(mat)):
                # mlin.append(" ")
                if j in numberchange or j in sizechange:
                    if propersplice:
                        mlin.append("X")
                    else:
                        mlin.append(splicecharacter)
                    mlin.append("  ")
                if i in numberchange:
                    if i in sizechange:
                        # >
                        if propersplice:
                            mlin.append(">")
                        else:
                            mlin.append(splicecharacter)
                    else:
                        # |
                        if propersplice:
                            mlin.append("-")
                        else:
                            mlin.append(splicecharacter)
                elif i in sizechange:
                    # *
                    if propersplice:
                        mlin.append("*")
                    else:
                        mlin.append(splicecharacter)
                if j + 1 < len(mat[i]):
                    mlin.append(" ")
            mlin.append("]")
            if printit:
                print("".join(mlin))
            if rerereturn:
                output.append("".join(mlin))
                output.append("\n")

        line = []
        line.append(str(i))
        line.append(":")
        if i < 10:
            line.append("   [")
        elif i < 100:
            line.append("  [")
        else:
            line.append(" [")
        for j in range(len(mat[i])):
            if j in numberchange:
                if j in sizechange:
                    # >
                    if propersplice:
                        line.append(" > ")
                    else:
                        line.append(" ")
                        line.append(splicecharacter)
                        line.append(" ")
                else:
                    # |
                    if propersplice:
                        line.append(" | ")
                    else:
                        line.append(" ")
                        line.append(splicecharacter)
                        line.append(" ")
            elif j in sizechange:
                # *
                if propersplice:
                    line.append(" * ")
                else:
                    line.append(" ")
                    line.append(splicecharacter)
                    line.append(" ")
            if mat[i][j] == 0:
                line.append(zerochar)
            elif mat[i][j] == 1:
                line.append(onechar)
            elif mat[i][j] == 2:
                line.append(twochar)
            elif mat[i][j] == -1 or mat[i][j]==255:
                line.append(negonechar)
            else:
                line.append(str(int(mat[i][j])))
            if j + 1 < len(mat[i]):
                line.append(" ")
        line.append("]")
        if printit:
            print("".join(line))
        if rerereturn:
            output.append("".join(line))
            output.append("\n")
    if rerereturn:
        return output


def printmatwithlinesvar(mat, fin, leftedge, rightedge, upedge, downedge):
    numberchange = []
    sizechange = []
    for i in range(len(fin) - 1):
        if len(fin[i]) != len(fin[i + 1]):
            numberchange.append(i + 1)
        cors1 = modsizedetector(fin[i])
        cors2 = modsizedetector(fin[i + 1])
        if cors1[0] != cors2[0] or cors1[1] != cors2[1]:
            sizechange.append(i + 1)
    print(numberchange)
    print(sizechange)
    for i in range(len(mat)):  # | or - is number change, * is size change, > is both
        if i >= upedge and i < downedge:
            if (i in sizechange or i in numberchange) and i != upedge:
                mlin = []
                for t in range(4):
                    mlin.append(" ")
                mlin.append("[")
                for j in range(len(mat)):
                    if j >= leftedge and j < rightedge:
                        # mlin.append(" ")
                        if j in numberchange or j in sizechange:
                            if propersplice:
                                mlin.append("X")
                            else:
                                mlin.append(splicecharacter)
                            mlin.append("  ")
                        if i in numberchange:
                            if i in sizechange:
                                # >
                                if propersplice:
                                    mlin.append(">")
                                else:
                                    mlin.append(splicecharacter)
                            else:
                                # |
                                if propersplice:
                                    mlin.append("-")
                                else:
                                    mlin.append(splicecharacter)
                        elif i in sizechange:
                            # *
                            if propersplice:
                                mlin.append("*")
                            else:
                                mlin.append(splicecharacter)
                        if j + 1 < len(mat[i]):
                            mlin.append(" ")
                mlin.append("]")
                print("".join(mlin))
            line = []
            line.append(str(i))
            line.append(":")
            if i < 10:
                line.append("  [")
            else:
                line.append(" [")
            for j in range(len(mat[i])):
                if j >= leftedge and j < rightedge:
                    if j == j:
                        if j in numberchange:
                            if j in sizechange:
                                # >
                                if propersplice:
                                    line.append(" > ")
                                else:
                                    line.append(" ")
                                    line.append(splicecharacter)
                                    line.append(" ")
                            else:
                                # |
                                if propersplice:
                                    line.append(" | ")
                                else:
                                    line.append(" ")
                                    line.append(splicecharacter)
                                    line.append(" ")
                        elif j in sizechange:
                            # *
                            if propersplice:
                                line.append(" * ")
                            else:
                                line.append(" ")
                                line.append(splicecharacter)
                                line.append(" ")
                    if mat[i][j] == 0:
                        line.append(zerochar)
                    elif mat[i][j] == 1:
                        line.append(onechar)
                    elif mat[i][j] == 2:
                        line.append(twochar)
                    elif mat[i][j] == -1:
                        line.append(negonechar)
                    else:
                        line.append(str(int(mat[i][j])))
                    if j + 1 < len(mat[i]):
                        line.append(" ")
            line.append("]")
            print("".join(line))


def inclusionbasedorder(set):
    neworder=[]
    for module in set:
        pos=0
        for orderedmodnum in range(len(neworder)):
            inside=True
            for poin in neworder[orderedmodnum]:
                if poin not in module:
                    inside=False
            if inside:
                pos=orderedmodnum+1
        neworder=neworder[0:pos]+[module]+neworder[pos:]
    return(neworder)



def reordermods(ordertype, fin, n, h):
    """Reorder the convex modules in convient ways. Possible orderings: point count(pc), bounding box (bb), min/max counts (mmc), number of intervals (intnum)"""
    reorder = []
    if ordertype == "pc":
        for i in range(1, n * h + 1):
            for el in fin:
                if len(el) == i:
                    reorder.append(el)
    if (
        ordertype == "bb"
    ):  # first ordered by point count, then ordered by bounding box size

        temp = []
        for i in range(1, n * h + 1):
            for el in fin:
                if len(el) == i:
                    temp.append(el)
        temp2 = []
        for i in range(n):
            for j in range(h):
                for el in temp:
                    cors = modsizedetector(el)
                    if cors[1] == j + 1 and cors[0] == i + 1:
                        temp2.append(el)
        return temp2
    if (
        ordertype == "to"
    ):  # same as bb but also in each group, forces them to be lined up by the bottom to the top
        for i in range(n):
            for j in range(h):
                for pointcount in range(1, n * h + 1):
                    # print(i,j,pointcount)
                    set = (
                        []
                    )  # set of modules with a sepcific amount of points and fixed bounding box
                    for el in fin:
                        cors = modsizedetector(el)
                        if (
                            cors[1] == j + 1
                            and cors[0] == i + 1
                            and len(el) == pointcount
                        ):
                            set.append(el)
                    if len(set) > 0:
                        # print("heres a set:",set)
                        ordered = orderset(set)
                        # print("now its ordered:", ordered)
                        for t in range(len(ordered)):
                            reorder.append(ordered[t])
    if (
        ordertype == "bbtwo"
    ):  # forcuses on not just the bounding box, but the location of the bounding box
        for j in range(h):
            for i in range(n):
                # print(i,j,pointcount)
                set = (
                    []
                )  # set of modules with a sepcific amount of points and fixed bounding box
                for el in fin:
                    cors = modsizedetector(el)
                    if cors[1] == j + 1 and cors[0] == i + 1:
                        set.append(el)
                if len(set) > 0:
                    # print("heres a set:",set)
                    ordered = orderset(set)
                    # print("now its ordered:", ordered)
                    for t in range(len(ordered)):
                        reorder.append(ordered[t])

    if ordertype=="awinclusion":
        for j in range(h):
            for i in range(n):
                set = []
                for el in fin:
                    cors = modsizedetector(el)
                    if cors[1] == j + 1 and cors[0] == i + 1:
                        set.append(el)
                if len(set) > 0:
                    ordered=inclusionbasedorder(set)
                    for t in range(len(ordered)):
                        reorder.append(ordered[t])


    if ordertype == "mmc":
        return fin
    if ordertype == "intnum":
        return fin
    if ordertype == "none":
        return fin
    return reorder


def dycke_paths(p1: Point, p2: Point) -> Iterator[List[Point]]:
    """Generate path from some upper left point to some down right point (credit to Alex for this)"""
    if p1 == p2:
        yield [p2]
        return
    if p1[0] < p2[0]:
        for path in dycke_paths((p1[0] + 1, p1[1]), p2):
            yield [p1] + path
    if p1[1] > p2[1]:
        for path in dycke_paths((p1[0], p1[1] - 1), p2):
            yield [p1] + path


def rank(iindex, jindex, fin, n, h):
    """Generate the rank of some specific combination of convex modules"""
    i = fin[iindex]
    j = fin[jindex]
    mins = []  # collection of the lowest elements of the poset
    maxs = []  # collection of the highest elements of the poset
    for test in range(len(i)):  # look through all the elements
        if [i[test][0] - 1, i[test][1]] in i or [
            i[test][0],
            i[test][1] - 1,
        ] in i:  # if theres a smaller element skip it
            pass
        else:
            mins.append(i[test])  # otherwise add it to minimals
        if [i[test][0] + 1, i[test][1]] in i or [
            i[test][0],
            i[test][1] + 1,
        ] in i:  # if theres a larger element skip it
            pass
        else:
            maxs.append(i[test])  # otherwise add it to maximums
    nl = len(mins)
    xl = len(maxs)
    mat = np.zeros((nl, xl))  # generate blank matrix
    for n in range(len(mins)):
        for x in range(len(maxs)):
            if (
                mins[n][0] <= maxs[x][0]
                and mins[n][1] <= maxs[x][1]
                and mins[n] in j
                and maxs[x] in j
            ):
                mat[n][
                    x
                ] = 1  # if theres a path from the min to the max change its apropriate spot in the matrix to 1
    rankk = np.linalg.matrix_rank(mat)  # calculate rank
    return rankk


def ranktwo(iindex, jindex, fin, n, h, mins, maxs):
    """Generate the rank of some specific combination of convex modules"""
    i = fin[iindex]
    j = fin[jindex]
    # mins = []  # collection of the lowest elements of the poset
    # maxs = []  # collection of the highest elements of the poset
    # for test in i:  # look through all the elements
    #     if [test[0] - 1, test[1]] in i or [
    #         test[0],
    #         test[1] - 1,
    #     ] in i:  # if theres a smaller element skip it
    #         pass
    #     else:
    #         mins.append(test)  # otherwise add it to minimals
    #     if [test[0] + 1, test[1]] in i or [
    #         test[0],
    #         test[1] + 1,
    #     ] in i:  # if theres a larger element skip it
    #         pass
    #     else:
    #         maxs.append(test)  # otherwise add it to maximums
    nl = len(mins)
    xl = len(maxs)
    mat = np.zeros((nl, xl))  # generate blank matrix
    for n in range(len(mins)):
        for x in range(len(maxs)):
            if (
                mins[n][0] <= maxs[x][0]
                and mins[n][1] <= maxs[x][1]
                and mins[n] in j
                and maxs[x] in j
            ):
                mat[n][
                    x
                ] = 1  # if theres a path from the min to the max change its apropriate spot in the matrix to 1
    rankk = np.linalg.matrix_rank(mat)  # calculate rank
    return rankk


def rowofmat(iindex, fin, n, h):
    i = fin[iindex]
    # print(fin)
    mins = []  # collection of the lowest elements of the poset
    maxs = []  # collection of the highest elements of the poset
    for test in i:  # look through all the elements
        if [test[0] - 1, test[1]] in i or [
            test[0],
            test[1] - 1,
        ] in i:  # if theres a smaller element skip it
            pass
        else:
            mins.append(test)  # otherwise add it to minimals
        if [test[0] + 1, test[1]] in i or [
            test[0],
            test[1] + 1,
        ] in i:  # if theres a larger element skip it
            pass
        else:
            maxs.append(test)  # otherwise add it to maximums
    row = []
    for j in range(len(fin)):
        row.append(ranktwo(iindex, j, fin, n, h, mins, maxs))
    return row


def createimage(points, width, hight):
    line1 = []
    for y in range(hight - 1, -1, -1):
        for x in range(width):
            if [x, y] in points:
                line1.append("X")
                if x < width:
                    if [x + 1, y] in points:
                        line1.append("----")
                    else:
                        line1.append("    ")
            else:
                if x < width:
                    line1.append("0    ")
                else:
                    line1.append("0")
        if y > 0:
            line1.append("\n")
            for x in range(width + 1):
                if [x, y] in points and [x, y - 1] in points:
                    line1.append("|")
                else:
                    line1.append(" ")
                if x < width:
                    line1.append("    ")
            line1.append("\n")
    return line1


def gennbyhfix(n, h):
    """Generate all convex mods that are exactly n wide and exacly h tall"""
    paths = []
    for path in dycke_paths((0, h), (n, 0)):
        # Generate all paths from the top left to bottom right of subrectangle
        paths.append(path)
    edgesleft = []
    edgesright = []
    for path in paths:
        templ = []
        tempr = []
        templ.append((0, h))
        for test in range(1, len(path)):
            if path[test - 1][1] - 1 == path[test][1]:
                templ.append(path[test])  # find the 'leftmost elements' of a path
            if path[test - 1 + 1][1] + 1 == path[test - 1][1]:
                tempr.append(path[test - 1])  # find the 'rightmost elements' of a path
        tempr.append((n, 0))
        # This is now a collection of just the left edges of the paths
        edgesright.append([tempr])
        # This is now a collection of just the right edges of the paths
        edgesleft.append([templ])
    fin = []
    for pathoutnum in range(len(paths)):
        for pathinnum in range(len(paths)):
            # Check every left path to every right path
            good = True
            for height in range(h + 1):
                for node in range(n + 1):
                    if (node, height) in paths[pathinnum] and node > edgesright[
                        pathoutnum
                    ][0][h - height][0]:
                        good = False  # check theres no parts of inner path thats past the outer path
                    if (node, height) in paths[pathoutnum] and edgesleft[pathinnum][0][
                        h - height
                    ][0] > node:
                        good = False  # same but vice versa
            if good:
                pointies = []
                for b in range(h + 1):
                    for a in range(n + 1):
                        if (
                            a <= edgesright[pathoutnum][0][h - b][0]
                            and a >= edgesleft[pathinnum][0][h - b][0]
                        ):
                            # Add all the points on the path edges and inbetween
                            pointies.append([a, b])
                fin.append(pointies)
    return fin  # return all the convex modules in the subrectangle


def get_fixconvexmod(n: int, h: int) -> Any:
    filenamecon = f"convex_{n}_{h}.mat"
    if isfile(filenamecon):
        with open(filenamecon, "rb") as file:
            return np.load(file)
    convexmod = gennbyhfix(n, h)
    with open(filenamecon, "wb") as file:
        np.save(file, convexmod)
    return convexmod


def genallnotworking(n, h):  # generate EVERY convex module
    mas = []

    h: int = h
    w: int = n

    # width to be the greater one
    if w < h:
        w, h = h, w

    for temp_h in range(1, h + 1):
        for temp_w in range(temp_h, w + 1):
            small = []
            print(
                f"Generating for subractangles size {temp_w} by {temp_h} out of {n} by {h}. "
            )
            # For every size of subrectangle, generate all the convex modules
            gend = gennbyhfix(temp_w - 1, (temp_h) - 1)
            for wider in range(n - temp_w + 1):
                # Find the amount to 'slide' the subrectangle around
                for higher in range(h - (temp_h) + 1):
                    for i in range(len(gend)):  # look at all the convex modules
                        temppoints = []
                        # Look at all the points in the convex module
                        for j in range(len(gend[i])):
                            # Slide them around by the apropriate amount
                            temppoints.append(
                                [gend[i][j][0] + wider, gend[i][j][1] + higher]
                            )
                        mas.append(temppoints)
            if temp_h != temp_w and temp_w <= h:
                for wider in range(n - temp_h + 1):
                    for higher in range(h - (temp_w) + 1):
                        for i in range(len(gend)):  # look at all the convex modules
                            temppointstwo = []
                            # Look at all the points in the convex module
                            for j in range(len(gend[i])):
                                # Slide them around by the apropriate amount
                                temppointstwo.append(
                                    [gend[i][j][1] + higher, gend[i][j][0] + wider]
                                )
                            mas.append(temppointstwo)
    return mas


def genall(n, h):  # generate EVERY convex module
    mas = []

    h: int = h
    w: int = n

    for temp_h in range(1, h + 1):
        for temp_w in range(1, w + 1):
            small = []
            if timeestimates:
                print(
                    f"Generating for subractangles size {temp_w} by {temp_h} out of {n} by {h}."
                )
            # For every size of subrectangle, generate all the convex modules
            gend = gennbyhfix(temp_w - 1, (temp_h) - 1)
            for wider in range(n - temp_w + 1):
                # Find the amount to 'slide' the subrectangle around
                for higher in range(h - (temp_h) + 1):
                    for i in range(len(gend)):  # look at all the convex modules
                        temppoints = []
                        # Look at all the points in the convex module
                        for j in range(len(gend[i])):
                            # Slide them around by the apropriate amount
                            temppoints.append(
                                [gend[i][j][0] + wider, gend[i][j][1] + higher]
                            )
                        mas.append(temppoints)
    return mas


def genmat(n, h, ordertype):  # generate the matrix
    fin = reordermods(ordertype, genall(n, h), n, h)
    matrix = np.zeros(
        (len(fin), len(fin)), dtype=np.ubyte
    )  # make the matrix (with all zeros)
    startstamp = time.time()
    lastwhole = 0
    for i in range(len(fin)):
        for j in range(len(fin)):
            if timeestimates:
                if 100 * i / len(fin) >= lastwhole:
                    percent = i / len(fin)
                    if percent == 0:
                        print(f"Generating ranks: {lastwhole}% of the way through!")
                    else:
                        snap = time.time()
                        timespent = snap - startstamp
                        percentleft = 1 - percent
                        total = timespent / percent * percentleft
                        print(
                            f"Generating ranks for {n} by {h}: {lastwhole}% of the way through! Time spent: {round(timespent)} seconds. Estimated time remaining: {round(total)} seconds left."
                        )
                    lastwhole = lastwhole + 1
            # Change the specific values to the apropriate value
            matrix[i][j] = rank(i, j, fin, n, h)
    matrix = matrix.T
    return [matrix, fin]


def get_matrix(n: int, h: int) -> Any:
    if h > n:
        t = h
        h = n
        n = t
    filename = f"matrix_{n}_{h}.mat"
    if isfile(filename):
        with open(filename, "rb") as file:
            return np.load(file)
    start = time.time()
    matrix = genmat(n, h,"bbtwo")[0]
    end = time.time()
    print(f"It took {round(end-start,3)}seconds to compute the {n} by {h} matrix!")
    with open(filename, "wb") as file:
        np.save(file, matrix)
    return matrix


def writeshittoafile(shit, filename):
    filename = f"{filename}.txt"
    if isfile(filename):
        with open(filename, "rb") as file:
            return np.load(file)
    with open(filename, "wb") as file:
        np.save(file, shit)


def checkrots(mat, fin):
    edge = [0]
    for i in range(len(fin) - 1):
        if len(fin[i]) != len(fin[i + 1]):
            edge.append(i + 1)
        else:
            cors1 = modsizedetector(fin[i])
            cors2 = modsizedetector(fin[i + 1])
            if cors1[0] != cors2[0] or cors1[1] != cors2[1]:
                edge.append(i + 1)
    edge.append(len(fin))
    metacorrect = True
    for x in range(len(edge)):
        for y in range(len(edge)):
            if edge[x] != len(fin) and edge[y] != len(fin):
                hormov = []
                vermov = []
                for t in range(len(fin)):
                    if t < edge[x] or t >= edge[x + 1]:
                        vermov.append(t)
                    if t < edge[y] or t >= edge[y + 1]:
                        hormov.append(t)
                new = np.delete(np.delete(mat, hormov, 0), vermov, 1)
                # printmatnice(new)
                correct = True
                size = new.shape
                # print(size)
                for inx in range(size[0]):
                    for iny in range(size[1]):
                        # print(inx,iny,size[0]-inx-1,size[1]-iny-1)
                        if (
                            new[inx][iny] != new[size[0] - inx - 1][size[1] - iny - 1]
                        ):  # 1,2
                            correct = False
                            printmatnice(new)
                            print(edge[x], edge[y])
                if correct:
                    # print("Rotates!")
                    pass
                else:
                    metacorrect = False
                    print("Doesn't Rotate :(")
                # print("")
    if metacorrect:
        print("everything rotates!")
    else:
        print("not everything rotates :(")


def subtract(mat,t,s,subt=1,flip=True):
    #print(t,s,flip)
    if flip:
        mat[:,t]=mat[:,t]-mat[:,s]*subt
    else:
        mat[t,:]=mat[t,:]-mat[s,:]*subt
    return(mat)


def multitracth(mat,target,listt):
    for sub in listt:
        #print(sub)
        #print(flip)
        #mat=subtract(mat,target,sub,flip)
        mat=subtract(mat,target,sub,flip=False)
    return(mat)


def multitractv(mat,target,listt):
    for sub in listt:
        #print(sub)
        #print(flip)
        #mat=subtract(mat,target,sub,flip)
        mat=subtract(mat,target,sub)
    return(mat)


def mult(mat,target,multiplier):
    mat[:,target]=mat[:,target]*multiplier
    return(mat)


def reduction(mat, n):
    mat=subtract(mat,14,6)
    mat=subtract(mat,23,6)
    mat=subtract(mat,15,7)
    mat=subtract(mat,17,7)
    mat=subtract(mat,25,7)
    mat=subtract(mat,18,8)
    mat=subtract(mat,26,8)
    mat=subtract(mat,14,9)
    mat=subtract(mat,15,10)
    mat=subtract(mat,25,10)
    mat=subtract(mat,17,11)
    mat=subtract(mat,25,11)
    mat=subtract(mat,18,12)
    mat=subtract(mat,14,13,-1)
    mat=subtract(mat,15,13,-1)
    mat=subtract(mat,22,13)
    mat=subtract(mat,25,13,-1)
    mat=subtract(mat,25,15)
    mat=subtract(mat,17,16,-1)
    mat=subtract(mat,18,16,-1)
    mat=subtract(mat,24,16)
    mat=subtract(mat,25,16,-1)
    mat=subtract(mat,25,17)
    mat=subtract(mat,22,19)
    mat=subtract(mat,23,19)
    mat=subtract(mat,24,20)
    mat=subtract(mat,26,20)
    return(mat)


def reduced(d,n):
    d=d.T
    t=0
    for i in range(0,n):
        d=subtract(d,2+i,0)
        d=subtract(d,2+t,1)
        t=t+n-i
    #0 2 5
    #1 7 11
    #2 
    #3 
    #4
    d=d.T
    return(d)


def calcdminuscainvb(mat,fin,w1,w2,w3):
    nums1 = []
    nums2 = []
    nums3 = []
    for i in range(w1):
        nums1.append(i)
    for i in range(w1, w2):
        nums2.append(i)
    for i in range(w2,w3):
        nums3.append(i)
    #print(nums1)
    #print(nums2)
    #print(nums3)
    A = np.delete(mat, nums2+nums3, 1)
    A = np.delete(A, nums2+nums3, 0)
    B = np.delete(mat, nums2+nums3, 0)
    B = np.delete(B, nums1+nums3, 1)
    C = np.delete(mat, nums2+nums3, 1)
    C = np.delete(C, nums1+nums3, 0)
    D = np.delete(mat, nums1+nums3, 1)
    D = np.delete(D, nums1+nums3, 0)
    #print("A")
    #printmatwithlines(A,fin,True,False,True)
    #print("B")
    #printmatnice(B)
    #print("C")
    #printmatnice(C)
    #print("D")
    #printmatnice(D)
    #print("A^-1")
    #printmatwithlines(np.linalg.inv(A),fin,True,True,True)
    #print("A^-1")
    #printmatnice(np.linalg.inv(A))
    print()
    #midtwo=np.matmul(C,np.linalg.inv(A))
    mid = np.matmul(np.linalg.inv(A), B)
    mul = np.matmul(C, mid)
    #print("A^-1B")
    #printmatnice(mid)
    #print("CA^-1")
    #printmatnice(midtwo)
    print("CA^-1B")
    printmatnice(mul)
    print("D")
    printmatnice(D)
    print("D-CA^-1B")
    printmatnice(np.subtract(D,mul))
    #print("det:", np.linalg.det(np.subtract(D,mul)))
    #print("")


def imredoingthisagain(mat,h):
    dist=0
    totalsize=2*(h**2+h)/2
    for i in range(1,h+1):
        totalsize=totalsize+(h+1-i)*(i**2+i)/2
    w2=int(totalsize)
    w1=int(totalsize-2-(h**2+h)/2)
    totalsize=int(totalsize)
    for a in range(2*h):
        prev=dist
        if a%2==0:
            b=a/2+1
            dist=dist+2*(h-b+1)
        if a%2==1:
            b=a/2+.5
            dist=dist+(h-b+1)*b*(b+1)/2
        print(a%2+1)
        print(b)
        calcdminuscainvb(mat,fin,int(prev),int(dist),totalsize)


def cordcalc(en,h):
    w=1
    a=1
    c=1
    e=1
    x=1
    while (w==1 and en>=2*(h-a+1)) or (w==2 and en>=(h-a+1)*(a**2/2+a/2)):
        if w==1:
            en=en-2*(h-a+1)
        if w==2:
            en=en-(h-a+1)*(a**2/2+a/2)
        w=w%2+1
        if w==1:
            a=a+1
    if w==1:
        while en>=2:
            en=en-2
            c=c+1
        while en>=1:
            en=en-1
            e=e+1
    if w==2:
        while en>=(a**2/2+a/2):
            en=en-a**2/2-a/2
            c=c+1
        while en>=a-e+1:
            en=en-(a-e+1)
            e=e+1
        while en>=1:
            en=en-1
            x=x+1
    return(w,a,c,e,x)


def validcoord(tw,ta,tc,te,tx,h):
    if tw!=1 and tw!=2:
        return(False)
    

def reversecordcalc(tw,ta,tc,te,tx,h):
    en=0
    w=1
    a=1
    c=1
    e=1
    x=1
    while w!=tw or a!=ta:
        if w==1:
            en=en+2*(h-a+1)
        if w==2:
            en=en+(h-a+1)*(a**2/2+a/2)
        w=w%2+1
        if w==1:
            a=a+1
        #print(en,w,a,c,e,x)
    #print('split')
    if w==1:
        while c!=tc:
            en=en+2
            c=c+1
            #print(en,w,a,c,e,x)
        while e!=te:
            en=en+1
            e=e+1
            #print(en,w,a,c,e,x)
    if w==2:
        while c!=tc:
            en=en+a**2/2+a/2
            c=c+1
            #print(en,w,a,c,e,x)
        while e!=te:
            en=en+(a-e+1)
            e=e+1
            #print(en,w,a,c,e,x)
        while x!=tx:
            en=en+1
            x=x+1
            #print(en,w,a,c,e,x)
    return(en)    


def thesimpleralgorithmfunction(mat,h,outputsubtractednumber=False):
    h=int(h)
    totalsize=2*(h**2+h)/2
    for i in range(1,h+1):
        totalsize=totalsize+(h+1-i)*(i**2+i)/2    
    bigL=[]
    for loc in range(int(totalsize)):
        cords=cordcalc(loc,h)
        v=cords[0]
        b=cords[1]
        d=cords[2]
        f=cords[3]
        y=cords[4]
        if v==1:
            typee=1
        elif v==2:
            if f==1:
                if y==1:
                    typee=2
                elif y>1:
                    typee=3
            if f>1:
                if y==1:
                    typee=4
                elif y>1:
                    typee=5
        if typee==1:
            l=[]
            modv=b*2+v
            for analysspot in range(3,modv):
                w=(analysspot-1)%2+1
                a=(analysspot-w)/2
                amount=b-a+1
                if w==1:
                    for part in range(int(amount)):
                        l.append(int(reversecordcalc(1,a,d+part,f,1,h)))
            if len(l)>0:
                mat=multitracth(mat,loc,l)
        if typee==2:
            l=[]
            modv=b*2+v
            for analysspot in range(3,modv):
                w=(analysspot-1)%2+1
                a=(analysspot-w)/2
                amount=(b-a+1)*((w)%2+1)
                if w==1:
                    for part in range(int(amount)):
                        l.append(int(reversecordcalc(1,a,d+math.floor(part/2),part%2+1,1,h))) #,d+math.floor(part/2),part%2+1,
                if w==2:
                    for part in range(int(amount)):
                        l.append(int(reversecordcalc(w,a,part+d,1,1,h)))
            mat=multitracth(mat,loc,l)
        if typee==3:
            l=[]
            modv=b*2+v
            for analysspot in range(3,modv):
                w=(analysspot-1)%2+1
                a=(analysspot-w)/2
                if w==1:
                    amount=(b-a+1)*2-1
                    for part in range(int(amount)):
                        if part%2==0 or (part%2==1 and part<=2*b-2*a-2*y+4):
                            l.append(int(reversecordcalc(1,a,d,1+part,1,h)))
                else:
                    amount=b+min(-y+1,-a+1,-1)
                    for part in range(int(amount)):
                        l.append(int(reversecordcalc(w,a,d+part,1,max(1,-b+y+part+a),h)))
            mat=multitracth(mat,loc,l)
        if typee==4:
            l=[]
            modv=b*2+v
            for analysspot in range(3,modv):
                w=(analysspot-1)%2+1
                a=(analysspot-w)/2
                if w==1:
                    amount=(b-a)*2+1
                    for part in range(int(amount)):
                        if part%2==0 or (part%2==1 and part>=2*f-3):
                            l.append(int(reversecordcalc(1,a,d+math.ceil((part)/2),1+(part+1)%2,1,h)))
                else:
                    amount=b-(max(a-1,f-1))-(max(y-a,0))
                    for part in range(int(amount)):
                        l.append(int(reversecordcalc(w,a,part+max(1,f-a+1)+d-1,max(min(a,f)-part,1),max(1,part+y+max(a,f)-b),h)))
            mat=multitracth(mat,loc,l)
        if typee==5:
            l=[]
            modv=b*2+v
            for analysspot in range(3,modv):
                w=(analysspot-1)%2+1
                a=(analysspot-w)/2
                if w==1:
                    amount=(b-a)*2
                    for part in range(int(amount)):
                        if (part%2==0 and part<=amount-2*y+3) or (part%2==1 and part>=2*f-3):
                            l.append(int(reversecordcalc(1,a,d+math.ceil((part)/2),1+(part+1)%2,1,h)))
                else:
                    amount=b-(max(a-1,f-1))-(max(y-a,0))
                    for part in range(int(amount)):
                        l.append(int(reversecordcalc(w,a,part+max(1,f-a+1)+d-1,max(min(a,f)-part,1),max(1,part+y+max(a,f)-b),h)))
            mat=multitracth(mat,loc,l)
        #print(loc,l)
        bigL.append([loc,l])
        #print(bigL)
    if outputsubtractednumber:
        #print(bigL)
        return(bigL)
    return(mat)


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def findbiandbc(points):
    ai=-234234234
    bi=-23452345
    ci=324523452345
    di=2489727498234
    for point in points:
        if point[0]==0 and point[1]<ci:
            ci=point[1]
        if point[0]==1 and point[1]>bi:
            bi=point[1]
        if point[0]==0 and point[1]>ai:
            ai=point[1]
        if point[0]==1 and point[1]<di:
            di=point[1]
    return(ai,bi,ci,di)


def equallist(list1,list2):
    for point in list1:
        if point not in list2:
            return(False)
    for point in list2:
        if point not in list1:
            return(False)
    return(True)


def parseintervals(module):
    pass


def thetruealgorithm(matrix,fin,n,h):
    h=int(h)
    n=int(n)
    totalsize=int(len(fin))    
    bigL=[]
    for j in range(int(len(fin))):
        subtractedrows=[]
        for k1low in range(h):
            for k0low in range(n):
                for k1high in range(h):
                    for k0high in range(n):
                        if k0low<=k0high and k1low<=k1high:
                            intersection=[]
                            for y in range(k1low,k1high+1):
                                for x in range(k0low,k0high+1):
                                    if [x,y] in fin[j]:
                                        intersection.append([x,y])
                            if intersection!=[]:
                                subtractedrows.append(intersection)
                            #print(intersection)
        #print(subtractedrows)
        unique=[]
        for sample in subtractedrows:
            if sample not in unique and sample!=fin[j]:
                unique.append(sample)
        
        
        subs=[]
        for row in unique:
            for k in range(int(len(fin))):
                #print(fin[k],'space',row)
                #if k==1 and row==[[1,0]]:
                    #print(fin[k])
                if fin[k]==row:
                    subs.append(k)
                    #print('called')
        #if j==209:
            #print(subs)
            #print(matrix[209][209])
#            for k in subs:
#                if matrix[k][209]!=0:
#                    print(matrix[k][209],k)
        matrix=multitracth(matrix,j,subs)
        #if j==209:
            #print(matrix[209][209])
        
        #mat=multitracth(mat,loc,l)
    return(matrix)

################################################

#                 _________________
#                /                /|
#               /                / |
#              /________________/ /|
#           ###|      ____      |//|
#          #   |     /   /|     |/.|
#         #  __|___ /   /.|     |  |_______________
#        #  /      /   //||     |  /              /|                  ___
#       #  /      /___// ||     | /              / |                 / \ \
#       # /______/!   || ||_____|/              /  |                /   \ \
#       #| . . .  !   || ||                    /  _________________/     \ \
#       #|  . .   !   || //      ________     /  /\________________  {   /  }
#       /|   .    !   ||//~~~~~~/   0000/    /  / / ______________  {   /  /
#      / |        !   |'/      /9  0000/    /  / / /             / {   /  /
#     / #\________!___|/      /9  0000/    /  / / /_____________/___  /  /
#    / #     /_____\/        /9  0000/    /  / / /_  /\_____________\/  /
#   / #                      ``^^^^^^    /   \ \ . ./ / ____________   /
#  +=#==================================/     \ \ ./ / /.  .  .  \ /  /
#  |#                                   |      \ \/ / /___________/  /
#  #                                    |_______\__/________________/
#  | :)                                 |               |  |  / /
#  |                                    |               |  | / /
#  |                                    |       ________|  |/ /________
#  |                                    |      /_______/    \_________/\
#  |                                    |     /        /  /           \ )
#  |                                    |    /OO^^^^^^/  / /^^^^^^^^^OO\)
#  |                                    |            /  / /
#  |                                    |           /  / /
#  |                                    |          /___\/
#  |hectoras                            |           oo
#  |____________________________________|

# Jerry <3 🥺

n, h = 4, 4

#$4\times 7$ was computed to be 46768052394597294518835023998024018972050519490560.
#5x6 933948142924834964884995615767572235578827309826215499065416883451592132806114172767180795158681083568701648877439332484306011786987681425437210453893523299475914752

reduce = False
timeestimates = True
dosingle = True
printmatrix = True
savematrix = True
order = "bbtwo"
printconvex = True
dostuff = True
twobyn = False
zerochar = "."
onechar = "1"
twochar = "2"
negonechar = "-"
propersplice = False
splicecharacter = " "

if dostuff:
    if dosingle:
        whole = genmat(n, h, order)
        mat = whole[0]
        if reduce:
            mat = reduction(mat, n)
        fin = whole[1]
        print(
            f"The {n} by {h} matrix has determinant: {int(np.linalg.det(mat))} and the length of the matrix is {int(math.sqrt(mat.size))}."
        )
        var = False
        if order == "bbtwo":
            var = True
        if printmatrix:
            print(mat)
    
        message = printmatwithlines(mat, fin, var, savematrix, printmatrix)
        if savematrix:
            width=20+6*len(message[0])
            height=20+7*len(message)
            message="".join(message)
            #print(message)
            font = ImageFont.truetype("C:/Users/revil/OneDrive/Documents/Thesis/VeraMono.ttf", size=10)

            img = Image.new('RGB', (width, height), color=(0,43,54))

            imgDraw = ImageDraw.Draw(img)

            bbox = imgDraw.textbbox((0, 0), message, font=font)
            textWidth = bbox[2] - bbox[0]
            textHeight = bbox[3] - bbox[1]

            xText = (width - textWidth) / 2
            yText = (height - textHeight) / 2
            imgDraw.multiline_text((10,10), message, font=font, fill=(204,204,194))
            img.save(f'C:/Users/revil/OneDrive/Documents/Thesis/{n}x{h}{order}.png')

        if printconvex:
            print("")
            file1 = open(f'C:/Users/revil/OneDrive/Documents/Thesis/{n}x{h}{order}convextstore.txt', 'w')
            L=[]
            for i in range(len(fin)):
                L.append(f"Number {i} is this:\n")
                L.append("".join(createimage(fin[i], n, h)))
                L.append("\n\n")
            file1.writelines(L)
            file1.close()
        # checkrots(mat,fin)
    else:
        for h in range(2, 3):
            for n in range(1, 1001):
                mat = get_matrix(n, h)
                print(
                    f"The {n} by {h} matrix has determinant: {int(np.linalg.det(mat))} and the length of the matrix is {int(math.sqrt(mat.size))}."
                )
                if printmatrix:
                    print(mat)
                print("")