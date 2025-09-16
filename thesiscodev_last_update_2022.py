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


def thesimplealgorithmfunction(mat,h):
    h=int(h)
    totalsize=2*(h**2+h)/2
    for i in range(1,h+1):
        totalsize=totalsize+(h+1-i)*(i**2+i)/2
    for loc in range(int(totalsize)):
        cords=cordcalc(loc,h)
        w=cords[0]
        a=cords[1]
        c=cords[2]
        e=cords[3]
        x=cords[4]
        if w==1:
            typee=1
        elif w==2:
            if e==1:
                if x==1:
                    typee=3
                elif x>1:
                    typee=4
            if e>1:
                if x==1:
                    typee=5
                elif x>1:
                    typee=6
        #print(loc,cords,typee)
        if typee==1:
            """
            0:
            1:
            2:
            3:
            4:
            5:
            6:
            7:
            8:
            9:
            
            15: 0,2
            16: 1,3
            17: 2,4
            18: 3,5
            19: 4,6
            20: 5,7
            21: 6,8
            22: 7,9
            
            35: 0,2,4 15,17
            36: 1,3,5 16,18
            37: 2,4,6 17,19
            38: 3,5,7 18,20
            39: 4,6,8 19,21
            40: 5,7,9 20,22
            
            59: 0,2,4,6 15,17,19 35,37
            60: 1,3,5,7 16,18,20 36,38
            61: 2,4,6,8 17,19,21 37,39
            62: 3,5,7,9 18,20,22 38,40

            83: 0,2,4,6,8 15,17,19,21 35,37,39 59,61
            84: 1,3,5,7,9 16,18,20,22 36,38,40 60,62
            """
            l=[]
            for sa in range(1,a):
                amount=a-sa+1
                #abandlength=2*(h-a+1)
                #print('h', h, 'sa',sa,'rev',reversecordcalc(1,sa,1,1,1,h))
                #start=reversecordcalc(1,sa,1,1,1,h)+loc-reversecordcalc(1,a,1,1,1,h)
                
                for part in range(amount):
                    l.append(int(reversecordcalc(1,sa,c+part,e,1,h)))
                    #print(loc,start+2*part)
                    #mat=subtract(mat,loc,int(start+2*part),flip=False)
            #print(loc,l)
            if len(l)>0:
                mat=multitracth(mat,loc,l)
            pass
        if typee==3:
            """
            10: 0,1                2
            11: 2,3                2
            12: 4,5
            13: 6,7
            14: 8,9
            
            23: 0,1,2,3 10,11 15,16              4 2 2
            26: 2,3,4,5 11,12 17,18              2,1,2
            29: 4,5,6,7 12,13 19,20
            32: 6,7,8,9 13,14 21,22
            
            41: 0,1,2,3,4,5 10,11,12 15,16,17,18 23,26 35,36           112233
            47: 2,3,4,5,6,7 11,12,13 17,18,19,20 26,29 37,38           223344
            53: 4,5,6,7,8,9 12,13,14 19,20,21,22 29,32 39,40           334455
            
            63: 0,1,2,3,4,5,6,7 10,11,12,13 15,16,17,18,19,20 23,26,29 35,36,37,38 41,47 59,60          8 4 6 3 4 2 2
            73: 2,3,4,5,6,7,8,9 11,12,13,14 17,18,19,20,21,22 26,29,32 37,38,39,40 47,53 61,62          2,1,2,3,2,6,2
            
            85: 0,1,2,3,4,5,6,7,8,9 10,11,12,13,14 15,16,17,18,19,20,21,22 23,26,29,32 35,36,37,38,39,40 41,47,53 59,60,61,62 63,73 83,84     10 5 8 4 6 3 4 2 2 (1)
            """
            l=[]
            modv=a*2+w
            for analysspot in range(3,modv):
                tw=(analysspot-1)%2+1
                ta=(analysspot-tw)/2
                amount=(a-ta+1)*((tw)%2+1)
                #print('h', h, 'sa',sa,'rev',reversecordcalc(1,sa,1,1,1,h))
                start=reversecordcalc(tw,ta,c,1,1,h)
                if tw==1:
                    for part in range(int(amount)):
                        #print(loc,c+part-part%2,part%2+1)
                        l.append(int(reversecordcalc(1,ta,c+math.floor(part/2),part%2+1,1,h))) #c+part-part%2
                        #print(loc,start+2*part)
                        #mat=subtract(mat,loc,int(start+part),flip=False)
                        pass
                if tw==2:
                    for part in range(int(amount)):
                        l.append(int(reversecordcalc(tw,ta,part+c,1,1,h)))
                        #print(loc,start+2*part)
                        #mat=subtract(mat,loc,int(reversecordcalc(tw,ta,part+c,1,1,h)),flip=False)
                        pass
            print(loc,l)
            mat=multitracth(mat,loc,l)
            pass
        if typee==4:
            """
            24: 0,1,2 10 15               3 1 1
            27: 2,3,4 11 17
            30: 4,5,6 12 19
            33: 6,7,8 13 21

            42: 0,1,2,3,4 10,11 15,16,17 23,27 35         5 2 3 2 1
            43: 0,1,2,4   10    15,17    24    35
            48: 2,3,4,5,6 11,12 17,18,19 26,30 37
            49: 2,3,4,6   11    17,19    27    37
            54: 4,5,6,7,8 12,13 19,20,21 29,33 39
            55: 4,5,6,8   12    19,21    30    39

            64: 0,1,2,3,4,5,6 10,11,12 15,16,17,18,19 23,26,30 35,36,37 41,48 59          7 3 5 3 3 2 1     f, f/2 round down, f-2, (f-2)/2 roudn up, f-4, (f-4)/2 round up 
            65: 0,1,2,3,4,6   10,11    15,16,17,19    23,27    35,37    42,49 59
            66: 0,1,2,4,6     10       15,17,19       24       35,37    43    59
            74: 2,3,4,5,6,7,8 11,12,13 17,18,19,20,21 26,29,33 37,38,39 47,54 61
            75: 2,3,4,5,6,8   11,12    17,18,19,21    26,30    37,39    48,55 61
            76: 2,3,4,6,8     11       17,19,21       27       37,39    49    61

            86: 0,1,2,3,4,5,6,7,8 10,11,12,13 15,16,17,18,19,20,21 23,26,29,33 35,36,37,38,39 41,47,54 59,60,61 63,74 83          9 4 7 4 5 3 3 2 1
            87: 0,1,2,3,4,5,6,8   10,11,12    15,16,17,18,19,21    23,26,30    35,36,37,39    41,48,55 59,61    64,75 83
            88: 0,1,2,3,4,6,8     10,11       15,16,17,19,21       23,27       35,37,39       42,49    59,61    65,76 83
            89: 0,1,2,4,6,8       10          15,17,19,21          24          35,37,39       43       59,61    66    83
            """                  #                 
            """
            #                    -2-1 0 1                         -1  0  1  2                 0 1 2
            #                    -1 0 1                            0  1  2                    1 2 3
            #                     0 1                              1  2                       2 3
            #                     1                                2                          3
            """
            #                     4,3,2,1                          4,3,2,1                    3,3,2,1           2,2,2,1
            #                     4,3,2,1                          4,3,2,1                    4,3,2,1           4,3,2,1
            l=[]
            modv=a*2+w
            for analysspot in range(3,modv):
                tw=(analysspot-1)%2+1
                ta=(analysspot-tw)/2
                if tw==1:
                    amount=(a-ta+1)*2-1
                else:
                    amount=min(a-ta+1,a-1)
                if tw==1:
                    start=reversecordcalc(tw,ta,c,1,1,h)
                    for part in range(int(amount)):
                        if part%2==0:
                            l.append(int(reversecordcalc(1,ta,c,1+part,1,h)))
                            #mat=subtract(mat,loc,int(start+part),flip=False)
                        if part%2==1 and x<=(amount-part)/2+1.5:
                            l.append(int(reversecordcalc(1,ta,c,1+part,1,h)))
                            #mat=subtract(mat,loc,int(start+part),flip=False)
                        pass
                if tw==2:
                    
                    for part in range(int(min(a-x+1,amount))):
                        l.append(int(reversecordcalc(tw,ta,c+part,1,max(1,-a+x+part+ta),h)))
                        #mat=subtract(mat,loc,int(reversecordcalc(tw,ta,c+part,1,max(1,-a+x+part+ta),h)),flip=False)
            #print(loc,l)
            mat=multitracth(mat,loc,l)
            pass
        if typee==5:
            """
            25: 1,2,3 11 16   #3,1
            28: 3,4,5 12 18
            31: 5,6,7 13 20
            34: 7,8,9 14 22

            44: 1,2,3,4,5 11,12 16,17,18 25,26 36    #5,3,1
            46: 1,3,4,5   12    16,18    28    36
            50: 3,4,5,6,7 12,13 18,19,20 28,29 38
            52: 3,5,6,7   13    18,20    31    38
            56: 5,6,7,8,9 13,14 20,21,22 31,32 40
            58: 5,7,8,9   14    20,22    34    40

            67: 1,2,3,4,5,6,7 11,12,13 16,17,18,19,20 25,26,29 36,37,38 44,47 60         #7...
            70: 1,3,4,5,6,7   12,13    16,18,19,20    28,29    36,38    46,50 60
            72: 1,3,5,6,7     13       16,18,20       31       36,38    52    60
            77: 3,4,5,6,7,8,9 12,13,14 18,19,20,21,22 28,29,32 38,39,40 50,53 62
            80: 3,5,6,7,8,9   13,14    18,20,21,22    31,32    38,40    52,56 62
            82: 3,5,7,8,9     14       18,20,22       34       38,40    58    62

            90: 1,2,3,4,5,6,7,8,9 11,12,13,14 16,17,18,19,20,21,22 25,26,29,32 36,37,38,39,40 44,47,53 60,61,62 67,73 84 122334455
            94: 1,3,4,5,6,7,8,9   12,13,14    16,18,19,20,21,22    28,29,32    36,38,39,40    46,50,53 60,62    70,77 84
            97: 1,3,5,6,7,8,9     13,14       16,18,20,21,22       31,32       36,38,40       52,56    60,62    72,80 84
            99: 1,3,5,7,8,9       14          16,18,20,22          34          36,38,40       58       60,62    82    84
            """ #y y y y, n y y y, n n y y, n n n y 
                #1,3,5,7
            l=[]
            modv=a*2+w
            for analysspot in range(3,modv):
                tw=(analysspot-1)%2+1
                ta=(analysspot-tw)/2
                if tw==1:
                    amount=(a-ta)*2+1
                    start=reversecordcalc(tw,ta,c,2,1,h)
                    for part in range(int(amount)):
                        if part%2==0:
                            #l.append(int(start+part))
                            l.append(int(reversecordcalc(1,ta,c+math.ceil((part)/2),1+(part+1)%2,1,h))) #modify using this guy but hes slightly wrong
                            #print(loc,start+part,reversecordcalc(1,ta,c+math.ceil((part)/2),1+(part+1)%2,1,h))
                            #mat=subtract(mat,loc,int(start+part),flip=False)
                        if part%2==1 and part>=(e-2)*2+1:
                            l.append(int(reversecordcalc(1,ta,c+math.ceil((part)/2),1+(part+1)%2,1,h)))
                            #print(loc,start+part,reversecordcalc(1,ta,c+math.ceil((part)/2),1+(part+1)%2,1,h))
                            #mat=subtract(mat,loc,int(start+part),flip=False)
                        pass
                else:
                    emaxamount=a-(max(ta-1,e-1))
                    trueamount=emaxamount-(max(x-ta,0))
                    for part in range(int(trueamount)):
                        #print(tw,ta,c+amount-x+1,1,max(1,x-a+1),h)
                        l.append(int(reversecordcalc(tw,ta,part+max(1,e-ta+1)+c-1,max(a-max(a-ta,a-e)-part,1),max(1,part+x+max(ta,e)-a),h)))
                        #print(loc,start+2*part)
                        #mat=subtract(mat,loc,int(reversecordcalc(tw,ta,part+max(1,e-ta+1)+c-1,max(a-max(a-ta,a-e)-part,1),max(1,part+x+max(ta,e)-a),h)),flip=False)
            #print(loc,l)
            mat=multitracth(mat,loc,l)
            pass
        if typee==6:
            """
            45: 1,2,3,4 11 16,17 25,27
            51: 3,4,5,6 12 18,19 28,30
            57: 5,6,7,8 13 20,21 31,33

            68: 1,2,3,4,5,6 11,12 16,17,18,19 25,26,30 36,37 44,48
            69: 1,2,3,4,6   11    16,17,19    25,27    37    45,49
            71: 1,3,4,5,6   12    16,18,19    28,30    36    46,51
            78: 3,4,5,6,7,8 12,13 18,19,20,21 28,29,33 38,39 50,54
            79: 3,4,5,6,8   12    18,19,21    28,30    39    51,55
            81: 3,5,6,7,8   13    18,20,21    31,33    38    52,57
            
            91: 1,2,3,4,5,6,7,8 11,12,13 16,17,18,19,20,21 25,26,29,33 36,37,38,39 44,47,54 60,61 67,74
            92: 1,2,3,4,5,6,8   11,12    16,17,18,19,21    25,26,30    36,37,39    44,48,55 61    68,75
            93: 1,2,3,4,6,8     11       16,17,19,21       25,27       37,39       45,49    61    69,76
            95: 1,3,4,5,6,7,8   12,13    16,18,19,20,21    28,29,33    36,38,39    46,50,54 60    70,78
            96: 1,3,4,5,6,8     12       16,18,19,21       28,30       36,39       46,51,55       71,79
            98: 1,3,5,6,7,8     13       16,18,20,21       31,33       36,38       52,57    60    72,81
            """  #12,21,22,31 
            l=[]
            modv=a*2+w
            for analysspot in range(3,modv):
                tw=(analysspot-1)%2+1
                ta=(analysspot-tw)/2
                if tw==1:
                    amount=(a-ta)*2
                    start=reversecordcalc(tw,ta,c,2,1,h)
                    for part in range(int(amount)):
                        if part%2==0 and part<=amount-2*x+3:
                            l.append(int(reversecordcalc(1,ta,c+math.ceil((part)/2),1+(part+1)%2,1,h)))
                            #print(start+part,reversecordcalc(tw,ta,c,2+math.ceil(part/2),1+part%2,h),start,part,c)
                            #mat=subtract(mat,loc,int(start+part),flip=False)
                        if part%2==1 and part>=(e-2)*2+1:
                            l.append(int(reversecordcalc(1,ta,c+math.ceil((part)/2),1+(part+1)%2,1,h)))
                            #mat=subtract(mat,loc,int(start+part),flip=False)
                        pass
                else:
                    emaxamount=a-(max(ta-1,e-1))
                    trueamount=emaxamount-(max(x-ta,0))
                    for part in range(int(trueamount)):
                        #print(tw,ta,c+amount-x+1,1,max(1,x-a+1),h)
                        l.append(int(reversecordcalc(tw,ta,part+max(1,e-ta+1)+c-1,max(a-max(a-ta,a-e)-part,1),max(1,part+x+max(ta,e)-a),h)))
                        #print(loc,start+2*part)
                        #mat=subtract(mat,loc,int(reversecordcalc(tw,ta,part+max(1,e-ta+1)+c-1,max(a-max(a-ta,a-e)-part,1),max(1,part+x+max(ta,e)-a),h)),flip=False)
            #print(loc,l)
            mat=multitracth(mat,loc,l)
            pass
        #if typee==5 or typee==6:
            """
            25: 11
            
            28: 12
            
            31: 13
            
            34: 14

            
            44: 11,12 25,26
            45: 11    25,27
            
            46: 12    28   

            50: 12,13 28,29
            51: 12    28,30

            52: 13    31   
            
            56: 13,14 31,32
            57: 13    31,33

            58: 14    34   
            

            67: 11,12,13 25,26,29 44,47
            68: 11,12    25,26,30 44,48
            69: 11       25,27    45,49
            
            70: 12,13    28,29    46,50
            71: 12       28,30    46,51
            
            72: 13       31       52   
            
            
            77: 12,13,14 28,29,32 50,53   
            78: 12,13    28,29,33 50,54
            79: 12       28,30    51,55
            
            80: 13,14    31,32    52,56   #222
            81: 13       31,33    52,57
            
            82: 14       34       58      #111
            
                        
            90: 11,12,13,14 25,26,29,32 44,47,53 67,73   2345 1234 123 12   1111 2111 211 21   1111 1111 111 11      >3  4  3  2
            91: 11,12,13    25,26,29,33 44,47,54 67,74   234  1234 123 12   111  2111 211 21   111  1112 112 12      >2  3  3  2
            92: 11,12       25,26,30    44,48,55 68,75   23   123  123 12   11   211  211 21   11   112  123 23      >1  2  2  2
            93: 11          25,27       45,49    69,76   2    12   12  12   1    21   21  21   1    12   23  34
             
            94: 12,13,14    28,29,32    46,50,53 70,77   345  234  123 12   111  211  321 32   111  111  111 11
            95: 12,13       28,29,33    46,50,54 70,78   34   234  123 12   11   211  321 32   11   112  112 12
            96: 12          28,30       46,51,55 71,79   3    23   123 12   1    21   321 32   1    12   123 23
               
            97: 13,14       31,32       52,56    72,80   45   34   23  12   11   21   32  43   11   11   11  11
            98: 13          31,33       52,57    72,81   4    34   23  12   1    21   32  43   1    12   12  12
            
            99: 14          34          58       82      5    4    3   2    1    2    3   4    1    1    1   1 
            """ 
            """
            #l=[]
            modv=a*2+w
            for analysspot in range(3,modv):
                tw=(analysspot-1)%2+1
                ta=(analysspot-tw)/2
                if tw==1:
                    pass
                else:
                    emaxamount=a-(max(ta-1,e-1))
                    trueamount=emaxamount-(max(x-ta,0))
                    for part in range(int(trueamount)):
                        #print(tw,ta,c+amount-x+1,1,max(1,x-a+1),h)
                        #l.append(int(reversecordcalc(tw,ta,part+max(1,e-ta+1)+c-1,max(a-max(a-ta,a-e)-part,1),max(1,part+x+max(ta,e)-a),h)))
                        #print(loc,start+2*part)
                        #mat=subtract(mat,loc,int(reversecordcalc(tw,ta,c+part,1,max(1,-a+x+part+ta),h)),flip=False)
            #print(loc,l)
            pass
            """
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
        if False:
            if j==103:
                subs.append(33)
            if j==116:
                subs.append(41)
            if j==123:
                subs.append(39)
            if j==136:
                subs.append(47)
            if j==166:
                subs.append(75)
            if j==169:
                subs.append(75)
            if j==170:
                subs.append(33)
            if j==176:
                subs.append(39)
            if j==196:
                subs.append(39)
            if j==198:
                subs.append(90)
            if j==199:
                subs.append(41)
            if j==200:
                subs.append(41)
            if j==203:
                subs.append(90)
            if j==209:
                print(subs)
                print('neon')
                subs.append(47)
                print(subs)

        
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

# download data
# 4x4???
# mat=genmat(n,h)
# for i in range(int(math.sqrt(mat.size))):
#     print(mat[i])

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

n, h = 2, 6

#$4\times 7$ was computed to be 46768052394597294518835023998024018972050519490560.
#5x6 933948142924834964884995615767572235578827309826215499065416883451592132806114172767180795158681083568701648877439332484306011786987681425437210453893523299475914752

reduce = False
timeestimates = True
dosingle = True
printmatrix = False
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
trimonebyonesandall = False  # keep off
rownumber = True
newview = False
reduceD=False
solveA=False
lastoneihope=False
rowcollum=False
thesimplealgorithm=False
stop=1
showsubtractedmods=False
scrapeunacountedzerostwowide=False
scrapeunacountedzerostest=False
scrapeunacountedzerosonewide=False
scrapenegorpostwowide=False
scrapenegorposonewide=False
scrapenotzerobblocstotalsize=False
whatsactuallyoutthere=False
fulltest=False
fulltesttwo=False
temptestin=False

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

            #textWidth, textHeight = imgDraw.textsize(message, font=font)
            #xText = (width - textWidth) / 2
            #yText = (height - textHeight) / 2
            bbox = imgDraw.textbbox((0, 0), message, font=font)
            textWidth = bbox[2] - bbox[0]
            textHeight = bbox[3] - bbox[1]

            xText = (width - textWidth) / 2
            yText = (height - textHeight) / 2
            imgDraw.multiline_text((10,10), message, font=font, fill=(204,204,194))
            img.save(f'C:/Users/revil/OneDrive/Documents/Thesis/{n}x{h}{order}.png')

            #imgDraw.multiline_text((10,10), message, font=font, fill=(204,204,194))

            #img.save(f'C:/Users/revil/OneDrive/Documents/Thesis/{n}x{h}{order}.png')
        if printconvex:
            print("")
            file1 = open('C:/Users/revil/OneDrive/Desktop/2x5convextstore.txt', 'w')
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
elif twobyn:
    n = 4
    w1 = int(n * (n + 1))
    w2 = int((n * (n**3 + 6 * n**2 + 35 * n + 30)) / 24)
    offset = 0
    if trimonebyonesandall:
        offset = 1
    rec = genmat(2, n, "bb")
    nums1 = []
    nums2 = []
    for i in range(w1):
        nums1.append(i)
    for i in range(w1, w2):
        nums2.append(i)
    if trimonebyonesandall:
        for i in range(2 * n):
            nums2.append(i)
        nums1.append(w2 - 1)
    A = np.delete(rec[0], nums2, 1)
    A = np.delete(A, nums2, 0)
    B = np.delete(rec[0], nums2, 0)
    B = np.delete(B, nums1, 1)
    C = np.delete(rec[0], nums2, 1)
    C = np.delete(C, nums1, 0)
    D = np.delete(rec[0], nums1, 1)
    D = np.delete(D, nums1, 0)
    Ainv = np.linalg.inv(A)
    mid = np.matmul(Ainv, B)
    mul = np.matmul(C, mid)
    r = np.subtract(D, np.matmul(C, np.matmul(np.linalg.inv(A), B)))
    t = np.zeros((w2 - w1 - offset, w2 - w1 - offset), dtype=np.ubyte)
    for i in range(w2 - w1 - offset):
        for j in range(w2 - w1 - offset):
            t[i][j] = r[i][j]
    print("Whole")
    printmatwithlines(rec[0], rec[1])
    print("A")
    printmatnice(A)
    print("B")
    printmatnice(B)
    print("C")
    printmatnice(C)
    print("D")
    printmatnice(D)
    print("Inverse of A")
    printmatnice(Ainv)
    # print("A^-1B")
    # printmatnice(mid)
    print("C A^-1 B")
    printmatnice(mul)
    print("Result")
    printmatnice(t)
    print("DET of result:", int(np.linalg.det(t)))
elif newview:
    n, h = 8, 2
    whole = genmat(h, n, "bb")
    mat = whole[0]
    fin = whole[1]
    print(
        f"The {n} by {h} matrix has determinant: {int(np.linalg.det(mat))} and the length of the matrix is {int(math.sqrt(mat.size))}."
    )
    part = 14  # 1-16
    across = int((part - 1) % 4)
    updown = int((math.floor((part - 1) / 4)) % 4)
    leftedge = 0  # the actual leftmost digit of what i want
    rightedge = 0  # one right of the leftmost digit of what i want
    upedge = 0
    downedge = 0
    if across == 0:
        rightedge = 2 * (n * (n + 1) / 2) - 2
    if across == 1:
        leftedge = 2 * (n * (n + 1) / 2) - 2
        rightedge = 2 * (n * (n + 1) / 2)
    if across == 2:
        leftedge = 2 * (n * (n + 1) / 2)
        rightedge = -1 * (n * (n + 1) / 2) + (
            1 / 24 * (n * (n**3 + 6 * n**2 + 35 * n + 30))
        )
    if across == 3:
        leftedge = -1 * (n * (n + 1) / 2) + (
            1 / 24 * (n * (n**3 + 6 * n**2 + 35 * n + 30))
        )
        rightedge = 1 / 24 * (n * (n**3 + 6 * n**2 + 35 * n + 30))
    if updown == 0:
        downedge = 2 * (n * (n + 1) / 2) - 2
    if updown == 1:
        upedge = 2 * (n * (n + 1) / 2) - 2
        downedge = 2 * (n * (n + 1) / 2)
    if updown == 2:
        upedge = 2 * (n * (n + 1) / 2)
        downedge = -1 * (n * (n + 1) / 2) + (
            1 / 24 * (n * (n**3 + 6 * n**2 + 35 * n + 30))
        )
    if updown == 3:
        upedge = -1 * (n * (n + 1) / 2) + (
            1 / 24 * (n * (n**3 + 6 * n**2 + 35 * n + 30))
        )
        downedge = 1 / 24 * (n * (n**3 + 6 * n**2 + 35 * n + 30))
    leftedge = int(leftedge)
    rightedge = int(rightedge)
    upedge = int(upedge)
    downedge = int(downedge)
    leftremove = []
    rightremove = []
    upremove = []
    downremove = []
    for i in range(leftedge):
        leftremove.append(i)
    for i in range(rightedge, int(1 / 24 * (n * (n**3 + 6 * n**2 + 35 * n + 30)))):
        rightremove.append(i)
    for i in range(upedge):
        upremove.append(i)
    for i in range(downedge, int(1 / 24 * (n * (n**3 + 6 * n**2 + 35 * n + 30)))):
        downremove.append(i)
    # printmatnice(mat)
    printmatwithlinesvar(mat, fin, leftedge, rightedge, upedge, downedge)
elif reduceD==True:
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
    #print(mat)
    totalsize=2*(h**2+h)/2
    for i in range(1,h+1):
        totalsize=totalsize+(h+1-i)*(i**2+i)/2
    w2=int(totalsize)
    w1=int(totalsize-2-(h**2+h)/2)
    print(w1)
    print(w2)
    nums1 = []
    nums2 = []
    for i in range(w1):
        nums1.append(i)
    for i in range(w1, w2):
        nums2.append(i)
    print(nums1)
    print(nums2)
    #A=[0,totalsize-2-(n**2+n)/2]
    A = np.delete(mat, nums2, 1)
    A = np.delete(A, nums2, 0)
    B = np.delete(mat, nums2, 0)
    B = np.delete(B, nums1, 1)
    C = np.delete(mat, nums2, 1)
    C = np.delete(C, nums1, 0)
    D = np.delete(mat, nums1, 1)
    D = np.delete(D, nums1, 0)
    print("A")
    #printmatwithlines(A,fin,True,False,True)
    print("B")
    #printmatnice(B)
    print("C")
    #printmatnice(C)
    print("D")
    #printmatnice(D)
    print("A^-1")
    #printmatwithlines(np.linalg.inv(A),fin,True,True,True)
    mid = np.matmul(np.linalg.inv(A), B)
    mul = np.matmul(C, mid)
    print("CA^-1B")
    #printmatnice(mul)
    print("D")
    #printmatnice(D)
    print("D-CA^-1B")
    printmatnice(np.subtract(D,mul))
    print("reduced D-CA^-1B")
    printmatnice(reduced(np.subtract(D,mul),h))
    #message = printmatwithlines(mat, fin, var, savematrix, printmatrix)
    if savematrix:
        width=20+6*len(message[0])
        height=20+7*len(message)
        message="".join(message)
        #print(message)
        font = ImageFont.truetype("C:/Users/revil/OneDrive/Documents/VeraMono.ttf", size=10)

        img = Image.new('RGB', (width, height), color=(0,43,54))

        imgDraw = ImageDraw.Draw(img)

        textWidth, textHeight = imgDraw.textsize(message, font=font)
        xText = (width - textWidth) / 2
        yText = (height - textHeight) / 2

        imgDraw.multiline_text((10,10), message, font=font, fill=(204,204,194))

        img.save(f'C:/Users/revil/OneDrive/Documents/ThesisWork/{n}x{h}{order}.png')
    if printconvex:
        print("")
        for i in range(len(fin)):
            print(f"Number {i} is this:")
            print("".join(createimage(fin[i], n, h)))
            print("")
elif solveA==True:
    whole = genmat(n, h, order)
    mat = whole[0]
    fin=whole[1]
    totalsize=2*(h**2+h)/2
    for i in range(1,h+1):
        totalsize=totalsize+(h+1-i)*(i**2+i)/2
    w2=int(totalsize)
    w1=int(totalsize-2-(h**2+h)/2)
    totalsize=int(totalsize)
    #print(w1)
    #print(w2)
    if 25!=25:
        calcdminuscainvb(mat,fin,10,15,totalsize)
        calcdminuscainvb(mat,fin,15,23,totalsize)
        calcdminuscainvb(mat,fin,23,35,totalsize)
        calcdminuscainvb(mat,fin,35,41,totalsize)
        calcdminuscainvb(mat,fin,41,59,totalsize)
        calcdminuscainvb(mat,fin,59,63,totalsize)
        calcdminuscainvb(mat,fin,63,83,totalsize)
    if 27==27:    
        calcdminuscainvb(mat,fin,14,21,totalsize)
        calcdminuscainvb(mat,fin,21,33,totalsize)
        calcdminuscainvb(mat,fin,33,51,totalsize)
        calcdminuscainvb(mat,fin,51,61,totalsize)
        calcdminuscainvb(mat,fin,61,91,totalsize)
        calcdminuscainvb(mat,fin,91,99,totalsize)
        calcdminuscainvb(mat,fin,99,139,totalsize)
        calcdminuscainvb(mat,fin,139,145,totalsize)
        calcdminuscainvb(mat,fin,145,190,totalsize)
        calcdminuscainvb(mat,fin,190,194,totalsize)
        calcdminuscainvb(mat,fin,194,236,totalsize)
        calcdminuscainvb(mat,fin,236,238,totalsize)
        calcdminuscainvb(mat,fin,238,266,totalsize)
    if False:
        nums1 = []
        nums2 = []
        for i in range(w1):
            nums1.append(i)
        for i in range(w1, w2):
            nums2.append(i)
        #print(nums1)
        #print(nums2)
        A = np.delete(mat, nums2, 1)
        A = np.delete(A, nums2, 0)
        nums2=[]
        for i in range(10,83):
            nums2.append(i)
        A2= np.delete(A,nums2,1)
        A2=np.delete(A2,nums2,0)
        printmatnice(A2)
        print(np.linalg.det(A2))
        nums25=[]
        for i in range(15,83):
            nums25.append(i)
        A25= np.delete(A,nums25,1)
        A25=np.delete(A25,nums25,0)
        printmatnice(A25)
        print(np.linalg.det(A25))
        nums3=[]
        for i in range(23,83):
            nums3.append(i)
        A3= np.delete(A,nums3,1)
        A3=np.delete(A3,nums3,0)
        printmatnice(A3)
        print(np.linalg.det(A3))
        nums4=[]
        for i in range(35,83):
            nums4.append(i)
        A4= np.delete(A,nums4,1)
        A4=np.delete(A4,nums4,0)
        printmatnice(A4)
        print(np.linalg.det(A4))
        nums5=[]
        for i in range(41,83):
            nums5.append(i)
        A5= np.delete(A,nums5,1)
        A5=np.delete(A5,nums5,0)
        printmatnice(A5)
        print(np.linalg.det(A5))
        nums6=[]
        for i in range(59,83):
            nums6.append(i)
        A6= np.delete(A,nums6,1)
        A6=np.delete(A6,nums6,0)
        printmatnice(A6)
        print(np.linalg.det(A6))
        nums7=[]
        for i in range(63,83):
            nums7.append(i)
        A7= np.delete(A,nums7,1)
        A7=np.delete(A7,nums7,0)
        printmatnice(A7)
        print(np.linalg.det(A7))
        printmatnice(A)
        print(np.linalg.det(A))

        # print(leftedge,rightedge,upedge,downedge,leftremove,rightremove,upremove,downremove)
if lastoneihope:
    whole = genmat(n, h, order)
    mat = whole[0]
    fin=whole[1]
    printmatnice(mat)
    imredoingthisagain(mat,h)

if rowcollum==True:
    whole = genmat(n, h, order)
    mat = whole[0]
    fin=whole[1]
    #printmatnice(mat)

    #print()
    #1x1,1x1 done
    if h==3:
        mat=subtract(mat,6,0,flip=False)
        mat=subtract(mat,6,1,flip=False)
        mat=subtract(mat,7,2,flip=False)
        mat=subtract(mat,7,3,flip=False)
        mat=multitract(mat,8,[4,5],flip=False)
        #2x1,2x1 done
        mat=subtract(mat,9,0,flip=False)
        mat=subtract(mat,9,2,flip=False)
        mat=subtract(mat,10,1,flip=False)
        mat=subtract(mat,10,3,flip=False)
        mat=subtract(mat,11,2,flip=False)
        mat=subtract(mat,11,4,flip=False)
        mat=subtract(mat,12,3,flip=False)
        mat=subtract(mat,12,5,flip=False)
        #1x2,1x2 done
        mat=subtract(mat,13,0,flip=False)
        mat=subtract(mat,13,1,flip=False)
        mat=subtract(mat,13,2,flip=False)
        mat=subtract(mat,13,3,flip=False)
        mat=subtract(mat,13,6,flip=False)
        mat=subtract(mat,13,7,flip=False)
        mat=subtract(mat,13,9,flip=False)
        mat=subtract(mat,13,10,flip=False)
        mat=subtract(mat,14,0,flip=False)
        mat=subtract(mat,14,1,flip=False)
        mat=subtract(mat,14,2,flip=False)
        mat=subtract(mat,14,6,flip=False)
        mat=subtract(mat,14,9,flip=False)
        mat=subtract(mat,14,6)
        mat=subtract(mat,14,9)
        mat=subtract(mat,15,1,flip=False)
        mat=subtract(mat,15,2,flip=False)
        mat=subtract(mat,15,3,flip=False)
        mat=subtract(mat,15,7,flip=False)
        mat=subtract(mat,15,10,flip=False)
        mat=subtract(mat,15,7)
        mat=subtract(mat,15,10)
        mat=subtract(mat,16,2,flip=False)
        mat=subtract(mat,16,3,flip=False)
        mat=subtract(mat,16,4,flip=False)
        mat=subtract(mat,16,5,flip=False)
        mat=subtract(mat,16,7,flip=False)
        mat=subtract(mat,16,8,flip=False)
        mat=subtract(mat,16,11,flip=False)
        mat=subtract(mat,16,12,flip=False)
        mat=subtract(mat,17,2,flip=False)
        mat=subtract(mat,17,3,flip=False)
        mat=subtract(mat,17,4,flip=False)
        mat=subtract(mat,17,7,flip=False)
        mat=subtract(mat,17,11,flip=False)
        mat=subtract(mat,17,7)
        mat=subtract(mat,17,11)
        mat=subtract(mat,18,3,flip=False)
        mat=subtract(mat,18,4,flip=False)
        mat=subtract(mat,18,5,flip=False)
        mat=subtract(mat,18,8,flip=False)
        mat=subtract(mat,18,12,flip=False)
        mat=subtract(mat,18,8)
        mat=subtract(mat,18,12)
        #2x2,2x2 done
        mat=subtract(mat,19,0,flip=False)
        mat=subtract(mat,19,2,flip=False)
        mat=subtract(mat,19,4,flip=False)
        mat=subtract(mat,19,9,flip=False)
        mat=subtract(mat,19,11,flip=False)
        mat=subtract(mat,20,1,flip=False)
        mat=subtract(mat,20,3,flip=False)
        mat=subtract(mat,20,5,flip=False)
        mat=subtract(mat,20,10,flip=False)
        mat=subtract(mat,20,12,flip=False)
        #1x3,1x3 done
        mat=subtract(mat,21,0,flip=False)
        mat=subtract(mat,21,1,flip=False)
        mat=subtract(mat,21,2,flip=False)
        mat=subtract(mat,21,3,flip=False)
        mat=subtract(mat,21,4,flip=False)
        mat=subtract(mat,21,5,flip=False)
        mat=subtract(mat,21,6,flip=False)
        mat=subtract(mat,21,7,flip=False)
        mat=subtract(mat,21,8,flip=False)
        mat=subtract(mat,21,9,flip=False)
        mat=subtract(mat,21,10,flip=False)
        mat=subtract(mat,21,11,flip=False)
        mat=subtract(mat,21,12,flip=False)
        mat=subtract(mat,21,13,flip=False)
        mat=subtract(mat,21,16,flip=False)
        mat=subtract(mat,21,19,flip=False)
        mat=subtract(mat,21,20,flip=False)
        mat=subtract(mat,22,0,flip=False)
        mat=subtract(mat,22,1,flip=False)
        mat=subtract(mat,22,2,flip=False)
        mat=subtract(mat,22,3,flip=False)
        mat=subtract(mat,22,4,flip=False)
        mat=subtract(mat,22,6,flip=False)
        mat=subtract(mat,22,7,flip=False)
        mat=subtract(mat,22,9,flip=False)
        mat=subtract(mat,22,10,flip=False)
        mat=subtract(mat,22,11,flip=False)
        mat=subtract(mat,22,13,flip=False)
        mat=subtract(mat,22,17,flip=False)
        mat=subtract(mat,22,19,flip=False)
        mat=subtract(mat,22,13)
        mat=subtract(mat,22,19)
        mat=subtract(mat,23,0,flip=False)
        mat=subtract(mat,23,1,flip=False)
        mat=subtract(mat,23,2,flip=False)
        mat=subtract(mat,23,4,flip=False)
        mat=subtract(mat,23,6,flip=False)
        mat=subtract(mat,23,9,flip=False)
        mat=subtract(mat,23,11,flip=False)
        mat=subtract(mat,23,14,flip=False)
        mat=subtract(mat,23,19,flip=False)
        mat=subtract(mat,23,6)
        mat=subtract(mat,23,19)
        mat=subtract(mat,24,1,flip=False)
        mat=subtract(mat,24,2,flip=False)
        mat=subtract(mat,24,3,flip=False)
        mat=subtract(mat,24,4,flip=False)
        mat=subtract(mat,24,5,flip=False)
        mat=subtract(mat,24,7,flip=False)
        mat=subtract(mat,24,8,flip=False)
        mat=subtract(mat,24,10,flip=False)
        mat=subtract(mat,24,11,flip=False)
        mat=subtract(mat,24,12,flip=False)
        mat=subtract(mat,24,15,flip=False)
        mat=subtract(mat,24,16,flip=False)
        mat=subtract(mat,24,20,flip=False)
        mat=subtract(mat,24,16)
        mat=subtract(mat,24,20)
        mat=subtract(mat,25,1,flip=False)
        mat=subtract(mat,25,2,flip=False)
        mat=subtract(mat,25,3,flip=False)
        mat=subtract(mat,25,4,flip=False)
        mat=subtract(mat,25,7,flip=False)
        mat=subtract(mat,25,10,flip=False)
        mat=subtract(mat,25,11,flip=False)
        mat=subtract(mat,25,15,flip=False)
        mat=subtract(mat,25,17,flip=False)
        mat=subtract(mat,25,7)
        mat=subtract(mat,25,10)
        mat=subtract(mat,25,11)
        mat=subtract(mat,25,15)
        mat=subtract(mat,25,17)
        mat=subtract(mat,26,1,flip=False)
        mat=subtract(mat,26,3,flip=False)
        mat=subtract(mat,26,4,flip=False)
        mat=subtract(mat,26,5,flip=False)
        mat=subtract(mat,26,8,flip=False)
        mat=subtract(mat,26,10,flip=False)
        mat=subtract(mat,26,12,flip=False)
        mat=subtract(mat,26,18,flip=False)
        mat=subtract(mat,26,20,flip=False)
        mat=subtract(mat,26,8)
        mat=subtract(mat,26,20)
        #2x3,2x3 done
    if h==5:
        mat=multitracth(mat,10,[0,1])
        mat=multitracth(mat,11,[2,3])
        mat=multitracth(mat,12,[4,5])
        mat=multitracth(mat,13,[6,7])
        mat=multitracth(mat,14,[8,9])

        mat=multitracth(mat,15,[0,2])
        mat=multitracth(mat,16,[1,3])
        mat=multitracth(mat,17,[2,4])
        mat=multitracth(mat,18,[3,5])
        mat=multitracth(mat,19,[4,6])
        mat=multitracth(mat,20,[5,7])
        mat=multitracth(mat,21,[6,8])
        mat=multitracth(mat,22,[7,9])
        mat=multitracth(mat,23,[0,1,2,3,10,11,15,16])
        mat=multitracth(mat,24,[0,1,2,10,15])
        mat=multitracth(mat,25,[1,2,3,11,16])
        #print('mat')
        #printmatnice(mat)
        #print('result')
        #printmatnice(result)
        #print('sub')
        #printmatnice(mat-result)
        mat=multitracth(mat,26,[2,3,4,5,11,12,17,18])
        mat=multitracth(mat,27,[2,3,4,11,17])
        mat=multitracth(mat,28,[3,4,5,12,18])
        mat=multitracth(mat,29,[4,5,6,7,12,13,19,20])
        mat=multitracth(mat,30,[4,5,6,12,19])
        mat=multitracth(mat,31,[5,6,7,13,20])
        mat=multitracth(mat,32,[6,7,8,9,13,14,21,22])
        mat=multitracth(mat,33,[6,7,8,13,21])
        mat=multitracth(mat,34,[7,8,9,14,22])
        if False:
            mat=multitractv(mat,24,[10,15])
            mat=multitractv(mat,25,[11,16])
            mat=multitractv(mat,27,[11,17])
            mat=multitractv(mat,28,[12,18])
            mat=multitractv(mat,30,[12,19])
            mat=multitractv(mat,31,[13,20])
            mat=multitractv(mat,33,[13,21])
            mat=multitractv(mat,34,[14,22])

        mat=multitracth(mat,35,[0,2,4,15,17])
        mat=multitracth(mat,36,[1,3,5,16,18])
        mat=multitracth(mat,37,[2,4,6,17,19])
        mat=multitracth(mat,38,[3,5,7,18,20])
        mat=multitracth(mat,39,[4,6,8,19,21])
        mat=multitracth(mat,40,[5,7,9,20,22])

        mat=multitracth(mat,41,[0,1,2,3,4,5,10,11,12,15,16,17,18,23,26,35,36])
        mat=multitracth(mat,42,[0,1,2,3,4,10,11,15,16,17,23,27,35])
        mat=multitracth(mat,43,[0,1,2,4,10,15,17,24,35])
        mat=multitracth(mat,44,[1,2,3,4,5,11,12,16,17,18,25,26,36])
        mat=multitracth(mat,45,[1,2,3,4,11,16,17,25,27])
        mat=multitracth(mat,46,[1,3,4,5,12,16,18,28,36])
        mat=multitracth(mat,47,[2,3,4,5,6,7,11,12,13,17,18,19,20,26,29,37,38])
        mat=multitracth(mat,48,[2,3,4,5,6,11,12,17,18,19,26,30,37])
        mat=multitracth(mat,49,[2,3,4,6,11,17,19,27,37])
        mat=multitracth(mat,50,[3,4,5,6,7,12,13,18,19,20,28,29,38])
        mat=multitracth(mat,51,[3,4,5,6,12,18,19,28,30])
        mat=multitracth(mat,52,[3,5,6,7,13,18,20,31,38])
        mat=multitracth(mat,53,[4,5,6,7,8,9,12,13,14,19,20,21,22,29,32,39,40])
        mat=multitracth(mat,54,[4,5,6,7,8,12,13,19,20,21,29,33,39])
        mat=multitracth(mat,55,[4,5,6,8,12,19,21,30,39])
        mat=multitracth(mat,56,[5,6,7,8,9,13,14,20,21,22,31,32,40])
        mat=multitracth(mat,57,[5,6,7,8,13,20,21,31,33])
        mat=multitracth(mat,58,[5,7,8,9,14,20,22,34,40])
        if False:
            mat=multitractv(mat,42,[23,35])
            mat=multitractv(mat,43,[10,35])
            mat=multitractv(mat,44,[26,36])
            mat=multitractv(mat,45,[11,16,17,25,27])
            mat=multitractv(mat,46,[12,36])
            mat=multitractv(mat,47,[])
            mat=multitractv(mat,48,[26,37])
            mat=multitractv(mat,49,[11,37])
            mat=multitractv(mat,50,[29,38])
            mat=multitractv(mat,51,[12,18,19,28,30])
            mat=multitractv(mat,52,[13,38])
            mat=multitractv(mat,53,[])
            mat=multitractv(mat,54,[29,39])
            mat=multitractv(mat,55,[12,39])
            mat=multitractv(mat,56,[32,40])
            mat=multitractv(mat,57,[13,20,21,31,33])
            mat=multitractv(mat,58,[14,40])
        #59: 0,2,4,6 15,17,19 35,37
        print('before')
        printmatnice(mat)
        mat=multitracth(mat,59,[0,2,4,6])
        print('one part')
        printmatnice(mat)
        mat=multitracth(mat,59,[15,17,19])
        print('two part')
        printmatnice(mat)
        mat=multitracth(mat,59,[35,37])
        print('done')
        printmatnice(mat)
        mat=multitracth(mat,60,[1,3,5,7,16,18,20,36,38])
        mat=multitracth(mat,61,[2,4,6,8,17,19,21,37,39])
        mat=multitracth(mat,62,[3,5,7,9,18,20,22,38,40])
        
        mat=multitracth(mat,63,[0,1,2,3,4,5,6,7,10,11,12,13,15,16,17,18,19,20,23,26,29,35,36,37,38,41,47,59,60])
        mat=multitracth(mat,64,[0,1,2,3,4,5,6,10,11,12,15,16,17,18,19,23,26,30,35,36,37,41,48,59])
        mat=multitracth(mat,65,[0,1,2,3,4,6,10,11,15,16,17,19,23,27,35,37,42,49,59])
        mat=multitracth(mat,66,[0,1,2,4,6,10,15,17,19,24,35,37,43,59])
        mat=multitracth(mat,67,[1,2,3,4,5,6,7,11,12,13,16,17,18,19,20,25,26,29,36,37,38,44,47,60])
        mat=multitracth(mat,68,[1,2,3,4,5,6,11,12,16,17,18,19,25,26,30,36,37,44,48])
        mat=multitracth(mat,69,[1,2,3,4,6,11,16,17,19,25,27,37,45,49])
        mat=multitracth(mat,70,[1,3,4,5,6,7,12,13,16,18,19,20,28,29,36,38,46,50,60])
        mat=multitracth(mat,71,[1,3,4,5,6,12,16,18,19,28,30,36,46,51])
        mat=multitracth(mat,72,[1,3,5,6,7,13,16,18,20,31,36,38,52,60])
        mat=multitracth(mat,73,[2,3,4,5,6,7,8,9,11,12,13,14,17,18,19,20,21,22,26,29,32,37,38,39,40,47,53,61,62])
        mat=multitracth(mat,74,[2,3,4,5,6,7,8,11,12,13,17,18,19,20,21,26,29,33,37,38,39,47,54,61])
        mat=multitracth(mat,75,[2,3,4,5,6,8,11,12,17,18,19,21,26,30,37,39,48,55,61])
        mat=multitracth(mat,76,[2,3,4,6,8,11,17,19,21,27,37,39,49,61])
        mat=multitracth(mat,77,[3,4,5,6,7,8,9,12,13,14,18,19,20,21,22,28,29,32,38,39,40,50,53,62])
        mat=multitracth(mat,78,[3,4,5,6,7,8,12,13,18,19,20,21,28,29,33,38,39,50,54])
        mat=multitracth(mat,79,[3,4,5,6,8,12,18,19,21,28,30,39,51,55])
        mat=multitracth(mat,80,[3,5,6,7,8,9,13,14,18,20,21,22,31,32,38,40,52,56,62])
        mat=multitracth(mat,81,[3,5,6,7,8,13,18,20,21,31,33,38,52,57])
        mat=multitracth(mat,82,[3,5,7,8,9,14,18,20,22,34,38,40,58,62])
        
        mat=multitracth(mat,83,[0,2,4,6,8,15,17,19,21,35,37,39,59,61])
        mat=multitracth(mat,84,[1,3,5,7,9,16,18,20,22,36,38,40,60,62])
        
        mat=multitracth(mat,85,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,26,29,32,35,36,37,38,39,40,41,47,53,59,60,61,62,63,73,83,84])
        mat=multitracth(mat,86,[0,1,2,3,4,5,6,7,8,10,11,12,13,15,16,17,18,19,20,21,23,26,29,33,35,36,37,38,39,41,47,54,59,60,61,63,74,83])
        mat=multitracth(mat,87,[0,1,2,3,4,5,6,8,10,11,12,15,16,17,18,19,21,23,26,30,35,36,37,39,41,48,55,59,61,64,75,83])
        mat=multitracth(mat,88,[0,1,2,3,4,6,8,10,11,15,16,17,19,21,23,27,35,37,39,42,49,59,61,65,76,83])
        mat=multitracth(mat,89,[0,1,2,4,6,8,10,15,17,19,21,24,35,37,39,43,59,61,66,83])
        mat=multitracth(mat,90,[1,2,3,4,5,6,7,8,9,11,12,13,14,16,17,18,19,20,21,22,25,26,29,32,36,37,38,39,40,44,47,53,60,61,62,67,73,84])
        mat=multitracth(mat,91,[1,2,3,4,5,6,7,8,11,12,13,16,17,18,19,20,21,25,26,29,33,36,37,38,39,44,47,54,60,61,67,74])
        mat=multitracth(mat,92,[1,2,3,4,5,6,8,11,12,16,17,18,19,21,25,26,30,36,37,39,44,48,55,61,68,75])
        mat=multitracth(mat,93,[1,2,3,4,6,8,11,16,17,19,21,25,27,37,39,45,49,61,69,76])
        mat=multitracth(mat,94,[1,3,4,5,6,7,8,9,12,13,14,16,18,19,20,21,22,28,29,32,36,38,39,40,46,50,53,60,62,70,77,84])
        mat=multitracth(mat,95,[1,3,4,5,6,7,8,12,13,16,18,19,20,21,28,29,33,36,38,39,46,50,54,60,70,78])
        mat=multitracth(mat,96,[1,3,4,5,6,8,12,16,18,19,21,28,30,36,39,46,51,55,71,79])
        mat=multitracth(mat,97,[1,3,5,6,7,8,9,13,14,16,18,20,21,22,31,32,36,38,40,52,56,60,62,72,80,84])
        mat=multitracth(mat,98,[1,3,5,6,7,8,13,16,18,20,21,31,33,36,38,52,57,60,72,81])
        mat=multitracth(mat,99,[1,3,5,7,8,9,14,16,18,20,22,34,36,38,40,58,60,62,82,84])
    print('')
    printmatwithlines(mat, fin, True, False, True)

if thesimplealgorithm==True:
    whole = genmat(n, h, order)
    mat = whole[0]
    fin=whole[1]
    #printmatnice(mat)

    #print("")
    result=thesimpleralgorithmfunction(mat,h)
    printmatnice(result)
    print(
            f"The {n} by {h} matrix has determinant: {int(np.linalg.det(result))} and the length of the matrix is {int(math.sqrt(mat.size))}."
        )
    #print('6')
    #printmatnice(savedddmat)
    #print('')
    #printmatnice(result)
    #printmatwithlines(result, fin, True, savematrix, printmatrix)
    #for i in range(205,280):
    #    l=[]
    #    for j in range(378,462):
    #        if result[i][j]==255:
    #            l.append("-")
    #        elif result[i][j]==0:
    #            l.append(".")
    #        elif result[i][j]==1:
    #            l.append(str(result[i][j]))
    #        elif result[i][j]>1:
    #            print('uhoh')
    #    message="".join(l)
    #    print(f"{i}: [{message}]")


    sum=0
    totalsize=2*(h**2+h)/2
    for i in range(1,h+1):
        totalsize=totalsize+(h+1-i)*(i**2+i)/2
    totalsize=int(totalsize)
    for a in range(totalsize):
        for b in range(totalsize):
            if a>b:
                sum=sum+result[a][b]
    print("bottom left sums to", sum)
    print('')
    trace=1
    for a in range(totalsize):
        for b in range(totalsize):
            if a==b:
                if result[a][b]==255:
                    trace=trace*-1
                else:
                    trace=trace*result[a][b]
    print("multiply trace to get:", trace)
    #printmatnice(result-savedddmat)

    # print("")
    # if printconvex:
    #     for i in range(len(fin)):
    #         print(f"Number {i} is this:")
    #         print("".join(createimage(fin[i],n,h)))
    #         print("")

if showsubtractedmods:
    whole = genmat(n, h, order)
    mat = whole[0]
    fin=whole[1]
    h=int(h)
    totalsize=2*(h**2+h)/2
    for i in range(1,h+1):
        totalsize=totalsize+(h+1-i)*(i**2+i)/2   
    #printmatnice(mat)

    #print("")
    result=thesimpleralgorithmfunction(mat,h,True)
    #print(result)
    if True: #visually print each convex module followed by the rows subtracted from them
        for i in range(len(result)):
            if i==88:
                print(result[i])
                el=result[i][0]
                print(f'For the row coresponding to this module number {el}:\n')
                print("".join(createimage(fin[el], n, h)))
                print('\nWe end up subtracting off the following modules:')
                for j in range(len(result[i][1])):
                    le=result[i][1][j]
                    print(f'\nModule number: {le}')
                    print("".join(createimage(fin[le], n, h)))

                for k in range(int(totalsize)):
                    if k not in result[i][1]:
                        sub=True
                        for point in fin[k]:
                            sub=sub and (point in fin[el])
                        if sub:
                            if k>=i:
                                print(f'\n We did not subtract off the following module with number {k}, however note its {k} which is bigger than {i}:')
                                print("".join(createimage(fin[k], n, h)))
                            else:
                                print(f'\n We did not subtract off the following module with number {k}:')
                                print("".join(createimage(fin[k], n, h)))
                if i!=len(result)-1:
                    print('\n \n \n \n')
    
    if False:#testing to see if all subtracted modules are submodules
        for i in range(len(result)):
            print(result[i])
            el=result[i][0]
            #print(f'For the row coresponding to this module number {el}:\n')
            #print("".join(createimage(fin[el], n, h)))
            #print('\nWe end up subtracting off the following modules:')
            for j in range(len(result[i][1])):
                le=result[i][1][j]
                sub=True
                for point in fin[le]:
                    sub=sub and (point in fin[el])
                if sub==False:
                    print(f'\nModule number: {le}')
            if i!=len(result)-1:
                print('\n \n')

if scrapenegorpostwowide:
    totalsize=2*(h**2+h)/2
    for i in range(1,h+1):
        totalsize=totalsize+(h+1-i)*(i**2+i)/2
    totalsize=int(totalsize)
    whole = genmat(n, h, order)
    mat = whole[0]
    fin=whole[1]
    a = np.ones( (totalsize,totalsize) )
    for i in range(totalsize):
        for j in range(totalsize):
            a[i][j]=mat[i][j]
    #for row in a:
    #    print(row)
    #printmatnice(mat)

    #print("")
    result=thesimpleralgorithmfunction(mat,h)
    #printmatnice(result)
    #printmatnice(fmatsave)
    #print(a[41],"~~~")
    for i in range(totalsize):
        for j in range(totalsize):
            wid=1
            side=0
            for point in fin[j]:
                if side==0:
                    side=point[0]
                if side!=0 and point[0]!=side:
                    wid=2
            if wid==2:
                if result[j][i]==255:
                    r=-1
                else:
                    r=result[j][i]
                if r!=0:
                    minsi = []  # collection of the lowest elements of the poset
                    maxsi = []  # collection of the highest elements of the poset
                    for test in fin[i]:  # look through all the elements
                        if [test[0] - 1, test[1]] in fin[i] or [
                            test[0],
                            test[1] - 1,
                        ] in fin[i]:  # if theres a smaller element skip it
                            pass
                        else:
                            minsi.append(test)  # otherwise add it to minimals
                        if [test[0] + 1, test[1]] in fin[i] or [
                            test[0],
                            test[1] + 1,
                        ] in fin[i]:  # if theres a larger element skip it
                            pass
                        else:
                            maxsi.append(test)
                    minsc = []  # collection of the lowest elements of the poset
                    maxsc = []  # collection of the highest elements of the poset
                    for test in fin[j]:  # look through all the elements
                        if [test[0] - 1, test[1]] in fin[j] or [
                            test[0],
                            test[1] - 1,
                        ] in fin[j]:  # if theres a smaller element skip it
                            pass
                        else:
                            minsc.append(test)  # otherwise add it to minimals
                        if [test[0] + 1, test[1]] in fin[j] or [
                            test[0],
                            test[1] + 1,
                        ] in fin[j]:  # if theres a larger element skip it
                            pass
                        else:
                            maxsc.append(test)
                    intervals=0
                    for lowpoint in minsi:
                        for highpoint in maxsi:
                            if lowpoint[0]<=highpoint[0] and lowpoint[1]<=highpoint[1]:
                                intervals=intervals+1
                    lowestheighti=h
                    lowestheightc=h
                    heighestheighti=-1
                    heighestheightc=-1
                    for point in fin[i]:
                        if point[1]<lowestheighti:
                            lowestheighti=point[1]
                        if point[1]>heighestheighti:
                            heighestheighti=point[1]
                    for point in fin[j]:
                        if point[1]<lowestheightc:
                            lowestheightc=point[1]
                        if point[1]>heighestheightc:
                            heighestheightc=point[1]
                    if (lowestheightc==lowestheighti and heighestheightc==heighestheighti) or (lowestheighti<lowestheightc and heighestheighti>heighestheightc):
                        if intervals==1 or intervals==3:
                            perdiction=1
                        if intervals==2:
                            perdiction=-1
                    else:
                        if intervals==2:
                            perdiction=1
                        if intervals==3:
                            perdiction=-1
                        if intervals==1:
                            perdiction=398450793475092345
                    if perdiction!=r:
                        print(f"input: {i}, row/comparison: {j}, original value: {a[j][i]}, actual result: {r}, perdiction: {perdiction}, intervals: {intervals}, lowestheighti{lowestheighti},lowestheightc{lowestheightc},heighestheighti{heighestheighti},heighestheightc{heighestheightc},minsi{minsi},maxsi{maxsi}")
                        if True:
                            print(f"The input {i} is this:")
                            print("".join(createimage(fin[i], n, h)))
                            print("")
                            print(f"The comparison {j} is this:")
                            print("".join(createimage(fin[j], n, h)))
                            print("")
                            print('')
                            print('')

if scrapenegorposonewide:
    totalsize=2*(h**2+h)/2
    for i in range(1,h+1):
        totalsize=totalsize+(h+1-i)*(i**2+i)/2
    totalsize=int(totalsize)
    whole = genmat(n, h, order)
    mat = whole[0]
    fin=whole[1]
    a = np.ones( (totalsize,totalsize) )
    for i in range(totalsize):
        for j in range(totalsize):
            a[i][j]=mat[i][j]
    #for row in a:
    #    print(row)
    #printmatnice(mat)

    #print("")
    result=thesimpleralgorithmfunction(mat,h)
    #printmatnice(result)
    #printmatnice(fmatsave)
    #print(a[41],"~~~")
    for i in range(totalsize):
        for j in range(totalsize):
            wid=1
            side=0
            for point in fin[j]:
                if side==0:
                    side=point[0]
                if side!=0 and point[0]!=side:
                    wid=2
            if wid==1:
                if result[j][i]==255:
                    r=-1
                else:
                    r=result[j][i]
                perdiction=1
                if perdiction!=r and r!=0:
                    print(f"input: {i}, row/comparison: {j}, original value: {a[j][i]}, actual result: {r}, perdiction:")
                    if True:
                        print(f"The input {i} is this:")
                        print("".join(createimage(fin[i], n, h)))
                        print("")
                        print(f"The comparison {j} is this:")
                        print("".join(createimage(fin[j], n, h)))
                        print("")
                        print('')
                        print('')
                            
if scrapeunacountedzerostwowide:
    totalsize=2*(h**2+h)/2
    for i in range(1,h+1):
        totalsize=totalsize+(h+1-i)*(i**2+i)/2
    totalsize=int(totalsize)
    whole = genmat(n, h, order)
    mat = whole[0]
    fin=whole[1]
    a = np.ones( (totalsize,totalsize) )
    for i in range(totalsize):
        for j in range(totalsize):
            a[i][j]=mat[i][j]
    #printmatnice(mat)

    #print("")
    result=thesimpleralgorithmfunction(mat,h)
    #printmatnice(result)
    print('ziong')
    file1 = open('C:/Users/revil/OneDrive/Desktop/dump3.txt', 'w')
    L=[]
    for j in range(totalsize):
        for i in range(totalsize):
            #if result[j][i]==0:
            if True:
                wid=1
                side=-3
                for point in fin[j]:
                    if side==-3:
                        side=point[0]
                    if side!=-3 and point[0]!=side:
                        wid=2
                if wid==2:
                    wid=1
                    side=-3
                    for point in fin[j]:
                        if side==-3:
                            side=point[0]
                        if side!=-3 and point[0]!=side:
                            wid=2
                    if wid==2:
                        if a[j][i]!=0:
                            unacountedforzero=True
                            incon=fin[i]
                            compmod=fin[j]
                            minsi = []  # collection of the lowest elements of the poset
                            maxsi = []  # collection of the highest elements of the poset
                            for test in incon:  # look through all the elements
                                if [test[0] - 1, test[1]] in incon or [
                                    test[0],
                                    test[1] - 1,
                                ] in incon:  # if theres a smaller element skip it
                                    pass
                                else:
                                    minsi.append(test)  # otherwise add it to minimals
                                if [test[0] + 1, test[1]] in incon or [
                                    test[0],
                                    test[1] + 1,
                                ] in incon:  # if theres a larger element skip it
                                    pass
                                else:
                                    maxsi.append(test)
                            minsc = []  # collection of the lowest elements of the poset
                            maxsc = []  # collection of the highest elements of the poset
                            for test in compmod:  # look through all the elements
                                if [test[0] - 1, test[1]] in compmod or [
                                    test[0],
                                    test[1] - 1,
                                ] in compmod:  # if theres a smaller element skip it
                                    pass
                                else:
                                    minsc.append(test)  # otherwise add it to minimals
                                if [test[0] + 1, test[1]] in compmod or [
                                    test[0],
                                    test[1] + 1,
                                ] in compmod:  # if theres a larger element skip it
                                    pass
                                else:
                                    maxsc.append(test)
                            #print(i,j,minsi,maxsi,minsc,maxsc)
                            #if intersection(incon,compmod)==[]:
                            #    unacountedforzero=False
                            flavor='none'
                            for tminc in minsc:
                                for tmaxc in maxsc:
                                    for tmini in minsi:
                                        for tmaxi in maxsi:
                                            if tmini[0]<=tmaxi[0] and tmini[1]<=tmaxi[1] and tminc[0]<=tmaxc[0] and tminc[1]<=tmaxc[1]:
                                                if tminc[0]>tmini[0] and tminc[1]>=tmini[1] and tmaxc[0]<=tmaxi[0] and tmaxc[1]<=tmaxi[1]:
                                                    unacountedforzero=False
                                                    flavor='a'
                                                
                                                if tminc[0]>=tmini[0] and tminc[1]>tmini[1] and tmaxc[0]<=tmaxi[0] and tmaxc[1]<=tmaxi[1]:
                                                    unacountedforzero=False
                                                    flavor='b'
                                                    #print(tminc,tmini,tmaxc,tmaxi)
                                                
                                                if tminc[0]>=tmini[0] and tminc[1]>=tmini[1] and tmaxc[0]<tmaxi[0] and tmaxc[1]<=tmaxi[1]:
                                                    unacountedforzero=False
                                                    flavor='c'
                                                
                                                if tminc[0]>=tmini[0] and tminc[1]>=tmini[1] and tmaxc[0]<=tmaxi[0] and tmaxc[1]<tmaxi[1]:
                                                    unacountedforzero=False
                                                    flavor='d'
                        
                            if unacountedforzero and j==174 and result[j][i]==0: #1212

                                L.append(str([i,j,a[j][i],result[j][i],unacountedforzero,flavor,minsi,maxsi,minsc,maxsc]))
                                L.append('\n')
                                if True:
                                    L.append(f"The input {i} is this:")
                                    L.append("\n")
                                    L.append("".join(createimage(fin[i], n, h)))
                                    L.append("\n")
                                    #L.append(f"The comparison {j} is this:")
                                    #L.append("\n")
                                    #L.append("".join(createimage(fin[j], n, h)))
                                    #L.append('\n')
                                    L.append("\n")
                                    #L.append("".join(createimage(fin[j], n, h)))
    #L.append("".join(createimage(fin[409], n, h)))
    file1.writelines(L)
    file1.close()
                                    #print(f"The comparison {j} is this:")
                                    #print("".join(createimage(fin[j], n, h)))
                                    #print("")
                                    #print('')
                                #print('')

if scrapeunacountedzerostest:
    totalsize=2*(h**2+h)/2
    for i in range(1,h+1):
        totalsize=totalsize+(h+1-i)*(i**2+i)/2
    totalsize=int(totalsize)
    whole = genmat(n, h, order)
    mat = whole[0]
    fin=whole[1]
    a = np.ones( (totalsize,totalsize) )
    for i in range(totalsize):
        for j in range(totalsize):
            a[i][j]=mat[i][j]
    #printmatnice(mat)

    #print("")
    result=thesimpleralgorithmfunction(mat,h)
    printmatnice(result)
    
    for j in range(totalsize):
        for i in range(totalsize):
            #if result[j][i]==0:
            if True:
                wid=1
                side=0
                for point in fin[j]:
                    if side==0:
                        side=point[0]
                    if side!=0 and point[0]!=side:
                        wid=2
                if wid==2:
                    sub=True
                    for point in fin[i]:
                        if point in fin[j]:
                            pass
                        else:
                            sub=False

                    
                    if j==42:
                        print(i,j,a[j][i],result[j][i],sub)
                        if True:
                            print(f"The input {i} is this:")
                            print("".join(createimage(fin[i], n, h)))
                            #print("")
                            print(f"The comparison {j} is this:")
                            print("".join(createimage(fin[j], n, h)))
                            #print("")
                            #print('')
                            #print('')

if scrapeunacountedzerosonewide:
    totalsize=2*(h**2+h)/2
    for i in range(1,h+1):
        totalsize=totalsize+(h+1-i)*(i**2+i)/2
    totalsize=int(totalsize)
    whole = genmat(n, h, order)
    mat = whole[0]
    fin=whole[1]
    a = np.ones( (totalsize,totalsize) )
    for i in range(totalsize):
        for j in range(totalsize):
            a[i][j]=mat[i][j]
    #printmatnice(mat)

    #print("")
    result=thesimpleralgorithmfunction(mat,h)
    printmatnice(result)
    
    for j in range(totalsize):
        for i in range(totalsize):
            #if result[j][i]==0:
            if True:
                wid=1
                side=-3
                for point in fin[j]:
                    if side==-3:
                        side=point[0]
                    if side!=-3 and point[0]!=side:
                        wid=2
                if wid==1:
                    if a[j][i]!=0:
                        inside=True
                        for point in fin[j]:
                            if point in fin[i]:
                                pass
                            else:
                                inside=False
                        if inside:
                            perdiction=1
                        else:
                            perdiction=0
                        if False:
                            unacountedforzero=True
                            incon=fin[i]
                            compmod=fin[j]
                            minsi = []  # collection of the lowest elements of the poset
                            maxsi = []  # collection of the highest elements of the poset
                            for test in incon:  # look through all the elements
                                if [test[0] - 1, test[1]] in incon or [
                                    test[0],
                                    test[1] - 1,
                                ] in incon:  # if theres a smaller element skip it
                                    pass
                                else:
                                    minsi.append(test)  # otherwise add it to minimals
                                if [test[0] + 1, test[1]] in incon or [
                                    test[0],
                                    test[1] + 1,
                                ] in incon:  # if theres a larger element skip it
                                    pass
                                else:
                                    maxsi.append(test)
                            minsc = []  # collection of the lowest elements of the poset
                            maxsc = []  # collection of the highest elements of the poset
                            for test in compmod:  # look through all the elements
                                if [test[0] - 1, test[1]] in compmod or [
                                    test[0],
                                    test[1] - 1,
                                ] in compmod:  # if theres a smaller element skip it
                                    pass
                                else:
                                    minsc.append(test)  # otherwise add it to minimals
                                if [test[0] + 1, test[1]] in compmod or [
                                    test[0],
                                    test[1] + 1,
                                ] in compmod:  # if theres a larger element skip it
                                    pass
                                else:
                                    maxsc.append(test)
                            #print(i,j,minsi,maxsi,minsc,maxsc)
                            #if intersection(incon,compmod)==[]:
                            #    unacountedforzero=False
                            if a[j][i]==0:
                                unacountedforzero=False
                                perdiction=0
                            if unacountedforzero:
                                for tminc in minsc:
                                    for tmaxc in maxsc:
                                        for tmini in minsi:
                                            for tmaxi in maxsi:
                                                if tminc[0]>tmini[0] and tminc[1]>=tmini[1] and tmaxc[0]<=tmaxi[0] and tmaxc[1]<=tmaxi[1]:
                                                    unacountedforzero=False
                                                    perdiction=0
                                                
                                                if tminc[0]>=tmini[0] and tminc[1]>tmini[1] and tmaxc[0]<=tmaxi[0] and tmaxc[1]<=tmaxi[1]:
                                                    unacountedforzero=False
                                                    perdiction=0
                                                
                                                if tminc[0]>=tmini[0] and tminc[1]>=tmini[1] and tmaxc[0]<tmaxi[0] and tmaxc[1]<=tmaxi[1]:
                                                    unacountedforzero=False
                                                    perdiction=0
                                                
                                                if tminc[0]>=tmini[0] and tminc[1]>=tmini[1] and tmaxc[0]<=tmaxi[0] and tmaxc[1]<tmaxi[1]:
                                                    unacountedforzero=False
                                                    perdiction=0
                                if unacountedforzero:
                                    pass
                            
                            
                        if result[j][i]!=perdiction:
                            print(i,j,a[j][i],result[j][i],perdiction)#add perdiciton
                            if True:
                                print(f"The input {i} is this:")
                                print("".join(createimage(fin[i], n, h)))
                                print("")
                                print(f"The comparison {j} is this:")
                                print("".join(createimage(fin[j], n, h)))
                                print("")
                                print('')
                                print('')

if whatsactuallyoutthere:
    totalsize=2*(h**2+h)/2
    for i in range(1,h+1):
        totalsize=totalsize+(h+1-i)*(i**2+i)/2
    totalsize=int(totalsize)
    whole = genmat(n, h, order)
    mat = whole[0]
    fin=whole[1]
    a = np.ones( (totalsize,totalsize) )
    for i in range(totalsize):
        for j in range(totalsize):
            a[i][j]=mat[i][j]
    #printmatnice(mat)

    #print("")
    result=thesimpleralgorithmfunction(mat,h)
    #printmatnice(result)
    print('ziong')
    file1 = open('C:/Users/revil/OneDrive/Desktop/2x7davafor261.txt', 'w')
    L=[]
    for j in range(totalsize):
        for i in range(totalsize):
            #if result[j][i]==0:
            if True:
                wid=1
                side=-3
                for point in fin[i]:
                    if side==-3:
                        side=point[0]
                    if side!=-3 and point[0]!=side:
                        wid=2
                if wid==2:
                    wid=1
                    side=-3
                    for point in fin[j]:
                        if side==-3:
                            side=point[0]
                        if side!=-3 and point[0]!=side:
                            wid=2
                    if wid==2:
                        if True:
                            incon=fin[i]
                            compmod=fin[j]
                            minsi = []  # collection of the lowest elements of the poset
                            maxsi = []  # collection of the highest elements of the poset
                            for test in incon:  # look through all the elements
                                if [test[0] - 1, test[1]] in incon or [
                                    test[0],
                                    test[1] - 1,
                                ] in incon:  # if theres a smaller element skip it
                                    pass
                                else:
                                    minsi.append(test)  # otherwise add it to minimals
                                if [test[0] + 1, test[1]] in incon or [
                                    test[0],
                                    test[1] + 1,
                                ] in incon:  # if theres a larger element skip it
                                    pass
                                else:
                                    maxsi.append(test)
                            minsc = []  # collection of the lowest elements of the poset
                            maxsc = []  # collection of the highest elements of the poset
                            for test in compmod:  # look through all the elements
                                if [test[0] - 1, test[1]] in compmod or [
                                    test[0],
                                    test[1] - 1,
                                ] in compmod:  # if theres a smaller element skip it
                                    pass
                                else:
                                    minsc.append(test)  # otherwise add it to minimals
                                if [test[0] + 1, test[1]] in compmod or [
                                    test[0],
                                    test[1] + 1,
                                ] in compmod:  # if theres a larger element skip it
                                    pass
                                else:
                                    maxsc.append(test)
                            bottomi=924792384
                            for iii in minsi:
                                bottomi=min(bottomi,iii[1])
                            topi=-23423423
                            for iii in maxsi:
                                topi=max(topi,iii[1])
                            bottomc=924792384
                            for ccc in minsc:
                                bottomc=min(bottomc,ccc[1])
                            topc=-23423423
                            for ccc in maxsc:
                                topc=max(topc,ccc[1])
                            typeee="other"

                            if bottomi==bottomc and topi==topc:
                                typeee="same"
                            if bottomi<bottomc and topi>topc:
                                typeee="surround"
                            if bottomi==bottomc and topi>topc:
                                typeee="pan"
                            if bottomi<bottomc and topi==topc:
                                typeee="skyscraper"
                            
                            
                            intervalnumber=0
                            for tminc in minsc:
                                for tmaxc in maxsc:
                                    if tminc[0]<=tmaxc[0] and tminc[1]<=tmaxc[1]:
                                        intervalnumber=intervalnumber+1
                            
                            b=-4234234
                            for point in fin[j]:
                                if point[0]==1 and point[1]>b:
                                    b=point[1]
                            
                            abutliketopleft=-4234234
                            for point in fin[j]:
                                if point[0]==0 and point[1]>abutliketopleft:
                                    abutliketopleft=point[1]

                            c=4234234
                            for point in fin[j]:
                                if point[0]==0 and point[1]<c:
                                    c=point[1]
                            
                            d=4234234
                            for point in fin[j]:
                                if point[0]==1 and point[1]<d:
                                    d=point[1]

                            low=bottomi
                            high=topi

                            perdiction=2948710752045

                            if typeee=="other" or int(a[j][i])==0:
                                perdiction=0

                            


                            if perdiction!=0:
                                if typeee=="same" or typeee=="surround":
                                    if intervalnumber==1 or intervalnumber==3:
                                        perdiction=1
                                    elif intervalnumber==2:
                                        perdiction=-1
                                    else:
                                        print('yuhoh1')
                                elif typeee=="pan" or typeee=="skyscraper":
                                    if intervalnumber==1 or intervalnumber==3:
                                        perdiction=-1
                                    elif intervalnumber==2:
                                        perdiction=1
                                    else:
                                        print('yuhoh2')
                                elif typeee=="other":
                                    print('yuhoh3')
                                else:
                                    print('yuhoh4')
                            



                            if j==181 and typeee=="skyscraper":
                                print(str([i,j,a[j][i],result[j][i],typeee,abutliketopleft,b,c,d,intervalnumber,perdiction]))
                                if True:
                                    pass
                                    #print(f"The input {i} is this:")
                                    #print("".join(createimage(fin[i], n, h)))
                                    #print(f"The comparison {j} is this:")
                                    #print("".join(createimage(fin[j], n, h)))
                                    #print("")
                                #L.append("\n\n")
                                    #L.append("".join(createimage(fin[j], n, h)))
    #L.append("".join(createimage(fin[409], n, h)))
    file1.writelines(L)
    file1.close()

if fulltest:
    totalsize=2*(h**2+h)/2
    for i in range(1,h+1):
        totalsize=totalsize+(h+1-i)*(i**2+i)/2
    totalsize=int(totalsize)
    whole = genmat(n, h, order)
    mat = whole[0]
    fin=whole[1]
    a = np.ones( (totalsize,totalsize) )
    for i in range(totalsize):
        for j in range(totalsize):
            a[i][j]=mat[i][j]
    #printmatnice(mat)

    #print("")
    result=thesimpleralgorithmfunction(mat,h)
    #printmatnice(result)
    print('ziong')
    file1 = open('C:/Users/revil/OneDrive/Desktop/2x7davafor261.txt', 'w')
    L=[]
    for j in range(totalsize):
        for i in range(totalsize):
            #if result[j][i]==0:
            if True:
                wid=1
                side=-3
                for point in fin[i]:
                    if side==-3:
                        side=point[0]
                    if side!=-3 and point[0]!=side:
                        wid=2
                if wid==2:
                    wid=1
                    side=-3
                    for point in fin[j]:
                        if side==-3:
                            side=point[0]
                        if side!=-3 and point[0]!=side:
                            wid=2
                    if wid==2:
                        if True:
                            incon=fin[i]
                            compmod=fin[j]
                            minsi = []  # collection of the lowest elements of the poset
                            maxsi = []  # collection of the highest elements of the poset
                            for test in incon:  # look through all the elements
                                if [test[0] - 1, test[1]] in incon or [
                                    test[0],
                                    test[1] - 1,
                                ] in incon:  # if theres a smaller element skip it
                                    pass
                                else:
                                    minsi.append(test)  # otherwise add it to minimals
                                if [test[0] + 1, test[1]] in incon or [
                                    test[0],
                                    test[1] + 1,
                                ] in incon:  # if theres a larger element skip it
                                    pass
                                else:
                                    maxsi.append(test)
                            minsc = []  # collection of the lowest elements of the poset
                            maxsc = []  # collection of the highest elements of the poset
                            for test in compmod:  # look through all the elements
                                if [test[0] - 1, test[1]] in compmod or [
                                    test[0],
                                    test[1] - 1,
                                ] in compmod:  # if theres a smaller element skip it
                                    pass
                                else:
                                    minsc.append(test)  # otherwise add it to minimals
                                if [test[0] + 1, test[1]] in compmod or [
                                    test[0],
                                    test[1] + 1,
                                ] in compmod:  # if theres a larger element skip it
                                    pass
                                else:
                                    maxsc.append(test)
                            bottomi=924792384
                            for iii in minsi:
                                bottomi=min(bottomi,iii[1])
                            topi=-23423423
                            for iii in maxsi:
                                topi=max(topi,iii[1])
                            bottomc=924792384
                            for ccc in minsc:
                                bottomc=min(bottomc,ccc[1])
                            topc=-23423423
                            for ccc in maxsc:
                                topc=max(topc,ccc[1])
                            typeee="other"

                            if bottomi==bottomc and topi==topc:
                                typeee="same"
                            if bottomi<bottomc and topi>topc:
                                typeee="surround"
                            if bottomi==bottomc and topi>topc:
                                typeee="pan"
                            if bottomi<bottomc and topi==topc:
                                typeee="skyscraper"
                            
                            
                            intervalnumber=0
                            for tmini in minsi:
                                for tmaxi in maxsi:
                                    if tmini[0]<=tmaxi[0] and tmini[1]<=tmaxi[1]:
                                        intervalnumber=intervalnumber+1
                            
                            b=-4234234
                            for point in fin[j]:
                                if point[0]==1 and point[1]>b:
                                    b=point[1]
                            
                            abutliketopleft=-4234234
                            for point in fin[j]:
                                if point[0]==0 and point[1]>abutliketopleft:
                                    abutliketopleft=point[1]

                            c=4234234
                            for point in fin[j]:
                                if point[0]==0 and point[1]<c:
                                    c=point[1]
                            
                            d=4234234
                            for point in fin[j]:
                                if point[0]==1 and point[1]<d:
                                    d=point[1]

                            low=bottomi
                            high=topi

                            perdiction=2948710752045
                            reason=0

                            if typeee=="other" or int(a[j][i])==0:
                                perdiction=0
                                reason=1

                            incon=fin[i]
                            compmod=fin[j]
                            minsi = []  # collection of the lowest elements of the poset
                            maxsi = []  # collection of the highest elements of the poset
                            for test in incon:  # look through all the elements
                                if [test[0] - 1, test[1]] in incon or [
                                    test[0],
                                    test[1] - 1,
                                ] in incon:  # if theres a smaller element skip it
                                    pass
                                else:
                                    minsi.append(test)  # otherwise add it to minimals
                                if [test[0] + 1, test[1]] in incon or [
                                    test[0],
                                    test[1] + 1,
                                ] in incon:  # if theres a larger element skip it
                                    pass
                                else:
                                    maxsi.append(test)
                            minsc = []  # collection of the lowest elements of the poset
                            maxsc = []  # collection of the highest elements of the poset
                            for test in compmod:  # look through all the elements
                                if [test[0] - 1, test[1]] in compmod or [
                                    test[0],
                                    test[1] - 1,
                                ] in compmod:  # if theres a smaller element skip it
                                    pass
                                else:
                                    minsc.append(test)  # otherwise add it to minimals
                                if [test[0] + 1, test[1]] in compmod or [
                                    test[0],
                                    test[1] + 1,
                                ] in compmod:  # if theres a larger element skip it
                                    pass
                                else:
                                    maxsc.append(test)
                            #print(i,j,minsi,maxsi,minsc,maxsc)
                            #if intersection(incon,compmod)==[]:
                            #    unacountedforzero=False
                            flavor='none'
                            for tminc in minsc:
                                for tmaxc in maxsc:
                                    for tmini in minsi:
                                        for tmaxi in maxsi:
                                            if tmini[0]<=tmaxi[0] and tmini[1]<=tmaxi[1] and tminc[0]<=tmaxc[0] and tminc[1]<=tmaxc[1]:
                                                if tminc[0]>tmini[0] and tminc[1]>=tmini[1] and tmaxc[0]<=tmaxi[0] and tmaxc[1]<=tmaxi[1]:
                                                    perdiction=0
                                                    reason=2
                                                
                                                if tminc[0]>=tmini[0] and tminc[1]>tmini[1] and tmaxc[0]<=tmaxi[0] and tmaxc[1]<=tmaxi[1]:
                                                    perdiction=0
                                                    reason=3
                                                
                                                if tminc[0]>=tmini[0] and tminc[1]>=tmini[1] and tmaxc[0]<tmaxi[0] and tmaxc[1]<=tmaxi[1]:
                                                    perdiction=0
                                                    reason=4
                                                
                                                if tminc[0]>=tmini[0] and tminc[1]>=tmini[1] and tmaxc[0]<=tmaxi[0] and tmaxc[1]<tmaxi[1]:
                                                    perdiction=0
                                                    reason=5

                            shape="other"
                            if abutliketopleft==b and c==d:
                                shape="square"
                            if abutliketopleft>b and c==d:
                                shape="b"
                            if abutliketopleft==b and c>d:
                                shape="q"
                            if abutliketopleft>b and c>d:
                                shape="lightning"
                            if shape=="other":
                                print('huhh how1')

                            
                            aibicidi=findbiandbc(fin[i])
                            ai=aibicidi[0]
                            bi=aibicidi[1]
                            ci=aibicidi[2]
                            di=aibicidi[3]

                            if typeee=="skyscraper":
                                if shape=="q" or shape=="lightning":
                                    perdiction=0
                                    reason=6
                                else:
                                    if shape=="square":
                                        if ci>=c+1:
                                            perdiction=0
                                            reason=7
                                    if shape=="b":
                                        if bi<ai and ci>=c+1:
                                            perdiction=0
                                            reason=8
                            
                            if typeee=="surround":
                                if shape!="square":
                                    perdiction=0
                                    reason=9
                                else:
                                    pass1=True
                                    for point in fin[j]:
                                        if point in fin[i]:
                                            pass
                                        else:
                                            pass1=False
                                    if pass1:
                                        pass2=True
                                        for point in fin[j]:
                                            if point in fin[i]:
                                                pass
                                            else:
                                                if point[1]>b:
                                                    if point[0]==0:
                                                        pass
                                                    else:
                                                        pass2=False
                                        if pass2:
                                            pass3=True
                                            for point in fin[j]:
                                                if point in fin[i]:
                                                    pass
                                                else:
                                                    if point[1]<c:
                                                        if point[0]==1:
                                                            pass
                                                        else:
                                                            pass3=False
                                            if pass3:
                                                pass

                                            else:
                                                perdiction=0
                                                reason=11
                                        else:
                                            perdiction=0
                                            reason=12
                                    else:
                                        perdiction=0
                                        reason=13

                            if typeee=="same":
                                if shape=="square" or shape=="lightning":
                                    pass
                                else:
                                    if shape=="q":
                                        if bi<ai and d+1<=ci and ci<=c-1:
                                            perdiction=0
                                            reason=15
                                    if shape=="b":
                                        if b+1<=bi and bi<=abutliketopleft-1 and ci>di:
                                            perdiction=0
                                            reason=16

                            if typeee=="pan":
                                if shape=="b" or shape=="lightning":
                                    perdiction=0
                                    reason=17
                                else:
                                    if shape=="square":
                                        if bi<=b-1:
                                            perdiction=0
                                            reason=18
                                    
                                    if shape=="q":
                                        if bi<=b-1 and ci>di:
                                            perdiction=0
                                            reason=19

                            if perdiction!=0:
                                if typeee=="same" or typeee=="surround":
                                    if intervalnumber==1 or intervalnumber==3:
                                        perdiction=1
                                        reason=20
                                    elif intervalnumber==2:
                                        perdiction=255
                                        reason=21
                                    else:
                                        print('yuhoh1')
                                elif typeee=="pan" or typeee=="skyscraper":
                                    if intervalnumber==1 or intervalnumber==3:
                                        perdiction=255
                                        reason=22
                                    elif intervalnumber==2:
                                        perdiction=1
                                        reason=23
                                    else:
                                        print('yuhoh2')
                                elif typeee=="other":
                                    print('yuhoh3')
                                else:
                                    print('yuhoh4')
                            



                            if j==75 and i==95:
                                print(str([i,j,a[j][i],result[j][i],perdiction,reason,shape,typeee,intervalnumber,abutliketopleft,b,c,d,ai,bi,ci,di]))
                                if True:
                                    print(f"The input {i} is this:")
                                    print("".join(createimage(fin[i], n, h)))
                                    print(f"The comparison {j} is this:")
                                    print("".join(createimage(fin[j], n, h)))
                                    print("")
                                    pass
                                #L.append("\n\n")
                                    #L.append("".join(createimage(fin[j], n, h)))
    #L.append("".join(createimage(fin[409], n, h)))
    file1.writelines(L)
    file1.close()
                                    #print(f"The comparison {j} is this:")
                                    #print("".join(createimage(fin[j], n, h)))
                                    #print("")
                                    #print('')
                                #print('')
# 
# dont do elements in a for loop do range(len(...))
if temptestin:
    whole = genmat(n, h, order)
    mat = whole[0]
    fin=whole[1]
    totalsize=int(len(fin))
    #print(totalsize)
    a = np.ones( (totalsize,totalsize) )
    for i in range(totalsize):
        for j in range(totalsize):
            a[i][j]=mat[i][j]
    matrix = np.ones( (totalsize,totalsize) )
    for i in range(totalsize):
        for j in range(totalsize):
            matrix[i][j]=mat[i][j]
    matrix=thetruealgorithm(matrix,fin,n,h)
    #printmatnice(matrix)
    
    for j in range(totalsize):
        print(i,j,matrix[j][i])
        print("".join(createimage(fin[j], n, h)))
        print('')
    print('yuhoh')
    print('yuhoh')
    print('yuhoh')
    print('yuhoh')
    print('yuhoh')
    list=[193, 194, 199, 200, 220, 222, 226, 228, 233, 234, 239, 240, 260, 262, 266, 268, 274, 275, 276, 277, 278, 284, 286, 292, 293, 301, 302, 304, 306, 307, 308, 313, 316, 318, 324, 325, 326, 327, 328, 334, 
336, 342, 343, 351, 352, 354, 356, 357, 358, 363, 366, 368, 407, 408, 410, 411, 412, 417, 418, 420, 421, 422, 427, 428, 447, 448, 467, 468, 469, 470, 471, 474, 477, 478, 479, 480, 482, 487, 488, 489, 490, 491, 494, 497, 498, 499, 500, 502, 508, 509, 510, 511, 512, 514, 515, 516, 517, 518, 519, 520, 521, 522, 528, 529, 530, 531, 532, 534, 536, 537, 538, 539, 540, 547, 549, 555, 556, 558, 560, 561, 563, 564, 565, 566, 567, 573, 574, 575, 576, 577, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 593, 594, 596, 597, 598, 599, 600, 601, 604, 607, 608, 609, 610, 611, 612, 613, 614, 615, 617, 619, 620, 621, 622, 623, 624, 625, 627, 633, 634, 635, 636, 637, 638, 639, 640, 641, 643, 645, 647, 651, 652, 653, 654, 655, 656, 660, 661, 662, 663, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 677]
    c = np.ones( (len(list),len(list)) )
    for i in range(len(list)):
        for j in range(len(list)):
            c[j][i]=-73
    for j in list:
        tl=[]
        for i in list:
            c[list.index(j)][list.index(i)]=matrix[j][i]
    #print(c)
    #print('d')
    d=rref2(c)
    #print(d)
    for j in range(len(d)):
        print(d[j])
    print(np.linalg.det(c))
    print(np.linalg.det(d))
                #print("".join(createimage(fin[j], n, h)))
                #print('')
                #print("".join(createimage(fin[i], n, h)))
                #print('')
                #print(matrix[j][i],j,i)
            
#                print(i)
#                print("".join(createimage(fin[i], n, h)))
#                print(j)
#                print("".join(createimage(fin[j], n, h)))
#                print("")
    #print(matrix[668][665])
    #print(matrix[665][665])
    #print(matrix[668][668])
    #print(matrix[665][668])
    #print(matrix[180][665])
    #print(matrix[180][668])
    #print("DSKLFJ:SDLFSD")
    #print("DSKLFJ:SDLFSD")
    #print("DSKLFJ:SDLFSD")
    #print("DSKLFJ:SDLFSD")
    #print("DSKLFJ:SDLFSD")
    #print("DSKLFJ:SDLFSD")
    #print("DSKLFJ:SDLFSD")

    #for i in range(totalsize):
    ##    if matrix[i][i]==0:
    #        print(matrix[i][i],i)
    #        print(i)
    #        print("".join(createimage(fin[i], n, h)))
    #print(matrix[47][9])


    """
    perdic = np.ones( (totalsize,totalsize) )
    for i in range(totalsize):
        for j in range(totalsize):
            perdic[i][j]=-90
    #printmatnice(perdic)
    if False:    
        for i in range(totalsize):
            for j in range(totalsize):
                incon=fin[i]
                compmod=fin[j]
                minsi = []  # collection of the lowest elements of the poset
                maxsi = []  # collection of the highest elements of the poset
                for test in incon:  # look through all the elements
                    if [test[0] - 1, test[1]] in incon or [
                        test[0],
                        test[1] - 1,
                    ] in incon:  # if theres a smaller element skip it
                        pass
                    else:
                        minsi.append(test)  # otherwise add it to minimals
                    if [test[0] + 1, test[1]] in incon or [
                        test[0],
                        test[1] + 1,
                    ] in incon:  # if theres a larger element skip it
                        pass
                    else:
                        maxsi.append(test)
                minsc = []  # collection of the lowest elements of the poset
                maxsc = []  # collection of the highest elements of the poset
                for test in compmod:  # look through all the elements
                    if [test[0] - 1, test[1]] in compmod or [
                        test[0],
                        test[1] - 1,
                    ] in compmod:  # if theres a smaller element skip it
                        pass
                    else:
                        minsc.append(test)  # otherwise add it to minimals
                    if [test[0] + 1, test[1]] in compmod or [
                        test[0],
                        test[1] + 1,
                    ] in compmod:  # if theres a larger element skip it
                        pass
                    else:
                        maxsc.append(test)
                iintervals=[]
                for tmini in minsi:
                    for tmaxi in maxsi:
                        if tmini[0]<=tmaxi[0] and tmini[1]<=tmaxi[1]:
                            iintervals.append([tmini,tmaxi])
                powerii=[]
                xxx = len(iintervals)
                for f in range(1 << xxx):
                    powerii.append([iintervals[g] for g in range(xxx) if (f & (1 << g))])
                #if i==73 and j==68:
                #    print(fin[i],fin[j])
                #    for iii in iintervals:
                #        print(iii)
                for colec in powerii:
                    if colec!=[]:
                        realized=[]
                        for intv in colec:
                            for pp0 in range(intv[0][0],intv[1][0]+1):
                                for pp1 in range(intv[0][1],intv[1][1]+1):
                                    if [pp0,pp1] not in realized:
                                        realized.append([pp0,pp1])
                        ismod=False
                        for kk in range(totalsize):
                            if equallist(fin[kk],realized):
                                ismod=True
                        
                        cont=True
                        for pointtttt in realized:
                            if pointtttt not in fin[j]:
                                cont=False
                        if cont and ismod:
                            #print(realized)
                            if i==73 and j==68:
                                print(realized)
                            if modsizedetector(realized)==modsizedetector(fin[j]):
                                perdic[j][i]=(-1)**(len(colec)+1)
                if perdic[j][i]==-90:
                    perdic[j][i]=0
    print('')
    #printmatnice(perdic)
    #for j in range(totalsize):
    #    for i in range(totalsize):
    #        if perdic[j][i]!=matrix[j][i]:
    #            #print(j,i,perdic[j][i],matrix[j][i])
    #            pass
    for j in range(totalsize):
        for i in range(totalsize):
            if i==70 and matrix[j][i]!=0:
                #print(i,j,matrix[j,i])
                pass
                #print(f"The comparison {j} is this:")
                #print("".join(createimage(fin[j], n, h)))for j in range(totalsize):
    for j in range(totalsize):
        for i in range(totalsize):
            if matrix[j][i]>1 or matrix[j][i]<-1:
                print(i,j,matrix[j,i])
                print("".join(createimage(fin[i], n, h)))
                print('')
                print("".join(createimage(fin[j], n, h)))
                pass


    #i=476
    #j=478
    #tests=[2, 32, 92, 197, 12, 23, 13, 373, 343, 5, 35, 95, 340, 240, 200, 8, 38, 194, 134, 98, 11, 89, 65, 41, 29, 24, 14]
    #for test in tests:
    #    print(matrix[test][476])
    #for j in range(totalsize):
    #    if matrix[j][139]==1:
    #        print(j)
    #        print("".join(createimage(fin[j], n, h)))
    #        print('')
    
    #print(matrix[478][476])
    
    #print(matrix[61][21])
    #for test in tests:
        #print(matrix[test][i])
    """