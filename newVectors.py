import catmaid as catmaid
import json
import numpy as np
import scipy.io
from tempfile import TemporaryFile
from StringIO import StringIO
import sys
from joblib import Parallel, delayed, dump, load
import time
import datetime
import itertools
import getpass


def printTimeDetails(i, n, st):
    # cute function that prints a prediction on how long it will take given
    # the amount of time the past skeletons have taken
    dt = time.time()-st
    sys.stdout.write(str(round(float(i)/n*100, 3)) +
                     '% complete | Elapsed: ' +
                     str(datetime.timedelta(seconds=round(dt))) +
                     ' | ETA: ' +
                     str(datetime.timedelta(seconds=(round(dt*n/float(i))
                                                     - round(dt)))) +
                     '          \r')
    sys.stdout.flush()


def DownLoadSkels(c):
    # downloads the skeleton JSON files to a folder, a time consuming process
    # that should be done overnight
    # wd = c.fetchJSON('http://catmaid.hms.harvard.edu/9/wiringdiagram/json')
    skList = c.skeleton_ids()
    np.savetxt('AAAskList', skList, delimiter=',')
    n = len(skList)
    i = 0
    st = time.time()
    for skid in skList:
        i += 1
        printTimeDetails(i, n, st)
        saveSkelJSON(skid, c)
    print ""


def ParallelDLS(c, n_jobs=-1):
    # Downloads Skeletons using Parallel library
    skList = c.skeleton_ids()
    np.savetxt('AAAskList', skList, delimiter=',')
    Parallel(n_jobs=n_jobs,verbose=50)(delayed(parallelSSJ)(skid,c) for skid in skList)


def saveSkelJSON(skid, c):
    outfile = open('skelJSONS/testfolder/sk' + str(skid) + '.json', 'w')
    skjs = c.fetchJSON('http://catmaid.hms.harvard.edu/9/skeleton/' +
                       str(skid) + '/json')
    json.dump(skjs, outfile, indent=4)


def parallelSSJ(skid, c):
    # Saves Skeletons as JSON file
    # Requires a connection object to be passed with it
    outfile = open('skelJSONS/testfolder/sk' + str(skid)
                   + '.json', 'w')
    skjs = c.fetchJSON('http://catmaid.hms.harvard.edu/9/skeleton/'
                       + str(skid) + '/json')
    json.dump(skjs, outfile, indent=4)
    outfile.close()


def GetNeurons(skList, c):
    # downloads the neurons and saves them in a dictionary with
    # Neurons[sknumb] = neuron
    Neurons = {}
    n = len(skList)
    i = 0
    st = time.time()
    for sk in skList:
        i += 1
        printTimeDetails(i, n, st)
        neuron = getNeuron(sk, c)
        Neurons[sk] = neuron

    print ""
    outfile = open('neurons.json', 'w')
    json.dump(Neurons, outfile, indent=4)
    return Neurons


def getNeuron(skid, c):
    sk = int(skid)
    skel = c.skeleton(sk)
    neuron = {}
    if skel is None:
            print str(skid)
            return neuron
    if len(skel['vertices'].keys()) >= 10:
            neuron = catmaid.Neuron(skel)
    return neuron


def parallelGetNeuron(skid, c):
    sk = int(skid)
    skel = c.skeleton(sk)
    neuron = {}
    if skel is None:
            print str(skid)
            return neuron
    if len(skel['vertices'].keys()) >= 10:
            neuron = catmaid.Neuron(skel)
    return neuron


def parallelGetNeurons(skList, c, n_jobs):
    Neurons = {}
    Neurons = Parallel(n_jobs=n_jobs, verbose=50)(delayed(parallelGetNeuron)(skid, c) for skid in skList)
    return Neurons


def GetEdgeLengths(skeletonList, c):
    # from a list of skeletonIDs it generates a list of the edgeLengths of
    # each skel
    distances = []
    for skid in skeletonList:
        sk = c.skeleton(skid)
        neuron = {}
        if len(sk['vertices'].keys()) >= 10:
                neuron = catmaid.Neuron(sk)
                dist = sum([neuron.distance(p, c) for (p, c)
                            in neuron.dgraph.edges_iter()])
        distances.append(dist)
    return np.array(distances)


def PreLoadedGetEdgeLengths(c, neurons={}):
    # if all the jsons are loaded through DownloadSkels(c) then this can be run
    # faster than GetEdgeLength
    skList = np.genfromtxt('AAAskList')
    distances = []
    n = len(skList)
    i = 0
    st = time.time()
    for skid in skList:
        i += 1
        printTimeDetails(i, n, st)
        dist = getDist(skid, neurons, c)
        distances.append(dist)

    dists = np.array(distances)
    skDists = np.column_stack((skList, dists))
    skDists2 = [[int(s), float(d)] for [s, d] in skDists]
    print ""
    return skDists2


def ParallelGetDist(c, neurons={}):
    skList = np.genfromtxt('AAAskList')
    distances = Parallel(n_jobs = 4)(delayed(getDist)(skid,neurons,c) for skid in skList)
    dists = np.array(distances)
    skDists = np.column_stack((skList, dists))
    skDists2 = [[int(s), float(d)] for [s, d] in skDists]
    return skDists2


def getDist(skid, neurons, c):
    infile = open('skelJSONS/sk' + str(int(skid)) + '.json', 'r')
    sk = json.load(infile)
    dist = 0.0
    if len(sk['vertices'].keys()) >= 10:
        if(neurons == {}):
            neuron = catmaid.Neuron(sk)
        else:
            neuron = neurons[skid]
        dist = sum([neuron.distance(p, c) for (p, c)
                   in neuron.dgraph.edges_iter()])
    return dist


def betweenDist(n1, v1, n2, v2):
        return sum([(n1.vertices[v1][k] - n2.vertices[v2][k]) ** 2.
                   for k in ('x', 'y', 'z')]) ** 0.5


def getCloseness(n1, n2, closeDist, c):
        # first get all nodes near
        n1ax = n1.axons
        n1den = n1.dendrites()
        n2ax = n2.axons
        n2den = n2.dendrites()
        n1axClose = []
        n2axClose = []
        n1denClose = []
        n2denClose = []
        # then go through edges and if a node near it is in add the edge dist
        for node1 in n1den.nodes():
            for ax in n2ax.keys():
                for node2 in n2ax[ax]['tree'].nodes():
                    d = betweenDist(n1, node1, n2, node2)
                    if d <= closeDist:
                        n1denClose.append(node1)
                        n2axClose.append(node2)
        for node1 in n2den.nodes():
            for ax in n1ax.keys():
                for node2 in n1ax[ax]['tree'].nodes():
                    d = betweenDist(n1, node2, n2, node1)
                    if d <= closeDist:
                        n1axClose.append(node2)
                        n2denClose.append(node1)
        n1denClose = np.unique(n1denClose)
        n2denClose = np.unique(n2denClose)
        n1axClose = np.unique(n1axClose)
        n2axClose = np.unique(n2axClose)
        # return n1axClose, n1denClose, n2axClose, n2denClose
        n1axDist = 0.0
        n1denDist = 0.0
        n2axDist = 0.0
        n2denDist = 0.0
        if(n1axClose != []):
            for ax in n1ax.keys():
                for (p, c) in n1ax[ax]['tree'].edges():
                    if (p in n1axClose) & (c in n1axClose):
                        n1axDist += n1.distance(p, c)
        if(n2axClose != []):
            for ax in n2ax.keys():
                for (p, c) in n2ax[ax]['tree'].edges():
                    if (p in n2axClose) & (c in n2axClose):
                        n2axDist += n2.distance(p, c)
        if(n1denClose != []):
            for (p, c) in n1den.edges():
                if (p in n1denClose) & (c in n1denClose):
                    n1denDist += n1.distance(p, c)
        if(n2denClose != []):
            for (p, c) in n2den.edges():
                if (p in n2denClose) & (c in n2denClose):
                    n2denDist += n2.distance(p, c)
        return [n1axDist, n1denDist, n2axDist, n2denDist]


def getDistBetween(axnode, denNodes, closeDist):
        # takes xyzidof 1 axnode and an array of all xyzid den nodes
        difmat = denNodes[:, :3] - axnode[:3]
        # can do as vector
        sqd = np.sum(difmat ** 2., 1)
        deninds = np.where(sqd <= closeDist)[0]
        return deninds


def getDistNumpy(n1, n2, closeDist):
    close = closeDist**2
    axe1 = n1.axons
    axe2 = n2.axons
    D1 = n1.dendrites()
    D2 = n2.dendrites()
    [n1den, n1axT] = getNumpyArrays(n1)
    [n2den, n2axT] = getNumpyArrays(n2)
    dens1 = np.empty([0, 4])
    dens1inds = np.array([])
    axs1 = np.empty([0, 4])
    dens2 = np.empty([0, 4])
    dens2inds = np.array([])
    axs2 = np.empty([0, 4])
    for n1ax in n1axT:
        whoClose = Parallel(n_jobs = 1)(delayed(getDistBetween)(ax,n2den,close) for ax in n1ax)
        axs1 = np.concatenate((axs1, n1ax[np.where(whoClose)[0]]))
        dens2inds = np.append(dens2inds, list(itertools.chain(*whoClose)))
    for n2ax in n2axT:
        whoClose = Parallel(n_jobs = 1)(delayed(getDistBetween)(ax,n1den,close) for ax in n2ax)
        axs2 = np.concatenate((axs2, n2ax[np.where(whoClose)[0]]))
        dens1inds = np.append(dens1inds, list(itertools.chain(*whoClose)))
    dens2inds = np.unique(dens2inds).astype(int)
    dens1inds = np.unique(dens1inds).astype(int)

    dens1 = n1den[dens1inds]
    dens2 = n2den[dens2inds]
    n1aD = 0.0
    n2aD = 0.0
    n1dD = 0.0
    n2dD = 0.0
    # axs1 = np.unique(axs1)
    # axs2 = np.unique(axs2)
    dens1 = np.unique(dens1)
    dens2 = np.unique(dens2)
    ax1 = axs1[:, 3]
    ax2 = axs2[:, 3]
    if(ax1 != []):
        for ax in axe1.keys():
            for (p, c) in axe1[ax]['tree'].edges():
                if (int(p) in ax1) & (int(c) in ax1):
                    n1aD += n1.distance(p, c)
    if(ax2 != []):
        for ax in axe2.keys():
            for (p, c) in axe2[ax]['tree'].edges():
                if(int(p) in ax2) & (int(c) in ax2):
                    n2aD += n2.distance(p, c)
    if(dens1 != []):
        for (p, c) in D1.edges():
            if (int(p) in dens1) & (int(c) in dens1):
                n1dD += n1.distance(p, c)
    if(dens2 != []):
        for (p, c) in D2.edges():
            if(int(p) in dens2) & (int(c) in dens2):
                n2dD += n2.distance(p, c)
    return n1aD, n1dD, n2aD, n2dD


def getNumpyArrays(n1):
    dendrites = n1.dendrites().nodes()
    vert = n1.vertices
    xyzden = np.zeros([len(dendrites), 4])
    for i in np.arange(len(dendrites)):
            xyzden[i, 0] = vert[dendrites[i]]['x']
            xyzden[i, 1] = vert[dendrites[i]]['y']
            xyzden[i, 2] = vert[dendrites[i]]['z']
            xyzden[i, 3] = int(dendrites[i])

    axons = n1.axons
    xyzaxToT = []
    for ax in axons.keys():
            axnods = axons[ax]['tree'].nodes()
            if axnods != []:
                xyzax = np.zeros([len(axnods), 4])
                for i in np.arange(len(axnods)):
                    xyzax[i, 0] = vert[axnods[i]]['x']
                    xyzax[i, 1] = vert[axnods[i]]['y']
                    xyzax[i, 2] = vert[axnods[i]]['z']
                    xyzax[i, 3] = int(axnods[i])
                xyzaxToT.append(xyzax)
    return xyzden, xyzaxToT


def runThroughNeurons(Neurons, closeDist):
    n = len(Neurons)
    neuronDistDict = dict()
    st = time.time()
    for i in np.arange(n-1):
        # printTimeDetails(i+1,n,st)
        if(Neurons[i] != {}):
            n1 = Neurons[i]
            neuronDistDict[n1.name] = dict()
            for j in np.arange(i+1, n):
                if(Neurons[j] != {}):
                    n2 = Neurons[j]
                    printTimeDetails(n*i+j, (n*(n+1))/2, st)
                    # print n1.name,n2.name
                    a1, d1, a2, d2 = getDistNumpy(n1, n2, closeDist)

                    neuronDistDict[n1.name][n2.name] = dict()
                    neuronDistDict[n1.name][n2.name]['ax1'] = a1
                    neuronDistDict[n1.name][n2.name]['den1'] = d1
                    neuronDistDict[n1.name][n2.name]['ax2'] = a2
                    neuronDistDict[n1.name][n2.name]['den2'] = d2
    return neuronDistDict


def execute():
    try:
        # first trys to access os environment elements
        c = catmaid.connect()
    except KeyError:
        Server = str(raw_input("Enter Catmaid Server: "))
        Proj = str(raw_input("Enter Catmaid Project: "))
        U_name = str(raw_input("Enter Catmaid UserName: "))
        P_word = getpass.getpass("Enter Catmaid Password: ")
        c = catmaid.Connection(Server, U_name, P_word, Proj)
    # first get skels from catmaid
    print "extracting Skeletons from catmaid...."
    # DownLoadSkels(c)
    skList = np.genfromtxt('AAAskList')
    # then use skelList to download Neurons
    print "saving Neurons to a dictionary"
    Neurons = GetNeurons(skList, c)
    # then get distances
    print "Calculating Neuron Length"
    Dists = PreLoadedGetEdgeLengths(c, Neurons)
    return skList, Neurons, Dists


def parallelExecute():
    try:
        # first trys to access os environment elements
        c = catmaid.connect()
    except KeyError:
        Server = str(raw_input("Enter Catmaid Server: "))
        Proj = str(raw_input("Enter Catmaid Project: "))
        U_name = str(raw_input("Enter Catmaid UserName: "))
        P_word = getpass.getpass("Enter Catmaid Password: ")
        c = catmaid.Connection(Server, U_name, P_word, Proj)
    ParallelDLS(c, -1)
    skList = np.genfromtxt('AAAskList')
    Neurons = parallelGetNeurons(skList, c, -1)
    return skList, Neurons

def GetDendDists(ConIds):
    # ConIds should be of the form <CONID><PRESKELID><POSTSKELID>
    # should be a list of all connectors from ORI defined cells to a common
    # target
    for conRow in ConIds:
        CID, PRESK, POSTSK = conRow
        

