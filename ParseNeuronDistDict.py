import numpy


def parseDistDict(ndd):
    nronsOG = ndd.keys()
    nrons = nronsOG
    for nr in nronsOG:
        if(nr in ndd.keys()):
            newrons = ndd[nr].keys()
            for new in newrons:
                if new not in nrons:
                    nrons.append(new)
    denadj = numpy.zeros([len(nrons), len(nrons)])
    axadj = numpy.zeros([len(nrons), len(nrons)])

    for nr in ndd.keys():
        inda = nrons.index(nr)
        for new in ndd[nr].keys():
            indb = nrons.index(new)
            axadj[inda][indb] = ndd[nr][new]['ax1']
            denadj[inda][indb] = ndd[nr][new]['den1']
            axadj[indb][inda] = ndd[nr][new]['ax2']
            denadj[indb][inda] = ndd[nr][new]['den2']
    return denadj, axadj
