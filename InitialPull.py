import catmaid
import scipy.io
import numpy
from httplib import BadStatusLine

# First setup a new Source
src = catmaid.get_source(cache=False)

print 'Pulling Adj and Skeletons from source'
adj, skels = src.get_graph()
scipy.io.savemat('adjacency.mat', mdict={'adjacency': adj})
scipy.io.savemat('skeletons.mat', mdict={'skeletons': skels})

print 'Pulling ApicalList and EMidOriRGBSpeed list from file'
fn_a = 'ApSkList'
fn_e = 'EMidOriRGBneuronIDSFTFspeed'
ApicalList = scipy.io.loadmat(fn_a+'.mat')[fn_a].astype(int)
EMidOriSpd = scipy.io.loadmat(fn_e+'.mat')[fn_e]

c = src._skel_source
somalist = []
COMList = []
print 'Creating SomaList and COMList'
for x in skels:
    st = str(x)
    print st
    try:
        urltemp = 'http://catmaid.hms.harvard.edu/9/skeleton/' +\
                  st + '/get-root'
        jsonA = c.fetchJSON(urltemp)
        rid = jsonA['root_id']
        listele = [0, jsonA['x'], jsonA['y'], jsonA['z'], 0]
        urltemp = 'http://catmaid.hms.harvard.edu/9/skeleton/' +\
                  st + '/node_count'
        jsonC = c.fetchJSON(urltemp)
        nops = jsonC['count']
        listele[4] = nops
        urltemp = 'http://catmaid.hms.harvard.edu/9/labels-for-node/treenode/' +\
                  str(rid)
        jsonB = c.fetchJSON(urltemp)
        soma = 0
        if('soma' in jsonB):
            soma = 1
            listele[0] = soma
            somalist.append(listele)
    except TypeError:
        print st + " has a problem!"
        listele = [0, 0, 0, 0, 0]
        somalist.append(listele)
    except BadStatusLine:
        print "Website trouble, try me again"

    # and COMList
    nr = src.get_neuron(x)
    COM = nr.center_of_mass
    Ori = numpy.nan
    Spd = numpy.nan
    if(x in EMidOriSpd[:, 0]):
        ind = EMidOriSpd[:, 0].tolist().index(x)
        Ori = EMidOriSpd[ind, 1]
        Spd = EMidOriSpd[ind, 8]
    COMList.append([int(x), COM['x'], COM['y'], COM['z'],
                    ApicalList.tolist().count(x), len(nr.nodes),
                    Ori, Spd])


scipy.io.savemat('somalist.mat', mdict={'somalist': somalist})
scipy.io.savemat('COMList.mat', {'COMList': COMList})
