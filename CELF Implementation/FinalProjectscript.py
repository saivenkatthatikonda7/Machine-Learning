import snap
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from heapq import *
import random
import operator

global smallWorld
probOfInfluence = 0.01

def genSmallWorld(N):
    M = 8000
    Graph = genCircle(N)
    Graph = connectNbrOfNbr(Graph, N)
    Graph = connectRandomNodes(Graph, M)
    return Graph

def connectNbrOfNbr(Graph, N):
    for i in range(N - 2):
        Graph.AddEdge(i, i + 2)
    Graph.AddEdge(N - 3, N - 1)
    Graph.AddEdge(N - 2, 0)
    Graph.AddEdge(N - 1, 1)
    return Graph

def genCircle(N):
    Graph = snap.PUNGraph.New()
    for i in range(N):
        Graph.AddNode(i)

    for i in range(N - 1):
        Graph.AddEdge(i, i + 1)
    Graph.AddEdge(N - 1, 0)
    return Graph

def connectRandomNodes(Graph, M):
    target_numedges = Graph.GetEdges() + M
    num_nodes = Graph.GetNodes()
    while Graph.GetEdges() < target_numedges:
        nodes = np.random.choice(num_nodes, 2, replace=False)
        if not Graph.IsEdge(nodes[0], nodes[1]):
            Graph.AddEdge(nodes[0], nodes[1])
    return Graph

smallWorld = genSmallWorld(12008)

scientificCollab = snap.LoadEdgeList(snap.PNGraph, "CA-HepPh.txt", 0, 1)

def getDataPointsToPlot(Graph):

    X, Y = [], []
    outdegree_list = []
    for eachNode in Graph.Nodes():
        outdegree_list.append(eachNode.GetOutDeg())
    count = Counter(outdegree_list)
    X_count = count.keys()
    Y_count = count.values()

    for i in range(len(outdegree_list)):
        X.append(i)
        if i in X_count:
            index = X_count.index(i)
            Y.append(Y_count[index])
        else:
            Y.append(0)

    return X, Y

x_smallWorld, y_smallWorld = getDataPointsToPlot(smallWorld)
plt.loglog(x_smallWorld, y_smallWorld, linestyle='dashed', color='r', label='Small World Network')

x_scientificCollab, y_scientificCollab = getDataPointsToPlot(scientificCollab)
plt.loglog(x_scientificCollab, y_scientificCollab, linestyle='dotted', color='b', label='Collaboration Network')
plt.show()

def calcClusteringCoefficient(Graph):

    c_arrays = []
    for NI in Graph.Nodes():
        k = NI.GetOutDeg()
        if k >= 2:
            k_denom = k * (k - 1) / 2
            neibor_nodes = []
            connected_neibor_edges = 0
            for Id in NI.GetOutEdges():
                neibor_nodes.append(Id)
            for id_n1 in range(len(neibor_nodes)):
                for id_n2 in range(id_n1 + 1, len(neibor_nodes)):
                    n1 = neibor_nodes[id_n1]
                    n2 = neibor_nodes[id_n2]
                    connected = Graph.IsEdge(n1, n2)
                    if connected:
                        connected_neibor_edges += 1
            c = connected_neibor_edges / float(k_denom)
            c_arrays.append(c)
        else:
            c_arrays.append(0)

    C = np.mean(c_arrays)

    return C

C_smallWorld = calcClusteringCoefficient(smallWorld)
C_scientificCollab = calcClusteringCoefficient(scientificCollab)

print('Clustering Coefficient for Small World Network: %f' % C_smallWorld)
print('Clustering Coefficient for Collaboration Network: %f' % C_scientificCollab)

########################################################################################################################

def getAnInfluenceSet(G, nodeIds):
    influenceSet = set(nodeIds)
    scheduled = set(nodeIds)

    while len(scheduled) > 0:
        randomNodeId = scheduled.pop()
        if not G.IsNode(randomNodeId):
            continue
        randomNode = G.GetNI(randomNodeId)
        for i in range(0, randomNode.GetOutDeg()):
            neighborNodeId = randomNode.GetOutNId(i)
            if neighborNodeId in influenceSet:
                continue
            trial = random.random()
            if trial < probOfInfluence:
                scheduled.add(neighborNodeId)
                influenceSet.add(neighborNodeId)

    return influenceSet

def CELF(G, k):
    S = []
    Q = []
    A = {}
    for node in G.Nodes():
        u = node.GetId()
        u_mg = getAnInfluenceSet(G, set([u]))
        u_updated = 0
        u_mg = len(u_mg)
        heappush(Q, (-u_mg, u, u_updated))
    while len(S) < k:
        u_mg, u, u_updated = heappop(Q)
        if u_updated == len(S):
            S.append(u)
        else:
            u_mg = getAnInfluenceSet(G, set(S).union(set([u]))) - getAnInfluenceSet(G, S)
            u_mg = len(u_mg)
            u_updated = len(S)

            heappush(Q, (-u_mg, u, u_updated))
            for v in G.Nodes():
                A[v.GetId()] = (len(getAnInfluenceSet(G, set(S).union(set([v.GetId()]))) - getAnInfluenceSet(G, S)))
            S.append(max(A.iteritems(), key=operator.itemgetter(1))[0])

    return S

########################################################################################################################

def getLengthInfluenceSet(G, nodeIds):
    influenceSet = set(nodeIds)
    scheduled = set(nodeIds)

    while len(scheduled) > 0:
        randomNodeId = scheduled.pop()
        if not G.IsNode(randomNodeId):
            continue
        randomNode = G.GetNI(randomNodeId)
        for i in range(0, randomNode.GetOutDeg()):
            neighborNodeId = randomNode.GetOutNId(i)
            if neighborNodeId in influenceSet:
                continue
            trial = random.random()
            if trial < 0.01:
                scheduled.add(neighborNodeId)
                influenceSet.add(neighborNodeId)

    return len(influenceSet)

nodelist = []
degree_centrality = {}
for node in smallWorld.Nodes():
    nodelist.append(node.GetId())

for node in nodelist:
    DegCentr = snap.GetDegreeCentr(smallWorld, node)
    degree_centrality[node] = DegCentr

tdc = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
getset = []
some = 5

resultset = [0]
while (some <= 30):
    for i in range(0, some):
        getset.append(tdc[i][0])
    some = some + 5
    resultset.append(sum([getLengthInfluenceSet(smallWorld, set([node])) for node in getset]))
    getset=[]

resultset = [i / 500 for i in resultset]

result_randomlist = [0]
for i in range(5, 35, 5):
    randomlist = random.sample(nodelist, i)
    result_randomlist.append(sum([getLengthInfluenceSet(smallWorld, set([node])) for node in randomlist]))

result_randomlist = sorted(result_randomlist)
divider = ((result_randomlist[-1] + 1) / 80)
result_randomlist = [i+1 / (divider+1) for i in result_randomlist]

########################################################################################################################

print "Starting CELF for Small World Network:"

smallWorld_CELF = [0]
for i in range(5, 35, 5):
    smallWorld_CELF.append(sum(CELF(smallWorld, i)))
smallWorld_CELF_sorted = sorted(smallWorld_CELF)
print("The sum of activated set sizes when you take 5 nodes is",smallWorld_CELF_sorted[1])
print("The sum of activated set sizes when you take 10 nodes is",smallWorld_CELF_sorted[2])
print("The sum of activated set sizes when you take 15 nodes is",smallWorld_CELF_sorted[3])
print("The sum of activated set sizes when you take 20 nodes is",smallWorld_CELF_sorted[4])
print("The sum of activated set sizes when you take 25 nodes is",smallWorld_CELF_sorted[5])
print("The sum of activated set sizes when you take 30 nodes is",smallWorld_CELF_sorted[6])

print(smallWorld_CELF)
smallWorld_CELF_scaled = [i/500 for i in smallWorld_CELF_sorted]

X = [0, 5, 10, 15, 20, 25, 30]
plt.plot(X, smallWorld_CELF_scaled, linestyle='dashed', color='r', label='CELF')
plt.plot(X, resultset, linestyle='dashed', color='g', label='Degree Centrality')
plt.plot(X, result_randomlist, linestyle='dashed', color='b', label='Random Nodes')
plt.legend('Comparison among CELF,  Degree Centrality and Random Nodes for Small World Network')
plt.xlabel('Target Set Size')
plt.ylabel('Active Set Size')
plt.show()

########################################################################################################################

scientificCollabpun = snap.LoadEdgeList(snap.PUNGraph, "CA-HepPh.txt", 0, 1)
scient_nodelist = []
scient_degree_centrality = {}
for node in scientificCollabpun.Nodes():
    scient_nodelist.append(node.GetId())

for node in scient_nodelist:
    DegCentrscient = snap.GetDegreeCentr(scientificCollabpun, node)
    scient_degree_centrality[node] = DegCentrscient

resultsetscient=[0]

tdcscient = sorted(scient_degree_centrality.items(), key=lambda x: x[1], reverse=True)
getsetscient = []
getsetscient.append(tdc[1][0])
resultsetscient.append(sum([getLengthInfluenceSet(scientificCollabpun, set([node])) for node in getsetscient]))
getsetscient=[]
getsetscient.append(tdc[2][0])
resultsetscient.append(sum([getLengthInfluenceSet(scientificCollabpun, set([node])) for node in getsetscient]))
#####

result_randomlistscient = [0]
for i in range(1,3):
    randomlistscient = random.sample(nodelist, i)
    result_randomlistscient.append(sum([getLengthInfluenceSet(scientificCollabpun, set([node])) for node in randomlistscient]))

result_randomlistscient = sorted(result_randomlistscient)



print "starting CELF for Scientific Collaboaration Network to get a set of 1 and 2 most influencial nodes:(it will take a couple of minutes for the execution)"
scient_1 =[0]
scient_1.append(sum(CELF(scientificCollab, 1)))
scient_1.append(sum(CELF(scientificCollab, 2)))
print( "the activated set size when you take 1 and 2 influencial nodes in scientific collabaration is",scient_1[0],scient_1[1])
print("starting CELF for different influencial node set sizes like [5,10,15,20,25,30] to plot the graph like the one in small world:") 
'''scientificCollab_CELF = [0]
for i in range(5, 35, 5):
    scientificCollab_CELF.append(sum(CELF(scientificCollab, i)))
scientificCollab_CELF_sorted = sorted(scientificCollab_CELF)
scientificCollab_CELF_scaled = [i/750 for i in scientificCollab_CELF]'''
scient_1 = [i/500 for i in scient_1]
X = [0,1,2]
plt.plot(X, scient_1, linestyle='dashed', color='r', label='CELF')
plt.plot(X, resultsetscient, linestyle='dashed', color='g', label='Degree Centrality')
plt.plot(X, result_randomlistscient, linestyle='dashed', color='b', label='Random Nodes')
plt.legend('Comparison among CELF,  Degree Centrality and Random Nodes for Scientific Collaboaration Network')
plt.xlabel('Target Set Size')
plt.ylabel('Active Set Size')
plt.show()
