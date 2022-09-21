# -*- coding: utf-8 -*-
"""
# License: BSD 3-clause
@author: Amo
"""
import numpy as np
import sys
import math
import random
import copy
import multiprocessing
#import Utils
from Utils import Dijkstra, Utils


class StoGraph(object):  
    class Node(object):
        def __init__(self,index):
            self.index = index
            self.neighbor = {}
            self.in_degree = 0
            self.out_degree = 0
        def myprint(self):
            print(self.index)
            for node in self.neighbor:
                print("{} {} {} {}".format(str(self.index), str(node) , str(self.neighbor[node][0]), str(self.neighbor[node][1])))       
    
    def __init__(self, path, vNum):
        self.nodes={}
        self.vNum = vNum
        self.EGraph = None
        
        for v in range(self.vNum):
             node = self.Node(str(v))
             node.neighbor={}
             self.nodes[str(v)]=node
             
        file1 = open(path, 'r') 
        while True: 
            line = file1.readline() 
            if not line: 
                break          
            ints = line.split()
            node1 = ints[0]
            node2 = ints[1]
            alpha = int(ints[2])
            beta = int(ints[3])
            if len(ints) == 6:
                mean = float(ints[4]) # mean of Weibull
                prob = float(ints[5]) # prob of edge
            elif len(ints) == 4:
                mean = Utils.getWeibull_mean(alpha,beta,500)
                prob = None
            else:
                sys.exit("stoGraph wrong input format")
            # para_2 = float(ints[3])
            
            if node1 in self.nodes:
                if node2 not in self.nodes[node1].neighbor:
                    self.nodes[node1].neighbor[node2]=[alpha,beta, mean, prob]
                    self.nodes[node1].out_degree += 1
                    self.nodes[node2].in_degree += 1
            else:
                sys.exit("non existing node") 
                
            if node2 not in self.nodes:
                sys.exit("non existing node") 
        
        #create mean graph
        temp_nodes  =  copy.deepcopy(self.nodes) 
        for node in temp_nodes:
            for tonode in temp_nodes[node].neighbor:
                temp_nodes[node].neighbor[tonode]=self.nodes[node].neighbor[tonode][2];
        
        self.EGraph = Graph(self.vNum, temp_nodes)
        
        #create unit graph
        temp_nodes  =  copy.deepcopy(self.nodes) 
        for node in temp_nodes:
            for tonode in temp_nodes[node].neighbor:
                temp_nodes[node].neighbor[tonode]=1;
        
        self.unitGraph = Graph(self.vNum, temp_nodes)
        
        #create random graph
        temp_nodes  =  copy.deepcopy(self.nodes) 
        for node in temp_nodes:
            for tonode in temp_nodes[node].neighbor:
                temp_nodes[node].neighbor[tonode]=random.random()
        self.randomGraph = Graph(self.vNum, temp_nodes)

    def EgraphShortest(self, source, destination = None):
        return Dijkstra.dijkstra(self.EGraph,source)
        
    def genMultiRealization(self, num, outfolder, edgeRatio, weightType, startIndex, distance ):
        #raise NotImplementedError()       
        for cout in range(num):
            #print(cout)
            outpath = "{}{}".format(outfolder, cout+startIndex)
            self.genOneRealization(outpath, edgeRatio, weightType, distance)


    def genMultiRealization_P(self, num, outfolder, edgeProb = None, weightType = "True", thread = 1, startIndex = 0, distance = True ):
        #raise NotImplementedError()
        block_size =int (num/thread);
        p = multiprocessing.Pool(thread)
            #print("222")
        p.starmap(self.genMultiRealization, ((block_size, outfolder, edgeProb, weightType, startIndex+i*block_size, distance) for i in range(thread)) )
        #print("333")
        p.close()
        p.join()
        
        
            
    def genOneRealization(self, outpath, edgeProb, weightType, distance ):
        #raise NotImplementedError()
        graph = Graph(self.vNum, copy.deepcopy(self.nodes))
        with open(outpath, 'w') as outfile:
            for node in self.nodes:
                
                graph.nodes[node].neighbor={}
                
                for tonode in self.nodes[node].neighbor:
                    
                    
                    
                    if edgeProb=="true":
                        if self.sampleEdge(node,tonode) is False:
                            continue
                        if weightType == "true":
                            weight = self.sampleTrueWeight(node, tonode)
                        else:
                            weight = self.sampleUniform(node, tonode)
                        graph.nodes[node].neighbor[tonode]=weight
                        outfile.write(node+" ") 
                        outfile.write(tonode+" ")
                        outfile.write(str(weight)+"\n") 
                    
                    if edgeProb == "uniform":
                        if(random.random() >= edgeProb):
                            continue
                        if weightType == "true":
                            weight = self.sampleTrueWeight(node, tonode)
                        else:
                            weight = self.sampleUniform(node, tonode)
                        graph.nodes[node].neighbor[tonode]=weight
                        outfile.write(node+" ") 
                        outfile.write(tonode+" ")
                        outfile.write(str(weight)+"\n") 
        outfile.close()
        
        if distance is True:
            with open(outpath+"_distance", 'w') as outfile:
                for node in self.nodes:                   
                    result = Dijkstra.dijkstra(graph, node);
                    for tonode in self.nodes:
                        if tonode in result:
                            string = result[tonode][0]+" "+result[tonode][1]+" "+result[tonode][2]+"\n" 
                            outfile.write(string) 
            outfile.close()

    def sampleTrueWeight(self, fromNode, toNode):
        alpha = self.nodes[fromNode].neighbor[toNode][0]
        beta = self.nodes[fromNode].neighbor[toNode][1]
        return Utils.getWeibull(alpha, beta)
        #raise NotImplementedError()
        
    def sampleUniform(self, fromNode, toNode):
        return random.random()
        #raise NotImplementedError()
    
    def sampleEdge(self, fromNode, toNode):
        if random.random() >= self.nodes[fromNode].neighbor[toNode][3]:
            return False
        else:
            return True
        
    def print(self):
        for node in self.nodes:
            self.nodes[node].myprint()
            
    #@staticmethod
    def GenStoGraph(self, outpath):
         with open(outpath, 'w') as outfile:
            for node in self.nodes:
                for toNode in self.nodes[node].neighbor:
                    alpha = str(self.nodes[node].neighbor[toNode][0])
                    beta = str(self.nodes[node].neighbor[toNode][1])
                    mean = str(self.nodes[node].neighbor[toNode][2])
                    #ber = str(1.0)
                    ber = str(1.0/self.nodes[toNode].in_degree)
                    string = node + " " + toNode +" "+alpha+" "+beta+" "+mean+" "+ber+"\n"
                    outfile.write(string)
         outfile.close()   
         
    def GenEmModel_Uniform(self, outpath):
        with open(outpath, 'w') as outfile:
           for node in self.nodes:
               for toNode in self.nodes[node].neighbor:
                   alpha = int(random.random()*10)+1
                   beta = int(random.random()*10)+1
                   mean = Utils.getWeibull(alpha, beta)
                   ber = random.random()/10
                   #ber = str(1.0/self.nodes[toNode].in_degree)
                   string = node + " " + toNode +" "+str(alpha)+" "+str(beta)+" "+str(mean)+" "+str(ber)+"\n"
                   outfile.write(string)
        outfile.close()   
        
    def GenEmModel_Approx(self, outpath, weight_q, prob_q):
        with open(outpath, 'w') as outfile:
           for node in self.nodes:
               for toNode in self.nodes[node].neighbor:
                   alpha = int(2*weight_q*self.nodes[node].neighbor[toNode][0]*random.random()+(1-weight_q)*self.nodes[node].neighbor[toNode][0])
                   beta = int(2*weight_q*self.nodes[node].neighbor[toNode][1]*random.random()+(1-weight_q)*self.nodes[node].neighbor[toNode][1])
                   mean = Utils.getWeibull(alpha, beta)
                   ber = 2*prob_q*self.nodes[node].neighbor[toNode][3]*random.random()+(1-prob_q)*self.nodes[node].neighbor[toNode][3]
                   #ber = str(1.0/self.nodes[toNode].in_degree)
                   string = node + " " + toNode +" "+str(alpha)+" "+str(beta)+" "+str(mean)+" "+str(ber)+"\n"
                   outfile.write(string)
        outfile.close()   
        
         #raise NotImplementedError()
    
                
class Graph(): 
 
    def __init__(self, vNum, nodes, path = None):
        self.vNum = vNum
        self.nodes = nodes
    
    def isConnected(self, node1, node2):
        checkedList=[]
        c_nodes=[]
        
        c_nodes.append(node1)
        #checkedList.append(node2)
        
        while len(c_nodes)>0:
            temp_node = []
            for node in c_nodes:
                for tonode in self.nodes[node].neighbor: 
                    if tonode == node2:
                        return True
                    if tonode not in checkedList:
                        temp_node.append(tonode) 
                checkedList.append(node) 
            c_nodes=copy.copy(temp_node)
        
        return False
    
    def print_(self):
        for node in self.nodes:
            for tonode in self.nodes[node].neighbor:
                print(node+" "+tonode+" "+str(self.nodes[node].neighbor[tonode]))
                
    def pathLength(self, x, y):
        #print(x)
        #print(y)
        if y is None:
            return None
        length = 0
        if y[0]!= x[0] or y[-1]!=x[1]:
            print(x)
            print(y)
            sys.exit("path y not for x") 
            return None;
        else:
            for i in range(len(y)-1):
                if y[i] in self.nodes and y[i+1] in self.nodes[y[i]].neighbor:
                    length += float(self.nodes[y[i]].neighbor[y[i+1]])
                else:
                    print(y[i]+" "+y[i+1])
                    print(y)
                    sys.exit("edge not existing") 
                    return None;
            return length
        #self.adjmatrix = {};


if __name__ == "__main__":
    #pass
    dataname = "hep"
    vnum=15233
    #stoGraph=StoGraph("data/hep/hep_raw_1", vnum)
    #stoGraph.GenStoGraph("data/hep/hep_model")
    
    stoGraph=StoGraph("data/{}/{}_model".format(dataname,dataname), vnum)
    #stoGraph.GenEmModel_Uniform("data/{}/{}_model_em_uniform".format(dataname,dataname))
    weight_q=1
    prob_q=1
    stoGraph.GenEmModel_Approx("data/{}/{}_model_em_approx_100_100".format(dataname,dataname), weight_q, prob_q)
    
    
    #stoGraph=StoGraph("data/{}/{}_model_em_uniform".format(dataname,dataname), vnum)
    #stoGraph.genMultiRealization_P(10, "data/{}/features/em_uniform_10000/".format(dataname), weightType='true', edgeProb= 'true',  thread=1, startIndex= 0, distance = True)
