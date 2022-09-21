# -*- coding: utf-8 -*-
"""
# License: BSD 3-clause
@author: Amo
"""
import numpy as np
import sys
import math
import random
#import time
import copy
import heapq
import multiprocessing
from Utils import Utils, Dijkstra
from shutil import copyfile
from USCO import USCO
from scipy.stats import powerlaw
from StoGraph import StoGraph, Graph
#from base import StructuredModel



    
    
class DE_USCO(USCO):
    
    class Realization(object):
        def __init__(self, graphPath, distancePath, vNum):
            self.tranTimes={}
            self.distance = {}
            self.nodes=set()
            self.vNum = vNum
            
            for v in range(self.vNum):
                 #node = self.Node(str(v))
                 #node.neighbor={}
                 self.tranTimes[str(v)]={}
                 self.distance[str(v)]={}
                 self.nodes.add(str(v))
                 
            file1 = open(graphPath, 'r') 
            while True: 
                line = file1.readline() 
                if not line: 
                    break          
                strings = line.split()
                node1 = (strings[0])
                node2 = (strings[1])
                time = float(strings[2])
                
                if node1 in self.tranTimes:
                    self.tranTimes[node1][node2]=time
                else:
                    sys.exit("non existing node") 
                    
                if node2 not in self.nodes:
                    sys.exit("non existing node") 
   
            file1 = open(distancePath, 'r') 
            while True: 
                line = file1.readline() 
                if not line: 
                    break          
                strings = line.split()
                node1 = (strings[0])
                node2 = (strings[1])
                time = float(strings[2])
                
                if node1 in self.distance:
                    self.distance[node1][node2]=time
                else:
                    sys.exit("non existing node") 
                    
                if node2 not in self.nodes:
                    sys.exit("non existing node") 
        def print(self):
            count = 0
            for node in self.nodes:
                for tonode in self.tranTimes[node]:
                    #print("{} {} {}".format(node,tonode, self.tranTimes[node][tonode]))
                    count += 1
            #for tonode in self.distance[node]:
                #print("{} {} {}".format(node,tonode, self.distance[node][tonode]))
            print(count)
            
    class Sample(object):
        
        class Query(object):
            def __init__(self, x_set, budget):
                self.x_set=x_set
                #print(x_set)
                self.budget=budget
            
        def __init__(self, query, y_set, inf):
            self.query=query
            self.decision=y_set
            self.inf=inf
            #raise NotImplementedError()
            
        def print(self):
            #x_set = self.query.x_set
            print(self.query.x_set)
            print(self.decision)
            print(self.inf)
            pass
            
    def kernel(self, realization, query, decision, wCover = False):
        x_set=query.x_set
        cover = []
        x_cover = set()
        value = 0
        for v in decision:
            if v not in query.x_set: # no verlap between decision and query.x_set
                cover.extend(realization.distance[v].keys())
        
        cover = set(cover)
        for v in x_set:
            if v in cover:
                value = value +1
                x_cover.add(v)
        
        if wCover is True:
            return value, x_cover
        else:
            return value
       
        
        #raise NotImplementedError()
        
    def solve_R(self, realizations, W, query):
        #print("solve_R {}".format(len(realizations))) 

        
        solution = []
        gains = []
        c_covers = []
        
            
        for i in range(len(realizations)):
            cover=set()
            #for v in query.x_set:
            #    cover[v]=0
            c_covers.append(cover)
            #print(c_coverOneGraph)
            
        for node in self.stoGraph.nodes:
            gain, node_cover = self.solve_R_margin(realizations, query, [node], W, c_covers) 
            heapq.heappush(gains, (-gain, node, node_cover))
        # 
        score_gain, node, node_cover = heapq.heappop(gains)
        solution.append(node)
        c_score = -score_gain
        #print("{} {}".format(set(node), -score_gain)) 

        c_cover = node_cover
        
        for _ in range(query.budget - 1):
            matched = False
            while not matched:
                _, current_node, _ = heapq.heappop(gains)
                score_gain, new_cover = self.solve_R_margin(realizations, query, [current_node], W, c_covers)
                heapq.heappush(gains, (-score_gain, current_node, new_cover))
                matched = gains[0][1] == current_node

            score_gain, node, c_cover = heapq.heappop(gains)
            c_score = c_score -  score_gain
            solution.append(node)
            #print("{} {}".format(solution, c_score)) 

        return set(solution)
        '''
        Define how to compute y = max w^T[f(x,y,r_1),...f(x,y,r_n)]
        '''
        raise NotImplementedError()
        
    def solve_R_margin(self, realizations, query , decision, W, c_covers):
        new_c_covers = copy.deepcopy(c_covers)
        total_gain = 0;
        for realization, cover, w in zip(realizations, new_c_covers, W):
            gain = 0
            _ , coveredNodes = self.kernel(realization, query, decision, wCover = True)
            for v in coveredNodes:
                if v not in cover:
                    gain += 1
                    cover.add(v)
            total_gain += w*gain
        return total_gain, new_c_covers
     
    
    def solveTrue(self, query, times = 1000):
        #x_set=query.x_set
        budget=query.budget
        #print("solveTrue {}".format(times))
        #print(x_set)
        #print(budget)
        solution = set()
        gains = []                  
            #print(c_coverOneGraph)           
        for node in self.stoGraph.nodes:
            gain = self.solveTrue_margin(query, set(), set([node]), times = times) 
            heapq.heappush(gains, (-gain, node))
        # 
        score_gain, best_node = heapq.heappop(gains)
        solution.add(best_node)
        c_score = -score_gain
        #print("{} {}".format(best_node, c_score)) 
 
        
        
        for _ in range(budget - 1):
            matched = False
            while not matched:
                _, current_node = heapq.heappop(gains)
                #score_gain = self.solveTrue_margin(query, set(solution), set([current_node]), times = times)
                score_gain = self.inf(query, solution.union(set([current_node])),times = times) - c_score
                heapq.heappush(gains, (-score_gain, current_node))
                matched = gains[0][1] == current_node
 
            score_gain, node = heapq.heappop(gains)
            c_score = c_score -  score_gain
            solution.add(node)
            #print("{} {} {}".format(solution, c_score, -score_gain)) 

        return solution, c_score
    
    def solveTrue_margin(self, query , decision, decision_new, times = 1000):
        return self.inf(query, decision.union(decision_new),times = times)-self.inf(query, decision, times = times)
        #pass
        
    def inf(self, query, decision, times = 2000):
        inf = 0.0
        for i in range(times):
            inf = inf + self.inf_single(query, decision)   
        return inf/times
    
    def inf_block(self, queries, decisions, times):
        infs = []
        for (query, decision) in zip(queries, decisions):
            infs.append(self.inf(query, decision, times = times))
            
        return infs
        
    def inf_single(self, query, decision):
        tstate = {} # current state
        fstate = {} # final time
        tTime = dict() # best time
        actTime = [] # all updated time
        for v in decision:
            if v not in query.x_set: # no overlap between decision and query.x_set
                tstate[v]=1
                heapq.heappush(actTime, (0.0, v))
                tTime[v]=0.0
            
            
        #print(tTime)
        
        while len(actTime)>0:
            current_node_time, current_node = heapq.heappop(actTime)  
            if current_node not in fstate:
                if current_node_time != tTime[current_node]:
                    sys.exit("current_node_time != tTime[current_node]") 
                fstate[current_node]=current_node_time
                self.solveTrue_spreadLocal(tstate, fstate, actTime, tTime, current_node, current_node_time)
        count = 0
        #cover={}
        for v in tstate:
            if  tstate[v]==1 and v in query.x_set:
                count += 1
                #cover[x]=tTime[x]
        #print(self.vNum-count)
        return count
    
    def solveTrue_spreadLocal(self, tstate, fstate, actTime, tTime, current_node, current_node_time):
        #print(tTime)
        #print(self.nodes[current_node].neighbor)
        for to_node in self.stoGraph.nodes[current_node].neighbor:
            if self.stoGraph.sampleEdge(current_node, to_node) is False:
                continue
            
            tranTime=self.stoGraph.sampleTrueWeight(current_node, to_node)
            if to_node in fstate:
                pass
            else:
                new_time = current_node_time+ tranTime
                if to_node in tstate:
                    
                    if new_time <tTime[to_node]:
                        tTime[to_node]=new_time
                        tstate[to_node]=tstate[current_node]
                        heapq.heappush(actTime, (new_time , to_node))
                        
                    if new_time == tTime[to_node]:
                        if tstate[current_node]==1:
                            tstate[to_node]=1
                            
                if to_node not in tstate:
                   # print(tTime)
                    tTime[to_node]=new_time
                    tstate[to_node]=tstate[current_node]
                    heapq.heappush(actTime, (new_time, to_node))
                    
    #def solveBatch(self, realizations, W, X, budgets, thread = 1):
    #    '''
    #    Define how to compute y = max w^T[f(x,y,r_1),...f(x,y,r_n)]
    #    '''
    #    raise NotImplementedError()
    
    def genQuery(self):
        raise NotImplementedError()
    
    def genSamples(self, Wpath, x_scale, y_scale, num, thread):
        samples = []
        X_size = powerlaw.rvs(2.5, scale = x_scale, size=num)
        Y_size = powerlaw.rvs(2.5, scale = y_scale, size=num)
        
        if thread == 1:
            for (x_size, y_size) in zip(X_size, Y_size):
                sample=self.genSample(int(x_size)+5, int(y_size)+5)
                samples.append(sample)
                print(len(samples))
        else:
            p = multiprocessing.Pool(thread)
            #print("222")
            samples=p.starmap(self.genSample, ((int(x_size)+5, int(y_size)+5) for (x_size, y_size) in zip(X_size, Y_size)))
            p.close()
            p.join()
        #count = 0
        with open(Wpath, 'w') as outfile:
            for sample in samples:
                string = ""
                for v in sample.query.x_set:
                    string =  string + v + " "
                string = string +"|"
                #outfile.write(string) 
                
                #string = ""
                for v in sample.decision:
                    string =  string + v + " "
                string = string +"|"
                #outfile.write(string) 
                
                #string = ""
                string = string + str(sample.inf) +"\n"
                outfile.write(string) 
                
                #print(string)

        outfile.close()

        
    def genSample(self, x_size, y_size):
        nodes = list(self.stoGraph.nodes.keys())
        random.shuffle(nodes)
                #print(new_topics)
        x_set=set(nodes[0:x_size])
        query=self.Sample.Query(x_set, y_size)
        y, value =self.solveTrue(query,500)
        
        sample=self.Sample(query, y, value)
        print("{} {}".format(x_size, y_size))
        
        return sample    
        #raise NotImplementedError()
    
    def readSamples(self, Rpath, num, Max, isRandom = True, RmaxSize = False):
        lineNums=(np.random.permutation(Max))[0:num] 
        
        #lineNums.sort()
        file1 = open(Rpath, 'r') 
        lineNum = 0
        queries = []
        decisions = []
        samples = []
        infs = []
        maxSize = 0
        
        while True:
            line = file1.readline() 
            if not line: 
                break 
            strings=line.split('|')
            if lineNum in lineNums:
                #print(str(lineNum))
                #print(trainLineNums)
                x_set = set(strings[0].split())
               
                decision = set(strings[1].split())    
                budget = len(decision)
                inf = float(strings[2])
                query=self.Sample.Query(x_set, budget)
                #print("query generated")
                sample = self.Sample(query, decision, inf)
                #print("sample generated")
                
                queries.append(query)
                decisions.append(decision)
                samples.append(sample) 
                infs.append(inf)
                
                if len(x_set) > maxSize:
                    maxSize = len(x_set)
                if len(decision) > maxSize:
                    maxSize = len(decision)   
                
            lineNum += 1 
        
        if RmaxSize  is True:
            return samples, queries, decisions, maxSize
        else:
            return samples, queries, decisions
        #raise NotImplementedError()
   
    def readRealizations(self, Rfolder, realizationsNum, indexes = None, realizationRandom = True, maxNum = None ):
        realizations = [] 
        if indexes is not None:
             realizationIndexes=indexes
        else:  
            if realizationRandom:
                if maxNum is None:
                    sys.exit("maxNum for specified when realizationRandom = True")
                    
                lineNums=(np.random.permutation(maxNum))[0:realizationsNum]
                realizationIndexes=lineNums
                #print("lineNums: {}".format(lineNums))
            else:
                for i in range(realizationsNum):
                    realizationIndexes.append(i)
        
        for i in realizationIndexes:             
            path_graph="{}/{}".format(Rfolder, i)
            path_dis="{}/{}_distance".format(Rfolder, i)
            realization=self.Realization(path_graph, path_dis, self.stoGraph.vNum)
            realizations.append(realization)
        #print(realizationIndexes)
        print("readRealizations done")
        return realizations, realizationIndexes
        
        #raise NotImplementedError()
    def test(self, TestSamples, TestQueries, TestDecisions, predDecisions, n_jobs, logpath = None, preTrainPathResult = None):
        trueInfs = []
        trueInfs_margin = []
        predInfs = []
        predInfs_margin = []
        ratios = []
        inter = []
        
        p = multiprocessing.Pool(n_jobs)
        block_size =int (len(TestQueries)/n_jobs)
        Ys=p.starmap(self.inf_block, ((TestQueries[i*block_size:min([len(TestQueries),(i+1)*block_size])], predDecisions[i*block_size:min([len(TestQueries),(i+1)*block_size])], 1000) for i in range(n_jobs) ))
        p.close()
        p.join()
        #decisions = []
        for inf_block in Ys:
            predInfs.extend(inf_block)
                
        #print(infs)  
       # print(predInfs)  
        #print(decisions)  
        #print(predDecisions)  
        for (sample, predDecision, predInf) in zip(TestSamples, predDecisions, predInfs):            
            #predInf = self.inf(query.x_set, predDecision, 1000)
            #predInfs.append(predInf)
            if sample.inf == 0 or len(sample.decision)==0:
                continue
            trueInfs.append(sample.inf)
            #trueInfs_margin.append(sample.inf-len(sample.decision))
            #predInfs_margin.append(predInf - len(sample.decision))
            ratios.append(predInf/sample.inf)
            inter.append(len(sample.decision.intersection(predDecision))/len(sample.decision))
        
        
        mean_ratios=np.mean(np.array(ratios))
        std_ratios=np.std(np.array(ratios))
        
        #mean_infs_margin=np.mean(np.array(trueInfs_margin))
        #std_infs_margin=np.std(np.array(trueInfs_margin))
        
        #mean_predInfs_margin=np.mean(np.array(predInfs_margin))
        #std_predInfs_margin=np.std(np.array(predInfs_margin))
        
        mean_infs=np.mean(np.array(trueInfs))
        std_infs=np.std(np.array(trueInfs))
        
        mean_predInfs=np.mean(np.array(predInfs))
        std_predInfs=np.std(np.array(predInfs))
        
        mean_inter=np.mean(np.array(inter))
        std_inter=np.std(np.array(inter))
        
        output = "Performance ratio: {} ({})".format(Utils.formatFloat(mean_ratios), Utils.formatFloat(std_ratios))
        Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)
        
        #output = "True Influence_margin: {} ({})".format(Utils.formatFloat(mean_infs_margin), Utils.formatFloat(std_infs_margin))
        #Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)
        #output = "Pred Influence_margin: {} ({})".format(Utils.formatFloat(mean_predInfs_margin), Utils.formatFloat(std_predInfs_margin))
        #Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)
        

        output = "True Influence: {} ({})".format(Utils.formatFloat(mean_infs), Utils.formatFloat(std_infs))
        Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)
        output = "Pred Influence: {} ({})".format(Utils.formatFloat(mean_predInfs), Utils.formatFloat(std_predInfs))
        Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)
        output = "Intersection: {} ({})".format(Utils.formatFloat(mean_inter), Utils.formatFloat(std_inter))
        Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)
        
        return Utils.formatFloat(mean_ratios)
        #return mean_ratios, std_ratios, mean_infs, std_infs, mean_predInfs, std_predInfs, mean_inter, std_inter
        
        
    def proximity(self, query):
        nodes = list(query.x_set)
        random.shuffle(nodes)
            #print(new_topics)
        decision=set(nodes[0:query.budget])
        
        return decision
   
   



def main():
    #pass
    dataname = "pl"
    stoGraphPath = "data/{}/{}_model".format(dataname, dataname)
    vNum = 768
    stoGraph = StoGraph(stoGraphPath, vNum)
    #stoGraph.print()
    
    de_usco = DE_USCO(stoGraph)
    x_scale = 40
    y_scale = 10
    num = 90 
    sample_type = "noOverlap"
    Wpath="data/{}/{}_DE_samples_{}_{}_{}_{}".format(dataname, dataname, x_scale, y_scale, sample_type, num)
    de_usco.genSamples(Wpath, x_scale, y_scale, num, 90)
    
    
def temp():
    dataname = "kro"
    Rpath = "data/{}/{}_DE_samples_200_10_2700"
    Wpath = "data/{}/{}_DE_samples_200_10_2700_new"
    file1 = open(Rpath, 'r') 
    file2 = open(Wpath, 'w')     
    while True:
        line = file1.readline() 
        if not line: 
            break 
        strings=line.split('|')
        
        new_string = strings[0]+"|"+strings[2]+"|"+strings[3]

        file2.write(new_string)

#g = Graph(9)
if __name__ == "__main__":
    pass
    main()
    