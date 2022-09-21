# -*- coding: utf-8 -*-
"""
# License: BSD 3-clause
@author: Amo
"""
import numpy as np
import sys
import random
#import time
import copy
import heapq
import multiprocessing
from Utils import Utils
from USCO import USCO
from scipy.stats import powerlaw
from StoGraph import StoGraph
#from base import StructuredModel



    
    
class DC_USCO(USCO):
    
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
            def __init__(self, a_set, budget, zero_inf):
                self.a_set=a_set
                #print(x_set)
                self.budget=budget
                self.zero_inf = zero_inf
            
        def __init__(self, query, p_set, inf):
            self.query=query
            self.decision=p_set
            self.inf=inf # num of nodes that are not activated by a
            #self.zero_inf = zero_inf
            #raise NotImplementedError()
            
        def print(self):
            #x_set = self.query.x_set
            print(self.query.a_set)
            print(self.query.zero_inf)
            print(self.decision)
            print(self.inf)
            pass
            
    def kernel(self, realization, query, decision, RpNodes = False):
        a_set=query.a_set
        p_set=decision
        protectedNodes = set()
        value = 0
        for v in self.stoGraph.nodes:
            v_a_distance = float("inf")
            for u in a_set:
                if v in realization.distance[u] and realization.distance[u][v]<v_a_distance:
                    v_a_distance = realization.distance[u][v]
                    
            v_p_distance = float("inf")
            for u in p_set:
                if v in realization.distance[u] and realization.distance[u][v]<v_a_distance:
                    v_p_distance = realization.distance[u][v]
                    
            if v_a_distance < v_p_distance:
                pass
            else:
                protectedNodes.add(v)
                value = value + 1

        
        if RpNodes is True:
            return value, protectedNodes
        else:
            return value
       
        
        #raise NotImplementedError()
        
    def solve_R(self, realizations, W, query):
        #print("solve_R {}".format(len(realizations))) 

        #print(W)
       # print("**")
        solution = set()
        gains = []
        c_aNodes = []
        
            
        for realization in realizations:
            aNode={}
            for v in self.stoGraph.nodes:
                c_distance = float("inf")
                for a in query.a_set:
                    if v in realization.distance[a] and realization.distance[a][v]<c_distance:
                        c_distance= realization.distance[a][v]
                        aNode[v]=c_distance
            #for v in query.x_set:
            #    cover[v]=0
            c_aNodes.append(aNode)
            #print(c_coverOneGraph)
            
        for node in self.stoGraph.nodes:
            gain, aNodes = self.solve_R_margin(realizations, query, set([node]), W, c_aNodes) 
            heapq.heappush(gains, (-gain, node, aNodes))
        # 
        score_gain, node, c_aNodes = heapq.heappop(gains)
        solution.add(node)
        c_score = -score_gain
        #print("{} {}".format(solution, -score_gain)) 

        #c_cover = node_cover
        
        for _ in range(query.budget - 1):
            matched = False
            while not matched:
                _, current_node, _ = heapq.heappop(gains)
                score_gain, new_c_aNodes = self.solve_R_margin(realizations, query, set([current_node]), W, c_aNodes)
                heapq.heappush(gains, (-score_gain, current_node, new_c_aNodes))
                matched = gains[0][1] == current_node

            score_gain, node, c_aNodes = heapq.heappop(gains)
            c_score = c_score -  score_gain
            solution.add(node)
            #print("{} {}".format(solution, c_score)) 
        
       # print("solve_R done") 
        return solution
        '''
        Define how to compute y = max w^T[f(x,y,r_1),...f(x,y,r_n)]
        '''
        raise NotImplementedError()
        
    def solve_R_margin(self, realizations, query , decision, W, c_aNodes):
        #print(W)
        new_c_aNodes = copy.deepcopy(c_aNodes)
        total_gain = 0;
        for realization, aNode, w in zip(realizations, new_c_aNodes, W):
            gain = 0
           # _ , new_aNode = self.kernel(realization, query, decision, RaNodes = True)
            new_aNode={}
            for v in aNode:
                v_p_distance = float("inf")
                for u in decision:
                    if v in realization.distance[u] and realization.distance[u][v]<v_p_distance:
                        v_p_distance = realization.distance[u][v]
                        
                if aNode[v] <= v_p_distance:
                    new_aNode[v]=aNode[v]
                else:
                    gain += 1
                    #aNode[v]=v_p_distance
            aNode=new_aNode
            #print(w)
            total_gain += w*gain
        return total_gain, new_c_aNodes
     
    
    def solveTrue(self, query, times = 1000):
        #a_set=query.a_set
        budget=query.budget
        #print("solveTrue {}".format(times))
        #print(x_set)
        #print(budget)
        solution = set()
        gains = []                  
            #print(c_coverOneGraph)           
        for node in self.stoGraph.nodes:
            #print(node) 
            gain = self.inf(query, solution.union({node}),times = times) - query.zero_inf
            heapq.heappush(gains, (-gain, node))
        # 
        score_gain, best_node = heapq.heappop(gains)
        solution.add(best_node)
        c_score = -score_gain + query.zero_inf
        #("{} {}".format(best_node, c_score)) 
 
        
        
        for _ in range(budget - 1):
            matched = False
            while not matched:
                _, current_node = heapq.heappop(gains)           
                score_gain = self.inf(query, solution.union({current_node}),times = times) - c_score
                heapq.heappush(gains, (-score_gain, current_node))
                matched = gains[0][1] == current_node
 
            score_gain, node = heapq.heappop(gains)
            c_score = c_score -  score_gain
            solution.add(node)
            #print("{} {} {}".format(solution, c_score, -score_gain)) 

        return solution, c_score
    
    def solveTrue_margin(self, query , decision, decision_new, times = 1000):
        raise NotImplementedError()
        #return self.inf(query, decision.union(decision_new),times = times)-self.inf(query, decision, times = times)
        #pass
        
    def inf(self, query, decision, times = 2000):
        inf = 0.0
        for i in range(times):
            #print(i)
            inf = inf + self.inf_single(query, decision)   
        return inf/times
    
    def inf_block(self, queries, decisions, times = 2000):
        infs = []
        for (query, decision) in zip(queries, decisions):
            infs.append(self.inf(query, decision, times = times))
            
        return infs
        
    def inf_single(self, query, decision):
        tstate = {} # current state
        fstate = {} # final time
        tTime = dict() # best time
        actTime = [] # all updated time
        a_set=query.a_set
        p_set=decision
        
        
        for v in a_set:
            tstate[v]=-1
            heapq.heappush(actTime, (0.0, v))
            tTime[v]=0.0
            
        for v in p_set:
            if v not in a_set:
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
            if  tstate[v]== -1:
                count += 1
                #cover[x]=tTime[x]
        #print(self.vNum-count)
        return self.stoGraph.vNum-count
    
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
                    
                    if new_time < tTime[to_node]:
                        tTime[to_node]=new_time
                        tstate[to_node]=tstate[current_node]
                        heapq.heappush(actTime, (new_time , to_node))
                        
                    if new_time == tTime[to_node]:
                        if tstate[current_node]==-1:
                            tstate[to_node]=-1
                            
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
    
    def genSamples(self, Wpath, scale_x, scale_y, num, thread):
        samples = []
        X_size = powerlaw.rvs(2.5, scale = scale_x, size=num)
        Y_size = powerlaw.rvs(2.5, scale = scale_y, size=num)
        
        if thread == 1:
            for (x_size, y_size) in zip(X_size, Y_size):
                
                sample=self.genSample(int(x_size)+5, int(y_size)+5)
                samples.append(sample)
                #print(len(samples))
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
                for v in sample.query.a_set:
                    string =  string + v + " "
                string = string +"|"

                
                string = string + str(sample.query.zero_inf) +"|"
                #string = ""
                for v in sample.decision:
                    string =  string + v + " "                  
                string = string +"|"
                
                #string = ""
                string = string + str(sample.inf) +"\n"
                outfile.write(string) 
                
                #print(string)

        outfile.close()
    '''    
    def genSamples(self, Wpath, num):
        samples = []
        X_size = powerlaw.rvs(2.5, scale = 200, size=num)
        Y_size = powerlaw.rvs(2.5, scale = 10, size=num)
        
        for (x_size, y_size) in zip(X_size, Y_size):
            sample=self.genSample(int(x_size), int(y_size))
            samples.append(sample)
            print(len(samples))
            
        #count = 0
        with open(Wpath, 'w') as outfile:
            for sample in samples:
                string = ""
                for v in sample.query.a_set:
                    string =  string + v + " "
                string = string +"|"
                #outfile.write(string) 
                
                #string = ""
                for v in sample.decision:
                    string =  string + v + " "
                string = string +"|"
                #outfile.write(string) 
                
                #string = ""
                string = str(sample.inf) +"\n"
                outfile.write(string) 
                
                #print(string)

        outfile.close()
    '''
        
    def genSample(self, x_size, y_size):
        #print("111")
        if x_size > self.stoGraph.vNum or y_size > self.stoGraph.vNum:
            sys.exit("x or y size too large")
            
        nodes = list(self.stoGraph.nodes.keys())
        random.shuffle(nodes)
                #print(new_topics)
        a_set=set(nodes[0:x_size])
        
        query=self.Sample.Query(a_set, y_size, 0)
        zero_inf = self.inf(query, set(), times = 1000)
        query.zero_inf=zero_inf
        #print("222")
        y, value =self.solveTrue(query, 500)     
        sample=self.Sample(query, y, value)
        print("genSample Done {} {}".format(x_size, y_size))
        
        #sample.print()
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
                a_set = set(strings[0].split())
                zero_inf = float(strings[1])
                decision = set(strings[2].split())    
                budget = len(decision)
                inf = float(strings[3])
                
                query=self.Sample.Query(a_set, budget, zero_inf)
                #print("query generated")
                sample = self.Sample(query, decision, inf)
                #print("sample generated")
                
                queries.append(query)
                decisions.append(decision)
                samples.append(sample) 
                infs.append(inf)
                if len(a_set) > maxSize:
                    maxSize = len(a_set)
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
        zeroInfs = []
        trueInfs = []
        predInfs = []
        inf_ratios = []
        true_diff = []
        pred_diff = []
        diff_ratios = []       
        inter = []
        
        testNum = len(TestSamples)
        
        p = multiprocessing.Pool(n_jobs)
        block_size =int (testNum/n_jobs)
        Ys=p.starmap(self.inf_block, ((TestQueries[i*block_size:min([testNum,(i+1)*block_size])], predDecisions[i*block_size:min([testNum,(i+1)*block_size])], 1000) for i in range(n_jobs) ))
        p.close()
        p.join()
        #decisions = []
        for inf_block in Ys:
            predInfs.extend(inf_block)
                
        #print(infs)  
       # print(predInfs)  
        #print(decisions)  
        #print(predDecisions)  
        for (query, sample, predInf, predDecision) in zip(TestQueries, TestSamples,  predInfs, predDecisions):            
            #predInf = self.inf(query.x_set, predDecision, 1000)
            #predInfs.append(predInf)
            if query.zero_inf == 0 or sample.inf==0 or len(sample.decision) == 0 or sample.inf == query.zero_inf:
                continue
            zeroInfs.append(query.zero_inf)
            trueInfs.append(sample.inf)
            inf_ratios.append(predInf/sample.inf)
            true_diff.append(sample.inf-query.zero_inf)
            pred_diff.append(predInf-query.zero_inf)
            diff_ratios.append((predInf-query.zero_inf)/(sample.inf-query.zero_inf))         
            inter.append(len(sample.decision.intersection(predDecision))/len(sample.decision))
        #print(len(zeroInfs))
        
        mean_zeroInfs=np.mean(np.array(zeroInfs))
        std_zeroInfs=np.std(np.array(zeroInfs))
        
        mean_trueInfs=np.mean(np.array(trueInfs))
        std_trueInfs=np.std(np.array(trueInfs))

        
        mean_predInfs=np.mean(np.array(predInfs))
        std_predInfs=np.std(np.array(predInfs))
        
        mean_inf_ratios=np.mean(np.array(inf_ratios))
        std_inf_ratios=np.std(np.array(inf_ratios))
        
        mean_true_diff=np.mean(np.array(true_diff))
        std_true_diff=np.std(np.array(true_diff))

        
        
        mean_pred_diff=np.mean(np.array(pred_diff))
        std_pred_diff=np.std(np.array(pred_diff))

        
        
        mean_diff_ratios=np.mean(np.array(diff_ratios))
        std_diff_ratios=np.std(np.array(diff_ratios))

        
        mean_inter=np.mean(np.array(inter))
        std_inter=np.std(np.array(inter))

        
        output = "zeroInfs: {} ({})".format(Utils.formatFloat(mean_zeroInfs), Utils.formatFloat(std_zeroInfs))
        Utils.writeToFile(logpath, output, toconsole = True, preTrainPathResult = preTrainPathResult)
        output = "trueInfs: {} ({})".format(Utils.formatFloat(mean_trueInfs), Utils.formatFloat(std_trueInfs))
        Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)
        output = "predInfs: {} ({})".format(Utils.formatFloat(mean_predInfs), Utils.formatFloat(std_predInfs))
        Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)
        output = "inf_ratios: {} ({})".format(Utils.formatFloat(mean_inf_ratios), Utils.formatFloat(std_inf_ratios))
        Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)
        output = "true_diff: {} ({})".format(Utils.formatFloat(mean_true_diff), Utils.formatFloat(std_true_diff))
        Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)
        output = "pred_diff: {} ({})".format(Utils.formatFloat(mean_pred_diff), Utils.formatFloat(std_pred_diff))
        Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)
        output = "diff_ratios: {} ({})".format(Utils.formatFloat(mean_diff_ratios), Utils.formatFloat(std_diff_ratios))
        Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)
        output = "inter: {} ({})".format(Utils.formatFloat(mean_inter), Utils.formatFloat(std_inter))
        Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)
        
        return Utils.formatFloat(mean_diff_ratios)
        #return results
        
        
        
    def proximity(self, query):
        nodes = []
        print("---------------")
        for v in query.a_set:
            print(v)
            print(self.stoGraph.nodes[v].neighbor.keys())
            nodes.extend(self.stoGraph.nodes[v].neighbor.keys())
        random.shuffle(nodes)
            #print(new_topics)
        decision = set()
        for node in nodes:
            if node not in query.a_set and node not in decision:
                decision.add(node)
            if len(decision)==query.budget:
                return decision
        print(nodes)
        print(query.a_set)
        sys.exit("no enough neighbors")
        #return decision        
        
   
   



def main():
    #pass
    dataname = "er"
    stoGraphPath = "data/{}/{}_model".format(dataname, dataname)
    vNum = 512
    stoGraph = StoGraph(stoGraphPath, vNum)
    #stoGraph.print()
    
    dc_usco = DC_USCO(stoGraph)
    x_scale = 10
    y_scale = 10
    num = 810 
    sample_type = "normal"
    Wpath="data/{}/{}_DC_samples_{}_{}_{}_{}".format(dataname, dataname, x_scale, y_scale, sample_type, num)
    dc_usco.genSamples(Wpath, x_scale, y_scale, num, 90)
    
        
    #a_set={'324', '766', '132', '610', '72', '902', '897', '100', '738', '708', '809', '180', '518', '285', '255'}
    
    #nodes = list(dc_usco.stoGraph.nodes.keys())
    #random.shuffle(nodes)
    #a_set = set(nodes[0:50])
    
    
    #nodes = list(dc_usco.stoGraph.nodes.keys())
    #random.shuffle(nodes)
    #decision = set(nodes[0:50])
    #decision = {'517', '420', '0', '128', '37', '513', '122', '8', '167'}
    #query=dc_usco.Sample.Query(a_set, 9, 0)
    #print(dc_usco.inf(query, decision, times=2000))
    
    #path = os.getcwd() 
    #data_path=path+"/data"
    #featurePath = "{}/{}/features/{}_{}".format(data_path, 'kro', 'true', 10000)
    #featureNum = 800
    #realizations, _ = dc_usco.readRealizations(featurePath, featureNum, maxNum = 10000)
    #new_decision=dc_usco.solve_R(realizations, np.ones(featureNum), query)
    #print(dc_usco.inf(query, new_decision, times=2000))
    
    
    '''
    print(dc_usco.inf(query, set(), times=1000))
    print()
    for i in range(10):
        nodes = list(dc_usco.stoGraph.nodes.keys())
        random.shuffle(nodes)
        r_decision = set(nodes[0:len(decision)])
        print(dc_usco.inf(query, r_decision, times=1000))
   
                #print(new_topics)
    '''            

def temp():
    a_set={}

#g = Graph(9)
if __name__ == "__main__":
    #pass
    main()
    