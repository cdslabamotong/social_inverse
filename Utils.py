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
import multiprocessing
import heapq
from os import listdir
from os.path import isfile, join

#from base import StructuredModel



class Utils(object):
    def writeToFile(path, string, toconsole = False, preTrainPathResult = None):
         logfile = open(path, 'a')
         logfile.write(string+"\n") 
         logfile.close() 
         if toconsole is True:
             print(string)
             
         if preTrainPathResult is not None:
             logfile = open(preTrainPathResult, 'a')
             logfile.write(string+"\n") 
             logfile.close() 
         
             
    def formatFloat(x):
        return "{:.4f}".format(x)
    
    def save_pretrain(path, weights, featureIndex, featurePath):
         with open(path+"/featureIndex", 'w') as outfile:
            for index, w in zip(featureIndex, list(weights)):
                outfile.write(str(index)+" "+str(w)+" "+"\n") 
         outfile.close()
         
    @staticmethod
    def getWeibull(alpha, beta):
        time = alpha*math.pow(-math.log(1-random.uniform(0, 1)), beta);
        if time >= 0:
            return math.ceil(time)+1
        else:
            sys.exit("time <0") 
            return None
         
    def getWeibull_mean(alpha, beta, times):
        time = 0
        for _ in range(times):
            time = time + Utils.getWeibull(alpha,beta)
            
        return time/times
    
    def higgsToRaw(inPath, outPath,vnum):
        file1 = open(inPath, 'r') 
        nodeMap = {}
        while True: 
            line = file1.readline() 
            if not line: 
                break   
            ints = line.split()
            node1 = ints[0]
            node2 = ints[1]
            if node1 not in nodeMap and len(nodeMap)<vnum:
                string = str(len(nodeMap))
                nodeMap[node1]=string
            if  len(nodeMap)== vnum:
                break;
                
            if node2 not in nodeMap and len(nodeMap)<vnum:
                string = str(len(nodeMap))
                nodeMap[node2]=string
            if  len(nodeMap)== vnum:
                break;  
        with open(outPath, 'w') as outfile:
            file1 = open(inPath, 'r')          
            while True: 
                line = file1.readline() 
                if not line: 
                    break   
                ints = line.split()
                node1 = ints[0]
                node2 = ints[1]
                if node1 in nodeMap and node2 in nodeMap:
                    alpha = int(random.random()*10)+1
                    beta = int(random.random()*10)+1
                    mean = str(Utils.getWeibull(alpha, beta))
                    ber = str(0.1)
                    string = nodeMap[node1] + " " + nodeMap[node2] +" "+str(alpha)+" "+str(beta)+" "+mean+" "+ber+"\n"
                    outfile.write(string)
            file1.close()
        outfile.close()
        
        
    def power2500ToRaw(inPath, outPath):
         with open(outPath, 'w') as outfile:
            file1 = open(inPath, 'r')          
            while True: 
                line = file1.readline() 
                if not line: 
                    break   
                ints = line.split("\"")
                node1 = ints[7]
                node2 = ints[9]
                alpha = int(random.random()*10)+1
                beta = int(random.random()*10)+1
                mean = str(Utils.getWeibull(alpha, beta))
                ber = str(0.1)
                string = node1 + " " + node2 +" "+str(alpha)+" "+str(beta)+" "+mean+" "+ber+"\n"
                outfile.write(string)
            file1.close()
         outfile.close()
         
    def hepToRaw(inPath, outPath):
        maxNode = 0
        with open(outPath, 'w') as outfile:
           file1 = open(inPath, 'r')          
           while True: 
               line = file1.readline() 
               if not line: 
                   break   
               ints = line.split()
               node1 = ints[0]
               node2 = ints[1]
               alpha = int(random.random()*10)+1
               beta = int(random.random()*10)+1
               mean = str(Utils.getWeibull(alpha, beta))
               ber = str(0.1)
               string = node1 + " " + node2 +" "+str(alpha)+" "+str(beta)+" "+mean+" "+ber+"\n"
               maxNode=max(maxNode, int(node1), int(node2))
               outfile.write(string)
           file1.close()
        outfile.close()
        print(maxNode)
        
    def readResultsToCvs(inPath, outPath):
         with open(outPath, 'w') as outfile:
            for f in listdir(inPath):
                for file in listdir(inPath+f):
                    if file == "result":
                        filePath = inPath+f+"/"+file
                        print(filePath)
                        file1 = open(filePath, 'r')     
                        lines = []
                        while True: 
                            line = file1.readline() 
                            if not line: 
                                break   
                            lines.append(line)
                        
                        dataname = lines[3].strip()
                        
                        words = lines[2].split()
                        source = words[0]
                        target = words[1]
                        
                        words = lines[4].split(",")
                        featureNum = words[0]
                        featureMethod = words[1]
                        betas = words[2].split()
                        beta = betas[1]
                        
                        traintype = lines[5].strip()
                        testtype = lines[6].strip()
                        
                        if target=="DC":
                            words = lines[13].split()
                            true_diff = float(words[1])
                            words = lines[14].split()
                            pred_diff = float(words[1])
                            result_opt= Utils.formatFloat(pred_diff/true_diff)
                            
                            words = lines[22].split()
                            true_diff = float(words[1])
                            words = lines[23].split(" ")
                            pred_diff = float(words[1])
                            result_ones= Utils.formatFloat(pred_diff/true_diff)
                            
                            words = lines[27].split()
                            summary_new = result_opt+" ("+result_ones+") "+words[2]
                            summary = lines[27].strip()
                            
                        if target=="DE":
                            words = lines[10].split()
                            true_diff = float(words[2])
                            words = lines[11].split()
                            pred_diff = float(words[2])
                            result_opt= Utils.formatFloat(pred_diff/true_diff)
                            
                            words = lines[15].split()
                            true_diff = float(words[2])
                            words = lines[16].split(" ")
                            pred_diff = float(words[2])
                            result_ones= Utils.formatFloat(pred_diff/true_diff)
                            
                            words = lines[19].split()
                            summary_new = result_opt+" ("+result_ones+") "+words[2]
                            summary = lines[19].strip()
                            
                        outfile.write(dataname+","+source+","+target+","+featureNum+","+featureMethod+","+beta+","+traintype+","+testtype+","+summary_new+","+summary+"\n")
                        
                        file1.close()                      
         outfile.close()
         
         
    def readResultsToCvsTime(inPath, outPath):
         with open(outPath, 'w') as outfile:
            for f in listdir(inPath):
                for file in listdir(inPath+f):
                    if file == "result":
                        filePath = inPath+f+"/"+file
                        print(filePath)
                        file1 = open(filePath, 'r')     
                        lines = []
                        while True: 
                            line = file1.readline() 
                            if not line: 
                                break   
                            lines.append(line)
                        
                        dataname = lines[3].strip()
                        
                        words = lines[2].split()
                        source = words[0]
                        target = words[1]
                        
                        words = lines[4].split(",")
                        featureNum = words[0]
                        featureMethod = words[1]
                        betas = words[2].split()
                        beta = betas[1]
                        
                        traintype = lines[5].strip()
                        testtype = lines[6].strip()
                        
                        words = lines[5].split(",")
                        nums = words[0].split()
                        train_num=float(nums[1])
                        
                        words = lines[6].split(",")
                        nums = words[0].split(":")
                        test_num=float(nums[1])
                        

                        
                        if target=="DC":
                            words = lines[13].split()
                            true_diff = float(words[1])
                            words = lines[14].split()
                            pred_diff = float(words[1])
                            result_opt= Utils.formatFloat(pred_diff/true_diff)
                            
                            words = lines[22].split()
                            true_diff = float(words[1])
                            words = lines[23].split(" ")
                            pred_diff = float(words[1])
                            result_ones= Utils.formatFloat(pred_diff/true_diff)
                            
                            words = lines[27].split()
                            summary_new = result_opt+" ("+result_ones+") "+words[2]
                            summary = lines[27].strip()
                            
                            words = lines[28].split()
                            nums = words[1].split(":")
                            train_time = float(nums[0])*3600+float(nums[1])*60+float(nums[2])
                            
                            words = lines[29].split()
                            nums = words[1].split(":")
                            test_time = (float(nums[0])*3600+float(nums[1])*60+float(nums[2]))/test_num
                            
                        if target=="DE":
                            words = lines[10].split()
                            true_diff = float(words[2])
                            words = lines[11].split()
                            pred_diff = float(words[2])
                            result_opt= Utils.formatFloat(pred_diff/true_diff)
                            
                            words = lines[15].split()
                            true_diff = float(words[2])
                            words = lines[16].split(" ")
                            pred_diff = float(words[2])
                            result_ones= Utils.formatFloat(pred_diff/true_diff)
                            
                            words = lines[19].split()
                            summary_new = result_opt+" ("+result_ones+") "+words[2]
                            summary = lines[19].strip()
                            
                            words = lines[20].split()
                            nums = words[1].split(":")
                            train_time = float(nums[0])*3600+float(nums[1])*60+float(nums[2])
                            
                            words = lines[21].split()
                            nums = words[1].split(":")
                            test_time = (float(nums[0])*3600+float(nums[1])*60+float(nums[2]))/test_num
                            
                        outfile.write(dataname+","+source+","+target+","+featureNum+","+featureMethod+","+beta+","+traintype+","+testtype+","+summary_new+","+summary+","+str(train_time)+","+str(test_time)+"\n")
                        
                        file1.close()                      
         outfile.close()
         
        
class Dijkstra(object): 
    
    @staticmethod
    def minDistance(dist, queue): 
        # Initialize min value and min_index as -1 
        minimum = float("Inf") 
        min_index = -1
          
        # from the dist array,pick one which 
        # has min value and is till in queue 
        for node in dist: 
            if dist[node] < minimum and node in queue: 
                minimum = dist[node] 
                min_index = node 
        return min_index 
    
    @staticmethod
    def printPath(graph, parent, j, path): 
          
        #Base Case : If j is source 
        if parent[j] == -1 :  
            #print(j," ",end='') 
            path.append(j);
            return
        Dijkstra.printPath(graph, parent , parent[j], path) 
        #print(j," ", end='') 
        path.append(j);
    
    @staticmethod
    def printSolution(graph, src, dist, parent): 
        #src = 0
        #print("Vertex \t\tDistance from Source\tPath") 
        result={};
        for node in dist: 
            #print("{} {} {} ".format(src, node, dist[node])), 
            if dist[node] != float("Inf"):
                path=[];
                path.append(src);
                path.append(node);
                path.append(str(dist[node]));
                Dijkstra.printPath(graph, parent, node, path) 
                #print()
                #print(path);
                result[node] = path;
        #print(result)
        return result;
    
    @staticmethod      
    def dijkstra(graph, src, results = None): 

        dist={}
        for node in graph.nodes:
            dist[node]=float("Inf")

        parent={}
        for node in graph.nodes:
           parent[node]=-1

        dist[src] = 0     
        queue = [] 
        for node in graph.nodes: 
            queue.append(node) 
              
        while queue: 

            u = Dijkstra.minDistance(dist,queue)  
            if u == -1:
                break     
            queue.remove(u)  
            for node in graph.nodes: 
                if node in graph.nodes[u].neighbor and node in queue: 
                    #print(graph.nodes[u].neighbor[node])
                    if dist[u] + graph.nodes[u].neighbor[node] < dist[node]: 
                        
                        dist[node] = dist[u] + graph.nodes[u].neighbor[node] 
                        parent[node] = u   
        
        # print the constructed distance array 
        result = Dijkstra.printSolution(graph, src, dist, parent) 
        if results != None:
            results[src]=result
            print(src)
            sys.stdout.flush()
            
        return result
    
    @staticmethod      
    def dijkstra_1(graph, src): 

        dist={}
        for node in graph.nodes:
            dist[node]=float("Inf")

        parent={}
        for node in graph.nodes:
           parent[node]=-1

        dist[src] = 0     
        queue = [] 
        for node in graph.nodes: 
            queue.append(node) 
              
        while queue: 

            u = Dijkstra.minDistance(dist,queue)  
            if u == -1:
                break     
            queue.remove(u)  
            for node in graph.nodes: 
                if node in graph.nodes[u].neighbor and node in queue: 
                    if dist[u] + graph.nodes[u].neighbor[node] < dist[node]: 
                        dist[node] = dist[u] + graph.nodes[u].neighbor[node] 
                        parent[node] = u   
        
        # print the constructed distance array 
        result = Dijkstra.printSolution(graph, src, dist, parent) 
       
        #print(src)
        #sys.stdout.flush()      
        return src, result
        # print the constructed distance array 

        

# Driver program
#g = Graph(9)
if __name__ == "__main__":
    #stoGraph=StoGraph("data/kro/kro", 1024)
    #Utils.power2500ToRaw("data/pl2500/power2500_raw","data/pl2500/power2500_raw_1")
    #Utils.hepToRaw("data/hep/hep_raw","data/hep/hep_raw_1")
    Utils.readResultsToCvsTime("results20/","temp20.csv")
    pass
