"""
==============================
StratLearner Training
==============================
"""



import argparse
import os
import sys
import random

from datetime import datetime

sys.path.insert(0,'..')
from DE import DE_USCO
from DC import DC_USCO
from StoGraph import StoGraph
from Utils import Utils

class Object(object):
    pass

def SI_NB(sourceT, targetT, target_USCO, TrainSamples, TestQueries):
    predDecisions = [] 
    score_map = {}
    
    for node in target_USCO.stoGraph.nodes.keys():
        score_map[node]={}
        for tonode in target_USCO.stoGraph.nodes.keys():
            score_map[node][tonode] = random.random()
    

    for sample in TrainSamples:
        if sourceT == 'DE':
            input_set= sample.query.x_set
        if sourceT == 'DC':
            input_set= sample.query.a_set
        for node in input_set:
            for tonode in sample.decision:
                score_map[node][tonode] += 1
  
            
            
    for query in TestQueries:   
        if targetT == 'DE':
            input_set= query.x_set
        if targetT == 'DC':
            input_set= query.a_set
        node_score={}
        for node in target_USCO.stoGraph.nodes.keys():
            node_score[node] = 0
            for v in input_set:
                node_score[node] = node_score[node] + score_map[v][node]
        #print(node_score)
        sortedNodeScore = sorted(node_score.items(), key=lambda kv: kv[1],reverse=True)
        decision = set()
        for item in sortedNodeScore:
            decision.add(item[0])
            if len(decision) == query.budget:
                break
            
        #print(str(query.budget) + " " +str(len(decision)))
        #print(decision)
        predDecisions.append(decision)
        
    return predDecisions

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--sourceT',  default='DE', 
                        choices=['DE','DC'])
    
    parser.add_argument(
        '--targetT',  default='DC', 
                        choices=['DE','DC'])
    
    
    parser.add_argument(
        '--dataname',  default='higgs10000', 
                        choices=['kro','pl2500', 'er', 'higgs10000', 'hep'])
    #parser.add_argument(
    #    '--vNum', type=int, default=768, choices=[1024,768,512],
    #                    help='kro 1024, power768 768, ER512 512')
    
    
   
        
    parser.add_argument(
        '--trainNum', type=int, default=270 
         , help='number of training data')  
    
    parser.add_argument(
        '--testNum', type=int, default=540, help='number of testing data')   
    
    parser.add_argument(
        '--testBatch', type=int, default=5, help='number of testing data')   
     
    parser.add_argument(
        '--thread', type=int, default=90, help='number of threads')
    
    parser.add_argument(
        '--output', default=False, action="store_true", help='if output prediction')
    
    
    parser.add_argument(
        '--pre_train', default=True ,action="store_true", help='if store a pre_train model')
    
    parser.add_argument(
        '--log_path', default='log',  help='if store a pre_train model')
    
    
    args = parser.parse_args()
    #utils= Utils()
    
    #problem ="ssp"
    
    sourceT = args.sourceT
    targetT = args.targetT
    
    
    dataname=args.dataname
    #vNum = args.vNum
    
    
    
    trainNum =args.trainNum
    testNum =args.testNum
    
    thread = args.thread
    
  
    

    if dataname == "kro":
        vNum=1024  
        sourceSample_maxPair = 2700
        targetSample_maxPair = 2700
        
    if dataname == "er":
        vNum=512 
        sourceSample_maxPair = 2700
        targetSample_maxPair = 2700
        
    if dataname == "higgs10000":
        vNum=10000
        sourceSample_maxPair = 2700
        targetSample_maxPair = 2700
        
    if dataname == "pl2500":
        vNum=2500
        sourceSample_maxPair = 810
        targetSample_maxPair = 810
        
    if dataname == "hep":
        vNum=15233
        sourceSample_maxPair = 810
        targetSample_maxPair = 810
        
    if sourceT == "DE":
        source_scale_x, source_scale_y, source_type  = 40, 10, "noOverlap"
        
    if sourceT == "DC":
        source_scale_x, source_scale_y, source_type  = 10, 10, "normal"                     
        
    if targetT == "DE":
        target_scale_x, target_scale_y, target_type = 40, 10, "noOverlap"
        
    if targetT == "DC":
        target_scale_x, target_scale_y, target_type = 10, 10, "normal"
    
    

    
    pre_train = args.pre_train
    preTrainPathResult = None

    
    #get data
    path = os.getcwd() 
    data_path=os.path.dirname(path)+"/data"
    sourceSample_path = "{}/{}/{}_{}_samples_{}_{}_{}_{}".format(data_path, dataname, dataname, sourceT, source_scale_x, source_scale_y, source_type, sourceSample_maxPair)
    targetSample_path = "{}/{}/{}_{}_samples_{}_{}_{}_{}".format(data_path, dataname, dataname, targetT, target_scale_x, target_scale_y, target_type, targetSample_maxPair)
    stoGraphPath = "{}/{}/{}_model".format(data_path,dataname,dataname)
    #featurePath = "{}/{}/features/{}_{}".format(data_path, dataname, featureGenMethod, maxFeatureNum)
    if args.log_path is not None:
        logpath=path+"/log"
    #print(data_path)
    #print(pair_path)
    #print(stoGraphPath)
    #print(featurePath)
    stoGraph = StoGraph(stoGraphPath, vNum)
    #stoGraph.print()
    

    if sourceT== "DE":
        source_USCO = DE_USCO(stoGraph)
    if sourceT== "DC":
        source_USCO = DC_USCO(stoGraph)
    if targetT== "DE":
        target_USCO = DE_USCO(stoGraph)   
    if targetT== "DC":
        target_USCO = DC_USCO(stoGraph)
        
    

    
    TrainSamples, TrainQueries, TrainDecisions, = source_USCO.readSamples(sourceSample_path, trainNum, sourceSample_maxPair)
    TestSamples, TestQueries, TestDecisions = target_USCO.readSamples(targetSample_path, testNum, targetSample_maxPair)
    

    
    
    
    
    
    
    #print(X_train)
    print("data fetched")
    #sys.exit()
    Utils.writeToFile(logpath, "data fetched")
  

    
    
    
    if pre_train is True:
        now = datetime.now()
        preTrainPath=path+"/pre_train/"+now.strftime("%d-%m-%Y-%H-%M-%S")+"/"
        if not os.path.exists(preTrainPath):
            os.makedirs(preTrainPath)
        #Utils.save_pretrain(preTrainPath, one_slack_svm.w, realizationIndexes, featurePath)
        Utils.writeToFile(logpath, preTrainPath, toconsole = True)   
        preTrainPathResult = preTrainPath+"/result"
    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True, preTrainPathResult = preTrainPathResult)
    
    Utils.writeToFile(logpath, "Making NB predictions ...", toconsole = True,preTrainPathResult = preTrainPathResult)
    
    
    
    predDecisions = SI_NB(sourceT, targetT, target_USCO, TrainSamples, TestQueries)
    Utils.writeToFile(logpath, "Testing ...", toconsole = True,preTrainPathResult = preTrainPathResult)
    
    
    #for decision, pred in zip(TestDecisions, predDecisions):
    #    print(decision)
    #    print(pred)
     #   print("=======================")
    #print(TestDecisions)
    
    Utils.writeToFile(logpath, sourceT+" "+targetT, toconsole = True, preTrainPathResult = preTrainPathResult) 
    Utils.writeToFile(logpath, dataname, toconsole = True, preTrainPathResult = preTrainPathResult)    
    #Utils.writeToFile(logpath, "featureNum: {}, featureGenMethod: {}, ".format(featureNum, featureGenMethod), toconsole = True,preTrainPathResult = preTrainPathResult)
    Utils.writeToFile(logpath, "trainNum: {}, {}_{} ".format(trainNum, source_scale_x, source_scale_y), toconsole = True,preTrainPathResult = preTrainPathResult)
    Utils.writeToFile(logpath, "testNum:{}, {}_{} ".format(testNum, target_scale_x, target_scale_y), toconsole = True,preTrainPathResult = preTrainPathResult)
    #Utils.writeToFile(logpath, "maxIter: {}, c: {} ".format(max_iter, C), toconsole = True,preTrainPathResult = preTrainPathResult)

    target_USCO.test(TestSamples, TestQueries, TestDecisions, predDecisions, thread, logpath = logpath, preTrainPathResult = preTrainPathResult )
    
    
    #mean_ratios, std_ratios, mean_infs, std_infs, mean_predInfs, std_predInfs, mean_inter, std_inter

    

    
    
    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True,preTrainPathResult = preTrainPathResult)
    
if __name__ == "__main__":
    #x, y = 0, 1
    #print(x)
    #print(y)
    main()