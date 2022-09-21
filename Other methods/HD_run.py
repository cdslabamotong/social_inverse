"""
==============================
StratLearner Training
==============================
"""



import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0,'..')
from DE import DE_USCO
from DC import DC_USCO
from Utils import Utils
from StoGraph import StoGraph

class Object(object):
    pass

def SI_HD(targetT, target_USCO,  TestQueries):
    predDecisions = [] 
    
    
    node_degree= {}
    for node in target_USCO.stoGraph.nodes:
        node_degree[node]=target_USCO.stoGraph.nodes[node].out_degree
        
    sortedNodeScore = sorted(node_degree.items(), key=lambda kv: kv[1],reverse=True)
    
    for query in TestQueries:   
        decision = set()
        for item in sortedNodeScore:
            decision.add(item[0])
            if len(decision) == query.budget:
                break
        predDecisions.append(decision)
        
    return predDecisions

def main():
    parser = argparse.ArgumentParser()
    
    
    
    parser.add_argument(
        '--targetT',  default='DC', 
                        choices=['DE','DC'])
    
    
    parser.add_argument(
        '--dataname',  default='kro', 
                        choices=['kro','pl2500', 'er', 'higgs10000', 'hep'])
        
    
    
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
    

    targetT = args.targetT
    
    
    dataname=args.dataname
    #vNum = args.vNum
    

    testNum =args.testNum

    
    thread = args.thread
    
    

        
                  
        
    if dataname == "kro":
        vNum=1024  
        targetSample_maxPair = 2700
        
    if dataname == "er":
        vNum=512 
        targetSample_maxPair = 2700
        
    if dataname == "higgs10000":
        vNum=10000
        targetSample_maxPair = 2700
        
    if dataname == "pl2500":
        vNum=2500
        targetSample_maxPair = 810
        
    if dataname == "hep":
        vNum=15233
        targetSample_maxPair = 810
                     
        
    if targetT == "DE":
        target_scale_x, target_scale_y, target_type = 40, 10, "noOverlap"
        
    if targetT == "DC":
        target_scale_x, target_scale_y, target_type = 10, 10, "normal"
    
    
    
    

    
    pre_train = args.pre_train
    preTrainPathResult = None

    
    #get data
    path = os.getcwd() 
    data_path=os.path.dirname(path)+"/data"
    #sourceSample_path = "{}/{}/{}_{}_samples_{}_{}_{}_{}".format(data_path, dataname, dataname, sourceT, source_scale_x, source_scale_y, source_type, sourceSample_maxPair)
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
    

    if targetT== "DE":
        target_USCO = DE_USCO(stoGraph)   
    if targetT== "DC":
        target_USCO = DC_USCO(stoGraph)
        
    
    #
    
    #TrainSamples, TrainQueries, TrainDecisions, = source_USCO.readSamples(sourceSample_path, trainNum, sourceSample_maxPair)
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
    
    Utils.writeToFile(logpath, "Making HD predictions ...", toconsole = True,preTrainPathResult = preTrainPathResult)
    
    
    
    predDecisions = SI_HD(targetT, target_USCO, TestQueries)
    Utils.writeToFile(logpath, "Testing ...", toconsole = True,preTrainPathResult = preTrainPathResult)
    
    
 
    
    Utils.writeToFile(logpath, targetT, toconsole = True, preTrainPathResult = preTrainPathResult) 
    Utils.writeToFile(logpath, dataname, toconsole = True, preTrainPathResult = preTrainPathResult)    
    #Utils.writeToFile(logpath, "featureNum: {}, featureGenMethod: {}, ".format(featureNum, featureGenMethod), toconsole = True,preTrainPathResult = preTrainPathResult)
    #Utils.writeToFile(logpath, "trainNum: {}, {}_{} ".format(trainNum, source_scale_x, source_scale_y), toconsole = True,preTrainPathResult = preTrainPathResult)
    Utils.writeToFile(logpath, "testNum:{}, {}_{} ".format(testNum, target_scale_x, target_scale_y), toconsole = True,preTrainPathResult = preTrainPathResult)
    #Utils.writeToFile(logpath, "maxIter: {}, c: {} ".format(max_iter, C), toconsole = True,preTrainPathResult = preTrainPathResult)

    target_USCO.test(TestSamples, TestQueries, TestDecisions, predDecisions, thread, logpath = logpath, preTrainPathResult = preTrainPathResult )
    
    
    #mean_ratios, std_ratios, mean_infs, std_infs, mean_predInfs, std_predInfs, mean_inter, std_inter

    

    
    
    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True,preTrainPathResult = preTrainPathResult)
    
if __name__ == "__main__":
    main()