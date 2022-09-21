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
from Utils import Utils
from StoGraph import StoGraph
from DE import DE_USCO
from DC import DC_USCO

class Object(object):
    pass

def SI_random(target_USCO, TestQueries):
    predDecisions = [] 
    for query in TestQueries:
         nodes = list(target_USCO.stoGraph.nodes.keys())
         random.shuffle(nodes)
         decision = set(nodes[0:query.budget])
         predDecisions.append(decision)
    return predDecisions

def main():
    parser = argparse.ArgumentParser()
    
   
    
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
        '--testNum', type=int, default=540, help='number of testing data')   
    
   
     
    parser.add_argument(
        '--thread', type=int, default=1, help='number of threads')
    
    parser.add_argument(
        '--output', default=False, action="store_true", help='if output prediction')
    
    
    parser.add_argument(
        '--pre_train', default=True ,action="store_true", help='if store a pre_train model')
    
    parser.add_argument(
        '--log_path', default='log',  help='if store a pre_train model')
    
    
    args = parser.parse_args()

    

    targetT = args.targetT
    
    
    dataname=args.dataname


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
    targetSample_path = "{}/{}/{}_{}_samples_{}_{}_{}_{}".format(data_path, dataname, dataname, targetT, target_scale_x, target_scale_y, target_type, targetSample_maxPair)
    stoGraphPath = "{}/{}/{}_model".format(data_path,dataname,dataname)
    #featurePath = "{}/{}/features/{}_{}".format(data_path, dataname, featureGenMethod, maxFeatureNum)
    if args.log_path is not None:
        logpath=path+"/log"

    stoGraph = StoGraph(stoGraphPath, vNum)
    #stoGraph.print()
    

    if targetT== "DE":
        target_USCO = DE_USCO(stoGraph)   
    if targetT== "DC":
        target_USCO = DC_USCO(stoGraph)
        
    
    #
    
   
    TestSamples, TestQueries, TestDecisions = target_USCO.readSamples(targetSample_path, testNum, targetSample_maxPair)
    
    #for sample in TrainSamples:
    #    sample.print()
    #sys.exit("stop")
    
    
    
    
    
    
    
    #print(X_train)
    print("data fetched")
    #sys.exit()
    Utils.writeToFile(logpath, "data fetched")
  
    #realizations, realizationIndexes = source_USCO.readRealizations(featurePath, featureNum, maxNum = maxFeatureNum)
    
    
    
    
    #**************************OneSlackSSVM
    #model = Model()
    #usco_Solver = USCO_Solver()
    #usco_Solver.initialize(realizations, source_USCO)
    
   
    #one_slack_svm = OneSlackSSVM(usco_Solver, verbose=verbose, C=C, tol=tol, n_jobs=thread,
                             #max_iter = max_iter, log = logpath)
    
    
    #one_slack_svm.fit(TrainQueries, TrainDecisions, initialize = False)
    
    
    
    if pre_train is True:
        now = datetime.now()
        preTrainPath=path+"/pre_train/"+now.strftime("%d-%m-%Y-%H-%M-%S")+"/"
        if not os.path.exists(preTrainPath):
            os.makedirs(preTrainPath)
        #Utils.save_pretrain(preTrainPath, one_slack_svm.w, realizationIndexes, featurePath)
        Utils.writeToFile(logpath, preTrainPath, toconsole = True)   
        preTrainPathResult = preTrainPath+"/result"
    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True, preTrainPathResult = preTrainPathResult)
    
    Utils.writeToFile(logpath, "Making random predictions ...", toconsole = True,preTrainPathResult = preTrainPathResult)
    
    
    
    predDecisions = SI_random(target_USCO, TestQueries)
    Utils.writeToFile(logpath, "Testing ...", toconsole = True,preTrainPathResult = preTrainPathResult)
    
    #print(TestDecisions)
    
    Utils.writeToFile(logpath, targetT, toconsole = True, preTrainPathResult = preTrainPathResult) 
    Utils.writeToFile(logpath, dataname, toconsole = True, preTrainPathResult = preTrainPathResult)    
    Utils.writeToFile(logpath, "testNum:{}, {}_{} ".format(testNum, target_scale_x, target_scale_y), toconsole = True,preTrainPathResult = preTrainPathResult)
   
    target_USCO.test(TestSamples, TestQueries, TestDecisions, predDecisions, thread, logpath = logpath, preTrainPathResult = preTrainPathResult )
    
    
    #mean_ratios, std_ratios, mean_infs, std_infs, mean_predInfs, std_predInfs, mean_inter, std_inter

    

    
    
    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True,preTrainPathResult = preTrainPathResult)
    
if __name__ == "__main__":
    main()