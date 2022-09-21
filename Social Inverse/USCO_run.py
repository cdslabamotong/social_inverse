"""
==============================
# License: BSD 3-clause
==============================
"""
import os
import sys
import multiprocessing
import argparse
import numpy as np
from datetime import datetime


from one_slack_ssvm import OneSlackSSVM
from subgradient_ssvm import SubgradientSSVM
from n_slack_ssvm import NSlackSSVM


sys.path.insert(0,'..')
from DE import DE_USCO
from DC import DC_USCO
from StoGraph import StoGraph
from USCO import USCO_Solver
from Utils import Utils




class Object(object):
    pass


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--sourceT',  default='DC', 
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
        '--featureNum', type=int, default=10,
                        help='number of features (random subgraphs) used in StratLearn ')
    parser.add_argument(
        '--featureGenMethod', default='uniform_1', \
             choices=['em_uniform','uniform_true','em_approx_50_50','em_approx_100_100','em_approx_10_10'], \
                help='the distribution used for generating features, the choices correspond phi_1^1, phi_0.01^1, phi_0.005^1, phi_+^+')
    
        
    parser.add_argument(
        '--trainNum', type=int, default=270, help='number of training data')  
    
    parser.add_argument(
        '--testNum', type=int, default=540, help='number of testing data')   
    
    parser.add_argument(
        '--testBatch', type=int, default=5, help='number of testing data')   
     
    parser.add_argument(
        '--thread', type=int, default=90, help='number of threads')
    
    parser.add_argument(
        '--beta', type=float, default=1, help='number of threads')
    
    parser.add_argument(
        '--output', default=False, action="store_true", help='if output prediction')
    
    
    parser.add_argument(
        '--pre_train', default=True ,action="store_true", help='if store a pre_train model')
    
    parser.add_argument(
        '--log_path', default='log',  help='if store a pre_train model')
    
    
    args = parser.parse_args()
    #utils= Utils.Utils
    
    #problem ="ssp"
    
    sourceT = args.sourceT
    targetT = args.targetT
    
    
    dataname=args.dataname
    #vNum = args.vNum
    
    
    
    trainNum =args.trainNum
    testNum =args.testNum
    #testBatch =args.testBatch
    #pairMax=2500
    
    thread = args.thread
    
    
    verbose=3
    #parameter used in SVM
    C = 0.0001
    tol=0.001
    max_iter = 10
    #alpha = 0.618
    #beta = alpha*alpha*(1-args.beta)
    beta = args.beta
    featureNum = args.featureNum
    featureGenMethod = args.featureGenMethod
    

    
   
    if dataname == "kro":
        vNum=1024  
        maxFeatureNum = 10000
        sourceSample_maxPair = 2700
        targetSample_maxPair = 2700
        
    if dataname == "er":
        vNum=512 
        maxFeatureNum = 10000
        sourceSample_maxPair = 2700
        targetSample_maxPair = 2700
        
    if dataname == "higgs10000":
        vNum=10000
        maxFeatureNum = 1000
        sourceSample_maxPair = 2700
        targetSample_maxPair = 2700
        
    if dataname == "pl2500":
        vNum=2500
        maxFeatureNum = 1000
        sourceSample_maxPair = 810
        targetSample_maxPair = 810
        
    if dataname == "hep":
        vNum=15233
        maxFeatureNum = 1000
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
    featurePath = "{}/{}/features/{}_{}".format(data_path, dataname, featureGenMethod, maxFeatureNum)
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
        
    
    #
    
    TrainSamples, TrainQueries, TrainDecisions, = source_USCO.readSamples(sourceSample_path, trainNum, sourceSample_maxPair)
    TestSamples, TestQueries, TestDecisions = target_USCO.readSamples(targetSample_path, testNum, targetSample_maxPair)
    
    #for sample in TrainSamples:
    #    sample.print()
    #sys.exit("stop")
    
    
    
    
    #print(X_train)
    print("data fetched")
    #sys.exit()
    Utils.writeToFile(logpath, "data fetched")
    '''
    for x, y in zip (X_train, Y_train):
        print(x)
        print(y)
        print()
    '''  
    realizations, realizationIndexes = source_USCO.readRealizations(featurePath, featureNum, maxNum = maxFeatureNum)
    #for realization in realizations:
    #    realization.print()
    #sys.exit("stop")
    
    
    #instance = SSP_InputInstance(stoGraphPath, featurePath, featureNum, vNum, 
                             #featureRandom = True, maxFeatureNum = maxFeatureNum,
                             #thread = thread)
    
    
    
    train_start_time = datetime.now()
    #model = Model()
    usco_Solver = USCO_Solver()
    usco_Solver.initialize(realizations, source_USCO)
    
    #**************************OneSlackSSVM
    #one_slack_svm = OneSlackSSVM(usco_Solver, verbose=verbose, C=C, tol=tol, n_jobs=thread, max_iter = max_iter, log = logpath)
    #one_slack_svm.fit(TrainQueries, TrainDecisions, beta, initialize = False)
    #w=one_slack_svm.w
    
    #**************************nSlackSSVM
    n_slack_svm = NSlackSSVM(usco_Solver, verbose=verbose, C=C, tol=tol, n_jobs=thread, max_iter = max_iter, log = logpath)
    n_slack_svm.fit(TrainQueries, TrainDecisions,  beta, initialize = False)
    w=n_slack_svm.w
    
    
    #**************************subgradient
    #subgradient_ssvm = SubgradientSSVM(usco_Solver, verbose=verbose, C=C, n_jobs=thread, max_iter = max_iter, log = logpath, learning_rate = 0.1, decay_exponent = 0)
    #subgradient_ssvm.fit(TrainQueries, TrainDecisions, initialize = False)    
    #w=subgradient_ssvm.w
    
    train_end_time = datetime.now()
    train_time = train_end_time  - train_start_time
    print(train_time)
    
    test_start_time = datetime.now()
    #**************************inference and save pretrain
    Utils.writeToFile(logpath, "Making SI predictions ...", toconsole = True,preTrainPathResult = preTrainPathResult)
    predDecisions = target_USCO.solve_R_batch(TestQueries, w, realizations, n_jobs=thread, offset = None)
    
    test_end_time = datetime.now()
    test_time = test_end_time - test_start_time
    print(test_time)
    
    if pre_train is True:
        now = datetime.now()
        preTrainPath=path+"/pre_train/"+now.strftime("%d-%m-%Y-%H-%M-%S")+"/"
        if not os.path.exists(preTrainPath):
            os.makedirs(preTrainPath)
        Utils.save_pretrain(preTrainPath, w, realizationIndexes, featurePath)
        Utils.writeToFile(logpath, preTrainPath, toconsole = True)   
        preTrainPathResult = preTrainPath+"/result"
    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True, preTrainPathResult = preTrainPathResult)
    
    
    Utils.writeToFile(logpath, "Testing ...", toconsole = True,preTrainPathResult = preTrainPathResult)
    
    #print(TestDecisions)
    
    Utils.writeToFile(logpath, sourceT+" "+targetT, toconsole = True, preTrainPathResult = preTrainPathResult) 
    Utils.writeToFile(logpath, dataname, toconsole = True, preTrainPathResult = preTrainPathResult)    
    Utils.writeToFile(logpath, "featureNum: {}, featureGenMethod: {}, beta: {}".format(featureNum, featureGenMethod, beta), toconsole = True,preTrainPathResult = preTrainPathResult)
    Utils.writeToFile(logpath, "trainNum: {}, {}_{} ".format(trainNum, source_scale_x, source_scale_y), toconsole = True,preTrainPathResult = preTrainPathResult)
    Utils.writeToFile(logpath, "testNum:{}, {}_{} ".format(testNum, target_scale_x, target_scale_y), toconsole = True,preTrainPathResult = preTrainPathResult)
    Utils.writeToFile(logpath, "maxIter: {}, c: {} ".format(max_iter, C), toconsole = True,preTrainPathResult = preTrainPathResult)
    
    Utils.writeToFile(logpath, "=================================", toconsole = True, preTrainPathResult = preTrainPathResult)
    
    ratio_pred = target_USCO.test(TestSamples, TestQueries, TestDecisions, predDecisions, thread, logpath = logpath, preTrainPathResult = preTrainPathResult )

    Utils.writeToFile(logpath, "=================================", toconsole = True, preTrainPathResult = preTrainPathResult)
        
    #randomW=np.random.rand(featureNum)
    randomW=np.ones(featureNum)
    randPredDecisions = target_USCO.solve_R_batch(TestQueries, randomW, realizations, n_jobs=thread, offset = None)
    ratio_ones=target_USCO.test(TestSamples, TestQueries, TestDecisions, randPredDecisions, thread, logpath = logpath, preTrainPathResult = preTrainPathResult )
    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True,preTrainPathResult = preTrainPathResult)
    
    Utils.writeToFile(logpath, "{} ({}) {}".format(ratio_pred, ratio_ones, now.strftime("%d-%m-%Y-%H-%M-%S")), toconsole = True,preTrainPathResult = preTrainPathResult)
    
    Utils.writeToFile(logpath, "train_time {} ".format(train_time), toconsole = True,preTrainPathResult = preTrainPathResult)
    Utils.writeToFile(logpath, "test_time {} ".format(test_time), toconsole = True,preTrainPathResult = preTrainPathResult)
    
    #mean_ratios, std_ratios, mean_infs, std_infs, mean_predInfs, std_predInfs, mean_inter, std_inter

    return "{} {} {}".format(ratio_pred, ratio_ones, now.strftime("%d-%m-%Y-%H-%M-%S"))

    
    
    
   
if __name__ == "__main__":
    main()
    