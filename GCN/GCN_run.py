from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

import GCN_utils
from GCN_utils import GCN
import sys
from datetime import datetime


sys.path.insert(0,'..')
from DE import DE_USCO
from DC import DC_USCO
from StoGraph import StoGraph
from Utils import Utils





def train(model, loader, optimizer, epoch=10):
    # Each epoch has a training and validation phase
    print("Epoch: {}".format(epoch))
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
            torch.set_grad_enabled(True)
            loader = data_loaders[phase]
            iters_per_epoch = len(loader)
        else:
            model.eval()  # Set model to evaluate mode
            loader = data_loaders[phase]
            iters_per_epoch = len(loader)

        for i, sample in enumerate(loader, start=epoch * iters_per_epoch):
            if args.cuda:
                input, target, adj, card = map(lambda x: x.cuda(), sample)
            else:
                input, target, adj, card = map(lambda x: x, sample)
    
            prediction = []
            for item in range(len(input)):
                result = model(input[item], adj[item])
                prediction.append(result)

    
            prediction = torch.stack(prediction, dim=0)
            loss = F.binary_cross_entropy_with_logits(prediction, target.squeeze())

        
    
            if phase == "train":
                optimizer.zero_grad() # zero the parameter (weight) gradients
                loss.backward() # backward pass to calculate the weight gradients
                optimizer.step() # update the weights
            else:
                output=loss.item()
            
                
        print("{} Losses: {:.04f}".format(phase, loss))
        
    return output
    
            
            
def test(model, loader, optimizer, epoch=0):
    model.eval()  # Set model to evaluate mode
    loader =loader
    iters_per_epoch = len(loader)
    
    true_export, pred_export, true_import = [], [], []
    
    for i, sample in enumerate(loader, start=epoch * iters_per_epoch):
            if args.cuda:
                input, target, adj, card = map(lambda x: x.cuda(), sample)
            else:
                input, target, adj, card = map(lambda x: x, sample)
    
            prediction = []
            for item in range(len(input)):
                result = model(input[item], adj[item])
                prediction.append(result)
                #results.append(result)
    
            prediction = torch.stack(prediction, dim=0)
            
            # calculate the loss between predicted and target keypoints
            loss = F.binary_cross_entropy_with_logits(prediction, target.squeeze())
            print("\n{} Loss: {}".format('Test', loss.item()))
            
            for pre, tru, inp, c in zip(prediction, target, input, card):
                #print("pre {}".format(GCN_utils.one_hot_to_str(inp.sum(dim=1).cpu().detach())))
                #print("tru {}".format(GCN_utils.one_hot_to_str(tru.squeeze().cpu().detach().numpy())))
                #print("inp {}".format(pre.cpu().detach()))
                #sys.exit()
                k=len(GCN_utils.one_hot_to_str(tru.squeeze().cpu().detach().numpy()))
               # print(k)
                true_import.append(torch.topk(inp.sum(dim=1), k).indices.numpy())
                pred_export.append(torch.topk(pre.cpu().detach(), k).indices.numpy())
                true_export.append(torch.topk(tru.squeeze(), k).indices.numpy())
                
    #X_test=[]
    #Y_test=[]
    Y_pred=[]
    for y_pred, y_test, x_test in zip(pred_export, true_export, true_import):
        Y_pred.append(set([str(x) for x in y_pred]))
    
    return Y_pred
    #GCN_utils.runTesting(X_test, Y_test, Y_pred, args, instance)
            
            
# Training the model


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
            '--sourceT',  default='DC', 
                            choices=['DE','DC'])
        
    parser.add_argument(
        '--targetT',  default='DE', 
                        choices=['DE','DC'])
    
    
    parser.add_argument(
        '--dataname',  default='kro', 
                        choices=['kro','pl2500', 'er', 'higgs10000', 'hep'])
    #parser.add_argument(
    #    '--vNum', type=int, default=1024, choices=[1024,768,512,4039],
    #                    help='kro 1024, power768 768, ER512 512')
        
    parser.add_argument(
        '--trainNum', type=int, default=180, help='number of training data') 
    parser.add_argument(
        '--valNum', type=int, default=90, help='number of validation data')   
    parser.add_argument(
        '--testNum', type=int, default=540, help='number of testing data')   
    
    parser.add_argument(
        '--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    
    parser.add_argument(
        '--thread', type=int, default=1, help='number of threads')
    
    parser.add_argument(
        '--output', action="store_true", help='if output prediction')
    
    
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed.')
    parser.add_argument(
        '--epochs', type=int, default=10,
                        help='Number of epochs to train.')
    parser.add_argument(
        '--lr', type=float, default=1e-2,
                        help='Initial learning rate.')
    parser.add_argument(
        '--weight_decay', type=float, default=1e-2,
                        help='Weight decay (L2 loss on parameters).')
    
    parser.add_argument(
        '--dropout', type=float, default=0.4,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument(
        "--train-only", action="store_true", default=False,
                        help="Only run training, no evaluation")
    parser.add_argument(
        "--eval-only", action="store_true", 
                        help="Only run evaluation, no training")
    parser.add_argument(
        "--batch-size", default=10,
                        help="Batch size of the training/testing set")
    
    parser.add_argument(
        '--pre_train', default=True ,action="store_true", help='if store a pre_train model')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    torch.set_num_threads(args.thread)
    
    sourceT = args.sourceT
    targetT = args.targetT
        
    dataname=args.dataname
    
    
    trainNum =args.trainNum
    valNum =args.valNum
    testNum =args.testNum
    #pairMax=2000
    totalNum=trainNum+valNum+testNum
    
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
    
    path = os.getcwd() 
    data_path=os.path.dirname(path)+"/data"
    sourceSample_path = "{}/{}/{}_{}_samples_{}_{}_{}_{}".format(data_path, dataname, dataname, sourceT, source_scale_x, source_scale_y, source_type, sourceSample_maxPair)
    targetSample_path = "{}/{}/{}_{}_samples_{}_{}_{}_{}".format(data_path, dataname, dataname, targetT, target_scale_x, target_scale_y, target_type, targetSample_maxPair)
    stoGraphPath = "{}/{}/{}_model".format(data_path, dataname, dataname)
    
    
    stoGraph = StoGraph(stoGraphPath, vNum)
        
    if sourceT== "DE":
        source_USCO = DE_USCO(stoGraph)
    if sourceT== "DC":
        source_USCO = DC_USCO(stoGraph)
    if targetT== "DE":
        target_USCO = DE_USCO(stoGraph)   
    if targetT== "DC":
        target_USCO = DC_USCO(stoGraph)
    
    logpath=path+"/log_gcn"
    
    trainVal_Samples, trainVal_Queries, trainVal_Decisions, maxTrainSize = source_USCO.readSamples(sourceSample_path, trainNum+valNum, sourceSample_maxPair, RmaxSize = True)
    #ValSamples, ValQueries, ValDecisions, maxValSize = source_USCO.readSamples(sourceSample_path, trainNum, sourceSample_maxPair, RmaxSize = True)
    TestSamples, TestQueries, TestDecisions, maxTestSize = target_USCO.readSamples(targetSample_path, testNum, targetSample_maxPair, RmaxSize = True)
   
    
    

    
    #topic_size=instance.stoCoverGraph.topicNum
    #node_size=instance.stoCoverGraph.nodeNum
    
    #args.topic_size = topic_size
    #args.node_size = node_size
    #vNum = len(instance.stoCoverGraph.vertices)
    #print(str(topic_size)+" "+str(node_size))
    
    
    #lineNums=(np.random.permutation(pairMax))
    #allLineNums=lineNums[0: trainNum+testNum+valNum]
    #trainLineNums=lineNums[0:trainNum]
    #testLineNums=lineNums[trainNum: trainNum+testNum]
    #valLineNums=lineNums[trainNum+testNum : trainNum+testNum+valNum]
    
    
    print('Building dataset...')
    trainVal_dataset = GCN_utils.Dataset(trainVal_Samples, stoGraphPath, sourceT, vNum,  train=True)
    test_dataset = GCN_utils.Dataset(TestSamples, stoGraphPath, targetT, vNum, train=True)
    #print(len(dataset.data))
    train_dataset, val_dataset = random_split(trainVal_dataset , (trainVal_dataset.total_size-valNum, valNum))
    #train_dataset, test_dataset = random_split(train_dataset, (len(train_dataset) - len(train_dataset)//5, len(train_dataset)//5))
    
    # Getting data loaders for training, validation, and testing
    train_loader = GCN_utils.get_loader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    valid_loader = GCN_utils.get_loader(val_dataset,  batch_size=valNum, num_workers=0, shuffle=False)
    test_loader = GCN_utils.get_loader(test_dataset, batch_size=test_dataset.total_size, num_workers=0, shuffle=False)
    
    data_loaders = {"train": train_loader, "val": valid_loader}
    
    
    
    # Model and optimizer
    model = GCN(vNum, dropout=args.dropout)
    
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    
    if args.cuda:
        model.cuda()
        
    t_loss=sys.maxsize
    for k in range(args.epochs):   
        c_loss=train(model, data_loaders, optimizer, epoch=k)
        #print(t_loss)
        if c_loss>t_loss:
            break
        else:
            #print("********")
            t_loss=c_loss
            #print(t_loss)
    # Generating testing report
    if not args.train_only:
        Y_pred= test(model, test_loader, optimizer, epoch=k)
        
        if args.pre_train is True:
                now = datetime.now()
                preTrainPath=path+"/pre_train/"+now.strftime("%d-%m-%Y-%H-%M-%S")+"/"
                if not os.path.exists(preTrainPath):
                    os.makedirs(preTrainPath)
                    Utils.writeToFile(args.logpath, preTrainPath, toconsole = True)   
                    preTrainPathResult = preTrainPath+"/result"
                    
                    
        Utils.writeToFile(logpath, "Testing GCN ...", toconsole = True)  
        Utils.writeToFile(logpath, sourceT+" "+targetT, toconsole = True) 
        Utils.writeToFile(logpath, dataname, toconsole = True)    
        Utils.writeToFile(logpath, "trainNum: {}, {}_{} ".format(trainNum, source_scale_x, source_scale_y), toconsole = True)
        Utils.writeToFile(logpath, "testNum:{}, {}_{} ".format(testNum, target_scale_x, target_scale_y), toconsole = True)     
        Utils.writeToFile(logpath, "lr: {}, seed: {}, batch-size: {}".format(args.lr, args.seed, args.batch_size), toconsole = True)    
        target_USCO.test(TestSamples, TestQueries, TestDecisions, Y_pred, args.thread, logpath = logpath)
            
