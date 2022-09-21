# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn.functional as F
#import SSP
import DSPN_utils
import sys


sys.path.insert(0,'..')
from DE import DE_USCO
from DC import DC_USCO
from StoGraph import StoGraph
from Utils import Utils
from datetime import datetime



 
def run(net, dataset,  optimizer, args, train=True, epoch=10, pool=None, valDataset=None, target_USCO=None):
    
    if train:
        net.train()    
        print("train")
        torch.set_grad_enabled(True)
    else:
        net.eval()
        torch.set_grad_enabled(False)

    #stop=False;
    
    iters_per_epoch=int(dataset.size/dataset.batch_size)   
    print("epoch {}".format(epoch))
    #min_loss=sys.maxsize
    Y_test = []
    Y_pred = []
    X_test = []
    for i in range(iters_per_epoch):
        print("batch {} {} {} ".format(epoch ,iters_per_epoch, i))

        sample=dataset.data[i*dataset.batch_size : (i+1)*dataset.batch_size]
        
        #print(len(dataset.data))
        #print(dataset.batch_size)
        #print(i)
        
        input=torch.stack([x[0] for x in sample])
        target_set=torch.stack([x[1] for x in sample])
        #target_mask=torch.stack([x[2] for x in sample])
        
        #print()
        (progress, masks, evals, gradn), (y_enc, y_label) = net(
            input, target_set, max_set_size=dataset.max_size)
                    
        
        # Only use representation loss with DSPN and when doing general supervised prediction, not when auto-encoding

        repr_loss = 10 * F.smooth_l1_loss(y_enc, y_label)
        loss = repr_loss
        
                   
        set_true, set_pred = [], []
        for i in range(len(target_set)):
            set_true.append(DSPN_utils.matrix_to_one_hot(target_set[i].detach().cpu(), vNum))
            set_pred.append(progress[-1][i].detach().cpu())
                            
      
        set_loss = []
        for i in range(len(set_pred)):
            set_pred[i].requires_grad=True
            #set_loss.append(F.smooth_l1_loss(set_pred[i], set_true[i]*1000000000))
            set_loss.append(F.binary_cross_entropy_with_logits(set_pred[i], set_true[i]))
            
        if args.no_cuda:
            set_loss = torch.tensor(set_loss, dtype=torch.float64, requires_grad=True)
        else:
            set_loss = set_loss.cuda()
        
        #loss = set_loss.mean()
        loss = set_loss.mean() + repr_loss.mean()
        print('\n set loss: ', set_loss.mean().item())
        print('repr loss: ', repr_loss.mean().item())
        
        if train:
            #print("optimizer.zero_grad()...")
            optimizer.zero_grad()
            #print("loss.backward()...")
            loss.backward()
            #print("optimizer.step()...")
            optimizer.step()
            #print("optimizer.step() done")
            my_lr_scheduler.step()      

        if train:
            sample_val=valDataset.data
            
            input_val=torch.stack([x[0] for x in sample_val]).detach().cpu()
            target_set_val=torch.stack([x[1] for x in sample_val]).detach().cpu()
          
            
            
            (progress_val, masks, evals, gradn), (y_enc_val, y_label_val) = net(
                input_val, target_set_val, valDataset.max_size)
                        
            
            # Only use representation loss with DSPN and when doing general supervised prediction, not when auto-encoding

            repr_loss_val = 10 * F.smooth_l1_loss(y_enc_val, y_label_val)
            #loss_val = repr_loss_val
            
                       
            set_true_val, set_pred_val = [], []
            for i in range(len(target_set_val)):
                set_true_val.append(DSPN_utils.matrix_to_one_hot(target_set_val[i].detach().cpu(), vNum))
                set_pred_val.append(progress_val[-1][i].detach().cpu())
                                
          
            set_loss_val = []
            for i in range(len(set_pred_val)):
                set_pred_val[i].requires_grad=True
                #print(set_pred_val[i].shape)
                #print(set_true_val[i].shape)
                set_loss_val.append(F.binary_cross_entropy_with_logits(set_pred_val[i], set_true_val[i]))
                
            if args.no_cuda:
                set_loss_val = torch.tensor(set_loss_val, dtype=torch.float64)
            else:
                set_loss_val = set_loss_val.cuda()
            
            print('\n set val loss: ', set_loss_val.mean().item())
            print('repr  val loss: ', repr_loss_val.mean().item())
            
            
            true_export_val = []
            pred_export_val = []
            true_import_val = []
    
            for p, s, pro in zip(target_set_val, input_val, progress_val[-1]):
                true_export_val.append(DSPN_utils.one_hot_to_number(p.detach().cpu()))
                true_import_val.append(DSPN_utils.one_hot_to_number(s.detach().cpu()))
                k=len(DSPN_utils.one_hot_to_number(p.detach().cpu()))
                pred_export_val.append(torch.topk(pro.cpu().detach(), k=k).indices.numpy())
            #utils.runTesting(true_import_val, true_export_val, pred_export_val, instance)
            
            #print(len(utils.one_hot_to_number(p.detach().cpu())))
            #print(len(torch.topk(pro.cpu().detach(), k=k).indices.numpy()))
            print("*********")
            '''
            if set_loss_val.mean().item()>min_loss:
                stop=True;
                break;
            else:
                min_loss=set_loss_val.mean().item()
            '''
           
        if not train:
            for p, s, pro in zip(target_set, input, progress[-1]):
                Y_test.append(DSPN_utils.one_hot_to_str(p.detach().cpu()))
                X_test.append(DSPN_utils.one_hot_to_str(s.detach().cpu()))
                #print("-----------")
                #k=vNum
                
                k=len(DSPN_utils.one_hot_to_number(p.detach().cpu()))
                Y_pred.append(set([str(x) for x in torch.topk(pro.cpu().detach(), k=k).indices.numpy()]))
                #sortedNodes = [str(x) for x in torch.topk(pro.cpu().detach(), k=k).indices.numpy()]
                #Y_pred.append(sortedNodes)
                #print(Y_pred[-1])
                #Y_pred_sortedNodes= DSPN_utils.one_hot_to_sortedDic(pro.cpu().detach())
                #Y_pred.append(SSP_Utils.prediction_from_sortedDic(instance.stoGraph.EGraph.nodes, DSPN_utils.one_hot_to_number(s.detach().cpu()), Y_pred_sortedDic))
                
                #one_hot_to_sortedDic

    print("Loop done")
    #print(len(pred_export))
    
    if train:
        return set_loss_val.mean().item()
    if not train:                        
        return Y_pred
    

    


if __name__ == "__main__":
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
    #    '--vNum', type=int, default=1024, choices=[1024,768,512, 4039],
    #                    help='kro 1024, power768 768, ER512 512')
        
    parser.add_argument(
        '--trainNum', type=int, default=180, help='number of training data') 
    parser.add_argument(
        '--valNum', type=int, default=90, help='number of validation data')   
    parser.add_argument(
        '--testNum', type=int, default=540, help='number of testing data')   
    
    parser.add_argument(
            '--testBatch', type=int, default=1, help='number of testing data')   
    
    parser.add_argument(
    '--thread', type=int, default=90, help='number of threads')
    parser.add_argument(
        '--output', action="store_true", help='if output prediction')
    
    
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train with"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-1, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=90, help="Batch size to train with"
    )
    
    parser.add_argument(
        "--inner-lr",
        type=float,
        default=10000000,
        help="Learning rate of DSPN inner optimisation",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="How many DSPN inner optimisation iteration to take",
    )
    
    
    parser.add_argument(
        "--no-cuda",
        default=True,
        action="store_true",
        help="Run on CPU instead of GPU (not recommended)",
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Only run training, no evaluation"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Only run evaluation, no training"
    )
    parser.add_argument("--multi-gpu", default=False, action="store_true", help="Use multiple GPUs")
    
    parser.add_argument(
        '--pre_train', default=True ,action="store_true", help='if store a pre_train model')
       
       
    
    args = parser.parse_args()
    
    #torch.set_num_threads(1)
    
    dataname=args.dataname
    #vNum = args.vNum
    
    trainNum =args.trainNum
    valNum =args.valNum
    testNum =args.testNum
    
    sourceT = args.sourceT
    targetT = args.targetT
    
    torch.set_num_threads(args.thread)
    
    
    
   #maxFeatureNum = 10000
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
            
    TrainSamples, TrainQueries, TrainDecisions, maxTrainSize = source_USCO.readSamples(sourceSample_path, trainNum, sourceSample_maxPair, RmaxSize = True)
    ValSamples, ValQueries, ValDecisions, maxValSize = source_USCO.readSamples(sourceSample_path, valNum, sourceSample_maxPair, RmaxSize = True)
    TestSamples, TestQueries, TestDecisions, maxTestSize = target_USCO.readSamples(targetSample_path, testNum, targetSample_maxPair, RmaxSize = True)
    
    max_size = max(maxTrainSize, maxValSize, maxTestSize)
        
    
    args.logpath=path+"/log_dspn"
    
    
    pre_train = args.pre_train
    preTrainPathResult = None
        
        
    print('Building dataset...')
    dataset_train = DSPN_utils.GetDataset(TrainSamples, sourceT, trainNum, args.batch_size, vNum, max_size, train=True)
    dataset_test = DSPN_utils.GetDataset(TestSamples, targetT, trainNum, args.batch_size, vNum, max_size, train=True)
    dataset_val = DSPN_utils.GetDataset(ValSamples, sourceT, trainNum, args.batch_size, vNum, max_size, train=False)
    print("max_set_size {}".format(max_size))
    
    
    
    args.set_size=max_size
    args.vNum=vNum
    net = DSPN_utils.build_net(args)
    
    if not args.no_cuda:
        net = net.cuda()
    
    if args.multi_gpu:
        net = torch.nn.DataParallel(net)
    
    
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    decayRate = 0.9
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    #optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=args.lr)
    
    min_loss=sys.maxsize
    for epoch in range(args.epochs):
        if not args.eval_only:
            print("-----------------------------training....")
            loss=run(net, dataset_train, optimizer, args, train=True, epoch=epoch, pool=None, valDataset=dataset_val)
            if loss < min_loss:
                min_loss=loss
                stop = False
            else:
                stop = True
        if epoch==args.epochs-1 or stop:
            print("-----------------------------testing....")
            Y_pred = run(net, dataset_test, optimizer, args, train=False, epoch=epoch, pool=None, target_USCO=target_USCO)
            
            if pre_train is True:
                now = datetime.now()
                preTrainPath=path+"/pre_train/"+now.strftime("%d-%m-%Y-%H-%M-%S")+"/"
                if not os.path.exists(preTrainPath):
                    os.makedirs(preTrainPath)
                    Utils.writeToFile(args.logpath, preTrainPath, toconsole = True)   
                    preTrainPathResult = preTrainPath+"/result"
        
        
            Utils.writeToFile(args.logpath, "Testing DSPN...", toconsole = True, preTrainPathResult = preTrainPathResult)  
            Utils.writeToFile(args.logpath, sourceT+" "+targetT, toconsole = True, preTrainPathResult = preTrainPathResult) 
            Utils.writeToFile(args.logpath, dataname, toconsole = True, preTrainPathResult = preTrainPathResult)    
            Utils.writeToFile(args.logpath, "trainNum: {}, {}_{} ".format(trainNum, source_scale_x, source_scale_y), toconsole = True, preTrainPathResult = preTrainPathResult)
            Utils.writeToFile(args.logpath, "testNum:{}, {}_{} ".format(testNum, target_scale_x, target_scale_y), toconsole = True, preTrainPathResult = preTrainPathResult)     
            Utils.writeToFile(args.logpath, "lr: {}, inner-lr: {}, iters: {}, batch-size: {}".format(args.lr, args.inner_lr, args.iters, args.batch_size), toconsole = True, preTrainPathResult = preTrainPathResult)    
            target_USCO.test(TestSamples, TestQueries, TestDecisions, Y_pred, args.thread, logpath = args.logpath, preTrainPathResult = preTrainPathResult)
            break
    
        if args.eval_only:
                break
    
    
    