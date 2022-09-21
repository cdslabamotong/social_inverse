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
sys.path.insert(0,'..')
from Utils import Dijkstra, Utils

class StructuredModel(object):
    """Interface definition for Structured Learners.

    This class defines what is necessary to use the structured svm.
    You have to implement at least joint_feature and inference.
    """
    def __repr__(self):
        return ("%s, size_joint_feature: %d"
                % (type(self).__name__, self.size_joint_feature))

    def __init__(self):
        """Initialize the model.
        Needs to set self.size_joint_feature, the dimensionality of the joint
        features for an instance with labeling (x, y).
        """
        self.size_joint_feature = None

    def _check_size_w(self, w):
        if w.shape != (self.size_joint_feature,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_joint_feature, w.shape))

    def initialize(self, X, Y, instance):
        # set any data-specific parameters in the model
        pass

    def joint_feature(self, x, y):
        raise NotImplementedError()

    def batch_joint_feature(self, X, Y, Y_true=None):
        joint_feature_ = np.zeros(self.size_joint_feature)
        if getattr(self, 'rescale_C', False):
            for x, y, y_true in zip(X, Y, Y_true):
                joint_feature_ += self.joint_feature(x, y, y_true)
        else:
            for x, y in zip(X, Y):
                joint_feature_ += self.joint_feature(x, y)
        return joint_feature_

    def _loss_augmented_djoint_feature(self, x, y, y_hat, w):
        # debugging only!
        x_loss_augmented = self.loss_augment(x, y, w)
        return (self.joint_feature(x_loss_augmented, y)
                - self.joint_feature(x_loss_augmented, y_hat))

    def inference(self, x, w, relaxed=None, constraints=None):
        raise NotImplementedError()

    def batch_inference(self, X, w, relaxed=None, constraints=None):
        # default implementation of batch inference
        if constraints:
            return [self.inference(x, w, relaxed=relaxed, constraints=c)
                    for x, c in zip(X, constraints)]
        return [self.inference(x, w, relaxed=relaxed)
                for x in X]

    def loss(self, y, y_hat):
        # hamming loss:
        if isinstance(y_hat, tuple):
            return self.continuous_loss(y, y_hat[0])
        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y] * (y != y_hat))
        return np.sum(y != y_hat)

    def batch_loss(self, Y, Y_hat):
        # default implementation of batch loss
        return [self.loss(y, y_hat) for y, y_hat in zip(Y, Y_hat)]

    def max_loss(self, y):
        # maximum possible los on y for macro averages
        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y])
        return y.size

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y is the result of linear programming
        if y.ndim == 2:
            raise ValueError("FIXME!")
        gx = np.indices(y.shape)

        # all entries minus correct ones
        result = 1 - y_hat[gx, y]
        if hasattr(self, 'class_weight'):
            return np.sum(self.class_weight[y] * result)
        return np.sum(result)

    def loss_augmented_inference(self, x, y, w, relaxed=None):
        print("FALLBACK no loss augmented inference found")
        return self.inference(x, w)

    def batch_loss_augmented_inference(self, X, Y, w, relaxed=None, n_jobs = 1):
        # default implementation of batch loss augmented inference
        raise NotImplementedError()
        #return [self.loss_augmented_inference(x, y, w, relaxed=relaxed)
        #        for x, y in zip(X, Y)]

    def _set_class_weight(self):
        if not hasattr(self, 'size_joint_feature'):
            # we are not initialized yet
            return

        if hasattr(self, 'n_labels'):
            n_things = self.n_labels
        else:
            n_things = self.n_states

        if self.class_weight is not None:

            if len(self.class_weight) != n_things:
                raise ValueError("class_weight must have length n_states or"
                                 " be None")
            self.class_weight = np.array(self.class_weight)
            self.uniform_class_weight = False
        else:
            self.class_weight = np.ones(n_things)
            self.uniform_class_weight = True


class USCO_Solver(StructuredModel):
    """Interface definition for Structured Learners.

    This class defines what is necessary to use the structured svm.
    You have to implement at least joint_feature and inference.
    """
    def __repr__(self):
        return ("%s, size_joint_feature: %d"
                % (type(self).__name__, self.size_joint_feature))

    def __init__(self):
        """Initialize the model.
        Needs to set self.size_joint_feature, the dimensionality of the joint
        features for an instance with labeling (x, y).
        """
        self.size_joint_feature = None

    def _check_size_w(self, w):
        if w.shape != (self.size_joint_feature,):
            raise ValueError("Got w of wrong shape. Expected %s, got %s" %
                             (self.size_joint_feature, w.shape))

    def initialize(self, realizations, USCO):
        # set any data-specific parameters in the model
        #self.featureNum = instance.featureNum
        self.size_joint_feature= len(realizations)
        self.realizations = realizations
        self.inference_calls = 0
        self.USCO=USCO
        #pass
    """
    def joint_feature(self, x, y):
        raise NotImplementedError()
    """ 
    #def computeFeature(self, x ,y):
    #    return self.USCO.computeFeature(self.realizations, x, y)
        
    def joint_feature(self, x, y, n_jobs = 1):
        if n_jobs == 1:
            return np.array(self.USCO.computeFeature(self.realizations, x, y),dtype=float)
        else:
            return self.USCO.computeFeature(self.realizations,x, y)
    
    def joint_feature_block(self, X, Y):
        #joint_feature_ = np.zeros(self.size_joint_feature)
        joint_feature_ = []
        for x, y in zip(X, Y):
                joint_feature_.append(self.joint_feature(x, y))
        return joint_feature_
    
    
    def batch_joint_feature(self, X, Y, Y_true=None, n_jobs=1):
        print("batch_joint_feature running {}".format(n_jobs))
        #print(Y)
        joint_feature_ = np.zeros(self.size_joint_feature)
        if getattr(self, 'rescale_C', False):
            for x, y, y_true in zip(X, Y, Y_true):
                joint_feature_ += self.joint_feature(x, y, y_true)
        else:
            #print("here " + str(n_jobs))
            if n_jobs == 1:
                for x, y in zip(X, Y):
                    #print(self.joint_feature(x, y))
                    joint_feature_ += self.joint_feature(x, y)
            else:       
                p = multiprocessing.Pool(n_jobs)
                block_size =int (len(X)/n_jobs)
                Ys=p.starmap(self.joint_feature_block, ((X[i*block_size:min([len(X),(i+1)*block_size])], Y[i*block_size:min([len(X),(i+1)*block_size])]) for i in range(n_jobs) ))
                p.close()
                p.join()
                for y_temp in Ys:
                    for feature in y_temp:
                        joint_feature_ += np.array(feature)
                    
        print("batch_joint_feature done")
        return joint_feature_
    
    def inference(self, x, w, relaxed=None, constraints=None):
       self.inference_calls += 1
       #print(w)
       #print("--")
       decision = self.USCO.solve_R(self.realizations, w, x)
       return decision
   

    
    def batch_inference(self, X, w, n_jobs, relaxed=None, constraints=None, ):
        #print("batch_inference running...")
        if n_jobs == 1:
            decisions = []
            for x in X:
                #print(x)
                decisions.append(self.inference(x, w))
        else:         
            p = multiprocessing.Pool(n_jobs)
            block_size =int (len(X)/n_jobs)
            #print(n_jobs)
            Ys=p.starmap(self.batch_inference, ((X[i*block_size:min([len(X),(i+1)*block_size])], w, 1) for i in range(n_jobs) ))
            p.close()
            p.join()
            decisions = []
            for decision_block in Ys:
                decisions.extend(decision_block)
        
        #print("batch_inference done")
        return decisions
        

   
  
    def loss_augmented_inference(self, x, y, w, relaxed=None):
        
        #print("FALLBACK no loss augmented inference found")
        #return self.inference(x, w)
        #print("loss_augmented_inference RUNNING")
        self.inference_calls += 1
        y_pre = self.inference(x, w)
        #print("loss_augmented_inference DONE")
        return y_pre
    
    def batch_loss_augmented_inference(self, X, Y, w, relaxed=None, n_jobs =1):
        #print("1111111111111")
        return self.batch_inference(X, w, n_jobs=n_jobs)
    
    #def predict_batch(self, X, realizations, w, n_jobs = 1):
    #    return self.USCO.solve_R_batch(X, w, realizations, n_jobs)
    
class USCO(object):
    
    '''
    class Pair(object):
        def __init__(self, x, y):
            #self.index = index
            self.x = x
            self.y = y
            #self.out_degree = 0    
    '''
    '''
    stoGraph: a stograph
    obeValue
    def test(self, X_test, Y_length, Y_pred, logpath= None)
    def genPairs(path, stoGraph, num):
    '''
    class Realization(object):
        def __init__(self):
            raise NotImplementedError()
            
    class Sample(object):
        def __init__(self):
            
            raise NotImplementedError()
    
    def __init__(self, stoGraph):
        self.stoGraph = stoGraph 
        #self.realizations, self.realizationIndexes = self.readRealizations(realizationPath, realizationNum)
        #self.realizations = None
        #self.realizationIndexes = None
        #self.pairs = None
        #self.samples = self.readSamples(samplePath, sampleNum)
        #self.samples = None
        #self.vNum=vNum
        
    #def readRealizations(self, realizationPath, realizationNum)
    
    def kernel(self, realization, query, y):
        raise NotImplementedError()
        
        
        
    def objValue(self, realizations, W, x, y):
        value = 0
        for realization, w, in zip(realizations, W):
            r_value= self.kernel(realization,x,y)
            value = value + w*r_value
        
        return value 
    
    def computeScore(self, x, y, w):
        feature = self.computeFeature(x, y)
        return w.dot(feature)
        
    def computeFeature(self, realizations, x, y):
        feature = []
        #print(self.featureNum)
        for realization in realizations:
                feature.append(self.kernel(realization, x, y))
        #print(feature)
        return feature  
    
   
        
    def solve_R(self, realizations, W, x):
        '''
        Define how to compute y = max w^T[f(x,y,r_1),...f(x,y,r_n)]
        '''
        raise NotImplementedError()
    
    def solve_R_batch(self, X, W, realizations, n_jobs=1, offset = None):
        #print("inference") 
        print("solve_R_batch RUNNING")     
        
        if n_jobs == 1:
            result =[]
            for x in X:
                result.append(self.solve_R(realizations, W, x))
            print("solve_R_batch DONE")
            return result
        else:
            #print("111")
            results={}
            p = multiprocessing.Pool(n_jobs)
            #print("222")
            results=p.starmap(self.solve_R, ((realizations, W, x) for x in X))
            p.close()
            p.join()
            print("solve_R_batch DONE")
            return results  
        
    def solveTrue(self, x, budget):
        '''
        Define how to compute the true solution
        '''
        raise NotImplementedError()
        
    def solveBatch(self, realizations, W, X, thread = 1):
        '''
        Define how to compute y = max w^T[f(x,y,r_1),...f(x,y,r_n)]
        '''
        raise NotImplementedError()
    
    def genQuery(self):
        raise NotImplementedError()
        
            
       
    def genSamples(self, Wpath, trainNum, testNum, Max = None):
        raise NotImplementedError()
    
    def readSamples(self, Rpath, trainNum, testNum, Max = None):
        raise NotImplementedError()
        
    
    def readRealizations(self, Rpath, fetureNum, Max = None):
        raise NotImplementedError()
    
    def test(self, samples, PredDecisions):
        raise NotImplementedError()
    
    



if __name__ == "__main__":
    pass

    #stoGraph=StoGraph("data/pl/pl_model", 768)
    #stoGraph.GenStoGraph("data/pl/pl_model")
    #stoGraph.genMultiRealization_P(10000, "data/pl/features/true_10000/", edgeProb= 0.1, weightType="true", thread=5, startIndex= 0, distance = True)
