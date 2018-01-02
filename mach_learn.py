'''
Created on Nov 22, 2011

@author: mllamosa
'''


from __future__ import print_function

from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from scipy.stats import linregress
from random import *
from sklearn.linear_model import Ridge

try:
	from pyfann import libfann
except:
	print ("Loading pyfann failed")

import numpy as np
from utilities import *
try:
    from svm import *
    from svmutil import *
except:
    "Loading libsvm failed"

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

try:
    from svmPython import *
except:
    print("Loading svmPython failed")

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
try:
	from sklearn import linear_model
except:
	print ("Loading linear_model module neighbors failed")
try:
        from sklearn import neighbors
except:
         print ("Loading sklearn module neighbors failed")
#from sklearn.cluster import KMeans

try:
	#from sklearn import pls
	from sklearn import cross_decomposition as pls
except:
	print ("Loading sklearn module pls failed")

class Sksvm():

    def __init__(self,parameters=({'sigma':1,'regularization':1}),details=({'mode':'nonmodular','kernel':'gaussian'})):
        #To include different kernels
        #Train and svm an return the model: b, alphas,width,
        self.model=np.array([])
        self.train=np.array([])
        self.parametersoptions=({'sigma':np.array([0.001,0.01,0.1,1,10,100,500,1000]),'regularization':np.array([0.001,0.01,0.1,1,10,100,500,1000])})
        self.detailsoptions=({'mode':None,'kernel':['gaussian','linear']})
        self.details=details
        self.parameters=parameters
        self.model = SVR(C=self.parameters['regularization'], gamma=1/(2*self.parameters['sigma']**2))

    def __call__(self,names=None):
        pass

    def copy(self):
        return Sksvm(parameters=self.parameters,details=self.details)

    def set_data(self,data):
        self.train = data

    def fit(self,data):
        #Train and svm an return the model: b, alphas,width,
        model = SVR(C=self.parameters['regularization'], gamma=1/(2*self.parameters['sigma']**2))
        model.fit(data[:,0:-1], data[:,-1])
        self.model=model
        self.train=data

    def set_parameters(self,parameters):
        self.parameters=parameters

    def set_parameters_subset(self,subset):
        '''Subset is a dict of parameter name:index'''
        for key in self.subset.keys():
            self.parameters[key]=(self.parametersoptions[key][subset[key]])

    def get_parameters_options(self):
        return self.parametersoptions

    def predict(self,data):
        return self.model.predict(data[:,0:-1])

    def crossvalidate(self,foldOut,list=None):
        if list:
            numberinstances=len(list)
            label=np.zeros(len(list))
        else:
            numberinstances=self.train.shape[0]
            label=np.zeros(numberinstances)
        instanceidx=np.array(range(numberinstances))
        shuffle(instanceidx)
        skf = StratifiedKFold(label, n_folds=foldOut)
        labelspredicted=np.array([]);
        for trainindex, testindex in skf:
            if list:
                idx_test = [item for idx in testindex for item in list[idx]]
                idx_train = [item for idx in trainindex for item in list[idx]]

            else:
                idx_test=instanceidx[testindex]
                idx_train=instanceidx[trainindex]

            datatrain = self.train[idx_train,:]
            datatest = self.train[idx_test,:]
            if not labelspredicted.any():
                idxs=idx_test
                svm=self.copy()
                svm.fit(datatrain)
                labelspredicted=svm.predict(datatest)

            else:
                svm=self.copy()
                svm.fit(datatrain)
                labelspredicted=np.concatenate((labelspredicted,np.array(svm.predict(datatest))))
                idxs=np.concatenate((idxs,np.array(idx_test)))
            tosort=np.vstack((idxs,labelspredicted))
        return  tosort[1,tosort[0,:].argsort()]

class RandF():

    def __init__(self,parameters=({'n_estimators':10,'min_samples_split':10}),details=({})):
        self.train=np.array([])
        self.parametersoptions=({'n_estimators':np.array(range(5,100,5)),'min_samples_split':np.array(range(2,50,5))})
        self.detailsoptions=({})
        self.details=details
        self.parameters=parameters

    def __call__(self,names=None):
        pass

    def copy(self):
        return RandF(parameters=self.parameters,details=self.details)

    def set_data(self,data):
        self.train = data

    def fit(self,data):
        #Train  model
        model = RandomForestRegressor(n_estimators=self.parameters['n_estimators'],min_samples_split=self.parameters['min_samples_split'],n_jobs=-1)
        model.fit(data[:,0:-1], data[:,-1])
        self.model = model
        self.train = data

    def set_parameters(self,parameters):
        self.parameters=parameters

    def set_parameters_subset(self,subset):
        '''Subset is a dict of parameter name:index'''
        for key in self.subset.keys():
            self.parameters[key]=(self.parametersoptions[key][subset[key]])

    def get_parameters_options(self):
        return self.parametersoptions

    def predict(self,data):
        return self.model.predict(data[:,0:-1])

    def crossvalidate(self,foldOut,list=None):
        if list:
            numberinstances=len(list)
            label=np.zeros(len(list))
        else:
            numberinstances=self.train.shape[0]
            label=np.zeros(numberinstances)
        instanceidx=np.array(range(numberinstances))
        shuffle(instanceidx)
        skf = StratifiedKFold(label, n_folds=foldOut)
        labelspredicted=np.array([]);
        for trainindex, testindex in skf:
            if list:
                idx_test = [item for idx in testindex for item in list[idx]]
                idx_train = [item for idx in trainindex for item in list[idx]]

            else:
                idx_test=instanceidx[testindex]
                idx_train=instanceidx[trainindex]

            datatrain = self.train[idx_train,:]
            datatest = self.train[idx_test,:]
            if not labelspredicted.any():
                idxs=idx_test
                model=self.copy()
                model.fit(datatrain)
                labelspredicted=model.predict(datatest)

            else:
                model=self.copy()
                model.fit(datatrain)
                labelspredicted=np.concatenate((labelspredicted,np.array(model.predict(datatest))))
                idxs=np.concatenate((idxs,np.array(idx_test)))
            tosort=np.vstack((idxs,labelspredicted))
        return  tosort[1,tosort[0,:].argsort()]


class kRidge():

    def __init__(self,parameters=({'alpha':0.01}),details=({})):
        self.train=np.array([])
        self.parametersoptions=({'alpha':np.array(np.arange(0.01,0.8,0.05))})
        self.detailsoptions=({})
        self.details=details
        self.parameters=parameters

    def __call__(self,names=None):
        pass

    def copy(self):
        return rRidge(parameters=self.parameters,details=self.details)

    def set_data(self,data):
        self.train = data

    def fit(self,data):
        #Train and svm an return the model: b, alphas,width,
        model = Ridge(alpha=self.parameters['alpha'])
        model.fit(data[:,0:-1], data[:,-1])
        self.model = model
        self.train = data

    def set_parameters(self,parameters):
        self.parameters=parameters

    def set_parameters_subset(self,subset):
        '''Subset is a dict of parameter name:index'''
        for key in self.subset.keys():
            self.parameters[key]=(self.parametersoptions[key][subset[key]])

    def get_parameters_options(self):
        return self.parametersoptions

    def predict(self,data):
        return self.model.predict(data[:,0:-1])

    def predict(self,data):
        return self.model.predict(data[:,0:-1])

    def crossvalidate(self,foldOut,list=None):
        if list:
            numberinstances=len(list)
            label=np.zeros(len(list))
        else:
            numberinstances=self.train.shape[0]
            label=np.zeros(numberinstances)
        instanceidx=np.array(range(numberinstances))
        shuffle(instanceidx)
        skf = StratifiedKFold(label, n_folds=foldOut)
        labelspredicted=np.array([]);
        for trainindex, testindex in skf:
            if list:
                idx_test = [item for idx in testindex for item in list[idx]]
                idx_train = [item for idx in trainindex for item in list[idx]]

            else:
                idx_test=instanceidx[testindex]
                idx_train=instanceidx[trainindex]

            datatrain = self.train[idx_train,:]
            datatest = self.train[idx_test,:]
            if not labelspredicted.any():
                idxs=idx_test
                model=self.copy()
                model.fit(datatrain)
                labelspredicted=model.predict(datatest)

            else:
                model=self.copy()
                model.fit(datatrain)
                labelspredicted=np.concatenate((labelspredicted,np.array(model.predict(datatest))))
                idxs=np.concatenate((idxs,np.array(idx_test)))
            tosort=np.vstack((idxs,labelspredicted))
        return  tosort[1,tosort[0,:].argsort()]



class rRidge():

    def __init__(self,parameters=({'alpha':0.01}),details=({})):
        self.train=np.array([])
        self.parametersoptions=({'alpha':np.array(np.arange(0.01,0.8,0.05))})
        self.detailsoptions=({})
        self.details=details
        self.parameters=parameters

    def __call__(self,names=None):
        pass

    def copy(self):
        return rRidge(parameters=self.parameters,details=self.details)

    def set_data(self,data):
        self.train = data

    def fit(self,data):
        #Train model
        model = Ridge(alpha=self.parameters['alpha'])
        model.fit(data[:,0:-1], data[:,-1])
        self.model = model
        self.train = data

    def set_parameters(self,parameters):
        self.parameters=parameters

    def set_parameters_subset(self,subset):
        '''Subset is a dict of parameter name:index'''
        for key in self.subset.keys():
            self.parameters[key]=(self.parametersoptions[key][subset[key]])

    def get_parameters_options(self):
        return self.parametersoptions

    def predict(self,data):
        return self.model.predict(data[:,0:-1])

    def predict(self,data):
        return self.model.predict(data[:,0:-1])

    def crossvalidate(self,foldOut,list=None):
        if list:
            numberinstances=len(list)
            label=np.zeros(len(list))
        else:
            numberinstances=self.train.shape[0]
            label=np.zeros(numberinstances)
        instanceidx=np.array(range(numberinstances))
        #randbinary = lambda n: [1 for b in range(1,n+1)]
        #randomclasslabel=randbinary(numberinstances)
        #randomclasslabel=randbinary(numberinstances)
        shuffle(instanceidx)
        skf = StratifiedKFold(label, n_folds=foldOut)
        labelspredicted=np.array([]);
        for trainindex, testindex in skf:
            if list:
                idx_test = [item for idx in testindex for item in list[idx]]
                idx_train = [item for idx in trainindex for item in list[idx]]

            else:
                idx_test=instanceidx[testindex]
                idx_train=instanceidx[trainindex]

            datatrain = self.train[idx_train,:]
            datatest = self.train[idx_test,:]
            if not labelspredicted.any():
                idxs=idx_test
                model=self.copy()
                model.fit(datatrain)
                labelspredicted=model.predict(datatest)

            else:
                model=self.copy()
                model.fit(datatrain)
                labelspredicted=np.concatenate((labelspredicted,np.array(model.predict(datatest))))
                idxs=np.concatenate((idxs,np.array(idx_test)))
            tosort=np.vstack((idxs,labelspredicted))
        return  tosort[1,tosort[0,:].argsort()]


class Svm():

    def __init__(self,parameters=({'sigma':1,'regularization':1}),details=({'mode':'nonmodular','kernel':'gaussian'})):
        #To include different kernels
        #Train and svm an return the model: b, alphas,width,
        self.model=np.array([])
        self.train=np.array([])
        self.parametersoptions=({'sigma':np.array([0.001,0.01,0.1,1,10,100,500,1000]),'regularization':np.array([0.001,0.01,0.1,1,10,100,500,1000])})
        self.detailsoptions=({'mode':['modular','nonmodular'],'kernel':['gaussian','linear']})
        self.details=details
        self.parameters=parameters

    def __call__(self,names=None):
        pass

    def copy(self):
    	return Svm(parameters=self.parameters,details=self.details)

    def set_data(self,data):
        self.train = data

    def fit(self,data):
        width,C=self.parameters['sigma'],self.parameters['regularization']
        mode=self.details['mode']
        #Train and svm an return the model: b, alphas,width,
        model = train_libsvm(data[:,0:-1],data[:,-1],width,C,mode)
        self.model=model
        self.train=data

    def set_parameters(self,parameters):
        self.parameters=parameters


    def set_parameters_subset(self,subset):
        '''Subset is a dict of parameter name:index'''
        for key in self.subset.keys():
            self.parameters[key]=(self.parametersoptions[key][subset[key]])

    def get_parameters_options(self):
        return self.parametersoptions

    def predict(self,data):
        return predict_libsvm(self.model,self.train,data)

    def crossvalidate(self,foldOut,list=None):
        if list:
            numberinstances=len(list)
            label=np.zeros(len(list))
        else:
            numberinstances=self.train.shape[0]
            label=np.zeros(numberinstances)
        instanceidx=np.array(range(numberinstances))
        #randbinary = lambda n: [1 for b in range(1,n+1)]
        #randomclasslabel=randbinary(numberinstances)
        shuffle(instanceidx)
        skf = StratifiedKFold(label, n_folds=foldOut)
        labelspredicted=np.array([]);
        for trainindex, testindex in skf:
            if list:
                idx_test = [item for idx in testindex for item in list[idx]]
                idx_train = [item for idx in trainindex for item in list[idx]]

            else:
                idx_test=instanceidx[testindex]
                idx_train=instanceidx[trainindex]

            datatrain = self.train[idx_train,:]
            datatest = self.train[idx_test,:]
            if not labelspredicted.any():
                idxs=idx_test
                svm=self.copy()
                svm.fit(datatrain)
                labelspredicted=svm.predict(datatest)

            else:
                svm=self.copy()
                svm.fit(datatrain)
                labelspredicted=np.concatenate((labelspredicted,np.array(svm.predict(datatest))))
                idxs=np.concatenate((idxs,np.array(idx_test)))
                tosort=np.vstack((idxs,labelspredicted))
        return  tosort[1,tosort[0,:].argsort()]

class Libsvm():

    def __init__(self,parameters=({'sigma':1,'regularization':1}),details=({'mode':'modular','kernel':'gaussian'})):
        #To include different kernels
        #Train and svm an return the model: b, alphas,width,
        self.model=np.array([])
        self.train=np.array([])
        self.parametersoptions=({'sigma':np.array([0.001,0.01,0.1,1,10,100,500,1000]),'regularization':np.array([0.001,0.01,0.1,1,10,100,500,1000])})
        self.detailsoptions=({'kernel':['gaussian','linear']})
        self.details=details
        self.parameters=parameters

    def __call__(self,names=None):
        pass

    def copy(self):
        return Libsvm(parameters=self.parameters,details=self.details)

    def set_data(self,data):
        self.train = data

    def fit(self,data):
        label,matrix = pre_libsvmdata(data)
        prob  = svm_problem(label,matrix)
        param = svm_parameter(pre_libsvmpara(self.parameters))
        #Train and svm an return the model: b, alphas,width,
        model = svm_train(prob, param)
        self.model=model
        self.train=data

    def predict(self,data):
        if self.model:
            label,matrix = pre_libsvmdata(data)
            p_label, p_acc, p_val = svm_predict(label,matrix, self.model)
            #label=[lab[0] for lab in p_label]
            return np.array(p_label)

        else:
            return None

    def set_parameters(self,parameters):
        self.parameters=parameters

    def set_parameters_subset(self,subset):
        '''Subset is a dict of parameter name:index'''
        for key in self.subset.keys():
            self.parameters[key]=(self.parametersoptions[key][subset[key]])

    def get_parameters_options(self):
        return self.parametersoptions

    def crossvalidate(self,foldOut,list=None):
        if list:
            numberinstances=len(list)
            label=np.zeros(len(list))
        else:
            numberinstances=self.train.shape[0]
            label=np.zeros(numberinstances)
        instanceidx=np.array(range(numberinstances))
        #randbinary = lambda n: [1 for b in range(1,n+1)]
        #randomclasslabel=randbinary(numberinstances)
        shuffle(instanceidx)
        skf = StratifiedKFold(label, n_folds=foldOut)
        labelspredicted=np.array();
        for trainindex, testindex in skf:

            if list:
                idx_test = [item for idx in testindex for item in list[idx]]
                idx_train = [item for idx in trainindex for item in list[idx]]

            else:
                idx_test=instanceidx[testindex]
                idx_train=instanceidx[trainindex]
            datatrain = self.train[idx_train,:]
            datatest = self.train[idx_test,:]
            if not labelspredicted.any():
                idxs=idx_test
                svm=self.copy()
                svm.fit(datatrain)
                labelspredicted=svm.predict(datatest)

            else:
                svm=self.copy()
                svm.fit(datatrain)

                labelspredicted=np.concatenate((labelspredicted,np.array(svm.predict(datatest))))
                idxs=np.concatenate((idxs,np.array(idx_test)))

        tosort=np.vstack((idxs,labelspredicted))
        return  tosort[1,tosort[0,:].argsort()]

class Mlr():
    def __init__(self,parameters=np.array([])):
        self.parameters=parameters
        self.model=np.array([])
        self.train=np.array([])
        self.name='mlr'
        self.parametersoptions = dict()

    def __call__(self,names=np.array([])):
        if self.model.any():
            print ("===>Model is below")
		#try:
            if len(names)>1:

                print (names[-1] + ' = ' + '%5.8f'%(self.model[0]) + '+' + '+'.join('%5.8f'%(self.model[idx]) + '*' + names[idx-1] for idx in range(1,len(self.model))))
            else:
                print ('Y = ' + '+'.join('%5.8f'%(self.model[idx])+ '*X' + str(idx) for idx in range(len(self.model))))
	    #except:
	#	        print 'Model matrix'
	#	        print self.model
        else:
	        print ('No model trained yet')

    def copy(self):
	       return Mlr()

    def get_parameters_options(self):
        return self.parametersoptions

    def set_parameters(self,parameters):
        pass

    def set_data(self,data):
        self.train = data

    def fit(self,data):
        #Train an mlr an return the model: coeficients,
	#numbexample,numvariable = data.shape
	#if numbexample<numvariable-1:
	#    print 'Singular data matrix, building the model with a reduce subset of', str(numbexample-1),' variables'

        #model = ols.ols(data[:,-1],data[:,0:-1])
        datawconstant=np.vstack((np.ones(data.shape[0]), data.T)).T
        model,residues, rank, s = np.linalg.lstsq(datawconstant[:,0:-1],datawconstant[:,-1])
        self.model=model
        self.train=data

    def get_name(self):
        return self.name


    def predict(self,data):
        coeficient=self.model
        prediction=[]
        datawconstant=np.vstack((np.zeros(data.shape[0]) + 1, data[:,0:-1].T)).T
        for vector in datawconstant:
            prediction.append(sum(vector*coeficient))
        return np.array(prediction)

    def crossvalidate(self,foldOut,list=None):

        if list:
            numberinstances=len(list)
            label=np.zeros(len(list))
        else:
            numberinstances=self.train.shape[0]
            label=np.zeros(numberinstances)
        instanceidx=np.array(range(numberinstances))
        #randbinary = lambda n: [1 for b in range(1,n+1)]
        #randomclasslabel=randbinary(numberinstances)
        shuffle(instanceidx)
        try:
            skf = StratifiedKFold(label, n_folds=foldOut)
        except:
            skf = StratifiedKFold(label, k=foldOut)
        labelspredicted=np.array([]);
        for trainindex, testindex in skf:

            if list:
                idx_test = [item for idx in testindex for item in list[idx]]
                idx_train = [item for idx in trainindex for item in list[idx]]

            else:
                idx_test=instanceidx[testindex]
                idx_train=instanceidx[trainindex]

            datatrain = self.train[idx_train,:]
            datatest = self.train[idx_test,:]

            if not labelspredicted.any():
                idxs=idx_test
                linear=self.copy()
                linear.fit(datatrain)
                labelspredicted=linear.predict(datatest)

            else:
                linear=self.copy()
                linear.fit(datatrain)
                labelspredicted=np.concatenate((labelspredicted,np.array(linear.predict(datatest))))
                idxs=np.concatenate((idxs,np.array(idx_test)))

        tosort=np.vstack((idxs,labelspredicted))
        return  tosort[1,tosort[0,:].argsort()]

class rTree():
	#>>>>>>> other

    def __init__(self,parameters=({'max_deep':3}),train=np.array([]),parametersoptions=({'max_deep':np.array(range(3,20,3))})):
        self.parameters=parameters
        self.model=np.array([])
        self.train=train
        self.parametersoptions=parametersoptions
        self.name='rTree'

    def __call__(self,names=np.array([])):
            pass

    def copy(self):
        return rTree(parameters=self.parameters,train=self.train,parametersoptions=self.parametersoptions)

    def set_parameters(self,parameters):
        self.parameters=parameters

    def get_parameters_options(self):
        return self.parametersoptions

    def set_data(self,data):
        self.train = data

    def fit(self,data):
        '''Train a tree an return the model'''
	#print "Parameter to fit",self.parameters['max_deep']
	#print "Size data",data.shape
        if self.parameters['max_deep'] > data.shape[1]:
	        self.parameters['max_deep'] = data.shape[1];

        clf = DecisionTreeRegressor(max_depth=self.parameters['max_deep'], max_features=data.shape[1]-1)
        clf = clf.fit(data[:,0:-1],data[:,-1])
        self.model = clf
        self.train = data

    def predict(self,data):
        return self.model.predict(data[:,0:-1])

    def crossvalidate(self,foldOut,list=None):

        if self.parameters['max_deep'] > self.train.shape[1]:
            self.parameters['max_deep'] = self.train.shape[1];
	#	self.parametersoptions=({'max_deep':np.array(range(3,self.train.shape[1]-1))})
        if list:
            numberinstances=len(list)
            label=np.zeros(len(list))
        else:
            numberinstances=self.train.shape[0]
            label=np.zeros(numberinstances)
        instanceidx=np.array(range(numberinstances))
        #randbinary = lambda n: [1 for b in range(1,n+1)]
        #randomclasslabel=randbinary(numberinstances)
        shuffle(instanceidx)
        try:
            skf = StratifiedKFold(label, n_folds=foldOut)
        except:
            skf = StratifiedKFold(label, k=foldOut)
        labelspredicted=np.array([]);
        for trainindex, testindex in skf:

            if list:
                idx_test = [item for idx in testindex for item in list[idx]]
                idx_train = [item for idx in trainindex for item in list[idx]]

            else:
                idx_test=instanceidx[testindex]
                idx_train=instanceidx[trainindex]

            datatrain = self.train[idx_train,:]
            datatest = self.train[idx_test,:]

            if not labelspredicted.any():
                idxs=idx_test
                dtree=rTree(self.parameters)
                dtree.fit(datatrain)
                labelspredicted=dtree.predict(datatest)

            else:
                dtree=self.copy()
                dtree.fit(datatrain)
                labelspredicted=np.concatenate((labelspredicted,np.array(dtree.predict(datatest))))
                idxs=np.concatenate((idxs,np.array(idx_test)))

        tosort=np.vstack((idxs,labelspredicted))
        return  tosort[1,tosort[0,:].argsort()]

class Tree():

    def __init__(self,parameters=({'max_deep':3}),train=np.array([]),parametersoptions=({'max_deep':np.array(range(3,20,3))})):
        self.parameters=parameters
        self.model=np.array([])
        self.train=train
        self.parametersoptions=parametersoptions
        self.name='tree'

    def __call__(self,names=np.array([])):
            pass

    def copy(self):
        return Tree(parameters=self.parameters,train=self.train,parametersoptions=self.parametersoptions)

    def set_parameters(self,parameters):
        self.parameters=parameters

    def set_data(self,data):
        self.train = data

    def get_parameters_options(self):
        return self.parametersoptions

    def fit(self,data):
        '''Train a tree an return the model'''
        if self.parameters['max_deep'] > data.shape[1]:
            self.parameters['max_deep'] = data.shape[1];

        self.train = data
        clf = DecisionTreeClassifier(max_depth=self.parameters['max_deep'])
        if np.unique(self.train[:,-1]).shape[0]>2:
	        labelclass=np.zeros(len(data[:,-1]))
	        labelclass[np.where(data[:,-1]>np.mean(self.train[:,-1]))]=1
	        print ("Classification tree must be trained on binary label, assigning labels based on mean and std values")
	        print ("Generating positive class > ",np.mean(self.train[:,-1])," (mean)")
        self.train[:,-1]=labelclass
	#else:
	#    try:
	#	self.train[:,-1]=np.array(map(float,self.train[:,-1]))
	#	print self.train[:,-1]
	#    except:
	#	print "Can not convert labels to binary integer"
        clf = clf.fit(self.train[:,0:-1],self.train[:,-1])
        self.model = clf

        def predict(self,data):
	        return self.model.predict(data[:,0:-1])

    def crossvalidate(self,foldOut,list=None):
        if self.parameters['max_deep'] > self.train.shape[1]:
        #       print self.train.shape
        #       print self.parametersoptions
                self.parameters['max_deep'] = self.train.shape[1];

        if list:
            numberinstances=len(list)
            label=np.zeros(len(list))
        else:
            numberinstances=self.train.shape[0]
            label=np.zeros(numberinstances)
        instanceidx=np.array(range(numberinstances))
        #randbinary = lambda n: [1 for b in range(1,n+1)]
        #randomclasslabel=randbinary(numberinstances)
        shuffle(instanceidx)
        skf = StratifiedKFold(label, n_folds=foldOut)
        labelspredicted=np.array([]);
        for trainindex, testindex in skf:

            if list:
                idx_test = [item for idx in testindex for item in list[idx]]
                idx_train = [item for idx in trainindex for item in list[idx]]

            else:
                idx_test=instanceidx[testindex]
                idx_train=instanceidx[trainindex]

            datatrain = self.train[idx_train,:]
            datatest = self.train[idx_test,:]

            if not labelspredicted.any():
                idxs=idx_test
                dtree=self.copy()
                dtree.fit(datatrain)
                labelspredicted=dtree.predict(datatest)

            else:
                dtree=Tree(self.parameters)
                dtree.fit(datatrain)
                labelspredicted=np.concatenate((labelspredicted,np.array(dtree.predict(datatest))))
                idxs=np.concatenate((idxs,np.array(idx_test)))

        tosort=np.vstack((idxs,labelspredicted))
        return  tosort[1,tosort[0,:].argsort()]

class Lasso():

    def __init__(self,parameters=({'alpha':0.1}),train=np.array([]),parametersoptions=({'alpha':np.arange(0.1,1,0.1)}),details=({})):
        #To include different kernels
        #Train and svm an return the model: b, alphas,width,
        self.model=np.array([])
        self.train=train
        self.parametersoptions=parametersoptions
        self.detailsoptions=({})
        self.details=details
        self.parameters=parameters

    def __call__(self,names=np.array([])):
        if self.model.any():
            try:
                if len(names)>1:
                    print (names[-1] + ' = ' + '%5.8f'%(self.model.coef_[0]) + '+' + '+'.join('%5.8f'%(self.model.coef_[idx]) + '*' + names[idx-1] for idx in range(1,len(self.model.coef_))))
                else:
                    print ('Y = ' + '+'.join('%5.88888888f'%(self.model.coef_[idx])+ '*X' + str(idx) for idx in range(len(self.model.coef_))))
            except:
                print ('Model matrix')
                print (self.model)
        else:
            print ('No model trained yet')

        print (self.model)

    def copy(self):
        return Lasso(parameters=self.parameters,train=self.train,parametersoptions=self.parametersoptions)

    def set_parameters(self,parameters):
        self.parameters = parameters

    def get_parameters_options(self):
        return self.parametersoptions

    def set_data(self,data):
        self.train = data

    def fit(self,data):
        #Train and mlr an return the model: coeficients,
        self.model = linear_model.LassoLars(alpha = self.parameters['alpha'])
        numbexample,numvariable = data.shape
        self.model.fit(data[:,0:-1],data[:,-1])
        self.train=data

    def get_name(self):
        return self.name

    def predict(self,data):
        return self.model.predict(data[:,0:-1])

    def crossvalidate(self,foldOut,list=None):
        if list:
            numberinstances=len(list)
            label=np.zeros(len(list))
        else:
            numberinstances=self.train.shape[0]
            label=np.zeros(numberinstances)
        instanceidx=np.array(range(numberinstances))
        #randbinary = lambda n: [1 for b in range(1,n+1)]
        #randomclasslabel=randbinary(numberinstances)
        shuffle(instanceidx)
        skf = StratifiedKFold(label, n_folds=foldOut)
        labelspredicted=np.array([]);
        for trainindex, testindex in skf:

            if list:
                idx_test = [item for idx in testindex for item in list[idx]]
                idx_train = [item for idx in trainindex for item in list[idx]]

            else:
                idx_test=instanceidx[testindex]
                idx_train=instanceidx[trainindex]
            datatrain = self.train[idx_train,:]
            datatest = self.train[idx_test,:]

            if not labelspredicted.any():
                idxs=idx_test
                mod=self.copy()
                mod.fit(datatrain)
                labelspredicted=mod.predict(datatest)

            else:
                mod=self.copy()
                mod.fit(datatrain)
                labelspredicted=np.concatenate((labelspredicted,np.array(mod.predict(datatest))))
                idxs=np.concatenate((idxs,np.array(idx_test)))

        tosort=np.vstack((idxs,labelspredicted))
        return  tosort[1,tosort[0,:].argsort()]

class ARDmlr():

    def __init__(self,parameters=dict()):
        #To include different kernels
        #Train and svm an return the model: b, alphas,width,
        self.model=np.array([])
        self.train=np.array([])
        self.parametersoptions=({})
        self.detailsoptions=({})
        self.details=({})
        self.parameters=parameters

    def __call__(self):
        pass

    def copy(self):
        return ARDmlr(self.parameters)

    def set_parameters(self,parameters):
        pass

    def set_data(self,data):
        self.train = data

    def get_parameters_options(self):
        pass

    def fit(self,data):
        #Train and mlr an return the model: coeficients,
        self.model = linear_model.ARDRegression(compute_score = True)
        self.model.fit(data[:,0:-1],data[:,-1])
        self.train=data

    def get_name(self):
        return self.name

    def predict(self,data):
        return self.model.predict(data[:,0:-1])

    def crossvalidate(self,foldOut,list=None):
        if list:
            numberinstances=len(list)
            label=np.zeros(len(list))
        else:
            numberinstances=self.train.shape[0]
            label=np.zeros(numberinstances)
        instanceidx=np.array(range(numberinstances))
        #randbinary = lambda n: [1 for b in range(1,n+1)]
        #randomclasslabel=randbinary(numberinstances)
        shuffle(instanceidx)
        skf = StratifiedKFold(label, n_folds=foldOut)
        labelspredicted=np.array([]);
        for trainindex, testindex in skf:

            if list:
                idx_test = [item for idx in testindex for item in list[idx]]
                idx_train = [item for idx in trainindex for item in list[idx]]

            else:
                idx_test=instanceidx[testindex]
                idx_train=instanceidx[trainindex]
            datatrain = self.train[idx_train,:]
            datatest = self.train[idx_test,:]

            if not labelspredicted.any():
                idxs=idx_test
                mod=self.copy()
                mod.fit(datatrain)
                labelspredicted=mod.predict(datatest)

            else:
                mod=self.copy()
                mod.fit(datatrain)
                labelspredicted=np.concatenate((labelspredicted,np.array(mod.predict(datatest))))
                idxs=np.concatenate((idxs,np.array(idx_test)))

        tosort=np.vstack((idxs,labelspredicted))
        return  tosort[1,tosort[0,:].argsort()]

class NNr():

    def __init__(self,parameters=({'n_neighbors':5,'weights':'distance'})):
        #To include different kernels
        #Train and svm an return the model: b, alphas,width,
        self.model=np.array([])
        self.train=np.array([])
        self.parametersoptions=({'n_neighbors':np.array([3,5,7,10]),'weights':np.array(['distance','uniform'])})
        self.detailsoptions=({})
        self.details=({})
        self.parameters=parameters

    def __call__(self):
        pass

    def copy(self):
        return NNr(self.parameters)

    def set_parameters(self,parameters):
        self.parameters = parameters

    def set_data(self,data):
        self.train = data

    def get_parameters_options(self):
        return self.parametersoptions

    def fit(self,data):
        #Train and mlr an return the model: coeficients,
        self.model = neighbors.KNeighborsRegressor(self.parameters['n_neighbors'], weights=self.parameters['weights'])
        self.model.fit(data[:,0:-1],data[:,-1])
        self.train=data

    def get_name(self):
        return self.name

    def predict(self,data):
        return self.model.predict(data[:,0:-1])

    def crossvalidate(self,foldOut,list=None):
        if list:
            numberinstances=len(list)
            label=np.zeros(len(list))
        else:
            numberinstances=self.train.shape[0]
            label=np.zeros(numberinstances)
        instanceidx=np.array(range(numberinstances))
        #randbinary = lambda n: [1 for b in range(1,n+1)]
        #randomclasslabel=randbinary(numberinstances)
        shuffle(instanceidx)
        skf = StratifiedKFold(label, n_folds=foldOut)
        labelspredicted=np.array([]);
        for trainindex, testindex in skf:

            if list:
                idx_test = [item for idx in testindex for item in list[idx]]
                idx_train = [item for idx in trainindex for item in list[idx]]

            else:
                idx_test=instanceidx[testindex]
                idx_train=instanceidx[trainindex]
                datatrain = self.train[idx_train,:]
            datatest = self.train[idx_test,:]

            if not labelspredicted.any():
                idxs=idx_test
                mod=self.copy()
                mod.fit(datatrain)
                labelspredicted=mod.predict(datatest)

            else:
                mod=self.copy()
                mod.fit(datatrain)
                labelspredicted=np.concatenate((labelspredicted,np.array(mod.predict(datatest))))
                idxs=np.concatenate((idxs,np.array(idx_test)))

        tosort=np.vstack((idxs,labelspredicted))
        return  tosort[1,tosort[0,:].argsort()]

class Kmeans():

    def __init__(self,parameters=({'k':10})):
        #To include different kernels
        #Train and svm an return the model: b, alphas,width,
        self.model=np.array([])
        self.train=np.array([])
        self.parametersoptions=({'k':np.array([5,10,20,30,50,80,100])})
        self.detailsoptions=({})
        self.details=({})
        self.parameters=parameters

    def __call__(self):
        pass

    def copy(self):
        return Kmeans(self.parameters)

    def set_parameters(self,parameters):
        self.parameters = parameters

    def set_data(self,data):
         self.train = data

    def get_parameters_options(self):
        return self.parametersoptions

    def fit(self,data):
        #Train and mlr an return the model: coeficients,
        self.model = KMeans(init='k-means++', k=self.parameters['k'], n_init=10)
        self.model.fit(data[:,0:-1])
        self.train=data

    def get_name(self):
        return self.name

    def predict(self,data):
        return self.model.predict(data[:,0:-1])

    def crossvalidate(self,foldOut,list=None):
        if list:
            numberinstances=len(list)
            label=np.zeros(len(list))
        else:
            numberinstances=self.train.shape[0]
            label=np.zeros(numberinstances)
        instanceidx=np.array(range(numberinstances))
        #randbinary = lambda n: [1 for b in range(1,n+1)]
        #randomclasslabel=randbinary(numberinstances)
        shuffle(instanceidx)
        skf = StratifiedKFold(label, n_folds=foldOut)
        labelspredicted=np.array([])
        for trainindex, testindex in skf:

            if list:
                idx_test = [item for idx in testindex for item in list[idx]]
                idx_train = [item for idx in trainindex for item in list[idx]]

            else:
                idx_test=instanceidx[testindex]
                idx_train=instanceidx[trainindex]
                datatrain = self.train[idx_train,:]
            datatest = self.train[idx_test,:]

            if not labelspredicted.any():
                idxs=idx_test
                mod=self.copy()
                mod.fit(datatrain)
                labelspredicted=mod.predict(datatest)

            else:
                mod=self.copy()
                mod.fit(datatrain)
                labelspredicted=np.concatenate((labelspredicted,np.array(mod.predict(datatest))))
                idxs=np.concatenate((idxs,np.array(idx_test)))

        tosort=np.vstack((idxs,labelspredicted))
        return  tosort[1,tosort[0,:].argsort()]

class Plsr():
    def __init__(self,parameters=({'factors':3}),train=np.array([]),parametersoptions=({'factors':np.array(range(5,100,5))})):
        self.model=np.array([])
        self.train=train
        self.parametersoptions=parametersoptions
        self.detailsoptions=({})
        self.details=({})
        self.parameters=parameters

    def __call__(self,names=None):
        pass

    def copy(self):
        return Plsr(parameters=self.parameters,train=self.train,parametersoptions=self.parametersoptions)

    def set_parameters(self,parameters):
        self.parameters = parameters

    def set_data(self,data):
        self.train = data

    def get_parameters_options(self):
        return self.parametersoptions

    def fit(self,data):
        if self.parameters['factors'] > data.shape[1]-1:
	        self.parameters['factors'] = data.shape[1]-1
        self.model = pls.PLSRegression(n_components = self.parameters['factors'])
        self.model.fit(data[:,0:-1],data[:,-1])
        self.train=data

    def get_name(self):
        return self.name

    def predict(self,data):
        return self.model.predict(data[:,0:-1])[:,0]

    def crossvalidate(self,foldOut,list=None):
        if self.parameters['factors'] > self.train.shape[1]-1:
                   self.parameters['factors'] = self.train.shape[1]-1;
        if list:
            numberinstances=len(list)
            label=np.zeros(len(list))
        else:
            numberinstances=self.train.shape[0]
            label=np.zeros(numberinstances)
        instanceidx=np.array(range(numberinstances))
        #randbinary = lambda n: [1 for b in range(1,n+1)]
        #randomclasslabel=randbinary(numberinstances)
        shuffle(instanceidx)
        skf = StratifiedKFold(label, n_folds=foldOut)
        labelspredicted=np.array([])
        for trainindex, testindex in skf:

            if list:
                idx_test = [item for idx in testindex for item in list[idx]]
                idx_train = [item for idx in trainindex for item in list[idx]]

            else:
                idx_test=instanceidx[testindex]
                idx_train=instanceidx[trainindex]
            datatrain = self.train[idx_train,:]
            datatest = self.train[idx_test,:]

            if not labelspredicted.any():
                idxs=idx_test
                mod=self.copy()
                mod.fit(datatrain)
                labelspredicted=mod.predict(datatest)

            else:
                mod=self.copy()
                mod.fit(datatrain)
                labelspredicted=np.concatenate((labelspredicted,np.array(mod.predict(datatest))))
                idxs=np.concatenate((idxs,np.array(idx_test)))
        tosort=np.vstack((idxs,labelspredicted))
        return  tosort[1,tosort[0,:].argsort()]

class Ffnn():

    def __init__(self,parameters=({'hdn':5,'layer':10,'maxepoch':500,'wdecay':0.2})):
        self.model=np.array([])
        self.train=np.array([])
        self.parametersoptions=({'hdn':np.array([3,5,10,20,30,50,80,100]),'layer':np.array([1]),'maxepoch':np.array([25,100,250,300,350,500,1000]),'wdecay':np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99])})
        self.detailsoptions=({})
        self.details=({})
        self.parameters=parameters

    def __call__(self):
        pass

    def copy(self):
        return Ffnn(self.parameters)

    def set_parameters(self,parameters):
        self.parameters = parameters

    def set_data(self,data):
        self.train = data

    def get_parameters_options(self):
        return self.parametersoptions

    def fit(self,data):
        #Train and mlr an return the model: coeficients,
        dataset = SupervisedDataSet(data.shape[1]-1, 1)
        for dat in data:
            dataset.appendLinked(dat[0:-1],dat[-1])

        model = buildNetwork(data[:,0:-1].shape[1], self.parameters['hdn'], 1, bias=True, hiddenclass=TanhLayer)
        trainer = BackpropTrainer(model, dataset=dataset,learningrate=0.01, momentum=0.1, weightdecay=self.parameters['wdecay'])
        trainer.trainUntilConvergence(maxEpochs=self.parameters['maxepoch'],continueEpochs=25,validationProportion=0.25)
        self.model = model
        self.train = data

    def get_name(self):
        return self.name

    def predict(self,data):
        dataset = SupervisedDataSet(data.shape[1]-1, 1)

        for dat in data:
            dataset.appendLinked(dat[0:-1],dat[-1])

        output = self.model.activateOnDataset(dataset)
        return np.reshape(output,output.shape[0])

    def crossvalidate(self,foldOut,list=None):
        if list:
            numberinstances=len(list)
            label=np.zeros(len(list))
        else:
            numberinstances=self.train.shape[0]
            label=np.zeros(numberinstances)
        instanceidx=np.array(range(numberinstances))
        #randbinary = lambda n: [1 for b in range(1,n+1)]
        #randomclasslabel=randbinary(numberinstances)
        shuffle(instanceidx)
        skf = StratifiedKFold(label, n_folds=foldOut)
        labelspredicted=np.array([]);
        for trainindex, testindex in skf:

            if list:
                idx_test = [item for idx in testindex for item in list[idx]]
                idx_train = [item for idx in trainindex for item in list[idx]]

            else:
                idx_test=instanceidx[testindex]
                idx_train=instanceidx[trainindex]
            datatrain = self.train[idx_train,:]
            datatest = self.train[idx_test,:]

            if not labelspredicted.any():
                idxs=idx_test
                mod=self.copy()
                mod.fit(datatrain)
                labelspredicted=mod.predict(datatest)

            else:
                mod=self.copy()
                mod.fit(datatrain)
                labelspredicted=np.concatenate((labelspredicted,np.array(mod.predict(datatest))))
                idxs=np.concatenate((idxs,np.array(idx_test)))
        tosort=np.vstack((idxs,labelspredicted))
        return  tosort[1,tosort[0,:].argsort()]

class Ffann():

    def __init__(self,parameters=({'hdn':5,'maxepoch':500})):
	    #self.model=None
        self.train=np.array([])
        self.parametersoptions=({'hdn':np.array([3,5,10,15,20]),'maxepoch':np.array([100,250,500,1000,2000,5000])})
        self.detailsoptions=({})
        self.details=({})
        self.parameters=parameters
        self.model = np.array([])

    def __call__(self):
        pass

    def copy(self):
        return Ffann(self.parameters)

    def set_parameters(self,parameters):
        self.parameters = parameters

    def set_data(self,data):
        self.train = data

    def get_parameters_options(self):
        return self.parametersoptions

    def fit(self,data):
        #Train a ffann an return the model
        self.train = data
        connection_rate = 1
        learning_rate = 0.7
        num_input = data.shape[1]-1
        num_hidden = self.parameters['hdn']
        num_output = 1
        desired_error = 0.0001
        max_iterations = self.parameters['maxepoch']
        iterations_between_reports = 100
        iterations_tunning = 20
        ann = libfann.neural_net()
        ann.create_sparse_array(connection_rate, (num_input, num_hidden, num_output))
        ann.set_learning_rate(learning_rate)
        ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
        trainid=random.sample(range(data.shape[0]),int(data.shape[0]*0.8))
        valid=np.setdiff1d(range(data.shape[0]),trainid)
        file=open('traindata_nne','wb')
        file.write(" ".join(map(str,[data[trainid,:].shape[0],data.shape[1]-1,1]))+ os.linesep)
        for dat in data[trainid,:]:
            file.write(" ".join(map(str,dat[0:-1])) + os.linesep)
            file.write(str(dat[-1]) + os.linesep)
        file.close()
        maxIt = max_iterations*0.7
        dEv = 0
        dEt = 1
        print ("Optimizing fase")
        ann.train_on_file('traindata_nne', int(max_iterations*0.7), iterations_between_reports, desired_error)
        print ("Tunning fase")
        while dEv < dEt + 0.001 and maxIt<=max_iterations:
            ann.train_on_file('traindata_nne', iterations_tunning, iterations_between_reports, desired_error)
            outval=np.array([ann.run(input) for input in data[valid,0:-1]])
            outtrain=np.array([ann.run(input) for input in data[trainid,0:-1]])
            dEv=np.sqrt(np.mean([data[valid[id],-1]-outval[id] for id in range(len(valid))])**2)
            dEt=np.sqrt(np.mean([data[trainid[id],-1]-outtrain[id] for id in range(len(trainid))])**2)
            #print "Errors===> Training",dEt,"Validation",dEv
            maxIt =  maxIt + iterations_tunning
        self.model = ann

    def get_name(self):
        return self.name

    def predict(self,data):
        output = [self.model.run(input) for input in data[:,0:-1]]
        return np.reshape(np.array(output),len(output))

    def crossvalidate(self,foldOut,list=None):
        if list:
            numberinstances=len(list)
            label=np.zeros(len(list))
        else:
            numberinstances=self.train.shape[0]
            label=np.zeros(numberinstances)
        instanceidx=np.array(range(numberinstances))
        #randbinary = lambda n: [1 for b in range(1,n+1)]
        #randomclasslabel=randbinary(numberinstances)
        shuffle(instanceidx)
        skf = StratifiedKFold(label, n_folds=foldOut)
        labelspredicted=np.array([]);
        for trainindex, testindex in skf:

            if list:
                idx_test = [item for idx in testindex for item in list[idx]]
                idx_train = [item for idx in trainindex for item in list[idx]]

            else:
                idx_test=instanceidx[testindex]
                idx_train=instanceidx[trainindex]
            datatrain = self.train[idx_train,:]
            datatest = self.train[idx_test,:]

            if not labelspredicted.any():
                idxs=idx_test
                mod=self.copy()
                mod.fit(datatrain)
                labelspredicted=mod.predict(datatest)

            else:
                mod=self.copy()
                mod.fit(datatrain)
                labelspredicted=np.concatenate((labelspredicted,np.array(mod.predict(datatest))))
                idxs=np.concatenate((idxs,np.array(idx_test)))
        tosort=np.vstack((idxs,labelspredicted))
        return  tosort[1,tosort[0,:].argsort()]

'''<<<<<<< local
functions={'svm':Svm(),'mlr':Mlr(),'libsvm':Libsvm(),'tree':Tree(),'lasso':Lasso(),'ardmlr':ARDmlr(),'nnr':NNr(),'kmeans':Kmeans(),'plsr':Plsr(),'ffnn':Ffnn(),'ffann':Ffann()}
======='''
functions={'randf':RandF(),'ridge':rRidge(),'svm':Svm(),'mlr':Mlr(),'libsvm':Libsvm(),'sksvm':Sksvm(),'tree':Tree(),'rtree':rTree(),'lasso':Lasso(),'ardmlr':ARDmlr(),'nnr':NNr(),'kmeans':Kmeans(),'plsr':Plsr(),'ffnn':Ffnn(),'ffann':Ffann()}
#>>>>>>> other

