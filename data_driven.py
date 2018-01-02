'''
created on Nov 22, 2011

@author: mllamosa
'''

from __future__ import print_function
import numpy as np
from scipy.stats import linregress
#from scikits.learn.metrics import roc_curve as roc, auc
from sklearn.metrics import roc_curve as roc, auc
from utilities import *
#from functions import *
from constants import *
import re
import pickle as cPickle
import sys
#import matplotlib


class Partition():
    def __init__(self,trainset=np.array([]),testset=np.array([]),valset=np.array([])):
        self.train=trainset
        self.test=testset
        self.val=valset

    def load(self,filepartitions,instname=np.array([])):
        file=open(filepartitions)
        data=file.read()
        file.close()
        train=np.array([])
        if re.search('Train',data):
		#try:
            trainname = data[re.search('Train',data).end()+1:].split('\n')[0].split(',')
            if instname.any():
				#print trainname[0:10],instname[0:10]
                train = [np.where(instname==item)[0][0] for item in trainname]
            else:
                train = map(int,trainname)
		#except (IndexError),Error:
		#	print "Can't find train instants in the data set",Error
		#	sys.exit(1)
        test=np.array([])

        if re.search('Test',data):
            try:
                testname = data[re.search('Test',data).end()+1:].split('\n')[0].split(',')
                if instname.any():
                    test = [np.where(instname==item)[0][0] for item in testname]
                else:
                    test = map(int,testname)
            except (IndexError):
                print ("Can't find test instants in the data set",Error)
                sys.exit(1)

        val=np.array([])

        if re.search('Val',data):
            try:
                valname = map(int,data[re.search('Val',data).end()+1:].split('\n')[0].split(','))
                if not instname.any():
                    val = [np.where(instname==item)[0][0] for item in valname]
                else:
                    val = map(int,valname)
            except (IndexError):
                print ("Can't find test instants in the data set",Error)
                sys.exit(1)

        train = np.array(train)
        test = np.array(test)
        val = np.array(val)

        if train.any():
                self.train=train

        if test.any():
                self.test=test

        if val.any():
                self.val=val

    def set(self,trainset=np.array([]),testset=np.array([]),valset=np.array([])):
        self.train=trainset
        self.test=testset
        self.val=valset
        print ("Setting train/set partition",len(self.train))

    def save(self,filepartition,istname=np.array([])):
        print ("===>Saving training/set partition to ",filepartition)
        file=open(filepartition,'w')

        if self.train.any() and len(self.train)>0:
            if len(istname) > 0:
                print >> file, 'Train' + ',' + ','.join(map(str,istname[self.train]))
            else:
                print >> file, 'Train' + ',' + ','.join(map(str,self.train))

        if self.test.any() and len(self.test):
            if len(istname) > 0:
                print >> file, 'Test' + ',' + ','.join(map(str,istname[self.test]))
            else:
                print >> file, 'Test' + ',' + ','.join(map(str,self.test))

        if self.val.any() and len(self.val)>0:
            if len(istname) > 0:
                print >> file, 'Val' + ',' + ','.join(map(str,istname[self.val]))
            else:
                print >> file, 'Val' + ',' + ','.join(map(str,self.val))
        file.close()

class Measure():
    def __init__(self,stats):
        if stats.any():
            self.accuracy=stats[0]
            self.precision=stats[1]
            self.fitness=stats[2]
            self.N=stats[3]
            self.p_value=stats[4]
            self.extra=stats[5]
        else:
            self.accuracy=float('NaN')
            self.precision=float('NaN')
            self.fitness=float('NaN')
            self.N=float('NaN')
            self.p_value=float('NaN')
            self.extra=float('NaN')

class Statistics():

    def __init__(self,trainstats=np.array([]),teststats=np.array([]),crossvalstats=np.array([])):
        self.train=Measure(trainstats)
        self.test=Measure(teststats)
        self.crossval=Measure(crossvalstats)


    def set_values(self,trainstats,teststats,crossvalstats):

        if trainstats.any():
            self.train=Measure(trainstats)
        if teststats.any():
            self.test=Measure(teststats)

        if crossvalstats.any():
            self.crossval=Measure(crossvalstats)



    def summary(self,labels=[1,2,4]):

        if len(np.unique(labels))>2:
             print ("".join(["Statistics            ","R2            ","STD           ","Slope         ","N        ","p-value    "]))
        else:
            if self.train.extra != float('NaN') and  self.crossval.extra != 'nan' and self.train.extra != np.float('nan'):
                print ("Cutoff of maximum F-score ",self.crossval.extra)
            print ("".join(["Statistics            ","AUC           ","Sensitivity   ","Specificity   ","N        ","Precision  "]))

        if self.train.accuracy:
            #print "Training       ", "%4.5f" %  self.train.accuracy,"%4.5f" %self.train.precision,"%4.5f" %self.train.fitness,"%1.0f" %self.train.N,"%4.5f" %self.train.p_value
            print ("       ".join(["Training       ", "%4.5f" %  self.train.accuracy,"%4.5f" %self.train.precision,"%4.5f" %self.train.fitness,"%1.0f" %self.train.N,"%4.5f" %self.train.p_value]))

        if self.crossval.accuracy:
                print ("       ".join(["Crossvalidation", "%4.5f" % self.crossval.accuracy,"%4.5f" % self.crossval.precision,"%4.5f" % self.crossval.fitness,"%1.0f" %self.crossval.N,"%4.5f" %self.crossval.p_value]))

        if self.test.accuracy:
                print ("       ".join(["Test           ", "%4.5f" % self.test.accuracy,"%4.5f" %self.test.precision,"%4.5f" %self.test.fitness,"%1.0f" %self.test.N,"%4.5f" %self.test.p_value]))

    def save(self,file):

        f=open(file,'w')

        print >> f, "Statistics        R2       std    fitting"

        if not self.train.accuracy.any():
            print >>f, "Training       ", "%4.5f" %  self.train.accuracy,"%4.5f" %self.train.precision,"%4.5f" %self.train.fitness,"%1.0f" %self.train.N

        if not self.crossval.accuracy.any():
            print >> f, "Crossvalidation", "%4.5f" % self.crossval.accuracy,"%4.5f" % self.crossval.precision,"%4.5f" % self.crossval.fitness,"%1.0f" %self.crossval.N

        if not self.test.accuracy.any():
            print >> f, "Test           ", "%4.5f" % self.test.accuracy,"%4.5f" %self.test.precision,"%4.5f" %self.test.fitness,"%1.0f" %self.test.N

        f.close()


class Model(object):
    def __init__(self,data=np.array([]),varnames=np.array([]),instnames=np.array([]),partition=Partition(),categories=np.array([]),groups=np.array([]),function=None,filterfunction=None,filtercorr=None,normalizefunction=None,nfo=3):

        self.partition = partition
        self.train = np.array([])
        self.test = np.array([])
        self.validation = np.array([])
        self.stats = np.array([])
        self.function = function
        self.categories = categories
        self.groups = groups
        self.varnames = varnames
        self.instnames = instnames
        self.varids = np.array([])
        self.varrelevance = np.array([])
        self.normalizer = normalizefunction
        self.filter = filterfunction
        self.filtercorr = filtercorr
        self.nfo = nfo
        self.cutoff = None
        self.__set_data__(data,filtering=True,normalizing=True)

    def dump(self,filename):
        FILE = open(filename, 'w')
        if self.function.__class__.__name__ not in ['Ffann','Libsvm']:
            cPickle.dump(self, FILE)
        else:
            print ('Couldnt dump to file model file',self.function.__class__.__name__)
            FILE.close()

    def save(self,file):
        if len(self.train) != 0:
            np.savetxt('%s.train'%file , self.train,fmt='%4.5f')
        if len(self.validation) != 0:
            np.savetxt('%s.xval'%file  , self.validation,fmt='%4.5f')
        if len(self.test) != 0:
            np.savetxt('%s.test'%file ,self.test,fmt='%4.5f')

        if self.stats:
            f=open('%s.perf'% file ,'w')
            for category in sorted(self.stats.keys()):
                if len(np.unique(self.data[:,-1]))>2:
                    print >> f, category,'Train_R2','Train_STD','Train_Slope','Train_N','Train_p-value','Test_R2','Test_STD','Test_Slope','Test_N','p_value','Val_R2','Val_STD','Val_Slope','Val_N','p_value'
                else:
                    print >> f, category,'Train_AUC','Train_sensitivity','Train_specificity','Train_N','Train_precision','Test_AUC','Test_sensitivity','Test_specificity','Test_N','Test_precision','Val_AUC','Val_sensitivity','Val_specificity','Val_N','Val_precision'
                    for group in sorted(self.stats[category].keys()):
                        print >> f,group, "%4.5f" % self.stats[category][group].train.accuracy,"%4.5f" % self.stats[category][group].train.precision,"%4.5f" % self.stats[category][group].train.fitness,"%1.0f" % self.stats[category][group].train.N,"%4.5f" % self.stats[category][group].train.p_value,\
                            "%4.5f" % self.stats[category][group].test.accuracy,"%4.5f" % self.stats[category][group].test.precision,"%4.5f" % self.stats[category][group].test.fitness,"%1.0f" % self.stats[category][group].test.N,"%4.5f" % self.stats[category][group].test.p_value,\
                            "%4.5f" % self.stats[category][group].crossval.accuracy,"%4.5f" % self.stats[category][group].crossval.precision,"%4.5f" % self.stats[category][group].crossval.fitness,"%1.0f" % self.stats[category][group].crossval.N,"%4.5f" % self.stats[category][group].crossval.p_value


        if self.varrelevance:
                print >>f, "Variable relevances"
                for key in sorted(self.varrelevance.keys()):
                    print >> f,"Var", key, '%1.5f'%self.varrelevance[key]

        f.close()

    def load_categories(self,filecategories):
        file = open(filecategories)
        categories = file.readline().split()
        file.close()
        groups = np.loadtxt(filecategories,skiprows=1,dtype='string')
        self.categories=dict()
        if len(categories)>1:
            for idx in range(len(categories)):
                self.categories[categories[idx]]=groups[:,idx]
        else:
                self.categories[categories[0]]=groups

    def set_function(self,function):
        self.function=function

	#if self.groups:
	#         trainidx = np.array([item for idx in self.partition.train for item in self.groups[idx]])
	#else:
	#         trainidx = self.partition.train

	#self.function.fit(self.data[trainidx,:])


    def set_data_subset(self,subset,filtering=False,normalizing=False):

        data=self.data[:,subset]
        self.varnames = self.varnames[subset]
        "Added new"
        f.varids = np.array(subset)
        f.nvariables = len(subset)-1
        self.__set_data__(data,filtering=False,normalizing=False)

    def __set_data__(self,data,filtering=False,normalizing=False):
        '''Set data and update function'''
        self.data=data
        if self.groups.any():
            trainidx = np.array([item for idx in self.partition.train for item in self.groups[idx]])
        else:
            trainidx = self.partition.train

        if self.data.any():
            self.nvariables = self.data.shape[1]-1
            self.ninstances = self.data.shape[0]
        else:
            self.nvariables = None
            self.ninstances = None

        if self.filter and filtering:
            print ("===>Removing constant values...")
            ids = self.filter.apply(self.data[trainidx,:])
             #ids = self.filter.apply(self.data)
            if ids.any():
                data = self.data[:,ids]
                varnames = self.varnames[ids]
                self.varids = ids
                self.nvariables = self.data.shape[1]-1

        if self.filtercorr and filtering:
             print ("===>Removing correlated values...")
             ids = self.filtercorr.apply(self.data[trainidx,:],0.99)
             #ids = self.filter.apply(self.data)
             #print 'Ids',(ids)
             if ids.any():
                self.data = self.data[:,ids]
                self.varnames = self.varnames[ids]
                self.varids = np.array(ids)
             self.nvariables = self.data.shape[1]-1

	#if (self.filtercorr or self.filter) and filtering:
	#	print '===>Saving data of non-coorelated variables to file ' + runfolder + os.sep + os.path.basename(datafile) + '_filter_noncorr.csv'
	#	data_dict = dict()
	#	for structid in range(len(self.instnames)):
	#		desc_dict = dict()
	#		for varid in range(len(self.varnames)):
	#			desc_dict[self.varnames[varid]] = data[structid,varid]
	#		data_dict[selt.instnames[structid]] = desc_dict
	#	descriptor_writer(runfolder + os.sep + os.path.basename(datafile) + '_filter_noncorr.csv',data_dict)

        if self.normalizer and normalizing:
                 print ("===>Normalizing the data...")
                 f.normalizer.map_data(self.data[trainidx,:])
                 self.data=self.normalizer.transmap_data(self.data)

	#if self.function != None and self.data != None :
	#    self.function.fit(self.data[trainidx,:])
        if not self.data.any():
            unction.set_data(self.data[trainidx,:])

            self.varnames = self.varnames

    def set_partition(self,partition):
        '''Set data partition and update function'''
        self.partition=partition
        if self.groups.any():
                 trainidx = np.array([item for idx in self.partition.train for item in self.groups[idx]])
        else:
                 trainidx = self.partition.train

        self.function.set_data(self.data[trainidx,:])
	#if self.function:
	#    self.function.fit(self.data[trainidx,:])

    def set_groups(self,groups):
        self.groups = groups

    def training(self):

        if self.groups.any():
            trainidx = np.array([item for idx in self.partition.train for item in self.groups[idx]])
        else:
            trainidx = self.partition.train

        #print 'Training shape',self.data[trainidx,:].shape
        self.function.fit(self.data[trainidx,:])
        self.train = self.function.predict(self.data[trainidx,:])

        if self.normalizer:
            self.train = self.normalizer.remap_data(self.train)

    def crossvalidating(self):
        if self.groups.any():
            list = []
            groupstrain = [self.groups[idx] for idx in self.partition.train]
            trainidx = np.array([item for idx in self.partition.train for item in self.groups[idx]])
            for group in groupstrain:
               listtmp = []
               for ele in group:
                   listtmp.append(np.where(trainidx==ele)[0][0])
               list.append(np.array(listtmp))
        else:
            trainidx = self.partition.train
            list=np.array([])
        list = np.array(list)

        if not self.function.train.any():
            self.function.set_data(self.data[trainidx,:])
        self.validation = self.function.crossvalidate(self.nfo,list)

        if self.normalizer:
            self.validation = self.normalizer.remap_data(self.validation)

    def testing(self):

        if self.groups.any():
            testidx = np.array([item for idx in self.partition.test for item in self.groups[idx]])
            trainidx = np.array([item for idx in self.partition.train for item     in self.groups[idx]])
        else:
            testidx = self.partition.test
        trainidx = self.partition.train
        if self.train.size == 0:
            print ("===>Fitting the model first")
            self.function.fit(self.data[trainidx,:])
            #print 'Test shape',self.data[testidx,:].shape
        self.test = self.function.predict(self.data[testidx,:])

        if self.normalizer:
            self.test = self.normalizer.remap_data(self.test)

    def similarity(self):
	    return tanimoto(self.data[:,0:-1])

    def cliff(self,cutoff=None):
	    ids = np.argsort(self.data[:,-1])
	    return cliffmatrix(self.data[ids,:],cutoff=cutoff),self.instnames[ids]

    def cliff_write_pp(self,filename,cutoff=None):
	    ids = np.argsort(self.data[:,-1])
	    cliffmatrix_write_pp(self.data[ids,:],filename,cutoff=cutoff)
	    file=open(filename + '.instlabels','w')
	    print >>file, " ".join(self.instnames[ids])
	    file.close

    def cliff_write(self,filename,cutoff=None):
             ids = np.argsort(self.data[:,-1])
             cliffmatrix_write(self.data[ids,:],filename,cutoff=cutoff)
             file=open(filename + '.instlabels','w')
             print >>file, " ".join(self.instnames[ids])
             file.close

    def summary(self):
        if self.stats:
            for category in sorted(self.stats.keys()):
                for groupkey in sorted(self.stats[category].keys()):
                    print ("Stats for category",category+", group", groupkey)
                    self.stats[category][groupkey].summary(self.data[:,-1])
            if self.varrelevance:
                print ("Variable relevances")
                for key in sorted(self.varrelevance.keys()):
                    print ("Var", key, '%1.5f'%self.varrelevance[key])
        else:
            print ('No stats, first run model.performance()')

    def predict(self,data):
        data = np.hstack((data,np.zeros((data.shape[0],1))))
        '''Temporary due to old models'''
        #print self.varids
        if self.varids.any():
            data = data[:,self.varids]
        if self.normalizer:
                data = self.normalizer.transmap_data(data)
                predictions = self.function.predict(data)
                predictions = self.normalizer.remap_data(predictions)
        else:
                predictions = self.function.predict(data)

        return predictions

    def response_surface(self, subset,varnames,file, show=None):
        import matplotlib
        if not show:
            matplotlib.use('Agg')
        #Two-var matrix
        if self.groups.any():
            trainidx = np.array([item for idx in self.partition.train for item in self.groups[idx]])
        else:
            trainidx = self.partition.train

        idxlabel = self.data.shape[1]-1
        #print np.hstack(([subset, idxlabel]))
        subset = subset-1
        datared = self.data[:,np.hstack(([subset, idxlabel]))]
        '''Make a copy of the normalization function to normalize the 2var models
            and reinitialize this mapping with the reduced dataset of two vars
        '''
        twovarnormalizer=self.normalizer.copy()
        if self.normalizer:
            dataredoriginal=self.normalizer.remap_data(self.data)[:,np.hstack(([subset, idxlabel]))]
        else:
            dataredoriginal=datared
        model2var = Model(dataredoriginal,self.partition,self.function,self.groups,None,twovarnormalizer)
        #Get normalized training data
        datanorm = datared[trainidx,:]
        #Get grid to predict ranging from min max normalized values
        maxX,minX = np.max(datanorm[:,0]),np.min(datanorm[:,0])
        maxY,minY = np.max(datanorm[:,1]),np.min(datanorm[:,1])
        X = np.arange(minX, maxX, (maxX-minX)/100)
        Y = np.arange(minY, maxY, (maxY-minY)/100)

        #Predict grid map
        X, Y = np.meshgrid(X, Y)
        dim=X.shape[0]
        X.resize([dim**2,1])
        Y.resize([dim**2,1])
        datatmp=np.hstack((X,Y))
        Znorm=model2var.predict(datatmp)
        #Important to concatenate like this, without converting Znorm to a matrix
        XYZ=np.vstack((X.T,Y.T,Znorm)).T
        if model2var.normalizer:
            XYZ=model2var.normalizer.remap_data(XYZ)
        X,Y,Z=XYZ[:,0],XYZ[:,1],XYZ[:,2]
        X=np.reshape(X,(dim,dim))
        Y=np.reshape(Y,(dim,dim))
        Z=np.reshape(Z,(dim,dim))
        #Normalization problems
        surface3D(X,Y,Z,varnames,file,show=None)

    def plot(self,file,show=None):
        import matplotlib
        if len(np.unique(self.data[:,-1]))<3:
            scatter_plot=auc_plot
        else:
            scatter_plot=scatter2D_plot
        #if not show:
            #matplotlib.use('Agg')

        if self.groups.any():
            trainidx = np.array([item for idx in self.partition.train for item in self.groups[idx]])
            testidx = np.array([item for idx in self.partition.test for item in self.groups[idx]])
        else:
            trainidx = self.partition.train
            testidx = self.partition.test

        if len(self.train) != 0:
            if self.normalizer:
                x,y=self.normalizer.remap_data(self.data[trainidx,-1]), self.train
		        #print x.min(),y.min()
            else:
                x,y=self.data[trainidx,-1], self.train
            scatter_plot(x,y,'Train',file,show)
            if self.categories:
                    for categoryidx in sorted(self.categories.keys()):
                        category = self.categories[categoryidx][trainidx]
                        for uniquegroup in np.unique(category):
                            idx=np.where(category==uniquegroup)
                            scatter_plot(x[idx],y[idx],'Train_'+ str(categoryidx) + '_' + str(uniquegroup),file,show)

        if len(self.test)!= 0:
            if self.normalizer:
                x,y=self.normalizer.remap_data(self.data[testidx,-1]), self.test
            else:
                x,y=self.data[testidx,-1], self.test

            scatter_plot(x,y,'Test',file,show)
            if self.categories:
                    for categoryidx in sorted(self.categories.keys()):
                        category = self.categories[categoryidx][testidx]
                        for uniquegroup in np.unique(category):
                            idx=np.where(category==uniquegroup)
                            scatter_plot(x[idx],y[idx],'Test_'+ str(categoryidx) + '_' + str(uniquegroup),file,show)

        if len(self.validation) != 0:
            if self.normalizer:
                x,y=self.normalizer.remap_data(self.data[trainidx,-1]), self.validation
            else:
                x,y=self.data[trainidx,-1], self.validation
            scatter_plot(x,y,'Validation',file,show)
            if self.categories:
                    for categoryidx in sorted(self.categories.keys()):
                        category = self.categories[categoryidx][trainidx]
                        for uniquegroup in np.unique(category):
                            idx=np.where(category==uniquegroup)
                            scatter_plot(x[idx],y[idx],'Val_'+ str(categoryidx) + '_' + str(uniquegroup),file,show)

    def univariatecorr(self,file,show=None):
        import matplotlib
        if not show:
             matplotlib.use('Agg')
        if not self.data.any():
            if self.normalizer:
                dataactual = self.normalizer.remap_data(self.data)
            else:
                dataactual = self.data
        for idx in range(dataactual.shape[1]-1):
            x,y=dataactual[:,idx], dataactual[:,-1]
            x[np.where(x>np.mean(x)*10)]=0
            scatterdens_plot(x,y,'UnivarCorr_' + self.varnames[idx],file,show)

    def copy(self,normalizing=True,filtering=True):
        if normalizing and filtering and self.normalizer :
            return Model(data=self.data,varnames=self.varnames,instnames=self.instnames,partition=self.partition,categories=self.categories,groups=self.groups,function=self.function.copy(),filterfunction=self.filter,normalizefunction=self.normalizer.copy(),nfo = self.nfo)
        else:
            return Model(data=self.data,varnames=self.varnames,instnames=self.instnames,partition=self.partition,categories=self.categories,groups=self.groups,function=self.function.copy(), nfo = self.nfo)

    def relevance(self):
        '''Individual variable relevance estimation by sensibility analysis'''
        varlist = range(self.data.shape[1])
        relevance = dict()
        for idx in range(len(varlist)-1):
            vartmp = list(varlist)
            vartmp.remove(idx)
            model=self.copy()
            model.__set_data__(self.data[:,vartmp])
            model.training()
            model.crossvalidating()
            model.performance()
            relevance[idx] = 1 - model.stats['all']['all'].crossval.accuracy
        self.varrelevance = relevance

    def pairmodels(self,file=None):
        numbervar=self.data.shape[1]-1
        crossvalmatrix=np.zeros((numbervar,numbervar))
        for i in range( numbervar):
            for j in range(i+1, numbervar,1):
                #datared=self.data[:,np.array([i,j, numbervar])]
                model2var=self.copy()
                model2var.__set_data__(self.data[:,(np.array([i,j, numbervar]))])
                model2var.crossvalidating()
                model2var.performance()
                crossvalmatrix[i,j]=model2var.stats['all']['all'].crossval.accuracy
        if file:
            print ("Saving to 2 var models stats to file", file)
            np.savetxt(file,crossvalmatrix,fmt='%4.5f')

        return crossvalmatrix

    def performance(self):
        stats=dict()
        stats['all']=({'all':Statistics(np.array([]),np.array([]),np.array([]))})
        if self.train.any():

                if self.groups.any():
                    trainidx = np.array([item for idx in self.partition.train for item in self.groups[idx]])
                else:
                    trainidx = self.partition.train

                label=self.data[trainidx,-1]
                if self.normalizer:
                    label = self.normalizer.remap_data(label)
                score = self.train

                stats['all']=({'all':Statistics(all_stats(label,score),np.array([]),np.array([]))})
                if self.categories:
                    for categoryidx in sorted(self.categories.keys()):
                        statstemp = dict()
                        category = self.categories[categoryidx][trainidx]
                        for uniquegroup in np.unique(category):
                            idx=np.where(category==uniquegroup)
                            statstemp[uniquegroup]=Statistics(all_stats(label[idx],score[idx]),np.array([]),np.array([]))
                        stats[categoryidx] = statstemp

        if  self.validation.any():

                if self.groups.any():
                    trainidx = np.array([item for idx in self.partition.train for item in self.groups[idx]])
                else:
                    trainidx = self.partition.train
                label=self.data[trainidx,-1]
                if self.normalizer:
                    label=self.normalizer.remap_data(label)
                score = self.validation
                stats['all']['all'].set_values(np.array([]),np.array([]),all_stats(label,score))
                self.cutoff =  stats['all']['all'].crossval.extra
                #print "Print cutoff",self.cutoff
        if self.categories:
                    for categoryidx in sorted(self.categories.keys()):
                        statstemp = dict()
                        category = self.categories[categoryidx][trainidx]
                        for uniquegroup in np.unique(category):
                            idx=np.where(category==uniquegroup)
                            try:
                                stats[categoryidx][uniquegroup].set_values(np.array([]),np.array([]),all_stats(label[idx],score[idx]))
                            except:
                                statstemp[uniquegroup] = Statistics(np.array([]),np.array([]),all_stats(label[idx],score[idx]))
                                stats[categoryidx] = statstemp

        if  self.test.any():

                if self.groups.any():
                    testidx = np.array([item for idx in self.partition.test for item in self.groups[idx]])
                else:
                    testidx = self.partition.test
                label=self.data[testidx,-1]
                if self.normalizer:
                    label = self.normalizer.remap_data(label)
                score = self.test
                stats['all']['all'].set_values(np.array([]),all_stats(label,score,cutoff=self.cutoff),np.array([]))
                if self.categories:
                    for categoryidx in sorted(self.categories.keys()):
                        statstemp = dict()
                        category = self.categories[categoryidx][testidx]
                        for uniquegroup in np.unique(category):
                            idx=np.where(category==uniquegroup)
                            try:
                                stats[categoryidx][uniquegroup].set_values(np.array([]),all_stats(label[idx],score[idx],self.cutoff),np.array([]))
                            except:
                                statstemp[uniquegroup] = Statistics(np.array([]),all_stats(label[idx],score[idx],self.cutoff),np.array([]))
                                stats[categoryidx] = statstemp

        self.stats=stats

    def optimize(self,opt):
        modelcopy = self.copy(normalizing=False,filtering=False)
        return opt.run(modelcopy)

