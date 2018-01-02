'''
Created on Oct 15, 2011

@author: mllamosa
'''
from __future__ import print_function
import numpy as np
from math import exp, sin, cos, log, sqrt, pi
from time import time
#from qsar import *
import scipy as sc
from scipy.stats import linregress
from sklearn.metrics import roc_curve as roc, auc, precision_recall_curve



LOGO='''
                                                                                  ,(((((/
                                                                                                                                                                 .((#&&&@%@&@
                                                                                                                                                                                                                                               (#%@%%&@@&@@@@@(*%.
                                                                                                                                                                                                                                                                                                                           ,(#%&&@@@@&&@&@*.
                                                                                                                                                                                                                                                                                                                                                                                                       (#%&&@@@@@@&@
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 .%##%&&&&@@&@@,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ,###%%%&&&@@@@@@@.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            .((#&&%%&&&&@@@&&&@@#
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              ((#%%&&   /%  ,&&&@@@
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             .#%%&&,    .,@&&     &&@@@
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          *#%%%,     ,&%&&&(      (@@%
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     .%%,      &&%&&&&&.        (@,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              #,   ,,(%#%#%%&&&&@.          @
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ###%&%%&%&%%&&&&&&&&.          .
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          /#(##%&%%&&&&&&&&&&..         .
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             /(((##%%%&&&&&%%&&%*,..       . .
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              *#%&&&&&&&&&&&@@@@&*..     ... ...
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ,%%%&%%%&&@@@@@@&&&&@&(,,,.....,,,.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      *##%%%%&&&&@&/.  /(###%%%%%%((,/#.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           /(#%%%%%%&(                *@&    &
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ,*(####&%&&/                      %%    /&,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   ./(#(((#&&&&(                .          .%      %
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       .(%(//(%(%%%%.          ..  .                 %.      %             .
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      .(#%#%%&%%,                  .,. *.. . .   .     *%%,%##/,% ..,...........
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 (&&%%&(.                       .. .    .           (%#%/........%%####%/      .
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              ...                   . * ..                   .
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              '''

class Lookup(dict):
    """
    a dictionary which can lookup value by key, or keys by value
    """
    def __init__(self, items=[]):
        """items can be a list of pair_lists or a dictionary"""
        dict.__init__(self, items)

    def get_key(self, value):

        """find the key(s) as a list given a value"""
        return [item[0] for item in self.items() if item[1] == value]

    def get_value(self, key):
        """find the value given a key"""
        return self[key]

def divisorsn():
        return set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def gaussian_kernel_(x1,x2,sigma):

    return exp(-np.linalg.norm(x1-x2)/2*sigma**2)

def gaussian_kernel(atomposition,dot,sigma):
    '''Apply Gaussian kernel to pair of vector and a point'''
    return np.exp(-(sum((atomposition-dot)**2)/(2*sigma**2)))

def vector2point_distance(vector,point):

    return np.sum((vector-point)**2,axis=1)**0.5

def sphe2cart(vector,ro,cita,sigma):

    sinSg=sin(sigma)
    cosSg=cos(sigma)
    cosCt=cos(cita)
    sinCt=sin(cita)
    x=vector[0]+ro*sinSg*cosCt
    y=vector[1]+ro*sinSg*sinCt
    z=vector[2]+ro*cosSg
    return x,y,z

def descriptor_writer(file,descriptordict):

    f=open(file, 'w')
    #f1=open(file+'_index','w')
    header=True
    for mol in  sorted(descriptordict.keys()):
        headerprop=True
        #print descriptordict[mol]
        for property in sorted(descriptordict[mol].keys()):
            if headerprop:
                dataentry=str(mol).split('/')[len(str(mol).split('/'))-1]
                #indexentry=str(mol).split('/')[len(str(mol).split('/'))-1]
                lagentry='Mol ID'
                headerprop=False

            #print descriptordict[mol][property]
            #Beware of dictionary dimensions  sorted(descriptordict[mol][property][0].keys())
            for lag in sorted(descriptordict[mol][property].keys()):
                dataentry = dataentry + ',' + '%4.6f'%(descriptordict[mol][property][lag])
                #dataentry = dataentry + ',' + str(descriptordict[mol][property][0][lag])
                #indexentry = indexentry + ',' + str(descriptordict[mol][property][1][lag])
                #lagentry = lagentry + ',' + 'auto_' + property + '_' + str(lag)
                lagentry = lagentry + ',' + '3DZern_' + property + '_' + str(lag)

        if header:
            print >> f,lagentry
            print >> f,dataentry

            #print >> f1,lagentry
            #print >> f1,indexentry
            header=False
        else:
            print >> f,dataentry
            #print >> f1,indexentry

    f.close()
    #f1.close()

def iso_writer(file,surface):
    pass

def cliffmatrix_write(data,filename,cutoff=None):
    labels = data[:,-1]
    data = data[:,0:-1]
    datasize = data.shape[0]
    file = open(filename,'w')
    for idx1 in range(datasize):
            vector = []
            for idx2 in range(idx1+1,datasize):
                denominator = 1-sum(data[idx1,:]*data[idx2,:])/(sum(data[idx1,:]**2) + sum(data[idx2,:]**2) - sum(data[idx1,:]*data[idx2,:]))
                if denominator != 0 :
                    numerator = abs(labels[idx1]-labels[idx2])
                    if numerator/denominator > 0:
                        vector.append(np.log(abs(labels[idx1]-labels[idx2])/denominator))
                    else:
                        vector.append(0)
                else:
                    vector.append(0)
            print >> file," ".join(map(str,vector))
    file.close()

def cliff_value(input):
    import numpy as np
    idx1=0
    idx2=1
    data,labels=input[0]
    value= round(np.log(abs(labels[idx1]-labels[idx2])/(1-np.sum(data[idx1]*data[idx2])/(np.sum(data[idx1]**2) + np.sum(data[idx2]**2) - np.sum(data[idx1]*data[idx2])))),3)
    return value

def cliffmatrix_write_pp(data,filename,cutoff=None):
    import math, sys, time
    import pp
    ncpus = 8
    labels = data[:,-1]
    data = data[:,0:-1]
    datasize = data.shape[0]
    file = open(filename,'w')
    inputvalues = []
    start_time = time.time()
    for idx1 in range(datasize):
                for idx2 in range(idx1+1,datasize):
                     inputvalues.append([[data[idx1],data[idx2]],[labels[idx1],labels[idx2]]])
                     if len(inputvalues) > 1000:
                        ppservers = ()
                        job_server = pp.Server(ncpus, ppservers=ppservers)
                        inputvalues = tuple(inputvalues)
                        #print "One input",inputvalues[0]
                        jobs = [(input, job_server.submit(cliff_value,(inputvalues,),(),("math","numpy",))) for input in inputvalues]
                        #for inputvalues,job in jobs:
                            #print "Input",inputvalues[1][0]-inputvalues[1][0],"output",job()
                        print >> file," ".join(map(str,([job() for inputvalues, job in jobs])))
                        inputvalues = []
                print ("Time elapsed: ", time.time() - start_time, "s")

    print ("Time elapsed: ", time.time() - start_time, "s")
#<<<<<<< local



#=======
#>>>>>>> other

def tanimoto(data):
         data = data[:,0:-1]
         datasize = data.shape[0]
         similvector = [round(abs(sum(data[idx1,:]*data[idx2,:])/(sum(data[idx1,:]**2) + sum(data[idx2,:]**2) - sum(data[idx1,:]*data[idx2,:]))),3) for idx1 in range(datasize) for idx2 in range(idx1+1,datasize)]
         similmatrix = np.zeros((datasize,datasize))
         index = 0
         for idx1 in range(datasize):
                 for idx2 in range(idx1+1,datasize):
                         similmatrix[idx1,idx2] = similvector[index]
                         index += 1
         return similmatrix

def cliffmatrix(data,cutoff=None):

    labels = data[:,-1]
    data = data[:,0:-1]
    datasize = data.shape[0]
    print ("Computing cliff vector")
    #salivector = [abs(labels[idx1]-labels[idx2])/(1-sum(data[idx1,:]*data[idx2,:])/(sum(data[idx1,:]**2) + sum(data[idx2,:]**2) - sum(data[idx1,:]*data[idx2,:]))) for idx1 in range(datasize) for idx2 in range(idx1+1,datasize)]
    salimatrix = np.zeros((datasize,datasize))
    index = 0
    for idx1 in range(datasize):
        for idx2 in range(idx1+1,datasize):
                                denominator = 1-sum(data[idx1,:]*data[idx2,:])/(sum(data[idx1,:]**2) + sum(data[idx2,:]**2) - sum(data[idx1,:]*data[idx2,:]))
                                if denominator != 0 :
                                        numerator = abs(labels[idx1]-labels[idx2])
                                        if numerator/denominator > 0:
                                                salimatrix[idx1,idx2] = np.log(abs(labels[idx1]-labels[idx2])/denominator)
                                        else:
                                                salimatrix[idx1,idx2] = 0
                                else:
                                        salimatrix[idx1,idx2] = 0
    #salimatrix = salimatrix + salimatrix.T
    if not cutoff:
        Number = 200
        vector = np.sort(np.ravel(salimatrix))[::-1]
        #Number = len(vector)
        if len(vector) > Number:
                    print ("Graph with only",Number)
                    cutoff = vector[Number]
                    #print vector[0:100]
        else:
                    print ("Graph with all numbers",len(vector))
                    cutoff = vector[-1]
        print ("Cliff cut-off", cutoff)

        if cutoff:
                cliffcutoff =  np.zeros(salimatrix.shape)
                indexes = np.where(salimatrix>cutoff)
                cliffcutoff[indexes]=1
                return cliffcutoff,indexes
        else:
                return salimatrix,[]

def cliff_graph(indexes,filename,labelsx=None,labelsy=None):
    import matplotlib.pyplot as plt
    import networkx as nx
    labeldict=({})
    DG=nx.DiGraph()
    print ("Generating nodes")
    for idx in range(len(indexes[0])):
        DG.add_weighted_edges_from([(indexes[0][idx],indexes[1][idx],1.5)])
        labeldict[indexes[0][idx]]=labelsy[indexes[0][idx]]
        labeldict[indexes[1][idx]]=labelsx[indexes[1][idx]]
        print ('Adding edge with nodes',labelsy[indexes[0][idx]],labelsx[indexes[1][idx]])
    nx.draw_spring(DG,font_size=8,alpha=0.7,node_size=300,labels=labeldict,width=0.5,with_labels=True)
    plt.savefig(filename + '_graph.pdf')

'''
def cliffmatrix_(data,cutoff=None):

    cliffmatrix = np.matrix([abs(data[:,-1]-label) for label in data[:,-1]])/(1-tanimoto(data[:,0:-1]))

    if cutoff:
        cliffcutoff =  np.zeros(cliffmatrix.shape)
        cliffcutoff[np.where(cliffmatrix>cutoff)]=1
        return cliffcutoff
    else:
        return cliffmatrix
'''
def dist(surface):
    distanceNill=True
    for dot in surface:
            if distanceNill:
                    distance=vector2point_distance(surface,dot)
                    distanceNill=False
            else:
                    #print surface.shape,dot.shape,dot,distance
                    distance=np.hstack((distance,vector2point_distance(surface,dot)))

        #print "Surface shape",surface.shape
    #print "Distance length", distance.shape
    return np.argsort(distance),np.sort(distance)

#def product(prop):
#
#
#    #t0=time()
#    product=np.array(prop*prop[0])
#    for prodidx in range(1,prop.shape[0]):
#        product=np.hstack((product,prop*prop[prodidx]))
#
#    #print time()-t0
#    return product

def product_(prop):

    #t0=time()
    size=prop.shape[0]
    product=np.zeros(size*size)
    print (product.shape)
    index=0
    for i in range(0,size):
        for j in range(i+1,size):
            #print i*size + j-1
            product[i*size + j-1]=prop[i]*prop[j]

        #print time()-t0
    return product

#def productfrom

def autocorr(surface,propertysurface,minlag,maxlag,step):


        #t0=time()
        surfacelen=surface.shape[0]
        distanceidx,distance=dist(surface)
        productproperty=product(propertysurface)
        lag=minlag
        maxcorrel=Lookup()
        maxidx=Lookup()
        while lag <= maxlag:
                lower=lag-step/2
                upper=lag+step/2
                if lag==minlag:
                    idxs=(np.nonzero(distance<=upper) and np.nonzero(distance>lower))[0]
                else:
                    idxs=(np.nonzero(distance<=upper))[0]

                idxskeep=(np.nonzero(distance>upper) )[0]
                propidxs=distanceidx[idxs]

                #Update distance and idxs
                distance=distance[idxskeep]
                distanceidx=distanceidx[idxskeep]

                if len(propidxs)>0:
                    #max=np.max(productproperty[propidxs])
                    protemp=productproperty[propidxs]
                    idx=np.argmax(protemp)
                    max=protemp[idx]
                    corridx=propidxs[idx]
                    i,j=np.divide(corridx,surfacelen)+1, np.remainder(corridx,surfacelen)+1
                    maxidx[lag]=str(i) + '_' + str(j)
                    maxcorrel[lag]=max
                else:
                #max=0
                #maxidx='na_na'
                    #print max
                    maxcorrel[lag]=0
                    maxidx[lag]='na_na'

                lag+=step
            #print time()-t0
        return maxcorrel,maxidx

def crosscorr(surface,propertysurface1,propertysurface2,minlag,maxlag,step):
        distanceNill=True
        for dot in surface:
            if distanceNill:
                distance=vector2point_distance(surface,dot)
                distanceNill=False
            else:
                #print surface.shape,dot.shape,dot,distance
                distance=np.vstack((distance,vector2point_distance(surface,dot)))

        autolen=propertysurface.shape[0]
        print (distance.shape)
        lag=minlag
        maxcorrel=Lookup()
        maxidx=Lookup()
        while lag <= maxlag:
            lower=lag-step/2
            upper=lag+step/2
            pairproduct=Lookup()
            pairarray=np.zeros((autolen,autolen))
            for i in range(autolen):
                #propertytemp=propertysurface[i]
                idxs=[idx for idx in range((i+1),autolen) if distance[i,idx]>=lower and distance[i,idx]<upper]
                for j in idxs:
                    key=str(i) + '_' + str(j)
                    pairproduct[key]=max(propertysurface1[i]*propertysurface2[j],propertysurface1[j]*propertysurface2[i])
                    pairarray[i,j]=pairproduct[key]
            max=np.max(pairarray)


            keymax=pairproduct.get_key(max)
            i,j=keymax[0].split('_')
            maxcorrel[lag]=pairarray[i,j]
            maxidx[lag]=keymax
            lag+=step
        return maxcorrel,maxidx

def read_descriptors(datafile):
    try:
       data = np.loadtxt(datafile, dtype='string', delimiter=',', skiprows=0)
    except MemoryError:
        try:
            fdata = open(datafile,'rb')
            data=fdata.readlines()
            #fdata.close()
            data = np.array([dat.split(",") for dat in data])
        except MemoryError:
            print ("Not enough memory to load entire data, trying to work with a fraction of the 100,000 instances")
            import random
            try:
                ids = random.sample(range(1,len(fdata.readlines())),100000)
                ids.append(0)
                ids.sort()

                data = np.array(data)[np.array(ids)]
                data = np.array([dat.split(",") for dat in data])
            except MemoryError:
                print ("Failed loading data, working with the first 1000 samples")
                fdata = open(datafile,'rb')
                data=np.array([fdata.readline().split(",") for idx in range(1000)])
            fdata.close()
    varnames, insnames, datared = data[0,1:],data[1:,0],data[1:,1:]
    matrix=[map(float,vector) for vector in datared]

    matrix=np.array(matrix)
    return matrix,varnames, insnames

def pre_libsvmdata(data):
    matrix=[map(float,vector[0:-1]) for vector in data]
    label=map(float,data[:,-1])
    return label,matrix

def pre_libsvmpara(parameters):
    try:
        c=str(parameters['regularization'])
    except:
        c='1'
    try:
        g=str(1/parameters['regularization']**2)
    except:
        g='1'

    return '-s 3 -c ' + c + ' -g ' + g

def surface3D(X,Y,Z,varnames,file=None,show=None):

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from matplotlib.pyplot import gca, figure, xticks, yticks, colorbar, savefig, savefig,xlabel, ylabel

    figure()
    ax = gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
            linewidth=0, antialiased=False)

    #ax.zlim(-1.01, 1.01)
    ax.set_zlim3d(np.min(Z)-(np.max(Z)-np.min(Z))/10,np.max(Z)+(np.max(Z)-np.min(Z))/10)
    xlabel(varnames[0],fontsize=16)
    ylabel(varnames[1],fontsize=16)
    ax.w_zaxis.set_major_locator(LinearLocator(10))
    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    colorbar(surf, shrink=0.5, aspect=5)
    savefig(file + '_response_surface.pdf',format='pdf')

def heatmap(data,labels=None,file=None):
    from  matplotlib.pyplot import imsave,imshow,savefig,cm,colorbar,figure,xticks,yticks
    #import Image
    data[np.where(np.isnan(data))]=0
    fig = figure()
    imshow(data,cmap=cm.jet,interpolation='nearest',alpha=0.85,origin='lower',extent=[0,len(labels),0,len(labels)])
    ax = fig.add_subplot(111)
    ax.set_xlabel('Functional property  --->')
    ax.set_ylabel('Functional property  --->')
    ax.set_title('SALI heatmap')
    #labels = np.hstack(("",labels))
    ax.set_xticklabels("")
    if len(labels)<21:
        ax.set_yticks(np.arange(0.5,len(labels),1))
    else:
        ax.set_yticks([])

    ax.set_yticklabels(labels,fontsize=6)
    #xticks(np.arange(0.5,len(labels)+0.5,1), labels,fontsize=6)
    #yticks(np.arange(1.5,len(labels)+1.5,1), labels,fontsize=6)
    cb = colorbar()
    savefig(file + '_heatmap.pdf',format='pdf')

    #import scipy
    #scipy.misc.imsave('outfile.jpg', data)
    #im = Image.fromarray(data)
    #im.save(file + '_heatmap33',"JPEG")

#local

def scatter2D_plot(x,y,flag,file=None,show=None,outliers=False):
#=======
#def scatter2D_plot(x,y,flag,file=None,show=None):
    #>>>>>>> other

    from matplotlib.pyplot import scatter,legend,plot,xlim,ylim, xlabel,ylabel,title, grid,savefig, figure, text
    from matplotlib import rc
    rc('xtick', labelsize=18)
    rc('ytick', labelsize=18)
    figure()
    #x,y=remap_data(self.data[self.partition.train,-1],[self.normalization.parameters[0][-1],self.normalization.parameters[1][-1]],self.normalization.method), self.train

    if outliers:
        ids = np.where(abs(x-y)<10*np.sqrt(np.mean((x-y)**2)))
        print ("Removing",len(x)-len(x[ids]),"outliers")
    x=x[ids]
    y=y[ids]
    min=np.min(np.hstack((x,y)))
    max=np.max(np.hstack((x,y)))
    #print x.min(),y.min()
    scatter(x,y,s=60,facecolor='white',edgecolor='blue',label='Data points',)
    plot([min-abs(max)/10,max+abs(max)/10],[min-abs(max)/10,max+abs(max)/10],'k',label='Perfect fit')
    #for idx in range(len(x)):
    #    text(x[idx],y[idx],idx,fontsize=12)
    gradient, intercept, r_value, p_value, std_err=linregress(x,y)
    yval=np.array([-999999,999999])*gradient+intercept
    plot([-999999,999999],yval,color='red',ls='--',label='Best fit',linewidth=2.0)
    b=r_value
    #xlim(min-abs(max)/10,max+abs(max)/10)
    #ylim(min-abs(max)/10,max+abs(max)/10)
    xlim(np.min(x)-abs(max)/20,max+abs(max)/10)
    ylim(np.min(x)-abs(max)/20,max+abs(max)/10)



    xlabel('Actual',size=18)
    ylabel('Predicted',size=18)
#<<<<<<< local
#    title('R$^{2}$ = ' + "%4.3f" %b**2 ,fontsize=18)
#=======
    title(flag + r' $R^2=$ ' + "%4.3f" %b**2 + r' $N=$'+"%1.0f" %len(x))
#>>>>>>> other
    legend(loc="upper left",prop={'size':14},scatterpoints=1,numpoints=2)
    #grid(True,alpha=0.3)
    savefig(file + '_' + flag +'_plot.pdf',format='pdf')

def auc_plot(x,y,flag,file=None,cutoff=None,show=None,linestyle='rx-.',include_baseline=True,equal_aspect=True):
    """ Method that generates a plot of the ROC curve
            Parameters:
                title: Title of the chart
                include_baseline: Add the baseline plot line if it's True
                equal_aspect: Aspects to be equal for all plot
    """
    try:
        fpr, tpr, thresholds_roc = roc(x,y)
        if not cutoff:
            precision, recall, thresholds = precision_recall_curve(x,y)
            precision[np.where(precision==0)]=0.000000001
            recall[np.where(recall==0)]=0.000000001
            F_score=2*(precision*recall)/(precision+recall)
            aucvalue,cutoff=round(auc(fpr, tpr),3),round(thresholds[np.where(F_score==max(F_score))][0],3)

        TPR=round(tpr[np.where((thresholds_roc-cutoff)==min(thresholds_roc-cutoff))][0],5)
        FPR=round(fpr[np.where((thresholds_roc-cutoff)==min(thresholds_roc-cutoff))][0],5)
        import pylab
        from matplotlib import pyplot
        pyplot.figure()
        pylab.clf()
        pylab.plot([x1 for x1 in np.hstack((0,fpr))], [y1 for y1 in np.hstack((0,tpr))],color='red',linewidth=8.0)
    #    pylab.plot([x1 for x1 in precision], [y1 for y1 in recall],color='blue',linewidth=2.0)
        if include_baseline:
            pylab.plot([0.0,1.0], [0.0,1.0],'k--')
            pylab.ylim((0,1))
            pylab.xlim((0,1))
            pylab.xticks(pylab.arange(0,1.1,.1),fontsize=16)
            pylab.yticks(pylab.arange(0,1.1,.1),fontsize=16)
            pylab.grid(True,alpha=0.5)
            if equal_aspect:
                    cax = pylab.gca()
                    cax.set_aspect('equal')
        #pylab.xlabel('1 - Specificity(red)/Precision(blue)')
        pylab.xlabel('1 - Specificity',fontsize=16)
        pylab.ylabel('Sensitivity',fontsize=16)

        if 'Train' == flag or 'Validation' == flag:
                    pylab.plot(FPR,TPR,'o',color='black')
                    pylab.figtext(FPR + 0.08,TPR - 0.08, str(cutoff) + "(" + ",".join(map(str,[1-FPR,TPR])) + ")" )

        pylab.title(flag + ' AUC=' + "%4.3f" %aucvalue + ' N='+'%1.0f' %len(x))
        pyplot.savefig(file + '_' + flag +'_aucplot.pdf',format='pdf')

    except:
        print ("Failed to generate AUC plots")


def auc_plot_multi(XY,flag="",file="Noname",cutoff=None,show=None,linestyle='rx-.',include_baseline=True,equal_aspect=True):
         
         """ Method that generates a plot of the ROC curve
             Parameters:
                 title: Title of the chart
                 include_baseline: Add the baseline plot line if it's True
                 equal_aspect: Aspects to be equal for all plot
         """
         import pylab
         from matplotlib import pyplot
         pyplot.figure()
         pylab.clf()
         color_list = ["b","g","r","c","m","y","k","w"]
         colorid = 0
         for xy in XY:  
             if colorid > len(color_list):
                    colorid = 0
                
             x,y = xy[:,0], xy[:,1]
             fpr, tpr, thresholds_roc = roc(x,y)
             aucvalue = round(auc(fpr, tpr),3)
        
             pylab.plot([x1 for x1 in np.hstack((0,fpr))], [y1 for y1 in np.hstack((0,tpr))],color=color_list[colorid],linewidth=1.0)
         #   pylab.plot([x1 for x1 in precision], [y1 for y1 in recall],color='blue',linewidth=2.0)
             if include_baseline:
                 pylab.plot([0.0,1.0], [0.0,1.0],'k--')
             pylab.ylim((0,1))
             pylab.xlim((0,1))
             pylab.xticks(pylab.arange(0,1.1,.1),fontsize=10)
             pylab.yticks(pylab.arange(0,1.1,.1),fontsize=10)
                
             #pylab.grid(False,alpha=0.5)
             if equal_aspect:
                 cax = pylab.gca()
                 cax.set_aspect('equal')
             #pylab.xlabel('1 - Specificity(red)/Precision(blue)')
             pylab.xlabel('1 - Specificity',fontsize=10)
             pylab.ylabel('Sensitivity',fontsize=10)
 
             pylab.title(flag + ' AUC=' + "%4.3f" %aucvalue + ' N='+'%1.0f' %len(x))
             colorid += 1
         ax = pylab.gca()
         ax.yaxis.set_ticks_position('left')
         ax.xaxis.set_ticks_position('bottom')
         ax.get_yaxis().set_tick_params(direction='out')
         ax.get_xaxis().set_tick_params(direction='out')
         pyplot.savefig(file + '_' + flag +'_aucplot.pdf',format='pdf')
         pylab.show()



def scatterdens_plot(x,y,flag,file=None,show=None):
    import matplotlib.cm as cm
    from matplotlib.pyplot import setp,getp,scatter,axis,legend,hexbin,plot,xlim,ylim, xlabel,ylabel,title, grid,savefig, figure
    from matplotlib import pyplot as plt
    fig = figure()
    ax = fig.add_subplot(111)
    min=np.min(np.hstack((x,y)))
    max=np.max(np.hstack((x,y)))
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    gsize = len(x)/100
    plt.hexbin(x,y,gridsize=gsize,cmap=cm.jet,vmin=0,vmax=gsize/50)
    #plt.hexbin(x,y,bins='log')
    plt.axis([xmin, xmax, ymin, ymax])
    #for tick in ax.xaxis.get_major_ticks():
    #   tick.label.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
       tick.label.set_fontsize(16)

    cb = plt.colorbar()
    cl = getp(cb.ax, 'ymajorticklabels')
    setp(cl, fontsize=16)
    cb.set_label('counts',fontsize=16)
    gradient, intercept, r_value, p_value, std_err=linregress(x,y)
    b=r_value
    #title(flag + ' R^2= ' + '%4.3f' %b**2 + ' N= ' + '%1.0f' %len(x) + '\n')
    print (file + '_' + flag +'_plot.pdf')
    savefig(file + '_' + flag +'_plot.pdf',format='pdf')

def shortest_path(connectionmatrix_orig):
    #[i,i] nodes are 0
    #none connected nodes are inf
    #conected nodes are 1
    connectionmatrix = connectionmatrix_orig.copy()
    connectionmatrix[connectionmatrix == 0]=np.inf
    n=connectionmatrix.shape[0]

    for i in range(n):
        for j in range(n):
            if i==j:
                connectionmatrix[i,j]=0

    for k in range(n):
        connectionmatrix = np.minimum(connectionmatrix, np.add.outer(connectionmatrix[:,k],connectionmatrix[k,:]) )
    connectionmatrix[connectionmatrix==np.inf]=0
    return connectionmatrix.astype(int)
    #adj=connectionmatrix
    #for k in range(n):
    #    adj = np.minimum(adj, np.tile(adj[:,k].reshape(-1,1),(1,n)) + np.tile(adj[k,:],(n,1)))
    #return adj

def surface_vertex_area(surface,connection,property):

    len_surface=surface.shape[0]
    triad=np.zeros((len_surface,5))
    vector_center_polar=np.zeros(3)
    for idx in range(len_surface):
        prop=np.average(np.hstack((property[connection[idx,0]-1],property[connection[idx,1]-1],property[connection[idx,2]-1])))
        coord=np.vstack((surface[connection[idx,0]-1],surface[connection[idx,1]-1],surface[connection[idx,2]-1]))
        center=centroid(sphe2cart_matrix(vector_center_polar,coord))
        area=trian_area(sphe2cart_matrix(vector_center_polar,coord))
        #print center,area,prop
        triad[idx]=np.hstack((cart2sphe(vector_center_polar,center),area,prop))

    #print np.sum(triad[:,3])

    return triad


def trian_area(vertexes):
    sides=np.zeros(3)
    #print vertexes
    sides[0]=np.linalg.norm(vertexes[0]-vertexes[1])
    sides[1]=np.linalg.norm(vertexes[0]-vertexes[2])
    sides[2]=np.linalg.norm(vertexes[1]-vertexes[2])
    s=np.sum(sides)/2
    #print s*(s-sides[0])*(s-sides[1])*(s-sides[2])
    return (s*(s-sides[0])*(s-sides[1])*(s-sides[2]))**0.5




def centroid(coord):

    centroid=np.zeros(3)
    for dot in coord:
        centroid=centroid+dot

    #print "Len coord",     len(coord)
    return centroid/len(coord)

def sphe2cart_matrix(vector,matrix):
    cart_matrix=np.zeros(3)
    for coord in matrix:
        cart_matrix=np.vstack((cart_matrix,sphe2cart(vector,coord[0],coord[1],coord[2])))
    return cart_matrix[1:]

def sphe2cart(vector,ro,phi,cita):

    sinPh=sin(phi)
    cosPh=cos(phi)
    cosCt=cos(cita)
    sinCt=sin(cita)
    x=vector[0]+ro*sinPh*cosCt
    y=vector[1]+ro*sinPh*sinCt
    z=vector[2]+ro*cosPh
    return np.array([x,y,z])


def cart2sphe(center,vector):
    #print vector

    ro=((center[0]-vector[0])**2+(center[1]-vector[1])**2+(center[2]-vector[2])**2)**0.5
    if (vector[2]-center[2])>0:
        phi=np.arctan((((center[0]-vector[0])**2+(center[1]-vector[1])**2)**0.5)/(center[2]-vector[2]))
    elif (vector[2]-center[2])==0:
        phi=pi/2
    elif (vector[2]-center[2])<0:
        phi=pi + np.arctan((((center[0]-vector[0])**2+(center[1]-vector[1])**2)**0.5)/(center[2]-vector[2]))
    if phi<0:
        phi=pi + np.remainder(phi,pi)
    if phi>pi:
        phi=np.remainder(phi,pi)
    if (vector[0]-center[0])>0 and (vector[1]-center[1])>0 :
        cita=np.arctan((vector[1]-center[1])/(vector[0]-center[0]))
    elif (vector[0]-center[0])>0 and (vector[1]+center[1])<0 :
        cita=2*pi + np.arctan((vector[1]-center[1])/(vector[0]-center[0]))
    elif (vector[0]-center[0])==0 :
        if (vector[1]-center[1])>=0:
            cita=pi/2
        elif (vector[1]-center[1])<0:
            cita=-pi/2
    elif (vector[0]-center[0])<0 :
        cita=pi + np.arctan((vector[1]-center[1])/(vector[0]-center[0]))
    return ro,phi,cita

def normalized_surface(surface):
        #Get center of mass

        #print surface.shape
        centermass=centroid(surface)
        surface_cart_centered=surface-centermass

        center=np.zeros(3)
        spheric=np.zeros(3)
        for dot in surface_cart_centered:
            spheric=np.vstack((spheric,cart2sphe(center,dot)))
        #print spheric
        spheric[:,0]=spheric[:,0]/spheric[:,0].max()
        #print spheric
        surface_norm=spheric[1:,:]
        #print surface_norm.shape
        return surface_norm


def connect_atoms(atoms,dist_table):
    #print generate connection table for molecule
    ctable=[]
    #loop over only the upper triangle
    #build connection table based on atom types and bond cutoffs
    distance_table=dist_table()
    atomlist=sorted(atoms.keys())
    for key in sorted(atoms.keys()):
        atom=atoms[key]
        atomlist.remove(key)
        for key1 in atomlist:
            atom1=atoms[key1]
            if atom.name=='C' and atom1.name=='H' and distance_table[key][key1] < 1.2:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='C' and atom1.name=='C' and distance_table[key][key1] < 1.2:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='C' and atom1.name=='C' and distance_table[key][key1] < 1.7:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='C' and atom1.name=='N' and distance_table[key][key1] < 1.7:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='N' and atom1.name=='C' and distance_table[key][key1] < 1.7:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='C' and atom1.name=='O' and distance_table[key][key1] < 1.7:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='O' and atom1.name=='C' and distance_table[key][key1] < 1.7:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='H' and atom1.name=='N' and distance_table[key][key1] < 1.2:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='N' and atom1.name=='H' and distance_table[key][key1] < 1.2:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='H' and atom1.name=='O' and distance_table[key][key1] < 1.2:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='O' and atom1.name=='H' and distance_table[key][key1] < 1.2:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='C' and atom1.name=='P' and distance_table[key][key1] < 2.0:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='P' and atom1.name=='C' and distance_table[key][key1] < 2.0:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='C' and atom1.name=='Zn' and distance_table[key][key1] < 2.5:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='Zn' and atom1.name=='C' and distance_table[key][key1] < 2.5:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='N' and atom1.name=='Zn' and distance_table[key][key1] < 2.5:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='Zn' and atom1.name=='N' and distance_table[key][key1] < 2.5:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='O' and atom1.name=='Zn' and distance_table[key][key1] < 2.5:
                temp=[key,key1]
                ctable.append(temp)

            elif atom.name=='Zn' and atom1.name=='O' and distance_table[key][key1] < 2.5:
                temp=[key,key1]
                ctable.append(temp)

def connection_table(atoms,dist_table):
    from constants import constant
    #Print generate connection table for molecule
    ctable=[]
    botable=[]
    #Created a constant object
    const=constant()
    #Loop over only the upper triangle
    #build connection table based on atom types and bond cutoffs
    distance_table=dist_table()
    atomlist=sorted(atoms.keys())
    #Atom key are the atom number
    for key in sorted(atoms.keys()):
        atom=atoms[key]
        atomlist.remove(key)
        for key1 in atomlist:
            atom1=atoms[key1]
            bond=atom.name + '-' + atom.name1
            bondInv=atom.name1 + '-' + atom.name
            try:
                bondtypes=sorted(const.get_value('bond',bond).keys())
            except:
                bondtypes=sorted(const.get_value('bond',bondInv).keys())
            #Check for connection using the largest bond distance per atom pair
            for btype in bondtypes:
                if  distance_table[key][key1] <= bondtypes[btype] + 0.01:
                    ctable.append([key,key1])
                    botable.append(btype)
                    break

    return  ctable,botable

def load_info_cif(ciffile):
    from pbc_dmat_v2 import Structure

    crys=Structure()
    #Atoms should be a dictionary
    distancematrix,atoms=distance_from_cif(ciffile)
    #In addition to connection table also bond type should be provide
    ctable,bondtype=connect_atoms(atoms,distancematrix)

def reduce_corr_slow(data,label,corr_cutoff = 0.95):
    import copy
    corr_vector = []
    for vector in data:
        corr_vector.append(np.corrcoef(np.hstack((vector,label)).T)**2)

class reduce_corr():
    import copy
    def __init__(self):
        pass

    def apply(self,data,corr_cutoff = 0.99):
            corr = np.corrcoef(data.T)**2
            ids = np.argsort(-corr[-1,:-1])
            corr = corr[:-1,:-1]
            ids_ini = np.array(range(corr.shape[0]))
            ids = ids.tolist()
            ids_nonredundant = []
            while len(ids)>0:
                id_sel = ids[0]
                ids_nonredundant.append(ids[0])
                ids_to_remove = np.where(corr[id_sel,:]>corr_cutoff)
                ids_redundant =  ids_ini[ids_to_remove]
                for id in ids_redundant:
                    try:
                        ids.remove(id)
                    except:
                        pass
            ids_nonredundant.append(data.shape[1]-1)
            return np.sort(np.array(ids_nonredundant))

class minmax():

    def __init__(self,parameter=None):
        self.parameters=None

    def copy(self):
        return minmax()

    def map_data(self,data):
        max = np.amax(data,axis=0)
        min = np.amin(data,axis=0)
        #print "Inside maxmin",max,min
    #for idx in range(data.shape[1]):
        #    datanormalize[:,idx]=2*(data[:,idx]-min[idx])/(max[idx]-min[idx]) - 1
        datanormalize = 2*(data-min)/(max-min) - 1
        self.parameters = [max,min]
        return  datanormalize

    def remap_data(self,data):

        '''Check if normalizing only label (1-D array) or all data matrix features and labels '''

        if len(data.shape)==1 :
            max=self.parameters[0][-1]
            min=self.parameters[1][-1]
            #print "MaxMin",max,min
        else:
            max=self.parameters[0]
            min=self.parameters[1]
            #print "MaxMin",max,min
            #print data
        databacknormalize=((data+1)*(max-min))/2 + min
        #print databacknormalize
        return  databacknormalize

    def transmap_data(self,data):
        #print np.asmatrix(data).shape
    #print self.parameters
        if len(data.shape)==1 :
            max=self.parameters[0][-1]
            min=self.parameters[1][-1]
        else:
            max=self.parameters[0]
            min=self.parameters[1]

        #print "MaxMin",max,min
        #for idx in range(data.shape[1]):
        #    datanormalize[:,idx]=2*(data[:,idx]-min[idx])/(max[idx]-min[idx]) - 1
        datanormalize=2*(data-min)/(max-min) - 1
        return  datanormalize

    def copy(self):
        return  minmax()

    def get_parameters(self):
        return self.parameters

class maxall():

    def __init__(self,parameter=None):
        self.parameters=None

    def copy(self):
        return maxall()

    def map_data(self,data):
        max = np.max(data[:,:-1],axis=1).reshape(data.shape[0],1).astype(float)
        #print "Inside maxmin",max,min
        #for idx in range(data.shape[1]):
        #    datanormalize[:,idx]=2*(data[:,idx]-min[idx])/(max[idx]-min[idx]) - 1
        datanormalize = np.divide(data[:,:-1],max)
        self.parameters = [max]
        return np.hstack((datanormalize,data[:,-1].reshape(data.shape[0],1)))

    def remap_data(self,data):

        '''Check if normalizing only label (1-D array) or all data matrix features and labels '''
        if len(data.shape)==1 :
            databacknormalize = data
        else:
            max = np.max(data[:,:-1],axis=1).reshape(data.shape[0],1).astype(float)
            databacknormalize = data[:,:-1]*max
            databacknormalize = np.hstack((databacknormalize,data[:,-1].reshape(data.shape[0],1)))
        #print databacknormalize
        return  databacknormalize

    def transmap_data(self,data):
        #print np.asmatrix(data).shape
        #print self.parameters
        if len(data.shape)==1 :
            datanormalize = data
        else:
            max = np.max(data[:,:-1],axis=1).reshape(data.shape[0],1).astype(float)
            datanormalize=np.divide(data[:,:-1],max)
            datanormalize = np.hstack((datanormalize,data[:,-1].reshape(data.shape[0],1)))
        return datanormalize

    def copy(self):
        return  maxall()

    def get_parameters(self):
        return self.parameters

class remove_constant_values():
    def __init__(self):
        pass
    def apply(self,data):
        idxs=[idx for idx in range(data.shape[1]-1) if len(np.unique(data[:,idx]))>1]
        idxs = np.hstack((np.array(idxs),data.shape[1]-1))
        #print idxs
        return idxs

def pair_product_matrix(vector):
    return np.dot(vector.T,vector)

def pair_product_matrix_(vector):
    vectorsize=len(vector)
    #matrix=np.zeros((vectorsize,vectorsize))
    matrix=[]
    for i in range(vectorsize):
        matrix.append(vector*vector[i])
        #for j in range(i+1,vectorsize,1):
        #    matrix[i,j]=vector[i]*vector[j]
    return np.array(matrix)

def dtable2ctable_bondorder(dtable,atoms):
    from constants import Bond as BondConstant
    numberatoms=len(atoms)
    ctable=np.zeros((numberatoms,numberatoms))
    for symbol1idx in range(numberatoms):
        symbol1=atoms[symbol1idx]
        for symbol2idx in range(symbol1idx+1,numberatoms,1):
            symbol2=atoms[symbol2idx]
            try:
                bondTypes=BondConstant[symbol1  + "-" + symbol2]
            except:
                try:
                    bondTypes=BondConstant[symbol2  + "-" + symbol1]
                except:
                    bondTypes=({1:1})

            bondtypeskeys=sorted(bondTypes.keys(),reverse=True)
            if dtable[symbol1idx,symbol2idx]<=bondTypes[bondtypeskeys[-1]]:
            	for bondkey in bondtypeskeys:
                    if dtable[symbol1idx,symbol2idx]<=bondTypes[bondkey]+0.02:
                        ctable[symbol1idx,symbol2idx]=bondkey
                        break

    return ctable

def dtable2ctable(dtable,atoms):
    from constants import Bond as BondConstant
    #from time import time
    #t0 = time()
    numberatoms=len(atoms)
    ctable=np.zeros((numberatoms,numberatoms))
    #print dtable.shape
    for symbol1idx in range(numberatoms):
        symbol1=atoms[symbol1idx]
        for symbol2idx in range(symbol1idx+1,numberatoms,1):
            symbol2=atoms[symbol2idx]
            try:
                bondTypes=BondConstant[symbol1  + "-" + symbol2]
            except:
                try:
                    bondTypes=BondConstant[symbol2  + "-" + symbol1]
                except:
                    #print "Can't find",symbol1  + "-" + symbol2,'either',symbol2  + "-" + symbol1,'bond at',dtable[symbol1idx,symbol2idx],'distance'
                    bondTypes=({1:1})

            bondtypeskeys=sorted(bondTypes.keys(),reverse=True)
            #print [symbol1idx,symbol2idx]
            if  dtable[symbol1idx,symbol2idx]>0.5 and dtable[symbol1idx,symbol2idx]<=bondTypes[bondtypeskeys[-1]]:
                ctable[symbol1idx,symbol2idx]=1
    #print (time() - t0), "sec"
    return ctable

def all_stats(labels,scores,cutoff=None):
    if np.unique(labels).shape[0]>1:
      #print np.unique(labels)
      if np.unique(labels).shape[0]==2:
       #print len(np.unique(labels))
       fpr, tpr, thresholds_roc = roc(labels,scores)
       precision, recall, thresholds = precision_recall_curve(labels,scores)
       precision[np.where(precision==0)]=0.000000001
       recall[np.where(recall==0)]=0.000000001
       if len(thresholds)>1:
        F_score=2*(precision*recall)/(precision+recall)
        try:
            if cutoff == None:
                #cutoff=round(thresholds_roc[np.where(abs(tpr-0.95)==min((abs(tpr-0.95))))][0],5)
                #print "Calculation cutoff of maximum F-score"
                cutoff=round(thresholds[np.where(F_score==max(F_score))][0],5)
            else:
                print ("Using cutoff from previous calculations",cutoff)
            aucvalue=round(auc(fpr, tpr),3)
            cutoff_id = np.where(abs(thresholds_roc-cutoff)==min(abs(thresholds_roc-cutoff)))
            cutoff_pre_id = np.where(abs(thresholds-cutoff)==min(abs(thresholds-cutoff)))
            TPR=round(tpr[cutoff_id][0],5)
            FPR=round(fpr[cutoff_id][0],5)
            PRE=round(precision[cutoff_pre_id][0],5)
            stats=aucvalue,TPR,1-FPR,len(labels),PRE,cutoff
        except:
            stats=float('NaN'),float('NaN'),float('NaN'),len(labels),float('NaN'),float('NaN')
       else:
        stats=float('NaN'),float('NaN'),float('NaN'),len(labels),float('NaN'),float('NaN')
      else:
            gradient, intercept, r_value, p_value, std_err = linregress(labels,scores)
            std_err=np.std((labels-scores))
            stats=r_value**2,std_err,gradient,len(labels),p_value,float('NaN')

    else:
        stats=[float('NaN'),float('NaN'),float('NaN'),len(labels),float('NaN'),float('NaN')]
    return np.array(stats)

def calc_dist(p1,p2):
    return sqrt((p2[0] - p1[0]) ** 2 +
                     (p2[1] - p1[1]) ** 2 +
                     (p2[2] - p1[2]) ** 2)

def test():
#    data=csv2libsvmdata(datafile)
    from utilities import read_descriptors
    data,varnames,dataid = read_descriptors('/home/cmse/fer19x/data/csiro/dataset/csv/nanographene-dataset_features_topologicalindex_qsar_IP.csv')
    ids = range(data.shape[1]-1)
    #ids_1 = reduce_corr(data[:,:-1],data[:,data.shape[1]-1:data.shape[1]])
    #print ids_1
    #ids = reduce_corr(data[:,np.array(ids_1)],data[:,data.shape[1]-1:data.shape[1]])
    #print ids
    label = data[:,data.shape[1]-1:data.shape[1]]
    for i in range(5):
        ids = reduce_corr(data)
        data = data[:,np.array(ids)]
        #print ids



def generate_higher_ranks(table):

    aaromtable2=[]
    aaromtable3=[]

    aaromtable = [vector for vector in table if vector[2] == 'A' or vector[2] == 'ar']

    for vector in aaromtable:
                for vektor in aaromtable:
                        if vector[0]==vektor[0] and vector[1] != vektor[1]:
                                if [vector[1],vektor[1],'A'] not in aaromtable2 and [vektor[1],vector[1], 'A'] not in aaromtable2:
                                        aaromtable2.append([vector[1],vektor[1],'A'])
    for vector in aaromtable:
                for vektor in aaromtable:
                        if vector[0] == vektor[1] and vector[1] != vektor[0]:
                                 if [vector[1],vektor[0],'A'] not in aaromtable2 and [vektor[0],vector[1], 'A'] not in aaromtable2:
                                        aaromtable2.append([vector[1], vektor[0], 'A'])
    for vector in aaromtable:
                for vektor in aaromtable:
                        if vector[1] == vektor [0] and vector[0] != vektor [1]:
                                 if [vector[0],vektor[1],'A'] not in aaromtable2 and [vektor[1],vector[0], 'A'] not in aaromtable2:
                                        aaromtable2.append([vector[0], vektor[1], 'A'])
    for vector in aaromtable:
                for vektor in aaromtable:
                        if vector[1] == vektor[1] and vector[0] != vektor [0]:
                                if [vector[0],vektor[0],'A'] not in aaromtable2 and [vektor[0],vector[0], 'A'] not in aaromtable2:
                                        aaromtable2.append([vector[0], vektor[0], 'A'])

    for vector in aaromtable2:
                aaromtable3.append(vector)
    for vector in table:
                aaromtable3.append(vector)

    return np.array(aaromtable3)


def generate_bond_table(coords):

    return np.array([ [i,j,'s'] for i in range(coords.shape[0]) for j in range(i,coords.shape[0]-1) if calc_dist(coords[i,],coords[j,])<1.6])
    #return [ [i,j] for i in range(coords.shape[0]) for j in range(i,coords.shape[0]-1) if distance_vector[i*coords.shape[0] + j] < 1.6]


if __name__ == '__main__':
     test()
