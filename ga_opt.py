#! /usr/bin/env python

'''
Title:
    General Genetic Algorithm (GA) for optimization tasks.

Date/Version:
    2011 12 07 v1.1

Author:
    Nick Trefiak

Description:
    Uses a binary chromosome to represent binary, integer and real values.
    Resolution for integer and real values should be course (e.g. 8bit or less).
    Fitness function should be provided externally.
    Chromo translation between binary and other forms not provided.
    Only performs minimization - if maximization is desired, then alter fitness
    function appropriately.

Usage:
    Import module into wrapper script.
    e.g. from general_GA import *
    or   from general_GA import GA
'''

__all__ = ['GA']

import logging
import random as r
import numpy as np

class Chromo:
    
    '''Store and perform operations on one chromosome.'''
    
    def __init__(self, nbit, bias, chromo=None):
        '''Initialize instance.'''
        if chromo is None:
            self.chromo = []
        else:
            self.chromo = chromo
        self.nbit = nbit
        self.bias = bias
        self.fitness = None

    def generate(self):
        '''Generate chromosome using the BIAS specified.'''
        self.chromo = [0]*self.nbit
        for i in range(self.nbit):
            if r.uniform(0, 1) > self.bias:
                self.chromo[i] = 1

    def mutate(self):
        '''Mutate a single random bit.'''
        bit = r.randint(0, self.nbit-1)
        self.chromo[bit] = int(not(self.chromo[bit]))
        return self.chromo

    def mutate_stepwise(self):
        '''Mutate each bit stepwise with probability 1/NBIT.'''
        prob = 1.0/float(self.nbit)
        for i in range(self.nbit):
            if r.uniform(0, 1) < prob:
                self.chromo[i] = int(not(self.chromo[i]))
        return self.chromo

    def mutate_stepwise_ensure(self):
        '''
        Mutate each bit stepwise with probability 1/NBIT,
        ensuring that at least 1 bit was mutated.
        '''
        mut = [0]*self.nbit
        prob = 1.0/float(self.nbit)
        while sum(mut) < 1:
            for i in range(self.nbit):
                if r.uniform(0, 1) < prob:
                    mut[i] = 1
        for i in range(self.nbit):
            if mut[i] == 1:
                self.chromo[i] = int(not(self.chromo[i]))
        return self.chromo

    def crossover(self, other):
        '''
        Perform a single point crossover with probability CR.
        Set bit selection ranges such that a crossover always occurs.
        Returns two children.
        '''
        bit = r.randint(1, self.nbit-2)
        child_a = self.chromo[0:bit] + other.chromo[bit:]
        child_b = other.chromo[0:bit] + self.chromo[bit:]
        return child_a, child_b

    def crossover_2p(self, other):
        '''
        Perform two point crossover with probability CR.
        Set bit selection ranges such that crossover always occurs.
        '''
        bit1 = r.randint(1, self.nbit-3)
        bit2 = r.randint(bit1, self.nbit-2)
        child_a = (self.chromo[0:bit1] + other.chromo[bit1:bit2] +
                  self.chromo[bit2:])
        child_b = (other.chromo[0:bit1] + self.chromo[bit1:bit2] +
                  other.chromo[bit2:])
        return child_a, child_b

    def simple_fitness(self):
        '''Use sum of chromosome as fitness for testing.'''
        self.fitness = sum(self.chromo)

    def target_fitness(self):
        '''Test optimization to a target chromo.'''
        targ = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        self.fitness = sum([abs(self.chromo[i]-targ[i]) for i in range(NBIT)])

class Population:
    
    '''Holds all individuals in one generation.'''
    
    def __init__(self, nbit, bias, npop, nway, selection_rate, fitfunc):
        self.pop = None
        self.nbit = nbit
        self.bias = bias
        self.npop = npop
        self.nway = nway
        self.selection_rate = selection_rate
        self.fitfunc = fitfunc
    
    def populate(self, parents = None, inipop = None):
        '''
        Create population of NPOP individuals.
        Pass list of parent chromo strings for generations > 0.
        '''
	
	if inipop != None:
		#print inipop.shape[0] , self.npop , inipop.shape[1] , self.nbit
		if inipop.shape[0] != self.npop or inipop.shape[1] != self.nbit:
	
			print 'Initial population dimension mismacth, new population is generated'
			inipop = None
		else:
			print 'Using initial population'
		
	if parents == None:
            		#for first generation - create random chromos
            		self.pop = []
            		for i in range(self.npop):
				if inipop == None:
					new_chromo = Chromo(self.nbit,self.bias)
				else:
					new_chromo = Chromo(self.nbit,self.bias,chromo=inipop[i])
				
				new_chromo.generate()
				new_chromo.fitness = self.fitfunc.fit(new_chromo.chromo)
                		self.pop.append(new_chromo)
                		#logging.info('chromo: %s %s %s', i, new_chromo.chromo,
            			#    new_chromo.fitness
        else:
            		#for all other generations - chromos come from children of previous
            		#generation, a.k.a parents of current generation
            		self.pop = []
            		for i in range(self.npop):
                		new_chromo = Chromo(self.nbit, self.bias, parents[i])
                		new_chromo.fitness = self.fitfunc.fit(new_chromo.chromo)
                		self.pop.append(new_chromo)
                		#logging.info('chromo: %s %s %s', i, new_chromo.chromo,
                		#             new_chromo.fitness)
	
    
    def sort_fit(self, individuals):
        '''Sort a given set of individuals by their fitness.'''
        fit = np.zeros((len(individuals)))
        for i in range(len(individuals)):
            fit[i] = individuals[i].fitness
        order = fit.argsort()
        ind_sort = []
        for i in range(len(individuals)):
            ind_sort.append(individuals[order[i]])
        return ind_sort

    def selection(self):
        '''
        N-way tournament selection based on fitness.
        Most fit competitor is selected with probability SR.
        '''
        #choose nway competitors at random from pop
        compindex = []
        for i in range(self.nway):
            compindex.append(r.randint(0, self.npop-1))
        competitors = []
        for i in range(self.nway):
            competitors.append(self.pop[compindex[i]])
        #sort competitors by fitness
        competitors_sort = self.sort_fit(competitors)
        #determine which competitor to return
        if r.uniform(0, 1) < self.selection_rate:
            return competitors_sort[0]
        else:
            index = r.randint(1, self.nway-1)
            return competitors_sort[index]


class GA:

    '''performs evolutionary algorithm on Population objects'''
    
    def __init__(self, fitfunc, nbit, npop=100, ngen=100, nway=8, elitism=True,
                 nelite=1, target=0.0, selection_rate=0.9, crossover_rate=0.7,
                 bias=0.5,inipop=None,runfolder=None):
        '''
        fitfunc - object which calculates the fitness of a chromosome,
                  which return the fitness as a float
        nbit - number of bits in chromo
        npop - number of chromos in one generation
        ngen - max number of generations to perform
        nway - tournament selection on n chromos
        target - stopping criterion for fitness
        selection_rate - selection rate
        crossover_rate - crossover rate
        bias - bias for initial chromo generation, range 0 to 1
        '''
        self.fitfunc = fitfunc
        self.nbit = nbit
        self.npop = npop
        self.ngen = ngen
        self.nway = nway
        self.elitism = elitism
        self.nelite = nelite
        self.target = target
        self.selection_rate = selection_rate
        self.crossover_rate = crossover_rate
        self.bias = bias
	self.inipop = inipop
	self.runfolder = runfolder
	self.best = self.Evolve()
    
    def Evolve(self):
        #start generation counter
        N = 0
        #create list to holds all generations
        gen = []
        #set very bad initial best fitness
        bestfitness = self.target + 1e6
        #perform evolution while conditions are not met
        while N < self.ngen and bestfitness > self.target:
            #logging.info('generation %s', N)
            #for the first generation
            print "Running generation",N+1,"..."
	    if N == 0:
                #create Population object
                gen.append(Population(self.nbit, self.bias, self.npop, self.nway,self.selection_rate, self.fitfunc))
                #fill population
                gen[-1].populate(inipop=self.inipop)
            #for all other generations
            else:
                #create Population object
                gen.append(Population(self.nbit, self.bias, self.npop, self.nway,self.selection_rate, self.fitfunc))
                #fill population
                gen[-1].populate(parents)

            #find best individual
            best = gen[-1].sort_fit(gen[-1].pop)

	    if self.runfolder != None:
	    	np.savetxt(self.runfolder + '/iniPopulation.csv',np.array([chr.chromo for chr in gen[-1].pop]),fmt='%s')
	    else:
		np.savetxt('iniPopulation.csv',np.array([chr.chromo for chr in gen[-1].pop]),fmt='%s')
	    
	    #set best fitness for current generation
            bestfitness = best[0].fitness
            #logging.info('    fitness %s', best[0].fitness)
            #logging.info('    best chromo %s', best[0].chromo)

            #create children from current population
            children=[]

            '''Perform elitism step - copy best nelite individuals unchanged
            into next generation.'''
            if self.elitism == True:
                for i in range(self.nelite):
                    children.append(best[i].chromo)
            '''perform crossover OR mutation, filling up pop as you go'''
            while len(children) < self.npop:
                if r.uniform(0, 1) < self.crossover_rate:
                    parent_a = gen[-1].selection()
                    parent_b = gen[-1].selection()
                    child_a, child_b = parent_a.crossover(parent_b)
                    children.append(child_a)
                    children.append(child_b)
                else:
                    parent_a = gen[-1].selection()
                    child = parent_a.mutate()
                    children.append(child)
            #assign children of current generation as parents of next generation
            #truncate in case more than npop children were created above
            parents = children[0:self.npop]    
            N += 1
	    print "Fitness = ", bestfitness 
        return gen[-1].sort_fit(gen[-1].pop)[0]


class simple_fitness:
    def __init__(self,multipleir):
    	self.multipleir = multipleir
    
    def fit(self,chromo):
	'''Use sum of chromosome as fitness for testing.'''
    	return sum(chromo)*self.multipleir

class target_fitness:

    def __init__(self):
    	pass
    '''Test optimization to a target chromo.'''
    def fit(self,chromo):
     	targ = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    	return sum([abs(chromo[i]-targ[i]) for i in range(len(chromo))])

def test():
    '''Test that the classes, methods, etc. work properly.'''
    logging.basicConfig(filename='test.log', level=logging.DEBUG)
    g = GA(fitfunc = simple_fitness(1), nbit = 20, npop = 50, bias = 0.5)
    print g.best.chromo, 1-g.best.fitness
    logging.info('best final individual: %s', g.best.__dict__)


if __name__ == '__main__':
    test()
