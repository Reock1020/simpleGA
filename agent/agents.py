from abc import ABCMeta, abstractmethod
from pickletools import optimize
import numpy as np
import random

class RuleAgent(object):
    def __init__(self,new_gene):
        self.LENGTH = 10 #遺伝子数
        self.MUTATION_RATE = 0.01 #突然変異
        # if i < N: #利己的か利他的か
        #     self.doctrine = 1 #利他的
        # else:
        #     self.doctrine = 0 #利己的
        # self.gene.insert(0,self.doctrine)
        if new_gene == None:
            self.gene = [random.randint(0,1) for i in range(self.LENGTH)]
        else:
            self.gene = new_gene



        self.n = 0

    def action(self):
        self_action = self.B_to_A()
        return self_action

    def B_to_A(self):
        self_belief = self.gene
        return self_belief

    def mutate(self):
        for i in range(self.LENGTH):
            if random.random() < self.MUTATION_RATE:
                self.gene[i] = 1-self.gene[i]

    def getVal(self,reward):
        # if self.gene[0] == 1:
        #     value = rewards[0]-rewards[1]
        # else:
        #     value = rewards[0]+rewards[1]-6
        return reward

    def getFitness(self,reward):
        result = self.getVal(reward)
        self.n = result
        return result
    
    def __str__(self):
        result = ""
        for g in self.gene:
            result += str(g)
        result += "\t" +str(self.n)
        return result
