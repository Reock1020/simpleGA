from abc import ABCMeta, abstractmethod
from pickletools import optimize
import numpy as np
import random

class RuleAgent(object):
    def __init__(self,new_gene,parent_pos):
        self.LENGTH = 14 #遺伝子数
        self.MUTATION_RATE = 0 #突然変異

        # if i < N: #利己的か利他的か
        #     self.doctrine = 1 #利他的
        # else:
        #     self.doctrine = 0 #利己的
        # self.gene.insert(0,self.doctrine)
        if new_gene == None:
            self.gene = [np.random.choice(range(2),p=np.array([0.5,0.5])) for i in range(self.LENGTH)]
        else:
            self.gene = new_gene

        if parent_pos == None:
            self.pos = random.randint(-50,50)
        else:
            self.pos = random.randint(parent_pos-5,parent_pos+5)

        self.my_fitness = 0
        self.pos_number = 0
        self.action_gene = []
        self.look_gene = []
        for i in range(int(self.LENGTH/2)):
            self.action_gene.append(self.gene[i])
        for i in range(int(self.LENGTH/2),self.LENGTH):
            self.look_gene.append(self.gene[i])
        
        self.character = sum(self.action_gene)/len(self.action_gene)


    def action(self):
        self_action = self.B_to_A()
        return self_action

    def B_to_A(self):
        self_belief = self.action_gene
        return self_belief

    def mutate(self):
        if random.random() < self.MUTATION_RATE:
            cnt=0
            if self.character > 0.5:
                while cnt > 0:
                    rnd = random.randrange(len(self.gene))
                    if self.gene[rnd] == 0:
                        self.gene[rnd] = 1-self.gene[rnd]
                        cnt+=1
            elif self.character < 0.5:
                while cnt > 0:
                    rnd = random.randrange(len(self.gene))
                    if self.gene[rnd] == 1:
                        self.gene[rnd] = 1-self.gene[rnd]
                        cnt+=1
            elif self.character == 0.5:
                while cnt > 0:
                    rnd = random.randrange(len(self.gene))
                    self.gene[rnd] = 1-self.gene[rnd]
                    cnt+=1

    def getVal(self,reward):
        # if self.gene[0] == 1:
        #     value = rewards[0]-rewards[1]
        # else:
        #     value = rewards[0]+rewards[1]-6
        return reward

    def getFitness(self,reward):
        result = self.getVal(reward)
        self.my_fitness = result
        return result
    
    def __str__(self):
        if self.character > 0.5:
            result = "利他的  "
        elif self.character < 0.5:
            result = "利己的  "
        else:
            result = "中立  "
        result += "action : "
        for g in self.action_gene:
            result += str(g)
        result += "\tlook : "
        for i in self.look_gene:
            result += str(i)
        result += "  fitness:" +str(self.my_fitness)
        result += "  pos:" +str(self.pos)
        return result


    def pos_update(self):
        self.pos =  random.randint(self.pos-2,self.pos+2)
