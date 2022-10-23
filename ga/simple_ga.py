from gettext import npgettext
import random 
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from agent import RuleAgent
from environments import RPD

class SimpleGA:
    def __init__(self):
        self.N = 10
        self.ITERATION  = 2
        self._initialize()
        self.env = RPD(self.ITERATION)
        self.total_number = sum(range(1,self.N+1))

    def _initialize(self):
        self.pool = [RuleAgent(None) for i in range(self.N)] #個体生成

    def _mutateAll(self):
        for i in range(self.N):
            self.pool[i].mutate()
    
    def _rankingSelection(self):
        pool_next = [] #次世代
        agent_rewards = [0]*self.N
        agent_fitness = [[0 for i in range(2)]for j in range(self.N)]
        for i in range(1,self.N):
                for j in range(i):
                    o1_action = self.pool[i].action()
                    o2_action = self.pool[j].action()
                    rewards = self.env.step(o1_action,o2_action)
                    agent_rewards[i] += rewards[0]
                    agent_rewards[j] += rewards[1]
        
        for i in range(self.N):
            agent_fitness[i][1] = i
            agent_fitness[i][0] = self.pool[i].getFitness(agent_rewards[i])
        agent_fitness.sort(key=lambda x: x[0])
        # print(agent_fitness)
        while len(pool_next) < self.N/2:
            rdm = random.random()
            for i in range(self.N):
                if rdm < (sum(range(0,i+2)))/self.total_number:
                    # print(rdm,(sum(range(0,i+2)))/self.total_number)
                    number = agent_fitness[i][1]
                    break
            offspring = copy.deepcopy(self.pool[number])
            pool_next.append(offspring)
        pool_next = self.intersection(pool_next)


        self.pool = pool_next[:]

    def intersection(self,pool_next):
        while len(pool_next) < self.N:
            rdm_border = random.randint(1,len(self.pool[0].gene))
            offspring1 = copy.deepcopy(self.pool[random.randrange(self.N)])
            offspring2 = copy.deepcopy(self.pool[random.randrange(self.N)])
            gene_len = len(offspring1.gene)
            new_gene = []
            for i in range(rdm_border):
                new_gene.append(offspring1.gene[i])
            for i in range(rdm_border,gene_len):
                new_gene.append(offspring2.gene[i])
            print(offspring1.gene,offspring2.gene,new_gene)
            pool_next.append(RuleAgent(new_gene))
        return pool_next



    def _tournamentSelection(self):
        pool_next = [] #次世代
        agent_rewards  = [0]*self.N
        while len(pool_next) < self.N:
            offspring1 = copy.deepcopy(self.pool[random.randrange(self.N)])
            offspring2 = copy.deepcopy(self.pool[random.randrange(self.N)])
            o1_action = offspring1.action()
            o2_action = offspring2.action()

            rewards = self.env.step([o1_action,o2_action])

            o1_fitness = offspring1.getFitness(rewards[0])
            o2_fitness = offspring2.getFitness(rewards[1])

            # if o1_fitness > o2_fitness and o1_fitness > 0:
            #     pool_next.append(offspring1)
            # elif o1_fitness == o2_fitness and o1_fitness > 0 and o2_fitness > 0:
            #     pool_next.append(offspring1)
            #     pool_next.append(offspring2)

            if o1_fitness > o2_fitness :
                pool_next.append(offspring1)
            else:
                pool_next.append(offspring2)
        self.pool = pool_next[:]

    def _printStatus(self, iteration):
        print("generation\t" +str(iteration))
        for c in self.pool:
            print("\t"+str(c))
    
    def evolve(self):
        for i in range(self.ITERATION):
            self._rankingSelection()
            self._printStatus(i)
            self._mutateAll()