from gettext import npgettext
import random 
import copy
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from agent import RuleAgent
from environments import RPD
from scipy.spatial import distance

class SimpleGA:
    def __init__(self,N):
        self.N = N
        self.ITERATION  = 100
        self.repeat_switch = True
        self.pos_plt_cnt =1
        self.pos_plt_switch = False
        self._initialize()
        self.env = RPD(self.ITERATION)
        self.total_number = sum(range(1,self.N+1))
        self.epi = 0
        self.s_pool = []
        self.a_pool = []
        self.n_pool = []
        self.s_similarity_sum = 0
        self.a_similarity_sum = 0
        self.n_similarity_sum = 0

        self.s_similarity_avg = 0
        self.a_similarity_avg = 0

        self.a_similarity_list_1 = []
        self.s_similarity_list_2 = []
        self.similarity_difference_list = []
        self.dispersion_rate_list = []

        self.character_range_difference = 0

        self.pre_pool = self.pool
        self.a = 0

    def _initialize(self):
        self.pool = [RuleAgent(None,None) for i in range(self.N)] #個体生成
    
    def _rankingSelection(self):
        pool_next = [] #次世代
        agent_rewards = [0]*self.N
        agent_fitness = [[0 for i in range(2)]for j in range(self.N)]
        for i in range(1,self.N): #総当たり戦
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
        print(agent_fitness)

        
        while len(pool_next) < self.N/2:
            rdm = random.random()
            for i in range(self.N):
                if rdm < (sum(range(0,i+2)))/self.total_number:
                    # print(rdm,(sum(range(0,i+2)))/self.total_number)
                    number = agent_fitness[i][1]
                    break
            offspring = copy.deepcopy(self.pool[agent_fitness[i][1]])
            pool_next.append(offspring)
        pool_next = self.intersection(pool_next)


        self.pool = pool_next[:]

    def intersection(self,pool_next):#交叉した子をすべて返す
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
            # print(offspring1.gene,offspring2.gene,new_gene)
            pool_next.append(RuleAgent(new_gene,None))
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

    def _rankingSelectionAtDistance(self):
        pool_next = [] #次世代
        agent_rewards = [0]*self.N
        agent_fitness = [[0 for i in range(2)]for j in range(self.N)]
        
        for i in range(0,self.N):#距離が10以下のエージェントとRPD
            cnt = 1
            for j in range(0,self.N):
                agent_distance = np.linalg.norm(self.pool[i].pos-self.pool[j].pos)
                if i!=j and agent_distance < 10:
                    o1_action = self.pool[i].action()
                    o2_action = self.pool[j].action()
                    rewards = self.env.step(o1_action,o2_action)
                    agent_rewards[i] += rewards[0]
                    cnt += 1
            agent_rewards[i] = agent_rewards[i]/cnt

        for i in range(self.N):
            agent_fitness[i][1] = i
            agent_fitness[i][0] = self.pool[i].getFitness(agent_rewards[i])
        agent_fitness.sort(key=lambda x: x[0])

        self._printStatus(self.epi)
        if self.epi%self.pos_plt_cnt==0:
            self.pos_plt(self.pool)

        n = self.N-1

        while len(pool_next) < self.N/2:#任意の数の上位を選択
            number = agent_fitness[n][1]
            offspring = copy.deepcopy(self.pool[number])
            offspring.pos = random.randint(offspring.pos-5,offspring.pos+5)
            pool_next.append(offspring)
            n=n-1
        self.pool = pool_next[:]
        n = 0
        while len(pool_next) < self.N:#選択した上位を交叉
            rdm = random.random()
            near_group_fitness = []

            for i in range(len(self.pool)):
                agent_distance = np.linalg.norm(self.pool[n].pos-self.pool[i].pos)
                if i!=n and agent_distance < 10:
                    near_group_fitness.append([self.pool[i].my_fitness,i])
            near_group_fitness.sort(key=lambda x: x[0])
            # print(near_group_fitness)

            for i in range(len(near_group_fitness)):#高い順に確率を上げる
                if rdm < (sum(range(0,i+2)))/sum(range(1,len(near_group_fitness)+1)):
                    # print(rdm,(sum(range(0,i+2)))/self.total_number)
                    number = near_group_fitness[i][1]
                    # print(number)
                    if number == n:
                        continue
                    break
                # print(number)

            min = 100000
            for i in range(len(self.pool)):
                agent_distance = np.linalg.norm(self.pool[n].pos-self.pool[i].pos)
                if agent_distance < min and n != i:
                    min = agent_distance
                    number = i

            print(number,n)
            offspring1 = copy.deepcopy(self.pool[number])
            offspring2 = copy.deepcopy(self.pool[n])
            n+=1
            new_agent = self.intersection2(offspring1,offspring2)
            print(new_agent.pos)
            pool_next.append(new_agent)

        for i in range(len(self.pool)):
            print(self.pool[i].pos)
        self.pool = pool_next[:]

    def intersection2(self,agent1,agent2):#子を返す
        rdm_border = random.randint(1,int(len(agent1.gene)/2)-1)
        gene_len = len(agent1.gene)
        new_action_gene = []
        new_look_gene = []
        new_gene = []
    
        for i in range(rdm_border):
            new_gene.append(agent1.gene[i])
        for i in range(rdm_border,int(gene_len/2)):
            new_gene.append(agent2.gene[i])
        for i in range(int(gene_len/2),rdm_border+int(gene_len/2)):
            new_gene.append(agent1.gene[i])
        for i in range(rdm_border+int(gene_len/2),gene_len):
            new_gene.append(agent2.gene[i])

        # print(offspring1.gene,offspring2.gene,new_gene)
        return RuleAgent(new_gene,agent2.pos)

    def _rankingSelectionAtDistance2(self):
        pool_next = [] #次世代
        self.pre_pool = self.pool
        agent_rewards = [0]*self.N
        agent_fitness = [[0 for i in range(2)]for j in range(self.N)]
        for i in range(0,self.N):
            cnt = 1
            min = 100000
            near_number = 0
            for a in range(self.N):#近くのエージェントとのRPD
                agent_distance = np.linalg.norm(self.pool[i].pos-self.pool[a].pos)
                if agent_distance < min and a!=i:
                    min = agent_distance
                    near_number = a
            o1_action = self.pool[i].action()
            o2_action = self.pool[near_number].action()
            rewards = self.env.step(o1_action,o2_action)
            agent_rewards[i] += rewards[0]


            # for j in range(0,self.N):総当たり戦
            #     agent_distance = np.linalg.norm(self.pool[i].pos-self.pool[j].pos)
            #     if i!=j and agent_distance < 10:
            #         o1_action = self.pool[i].action()
            #         o2_action = self.pool[j].action()
            #         rewards = self.env.step(o1_action,o2_action)
            #         agent_rewards[i] += rewards[0]
            #         cnt += 1
            # agent_rewards[i] = agent_rewards[i]/cnt

        for i in range(self.N):
            agent_fitness[i][1] = i
            agent_fitness[i][0] = self.pool[i].getFitness(agent_rewards[i])
        agent_fitness.sort(key=lambda x: x[0])

        self._printStatus(self.epi)
        # if self.epi%self.pos_plt_cnt==0: self.pos_plt(self.pool)

        # if self.pos_plt_switch:
        #     self.pos_plt(self.pool)

        n = self.N-1

        while len(pool_next) < self.N*0.7:#任意の数の上位を選択
            number = agent_fitness[n][1]
            offspring = copy.deepcopy(self.pool[number])
            offspring.pos_update()
            pool_next.append(offspring)
            n=n-1
        self.pool = pool_next[:]
        n = 0
        while len(pool_next) < self.N:#選択した上位を交叉
            rdm = random.random()
            min = 100000
            for i in range(len(self.pool)):
                agent_distance = np.linalg.norm(self.pool[n].pos-self.pool[i].pos)
                if agent_distance < min and n != i:
                    min = agent_distance
                    number = i
            offspring1 = copy.deepcopy(self.pool[number])
            offspring2 = copy.deepcopy(self.pool[n])
            n+=1
            new_agent = self.intersection2(offspring1,offspring2)
            pool_next.append(new_agent)
        self.pool = pool_next[:]
        

    def _rankingSelectionAtDistance3(self):
        pool_next = [] #次世代
        agent_rewards = [0]*self.N
        agent_fitness = [[0 for i in range(2)]for j in range(self.N)]
        for i in range(0,self.N):
            cnt = 1
            min = 100000
            near_number = 0
            for j in range(0,self.N):#総当たり戦
                agent_distance = np.linalg.norm(self.pool[i].pos-self.pool[j].pos)
                if i!=j and agent_distance < 10:
                    o1_action = self.pool[i].action()
                    o2_action = self.pool[j].action()
                    rewards = self.env.step(o1_action,o2_action)
                    agent_rewards[i] += rewards[0]
                    cnt += 1
            agent_rewards[i] = agent_rewards[i]/cnt

        for i in range(self.N):
            agent_fitness[i][1] = i
            agent_fitness[i][0] = self.pool[i].getFitness(agent_rewards[i])
        agent_fitness.sort(key=lambda x: x[0])

        self._printStatus(self.epi)
        # if self.epi%self.pos_plt_cnt==0:
        #     self.pos_plt(self.pool)

        # if self.pos_plt_switch:
        #     self.pos_plt(self.pool)

        n = self.N-1

        while len(pool_next) < self.N*0.7:#任意の数の上位を選択
            number = agent_fitness[n][1]
            offspring = copy.deepcopy(self.pool[number])
            offspring.pos_update()
            pool_next.append(offspring)
            n=n-1
        self.pool = pool_next[:]
        n = 0
        while len(pool_next) < self.N:#選択した上位を交叉
            rdm = random.random()
            min = 100000
            for i in range(len(self.pool)):
                agent_distance = np.linalg.norm(self.pool[n].pos-self.pool[i].pos)
                if agent_distance < min and n != i:
                    min = agent_distance
                    number = i
            offspring1 = copy.deepcopy(self.pool[number])
            offspring2 = copy.deepcopy(self.pool[n])
            n+=1
            new_agent = self.intersection2(offspring1,offspring2)
            pool_next.append(new_agent)
        self.pool = pool_next[:]

    def calc_dispersion_rate(self,new_pool):
        sum = 0
        for i in range(self.N-1):
            agent1 = new_pool[i].character
            agent2 = new_pool[i+1].character
            if (agent1 < 0.5 and agent2 > 0.5) or (agent1 > 0.5 and agent2 < 0.5):
                sum += 1
        return sum
    
    def pool_allocation(self,new_pool):
        self.s_pool = []
        self.a_pool = []
        self.n_pool = []
        for i in range(self.N):
            if new_pool[i].character > 0.5:
                self.a_pool.append(new_pool[i])
            elif new_pool[i].character < 0.5:
                self.s_pool.append(new_pool[i])
            else:
                self.n_pool.append(new_pool[i])


    def calc_similarity(self,c):
        if c.character > 0.5:
            d = self.calc_hamming_distance_average(self.a_pool,c)
            self.a_similarity_sum+=d
        elif c.character < 0.5:
            d = self.calc_hamming_distance_average(self.s_pool,c)
            self.s_similarity_sum+=d
        else:
            d = self.calc_hamming_distance_average(self.n_pool,c)
            self.n_similarity_sum+=d
        return d

    def calc_hamming_distance_average(self,affiliation_pool,c):
        sum = 0
        for i in range(len(affiliation_pool)):
            sum += distance.hamming(c.look_gene,affiliation_pool[i].look_gene)*len(c.look_gene)
        if len(affiliation_pool)!=1:
            sum = sum/(len(affiliation_pool)-1)
        else:
            sum = sum/len(affiliation_pool)
        return sum
    
    def calc_hamming_distance_average2(self,affiliation_pool,c):
        sum = 0
        for i in range(len(affiliation_pool)):
            sum += distance.hamming(c.look_gene,affiliation_pool[i].look_gene)*len(c.look_gene)
        
        sum = sum/(len(affiliation_pool))
        
        return sum

    def _printStatus(self, iteration):
        new_pool = []
        agent_pos = [[0 for i in range(2)]for j in range(self.N)]
        for i in range(self.N):
            agent_pos[i][1] = i
            agent_pos[i][0] = self.pool[i].pos
        agent_pos.sort(key=lambda x: x[0])

        for i in range(self.N):
            new_pool.append(self.pool[agent_pos[i][1]])
        dispersion_rate = self.calc_dispersion_rate(new_pool)
        self.pos_plt_switch = False
        if dispersion_rate < 10:
            self.pos_plt_switch = True
        if dispersion_rate == 0:
            self.repeat_switch = False
        self.dispersion_rate_list.append(dispersion_rate)

        if dispersion_rate == 0:
            return 

        self.pool = new_pool
        # print("generation : " +str(iteration)+"\tばらつき:"+str(dispersion_rate))
        self.pool_allocation(new_pool)
        

        for c in new_pool:
            similarity = self.calc_similarity(c)
            # print("\t"+str(c)+"\t  similarity : "+str(similarity))

        s_similarity_avg,a_similarity_avg,n_similarity_avg = self.calc_similarity_avg()
        self.s_similarity_avg = s_similarity_avg
        self.s_similarity_list_2.append(self.s_similarity_avg)
        self.a_similarity_avg = a_similarity_avg
        self.a_similarity_list_1.append(self.a_similarity_avg)
        # print("利己的similarity : "+str(s_similarity_avg),"利他的similarity : "+str(a_similarity_avg))
        self.character_range_difference = self.calc_character_range_difference()
        self.similarity_difference_list.append(self.character_range_difference)

        # print("性格の距離 : "+str(self.character_range_difference))

    def calc_character_range_difference(self):
        sum = 0
        for i in range(len(self.a_pool)):
            sum += self.calc_hamming_distance_average2(self.s_pool,self.a_pool[i])
        avg = sum/len(self.a_pool)
        return avg

        
    def calc_similarity_avg(self):
        s_similarity_avg = self.s_similarity_sum/len(self.s_pool)
        self.s_similarity_sum = 0
        a_similarity_avg = self.a_similarity_sum/len(self.a_pool)
        self.a_similarity_sum = 0
        return s_similarity_avg,a_similarity_avg,0

    def sim_plt(self,episode_list):

        fig, ax1 = plt.subplots(figsize=(10,8))
        ax1.bar(episode_list,self.dispersion_rate_list, align="center", color="royalblue", linewidth=0)
        ax2 = plt.twinx()
        ax2.plot(episode_list,self.s_similarity_list_2,linewidth=4,c="red",label = 'selfish')
        ax2.plot(episode_list,self.a_similarity_list_1,linewidth=4,c="blue",label='altruistic')
        ax2.plot(episode_list,self.similarity_difference_list,linewidth=4,c="green",label='difference')
        plt.legend()
        plt.show()

    def pos_plt(self,pool):
        x_pos_list0 = []
        x_pos_list1 = []
        x_pos_list2 = []
        y_pos_list0 = []
        y_pos_list1 = []
        y_pos_list2 = []
        
        for i in range(len(pool)):
            judge = self.pool[i].character
            if judge > 0.5:
                x_pos_list1.append(pool[i].pos)
                y_pos_list1.append(0)
            elif judge < 0.5:
                x_pos_list0.append(pool[i].pos)
                y_pos_list0.append(0)
            elif judge == 0.5:
                x_pos_list2.append(pool[i].pos)
                y_pos_list2.append(0)

        fig, ax = plt.subplots(figsize=(10,8))
        ax.scatter(x_pos_list0,y_pos_list0,c="red",label = 'selfish')
        ax.scatter(x_pos_list1,y_pos_list1,c="blue",label='altruistic')
        ax.scatter(x_pos_list2,y_pos_list2,c="green",label='neutral')
        plt.legend()
        plt.show()

    def _mutateAll(self):
        for i in range(self.N):
            self.pool[i].mutate2()

    
    def evolve(self):
        episode_list = []
        max = 0
        for i in range(self.ITERATION):
            # for j in range(self.N): self.pool[j].step()
            episode_list.append(i)
            self._rankingSelectionAtDistance2()
            self._mutateAll()
            if self.dispersion_rate_list[i] == 0:
                self.dispersion_rate_list.pop(i)
                episode_list.pop(i)
                break
            self.epi+=1
            sa = self.similarity_difference_list[i]-self.a_similarity_list_1[i]-self.s_similarity_list_2[i]
            if i == 0:max = sa
            if max <= sa:
                max = sa
                evaluation_pool= copy.deepcopy(self.pre_pool)
                self.a = i
        # self.sim_plt(episode_list)
        return max,evaluation_pool
        