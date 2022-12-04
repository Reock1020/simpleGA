from ctypes import c_char


class RPD(object):
    def __init__(self,episode_size):
        self.episode_size=episode_size
        self.episode_counter = 0
        self.c_c = 9 #協力協力
        self.c_b = 0
        self.b_c = 10
        self.b_b = 1

        self.s_total_reward = 0
        self.o_total_reward = 0

        self.str = ["裏切り","協力"]

    def step(self,s_action,o_action):
        l = len(s_action)
        s_reward_total = 0
        o_reward_total = 0
        for i in range(l):
            self_reward,opp_reward=self.calc_reward(s_action[i],o_action[i])
            s_reward_total += self_reward
            o_reward_total += opp_reward

        done = self.episode_counter==self.episode_size
        self.episode_counter+=1
        
        
        # print("reward(",self_reward," ,",opp_reward,")")
        return ([s_reward_total,o_reward_total])

    def calc_reward(self,s_action,o_action):
        if s_action==1 and o_action==1:
            s_reward = self.c_c
            o_reward = self.c_c
        elif s_action==0 and o_action==1:
            s_reward = self.b_c
            o_reward = self.c_b
        elif s_action==1 and o_action==0:
            s_reward = self.c_b
            o_reward = self.b_c
        else :
            s_reward = self.b_b
            o_reward = self.b_b
        
        return s_reward,o_reward


    def step_all_agents(self,rewards):
        self.s_total_reward+=rewards[0]
        self.o_total_reward+=rewards[1]
        print("total_reward==> ",self.s_total_reward," , ",self.o_total_reward)
        print("____________________________________________________")
        

    def reset(self):
        raise NotImplementedError("reset must be explicitly overridden")

    def render(self):
        pass
    