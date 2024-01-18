## MADDPG implementation using ddpg_agent.py for Actor-Critic

import torch

import random
from collections import namedtuple, deque
import numpy as np
from ddpg import Agent
from buffer import ReplayBuffer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # mini batch size  #128
UPDATE_FREQ = 1
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters

class MADDPG():
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random.seed(random_seed)
        
        # creating agents and store them into agents list
        # Model Hyper-parameters
        # self.hidden1_actor = 16
        # self.hidden2_actor = 8
        # # obsFull + actionFull = state_size + num_agent * action_size(of each) 
        # self.input_critic = state_size + 2*action_size      
        # self.hidden1_critic = 32
        # self.hidden2_critic = 16 
        
        # creating agents and store them into agents list
        # self.agents = [Agent(state_size, self.hidden1_actor, self.hidden2_actor, action_size, self.input_critic, self.hidden1_critic, self.hidden2_critic) for i in range(num_agents)]

        self.agents = [Agent(state_size, action_size, num_agents, random_seed) for i in range(num_agents)]
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.step_count = 0
        
    # reset each agent's noise
    def reset(self):
        for agent in self.agents:
            agent.reset()
            
    # state: state array for all agents [agent_no, state_size]
    # output: actions of all agents [agent_no, action of an agent]
    def act(self, state, i_episode, add_noise=True):
        actions = []
        for agent_state, agent in zip(state, self.agents):
            action = agent.act(agent_state, i_episode, add_noise)
            action = np.reshape(action, newshape=(-1))
            actions.append(action)
        actions = np.stack(actions)
        return actions
        
    # store states, actions, etc into ReplayBuffer and trigger training regularly
    # state and new_state : state of all agents [agent_no, state of an agent]
    # action: action of all agents [agent_no, action of an agent]
    # reward: reward of all agents [agent_no]
    # dones: dones of all agents [agent_no]
    def step(self, i_episode, state, action, reward, next_state, done):
        full_state = np.reshape(state, newshape=(-1))
        next_full_state = np.reshape(next_state, newshape=(-1))
        
        self.memory.add(state, full_state, action, reward, next_state, next_full_state, done)
        
        self.step_count = ( self.step_count + 1 ) % UPDATE_FREQ
        
        if len(self.memory) > BATCH_SIZE and i_episode > 500:
            for l_cnt in range(3):
                for agent in self.agents:
                    experiences = self.memory.sample()
                    self.learn(experiences, agent, GAMMA)

                for agent in self.agents:
                    agent.soft_update(agent.actor_local, agent.actor_target, TAU)
                    agent.soft_update(agent.critic_local, agent.critic_target, TAU)

    # execute learning on an agent    
    def learn(self, experiences, agent, GAMMA):
        # batch dataset for training
        states, full_states, actions, rewards, next_states, next_full_states, dones = experiences
                
        # compute NO-NOISE action using target actor and current state - [batch_size, # of agent, action size]
        # this will be used as input on critic local network
        actor_target_actions = torch.zeros(actions.shape, dtype=torch.float, device=device)
        for agent_idx, agent_i in enumerate(self.agents):
            if agent == agent_i:
                agent_id = agent_idx
            agent_i_current_state = states[:,agent_idx]
            actor_target_actions[:,agent_idx,:] = agent_i.actor_target.forward(agent_i_current_state)
        actor_target_actions = actor_target_actions.view(BATCH_SIZE, -1)


        agent_state = states[:,agent_id,:]
        agent_action = actions[:,agent_id,:]
        agent_reward = rewards[:,agent_id].view(-1,1)
        agent_done = dones[:,agent_id].view(-1,1)

        
        # replace action of the specific agent with actor_local output (NOISE removal)
        actor_local_actions = actions.clone()
        actor_local_actions[:, agent_id, :] = agent.actor_local.forward(agent_state)
        actor_local_actions = actor_local_actions.view(BATCH_SIZE, -1)        
        actions = actions.view(BATCH_SIZE, -1)       

        
        agent_experience = (full_states, actions, actor_local_actions, actor_target_actions,
                            agent_state, agent_action, agent_reward, agent_done,
                            next_states, next_full_states)

        agent.learn(agent_experience, GAMMA)

    def save(self):
        for idx, agent in enumerate(self.agents):
            chk_actor_filename = 'checkpoint_agent{}_actor.pth'.format(idx)
            chk_critic_filename = 'checkpoint_critic{}_critic.pth'.format(idx)
            torch.save(agent.actor_local.state_dict(), chk_actor_filename)
            torch.save(agent.critic_local.state_dict(), chk_critic_filename)
        
