#!/usr/bin/env python
# coding: utf-8

# In[15]:


import gym
import random
import numpy as np
import time
from IPython.display import clear_output


# In[17]:


env = gym.make('FrozenLake-v0')


# In[18]:


env.render()


# In[19]:


action_space_size = env.action_space.n


# In[21]:


state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))


# In[24]:


num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay = 0.001


# In[25]:


rewards_all_episodes = []  #hold all the rewards from each episode
#q learning algo
for episode in range(num_episodes): #every thing that happens in a episode
    state = env.reset() #reset the state of envt to start
    done = False #track if the episode is finished
    rewards_current_episode = 0 
    
    for step in range(max_steps_per_episode): #every thing that happens in a single time step in a episode

    # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1) #whether the agent will explore 
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:]) # exploit envt
        else:
            action = env.action_space.sample() #explore
        
        # action is choosen
        new_state, reward, done, info = env.step(action)#execute the action
        
        #update q table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) +             learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        state = new_state #set current state to new
        rewards_current_episode += reward #update the reward
        if done == True: #check if last action ended the episode
            break
            
    exploration_rate = min_exploration_rate +     (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode) #decay exploration rate
    rewards_all_episodes.append(rewards_current_episode) #append the episode reward


# In[37]:


# Calculate and print the average reward per thousand episodes
rewards_per_thosand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000

print("Average reward per thousand episodes\n")
for r in rewards_per_thosand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

print("")    
print(q_table)


# In[38]:


#we will play 3 episodes of the game
for episode in range(3):
    state = env.reset()
    done = False
    print("EPISODE ", episode+1, "\n\n\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)
        action = np.argmax(q_table[state,:])        
        new_state, reward, done, info = env.step(action)  
        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                print("You reached the goal")
                time.sleep(3)
            else:
                print("You fell through a hole")
                time.sleep(3)
                clear_output(wait=True)
            break
        state = new_state            
env.close()


# In[ ]:




