import os
from datetime import datetime
from tqdm import tqdm
import sys
from pathlib import Path

print(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
print(sys.path)

import torch
import numpy as np

import gym
import ma_gym
from ma_gym.wrappers import Monitor

from agent.models.PPO import PPO

AGENT_NB = 5


################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    episode = 10000000
    max_ep_len = 1000  # max timesteps in one episode
    max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps

    def train_monitor_callable(episode_id):
        return True if episode_id >= episode - 20 else False

    env_name = f'MACombat-v{AGENT_NB}'
    gym.envs.register(
        id=env_name,
        entry_point='ma_gym.envs.combat:Combat',
        kwargs={'grid_shape': (15, 15), 'n_agents': AGENT_NB, 'n_opponents': AGENT_NB, 'step_cost': -0.5}
        # It has a step cost of -0.2 now
    )
    env = gym.make(env_name)
    env = Monitor(env, directory='recordings', force=True, video_callable=train_monitor_callable)

    agent_nb = env.n_agents

    has_continuous_action_space = False  # continuous action space; else discrete

    print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)  # save model frequency (in num timesteps)

    # #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = 10000
    K_epochs = 2  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0002  # learning rate for actor network
    lr_critic = 0.0002  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################

    # state space dimension
    state_dim = env.observation_space[0].shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        # action_dim = env.action_space.n
        action_dim = env.action_space[0].n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim + agent_nb, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                    has_continuous_action_space, 0)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward,loss,test_reward,win_rate\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    for _ in tqdm(range(episode)):

        states = np.array(env.reset())

        current_ep_reward = 0
        done = False

        while not done:
            loss = None
            states = torch.tensor(states, dtype=torch.float32)
            actions_tensor = -1 * torch.ones(agent_nb, )
            rest_agents = agent_nb
            for ind_state in states:
                state_with_c = torch.cat(
                    (ind_state,
                     actions_tensor), -1)
                if rest_agents != agent_nb:
                    ppo_agent.buffer.next_states.append(state_with_c)
                ind_action = ppo_agent.select_action(state_with_c)
                actions_tensor[-rest_agents] = ind_action
                rest_agents -= 1
                if rest_agents != 0:
                    ppo_agent.buffer.rewards.append(0)
                    ppo_agent.buffer.is_terminals.append(False)

            states, reward, done, healths = env.step(actions_tensor.tolist())
            states = np.array(states)
            reward = sum(reward)
            done = all(done)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            ppo_agent.buffer.next_states.append(torch.cat(
                (torch.tensor(states, dtype=torch.float32)[0],
                 -1 * torch.ones(agent_nb, )), -1))

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                loss = ppo_agent.update()

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
                avg_test_reward, win_rate = test(ppo_agent)

                log_f.write('{},{},{},{},{},{}\n'.format(
                    i_episode, time_step, log_avg_reward, loss, avg_test_reward, win_rate))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t Timestep : {} \t Average Reward : {} \t Loss : {}".format(i_episode, time_step,
                                                                                        print_avg_reward, loss))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


@torch.inference_mode()
def test(agent):

    ################## hyperparameters ##################

    total_test_episodes = 32    # total num of testing episodes

    #####################################################


    env_name = f'MACombat-v{AGENT_NB}'
    env = gym.make(env_name)


    agent_nb = env.n_agents

    # initialize a PPO agent
    ppo_agent = agent

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    test_win_episode = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        states = np.array(env.reset())
        done = False

        while not done:
            states = torch.tensor(states, dtype=torch.float32)
            actions_tensor = -1 * torch.ones(agent_nb, )
            rest_agents = agent_nb
            for ind_state in states:
                state_with_c = torch.cat(
                    (ind_state,
                     actions_tensor), -1)

                ind_action = ppo_agent.select_action(state_with_c)
                actions_tensor[-rest_agents] = ind_action
                rest_agents -= 1

            states, reward, done, healths = env.step(actions_tensor.tolist())
            states = np.array(states)
            reward = sum(reward)
            done = all(done)
            ep_reward += reward

        # clear buffer
        ppo_agent.buffer.clear()
        if sum(env.opp_health.values()) == 0:
            test_win_episode += 1
        test_running_reward += ep_reward

    env.close()

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    win_rate = test_win_episode / total_test_episodes
    print("average test reward : " + str(avg_test_reward))
    print(f"win_rate :{win_rate}")
    return avg_test_reward, win_rate

if __name__ == '__main__':
    train()
