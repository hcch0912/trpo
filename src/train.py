#! /usr/bin/env python3
"""
PPO: Proximal Policy Optimization

Written by Patrick Coady (pat-coady.github.io)

PPO uses a loss function and gradient descent to approximate
Trust Region Policy Optimization (TRPO). See these papers for
details:

TRPO / PPO:
https://arxiv.org/pdf/1502.05477.pdf (Schulman et al., 2016)

Distributed PPO:
https://arxiv.org/abs/1707.02286 (Heess et al., 2017)

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

And, also, this GitHub repo which was helpful to me during
implementation:
https://github.com/joschu/modular_rl

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""
import gym
import numpy as np
from gym import wrappers
from policy import Policy
from value_function import NNValueFunction
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import signal
import collections
from copy import deepcopy

class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


# def init_gym(env_name):
#     """
#     Initialize gym environment, return dimension of observation
#     and action spaces.

#     Args:
#         env_name: str environment name (e.g. "Humanoid-v1")

#     Returns: 3-tuple
#         gym environment (object)
#         number of observation dimensions (int)
#         number of action dimensions (int)
#     """
#     env = gym.make(env_name)
#     obs_dim = env.observation_space.shape[0]
#     act_dim = env.action_space.shape[0]

#     return env, obs_dim, act_dim

def get_traj_n(act_trajs):
    act_traj = []

    for i in range(len(act_trajs)):
        act_traj.append([])
        for j in range(len(act_trajs)):
            if i != j:
     
                a = deepcopy(act_trajs[j])
                act_traj[i].append(a)
 
    return np.array(act_traj)

def run_episode(env, policys, scaler, action_dim, timesteps, animate=False):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset()
    observes = [[] for i in range(len(obs))] 
    actions = [[] for i in range(len(obs))]
    rewards = [[] for i in  range(len(obs))] 
    intents = [[] for i in range(len(obs))]
    

    act_trajs = [collections.deque(np.zeros(( timesteps, action_dim)), 
                maxlen = timesteps) for _ in range(len(obs))]
    done = False
    step = 0.0
    # scale, offset = scaler.get()
    # scale[-1] = 1.0  # don't scale time step feature
    # offset[-1] = 0.0  # don't offset time step feature
    while not done:
        if animate:
            env.render()

        act_traj_n = get_traj_n(deepcopy(act_trajs))
        obs = [o.astype(np.float32).reshape((1, -1)) for o in obs]
        obs = [np.append(o, [[step]], axis=1) for o in obs]  # add time step feature
        # unscaled_obs.append(obs)
        # obs = (obs - offset) * scale  # center and scale observations
        # observes.append(obs)
        # observes = [observes[i].append(obs[i]) for i in range(len(obs)) ]
        for i, policy in enumerate(policys):
            # print(act_traj_n[i])
            # print(policy.sample_intent)
            intent = policy.sample_intent([np.array(act_traj_n[i]).reshape(1, -1).astype(np.float32)])[0]
            action = policy.sample_action(obs[i], intent).reshape((1, -1)).astype(np.float32)
            observes[i].append(obs[i])
            actions[i].append(action)
            intents[i].append(intent)
            act_trajs.append(np.array(act_traj_n[i]))

        obs, reward, done, _ = env.step(np.squeeze([action[-1] for action in actions], axis=0))
        for i in range(len(policys)):
            if not isinstance(reward[i], float):
                reward[i] = np.asscalar(np.asarray(reward[i]))
            rewards[i].append(reward[i])
        step += 1e-3  # increment time step feature

    return (observes, actions, np.array(rewards, dtype=np.float64),
            intents, act_trajs)


def run_policy(env, policys, scaler, logger, action_dim, timesteps, episodes):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories_n = [[] for i in range(len(policys))]
    for e in range(episodes):
        
        for i in range(len(trajectories_n)):
            observes, actions, rewards, intents, act_trajs = run_episode(env, policys, scaler, action_dim, timesteps)
            total_steps += len(observes[0])
            trajectory = {'observes': observes,
                          'actions': actions,
                          'rewards': rewards,
                          'intents': intents,
                          'act_trajs': act_trajs}
            trajectories_n[i].append(trajectory)        
    
   
    # logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
    #             'Steps': total_steps})

    return trajectories_n


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories_n, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectories in trajectories_n:
        for trajectory in trajectories:
            if gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = trajectory['rewards'] * (1 - gamma)
            else:
                rewards = trajectory['rewards']
            disc_sum_rew = discount(rewards, gamma)
            trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories_n, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectories in trajectories_n:
        for trajectory in trajectories:
            observes = trajectory['observes']
            intents = trajectory['intents']
            # print(np.array(observes).shape)
            # print(np.array(intents).shape)
            # print(np.array(observes[0]).shape, np.array(intents[0]).shape)

            values = [val_func.predict(np.concatenate([observe, intent], axis = 2).reshape(1,-1)) 
                     for observe, intent in zip(observes, intents)]
            # print(values)
            trajectory['values'] = values


def add_gae(trajectories_n, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectories in trajectories_n:
        for trajectory in trajectories:
            if gamma < 0.999:  # don't scale for gamma ~= 1
                # print(trajectory)
                rewards = trajectory['disc_sum_rew']
            else:
                rewards = trajectory['rewards']
            values = trajectory['values']
            # temporal differences
            tds = [reward - value for reward, value in zip(rewards, values)]
            tds =  [td + dis for td, dis in zip(tds, np.append([value* gamma for value in values[1:] ], 0) )]
            advantages = [discount(t, gamma * lam) for t in tds]
            trajectory['advantages'] = advantages


def build_train_set(trajectories_n):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes_n = []
    actions_n = []
    intents_n = []
    act_trajs_n = []
    disc_sum_rew_n = []
    advantages_n = []
    for trajectories in trajectories_n:
        observes_n.append([traj['observes']  for traj in trajectories])
        actions_n.append([traj['actions'] for traj in trajectories ])
        intents_n.append([traj['intents'] for traj in trajectories])
        act_trajs_n.append([traj['act_trajs'] for traj in trajectories])
        disc_sum_rew_n.append([traj['disc_sum_rew'] for traj in trajectories])
        advantages = [traj['advantages'] for traj  in trajectories]
        # normalize advantages
        advantages = [(adv - np.mean(adv)) / (np.std(adv) + 1e-6) for adv in advantages]

        advantages_n.append(advantages)
    return observes_n, actions_n, intents_n, act_trajs_n,  advantages_n, disc_sum_rew_n


def log_batch_stats(observes, actions, intents, act_trajs, advantages, disc_sum_rew, logger, episode):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_intent': np.mean(intents),
                '_min_intent': np.min(intents),
                '_max_intent': np.max(intents),
                '_std_intent': np.mean(np.var(intents, axis=0)),
                # '_mean_act_traj': np.mean(act_trajs),
                # '_min_act_traj': np.min(act_trajs),
                # '_max_act_traj': np.max(act_trajs),
                # '_std_act_traj': np.mean(np.var(act_trajs, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode
                })

def make_env(scenario_name,  benchmark=False):
    from Env.multiagent.environment import MultiAgentEnv
    import Env.multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def main(num_episodes, gamma, lam, kl_targ, batch_size, hid1_mult, policy_logvar,
        scenario,  num_agents, action_dim, timesteps):
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
        hid1_mult: hid1 size for policy and value_f (mutliplier of obs dimension)
        policy_logvar: natural log of initial policy variance
    """
    killer = GracefulKiller()
    # env, obs_dim, act_dim = init_gym(env_name)
    env = make_env(scenario)
    obs_dims = env.observation_space
    act_dims = [env.action_space[0].n for i in range(env.n)]
   
    obs_dims = [obs_dim.shape[0] + 1 for obs_dim in obs_dims]  # add 1 to obs dimension for time step feature (see run_episode())
  
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    logger = Logger(logname=scenario, now=now)
    aigym_path = os.path.join('/tmp', scenario, now)
    # env = wrappers.Monitor(env, aigym_path, force=True)
    scaler = Scaler(obs_dims)
    val_func = NNValueFunction(obs_dims[0]+act_dims[0], hid1_mult)
    policys = []

    for i in range(num_agents):
        policys.append(Policy(i, obs_dims[i], act_dims[0], kl_targ, hid1_mult, policy_logvar, num_agents-1, timesteps))
    # run a few episodes of untrained policy to initialize scaler:
    run_policy(env, policys, scaler, logger,  act_dims[0], timesteps, episodes=5)
    episode = 0
    while episode < num_episodes:
        trajectories = run_policy(env, policys, scaler, logger, act_dims[0],timesteps, episodes=batch_size)
        episode += len(trajectories)
        add_value(trajectories, val_func)  # add estimated values to episodes
        add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
        add_gae(trajectories, gamma, lam)  # calculate advantage
        # concatenate all episodes into single NumPy arrays
        observes, actions, intents, act_trajs,  advantages, disc_sum_rew = build_train_set(trajectories)
        # add various stats to training log:
        # log_batch_stats(observes, actions,intents, act_trajs,  advantages, disc_sum_rew, logger, episode)
        for i, policy in enumerate(policys):

            policy.update(observes[i], actions[i], intents[i], act_trajs[i], advantages[i], logger)  # update policy
            val_func.fit(observes[i]+intents[i], disc_sum_rew[i], logger)  # update value function
        logger.write(display=True)  # write logger results to file and stdout
        if killer.kill_now:
            if input('Terminate training (y/[n])? ') == 'y':
                break
            killer.kill_now = False
    logger.close()
    for policy in policys:
        policy.close_sess()
    val_func.close_sess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    # parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=1000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)
    parser.add_argument('-m', '--hid1_mult', type=int,
                        help='Size of first hidden layer for value and policy NNs'
                             '(integer multiplier of observation dimension)',
                        default=10)
    parser.add_argument('-v', '--policy_logvar', type=float,
                        help='Initial policy log-variance (natural log of variance)',
                        default=-1.0)
    parser.add_argument('--scenario', default ='simple_push')
    parser.add_argument('--num_agents', type=int, default = 2, help = "number of agents")
    parser.add_argument('--action_dim', type = int, default = 5,help = "action dimension")
    parser.add_argument('--timesteps', type = int,  default = 3, help = "the max length of the act traj buffer")


    args = parser.parse_args()
    main(**vars(args))
