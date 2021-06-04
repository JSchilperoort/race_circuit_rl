import numpy as np
import time
import csv
import sys
import os
import multiprocessing

from environment import RacingEnv
from agent import Agent


def write_to_file(filename, rewards):
    with open(filename, 'w', newline='', encoding='utf-8') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(rewards)


def calc_eta(e, episodes, t0):
    time_passed = time.time() - t0
    h = int(np.floor(((time_passed / (e + 0.00001)) * (episodes - (e + 0.00001))) / 3600))
    m = int(np.floor(((time_passed / (e + 0.00001)) * (episodes - (e + 0.00001))) / 60) - h * 60)
    s = int(np.floor(((time_passed / (e + 0.00001)) * (episodes - (e + 0.00001)))) - h * 3600 - m * 60)
    if h < 10:
        h = "0{}".format(h)
    if m < 10:
        m = "0{}".format(m)
    if s < 10:
        s = "0{}".format(s)
    eta = "{0}:{1}:{2}".format(h, m, s)
    return eta


def test_env(env, agent):
    """run and render the environment continuously for one lap of the circuit"""

    env.checkpoint = 1
    env.x, env.y = env.midpoint(env.getCheckpoint(env.checkpoint))
    angle_noise = ((np.random.rand() - 0.5) * 10) / 180 * np.pi
    env.angle = -env.checkpointAngle(env.checkpoint) + np.pi / 2 + angle_noise
    env.speed = 0
    env.t = 0
    state = env.get_state()
    env.set_n_steps(9999999)

    rewards = list()
    n_episodes = 0

    while True:
        n_episodes += 1
        action = agent.act(state)

        state, reward, _, _ = env.step(action)
        rewards.append(reward)
        env.render()

        if env.checkpoint == 0:
            break

    return rewards, len(env.lwall)


def write_intermediate_output(alg, index, t1, e, sum_v_loss, sum_p_loss, sum_rewards, eta, directory_name):
    filename = os.path.join(directory_name, "running_output-{0}-{1}.txt".format(alg, index))
    file_object = open(filename, 'a', encoding='utf-8')
    runtime = time.time() - t1
    m = int(np.floor(runtime / 60))
    s = int(np.floor(runtime % 60))
    file_object.write('Epoch: {0}\t|\tRuntime: {1}:{2}\t|\tValue loss: {3:.3f}\t|\tPolicy loss: {4:.3f}\t|\t'
                      'Rewards: {5}\t|\tEta:{6}\n'.format(e, m, s, sum_v_loss, sum_p_loss, sum_rewards, eta))
    file_object.close()


def perform_test(parameters):
    """perform_test receives a parameter string separated by dashes and runs an experiment based on these parameters"""

    # retrieve parameters back from dash-separated parameter string
    alg, index, directory_name, n_episodes, n_layers_actor, n_units_actor, n_layers_critic, n_units_critic = parameters.split('-')
    index = int(index)
    n_episodes = int(n_episodes)
    n_layers_actor = int(n_layers_actor)
    n_units_actor = int(n_units_actor)
    n_layers_critic = int(n_layers_critic)
    n_units_critic = int(n_units_critic)

    # initialize environment and learning agent
    env = RacingEnv()
    agent = Agent(env, alg, n_layers_actor, n_units_actor, n_layers_critic, n_units_critic)

    ep_reward = list()
    sum_v_loss = sum_p_loss = sum_rewards = 0

    t0 = t1 = time.time()

    # initialize standard deviation sampling parameter for SPG (ranges from 1 till 0.05)
    sigma = 1
    d_sigma = (sigma - 0.05) / n_episodes

    for s in range(n_episodes):
        # after 19.5 hours, training halts automatically
        if (time.time() - t0) > (19.5 * 3600):
            break

        total_reward = 0
        all_vl = list()
        all_pl = list()
        state = env.reset()
        done = False

        # 'done' means the agent has completed 200 steps in the environment
        while not done:
            # sample action from network
            action = agent.act(state)

            # take action, observe and save environment state
            next_state, reward, done, dead = env.step(action)
            agent.savexp(state, next_state, action, dead, reward)

            state = next_state
            total_reward += reward

        ep_reward.append(total_reward)
        sum_rewards += total_reward

        sigma -= d_sigma

        # train actor and critic 10 times
        for _ in range(10):
            vl, pl = agent.train(sigma)
            all_vl.append(vl)
            all_pl.append(pl)

        sum_v_loss += (sum(all_vl) / len(all_vl))
        sum_p_loss += (sum(all_pl) / len(all_pl))

        # write to output file every 100 episodes
        if (s + 1) % 100 == 0:
            eta = calc_eta(s, n_episodes, t0)
            write_intermediate_output(alg, index, t1, s, sum_v_loss / 100, sum_p_loss / 100, sum_rewards / 100, eta,
                                      directory_name)
            sum_v_loss = sum_p_loss = sum_rewards = 0
            t1 = time.time()

    print("Writing data...")
    filename = os.path.join(directory_name, "{0}-{1}.csv".format(alg, index))
    write_to_file(filename, ep_reward)

    # test_env(env, agent)

    return ep_reward


def run_test_parallel(directory_name, n_episodes, n_layers_actor, n_units_actor, n_layers_critic, n_units_critic):
    """run 24 tests in parallel(or less if insufficient number of cpus)"""

    algorithms = ['spg', 'spg', 'spg', 'spg', 'spg', 'spg', 'tdspg', 'tdspg', 'tdspg', 'tdspg', 'tdspg', 'tdspg',
                  'td3', 'td3', 'td3', 'td3', 'td3', 'td3', 'dpg', 'dpg', 'dpg', 'dpg', 'dpg', 'dpg']

    algorithms = ['spg']
    # create list of parameter strings separated by dashes to pass on in parallel
    parameters = [x + '-' + str(i) + '-' + directory_name + "-" + n_episodes + "-" + n_layers_actor + "-" +
                  n_units_actor + "-" + n_layers_critic + "-" + n_units_critic for i, x in enumerate(algorithms)]

    # set the numer of cpus as either the required or maximal amount
    n_cpus = min(multiprocessing.cpu_count(), len(algorithms))
    print("Spinning up {} cpus...".format(n_cpus))

    # initialize parallel tests
    t0 = time.time()
    with multiprocessing.Pool(n_cpus) as p:
        data = p.map(perform_test, parameters)
    print("Running took {0:.2f} seconds".format(time.time() - t0))


def main(argv):
    """read and assign input parameters"""

    directory_name = argv[0]
    n_episodes = argv[1]
    n_layers_actor = argv[2]
    n_units_actor = argv[3]
    n_layers_critic = argv[4]
    n_units_critic = argv[5]

    # create directory to store results
    if not os.path.isdir(directory_name):
        os.makedirs(directory_name)

    # start tests
    print("Start run_test()..")
    run_test_parallel(directory_name, n_episodes, n_layers_actor, n_units_actor, n_layers_critic, n_units_critic)


if __name__ == "__main__":
    # discard first argument 'main.py'
    main(sys.argv[1:])
