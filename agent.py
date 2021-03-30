import argparse
import math

import pygame
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import save_model, load_model
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from tensorflow.python.keras.optimizer_v2.adam import Adam

from F9LanderCORE import Options, World, Simulation, Rocket, Platform

ACTIONS = 4

"""
0: L
1: M
2: R
3: L+M
4: R+M
5: L+M+R
6: None
"""

BATCH_SIZE = 64
FEATURE_SIZE = 7
# MIN_STEPS = 30
GAMMA = 0.99
OBSERVE_PERIOD = 4096


class Agent:

    def __init__(self, load_weights=False):
        self.load_weights = load_weights
        self.memory = deque(maxlen=10000)
        self.steps = 0
        self.eps = 1.0
        self.eps_decay = 0.998
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.tau = 0.05

        if not self.load_weights:
            self.model = self.create_model()
            self.target_model = self.create_model()
        else:
            self.model = load_model('current_best.h5')

    def target_train(self):
        if not self.load_weights:
            weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)

            self.target_model.set_weights(target_weights)
        else:
            return None

    def act(self, state):

        if self.eps < random.uniform(0, 1) or self.load_weights:
            result = self.model.predict(state)
            best = np.argmax(result)
        else:
            best = self.explore()

        return best

    def create_model(self):
        model = Sequential()
        model.add(Dense(150, input_dim=FEATURE_SIZE, activation="relu"))
        model.add(Dense(120, activation="relu"))
        model.add(Dense(ACTIONS, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

        return model

    def get_sample(self, sample):
        self.memory.append(sample)
        self.steps = self.steps + 1

    def get_action_by_user(self):
        key = pygame.key.get_pressed()
        keys = [key[pygame.K_w], key[pygame.K_a], key[pygame.K_d], key[pygame.K_n]]
        action = convert_action_inverse(keys)
        print("ACTION!!!!!", action)
        print("KEYS!!!!!", keys)

        return action

    def explore(self):
        event = random.randint(0, ACTIONS - 1)
        # print("Exploring %s" % str(event))

        return event

    def backward(self):

        if self.steps > OBSERVE_PERIOD and not self.load_weights:
            self.eps *= self.eps_decay
            sample = random.sample(self.memory, BATCH_SIZE)
            inputs = np.zeros((BATCH_SIZE, FEATURE_SIZE))

            targets = np.zeros((inputs.shape[0], ACTIONS))

            for i, mini_batch in enumerate(sample):
                state_t0 = mini_batch[0]
                action_t0 = mini_batch[1]
                reward_t0 = mini_batch[2]
                state_t1 = mini_batch[3]

                inputs[i:i + 1] = state_t0
                prediction = self.model.predict(state_t0)

                # print("prediction: ", prediction)

                targets[i] = prediction
                q_sa = self.model.predict(state_t1)

                if state_t1 is not None:
                    targets[i, action_t0] = reward_t0 + GAMMA * np.max(q_sa)
                else:
                    targets[i, action_t0] = reward_t0

            self.model.fit(inputs, targets, batch_size=BATCH_SIZE, epochs=1)

    def save_model_weights(self):
        save_model(self.model, "current_best.h5")
        save_model(self.target_model, "current_best_target.h5")


def get_reward(state, angle, posY, posX, vY, vX, contact, action, av, prev_shaping):
    if state == "dnf":
        return -1, None

    reward = 0
    shaping = \
        - 100 * np.sqrt((posX - 0.5) ** 2 + posY ** 2) \
        - 100 * np.sqrt(vX ** 2 + vY ** 2) \
        - 100 * abs(angle) + 10 * contact  # And ten points for legs contact, the idea is if you
    # lose contact again after landing, you get negative reward
    if prev_shaping is not None:
        reward = shaping - prev_shaping

    # reward -= m_power * 0.30  # less fuel spent is better, about -30 for heuristic landing
    # reward -= s_power * 0.03

    if state == "destroyed":
        reward = -100
    if state == "landed":
        reward = +100

    return reward, shaping


def convert_action(best):
    """
    0: L
    1: M
    2: R
    3: L+M
    4: R+M
    5: L+M+R
    6: None
    """

    """
    0: None
    1: L+M
    2: R+M
    3: L+M+R
    """

    if best == 0:
        return [0, 0, 0, 0]
    elif best == 1:
        return [0, 1, 0, 0]
    elif best == 2:
        return [0, 0, 1, 0]
    elif best == 3:
        return [1, 1, 1, 0]
    else:
        return [0, 0, 0, 0]

    # if best == 6:
    #     return [0, 0, 0, 0]
    # elif best == 5:
    #     return [1, 1, 1, 0]
    # elif best == 4:
    #     return [1, 0, 1, 0]
    # elif best == 3:
    #     return [1, 1, 0, 0]
    # elif best == 2:
    #     return [0, 1, 0, 0]
    # elif best == 1:
    #     return [0, 0, 1, 0]
    # elif best == 0:
    #     return [1, 0, 0, 0]
    # else:
    #     return [0, 0, 0, 0]


def convert_action_inverse(keys):
    if keys == [0, 0, 0, 0]:
        return 0
    elif keys == [1, 1, 1, 0]:
        return 3
    elif keys == [0, 1, 0, 0]:
        return 1
    elif keys == [0, 0, 1, 0]:
        return 2
    else:
        return 0


def train_iters(episodes=10000, load=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--socket", action="store_true", help="Run game in socket mode")
    parser.add_argument("-i", "--ip", type=str, default='127.0.0.1', help="IP address for socket mode")
    parser.add_argument("-p", "--port", type=int, default=50007, help="Port")
    parser.add_argument("-d", "--display", action="store_true", help="Run without graphics. Text output only.")
    parser.add_argument("-t", "--test", type=int, default=-42, help="Test mode. Enter iterations number.")  # 42000
    args = parser.parse_args()

    if args.test > 0:
        test_iterations = args.test
        log_file = open("./log/log.txt", "w")
        log_file.write("[")

    options = Options(args.socket, args.ip, args.port, args.display)
    world = World(options)
    simulation = Simulation(options, max_steps=episodes)
    entities = [Rocket(world), Platform(world)]
    agent = Agent(load_weights=load)

    trial_len = 750
    rewards = []

    while simulation.running:
        simulation.__restart__(world, entities)

        action = [0, 0, 0, 0]
        report = simulation.step(world, entities, action=convert_action(action))

        state = np.array(
            [report["Angular Velocity"], report["Vx"], report["Vy"], report["Posx"], report["Posy"],
             report["Angle"], report["Contact"]]).reshape(1, FEATURE_SIZE)

        prev_shaping = None

        current_state_rewards = []
        x_pos = []
        y_pos = []
        angle = []
        av = []

        for step in range(trial_len):
            action = agent.act(state)
            report = simulation.step(world, entities, action=convert_action(action))

            new_state = np.array(
                [report["Angular Velocity"], report["Vx"], report["Vy"], report["Posx"], report["Posy"],
                 report["Angle"], report["Contact"]]).reshape(1, FEATURE_SIZE)

            reward, shaping = get_reward(report["State"], report["Angle"], state[0][4], state[0][3], state[0][2],
                                         report["Vx"], report["Contact"], action, report["Angular Velocity"],
                                         prev_shaping)
            prev_shaping = shaping

            current_state_rewards.append(reward)
            x_pos.append(state[0][3])
            y_pos.append(state[0][4])
            angle.append(report["Angle"])
            av.append(report["Angular Velocity"])

            print(report)
            print(reward)

            agent.get_sample((state, action, reward, new_state))

            state = new_state

            agent.backward()
            agent.target_train()

            if not load:
                agent.save_model_weights()

            if report["State"] != "none":
                break

        rewards.append(np.sum(current_state_rewards))

        idx = 0

        data_to_plot = [
            (current_state_rewards, "Step #", "Reward"),
            (x_pos, "Step #", "x_pos"),
            (y_pos, "Step #", "y_pos"),
            (angle, "Step #", "angle"),
            (av, "Step #", "angular velocity"),
            (rewards, "Episode #", "Reward"),
        ]

        plt.figure(0)
        for i in range(3):
            for j in range(2):
                ax = plt.subplot2grid((3, 2), (i, j))
                ax.plot(range(len(data_to_plot[idx][0])), data_to_plot[idx][0])
                ax.set_xlabel(data_to_plot[idx][1])
                ax.set_ylabel(data_to_plot[idx][2])
                idx += 1

        plt.draw()
        plt.pause(0.001)

        print("Average Reward: %s" % np.mean(current_state_rewards))


if __name__ == '__main__':
    train_iters(1000000)
