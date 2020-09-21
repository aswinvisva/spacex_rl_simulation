import argparse
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import save_model, load_model
import numpy as np
import random
from collections import deque

from F9LanderCORE import Options, World, Simulation, Rocket, Platform

ACTIONS = 7

"""
0: L
1: M
2: R
3: L+M
4: R+M
5: L+M+R
6: None
"""

MEMORY_SIZE = 1024
BATCH_SIZE = 128
FEATURE_SIZE = 6
# MIN_STEPS = 30
GAMMA = 0.975
OBSERVE_PERIOD = 256


class Agent:

    def __init__(self, load_weights=False):
        self.load_weights = load_weights
        self.memory = deque()
        self.steps = 0
        self.eps = 0.65
        self.eps_decay = 0.975

        if not self.load_weights:
            self.model = Sequential()
            self.model.add(Dense(16, activation='linear', input_dim=FEATURE_SIZE))
            self.model.add(Dense(8, activation='linear'))
            self.model.add(Dense(ACTIONS, activation='linear', kernel_initializer='he_normal'))
            self.model.compile(loss='mse', optimizer='adam')
        else:
            self.model = load_model('rl_best_agent.h5')

    def get_prediction(self, state):

        if self.eps < random.uniform(0, 1):
            result = self.model.predict(state)
            print(result)
            best = np.argmax(result)
        else:
            best = self.explore()

        return best

    def get_sample(self, sample):
        self.memory.append(sample)
        self.steps = self.steps + 1
        if len(self.memory) > MEMORY_SIZE:
            self.memory.popleft()

    def explore(self):
        print("Exploring...")
        return random.randint(0, FEATURE_SIZE - 1)

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


def get_reward(state, dist, angle, posY, vY, fuel, av):
    reward = 0

    reward += math.cos(1.25 * angle)  # How close to 0 degrees

    reward += math.cos(1.25 * av)  # How close to 0 degrees

    reward -= np.tanh(abs(dist))

    reward -= np.tanh(abs(posY))

    reward -= np.tanh(abs(vY))

    reward += fuel / 10

    if state == "landed":
        reward += 1
    elif state == "destroyed":
        reward -= 1

    print("Reward", reward)

    return reward


def convert_action(best):
    if best == 6:
        return [0, 0, 0, 0]
    elif best == 5:
        return [1, 1, 1, 0]
    elif best == 4:
        return [1, 0, 1, 0]
    elif best == 3:
        return [1, 1, 0, 0]
    elif best == 2:
        return [0, 1, 0, 0]
    elif best == 1:
        return [0, 0, 1, 0]
    elif best == 0:
        return [1, 0, 0, 0]
    else:
        return [0, 0, 0, 0]


def train_iters(episodes=10000, load=False):
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
    memory = []
    action = [0, 0, 0, 0]
    report = simulation.step(world, entities, action=convert_action(action))
    state = np.array(
        [report["Angular Velocity"], report["Vx"], report["Vy"], report["Posx"], report["Posy"], report["Angle"]]).reshape(
        1,
        FEATURE_SIZE)
    while simulation.running:
        action = agent.get_prediction(state)
        report = simulation.step(world, entities, action=convert_action(action))
        new_state = np.array(
            [report["Angular Velocity"], report["Vx"], report["Vy"], report["Posx"], report["Posy"], report["Angle"]]).reshape(1,
                                                  FEATURE_SIZE)

        print(report)

        if report["State"] != "none":
            for item in memory:
                reward = get_reward(report["State"], item[3], item[4], item[2][0][4], item[2][0][2], item[2][0][0],
                                    item[5])
                agent.get_sample((item[0], item[1], reward, item[2]))

            agent.backward()
        else:
            memory.append((state, action, new_state, report["Dist"], report["Angle"], report["Angular Velocity"]))

        state = new_state
        agent.save_model_weights()


if __name__ == '__main__':
    train_iters(1000000)
