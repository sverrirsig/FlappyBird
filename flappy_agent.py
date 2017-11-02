from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import numpy
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import os


class FlappyAgent:
    def __init__(self):
        self.y_pos_intervals = [x[-1] for x in numpy.array_split(numpy.array(range(0, 388)), 15)]
        self.top_y_gap_intervals = [x[-1] for x in numpy.array_split(numpy.array(range(25, 193)), 15)]
        self.velocity_intervals = [x[-1] for x in numpy.array_split(numpy.array(range(-8, 11)), 15)]
        self.horizontal_distance_next_pipe = [x[-1] for x in numpy.array_split(numpy.array(range(3, 284)), 15)]

        self.states = list(itertools.product(*[
            self.y_pos_intervals,
            self.top_y_gap_intervals,
            self.horizontal_distance_next_pipe,
            self.velocity_intervals,
        ]))

        self.actions = [0, 1]
        self.Q = {}
        self.pi = {}
        for state in self.states:
            self.pi[state] = random.randint(0, 1)
            for action in self.actions:
                self.Q[(state, action)] = 0

        self.observations = []

    def reward_values(self):
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}
    
    def observe(self, s1, a, r, s2, end):
        print("Observe should be overwritten!")
        return

    def training_policy(self, state):
        greedy_action = self.pi[state]

        if random.uniform(0, 1) < (1 - self.epsilon + (self.epsilon / len(self.actions))):
            return greedy_action
        else:
            return [x for x in self.actions if x != greedy_action][0]

    def policy(self, state):
        return self.pi[state]

    def update_policy_fixed(self, s1):
        if self.Q[(s1, 0)] > self.Q[(s1, 1)]:
            self.pi[s1] = 0
        else:
            self.pi[s1] = 1

    def update_policy_average(self, s1):
        if self.Q[(s1, 0)][0] > self.Q[(s1, 1)][0]:
            self.pi[s1] = 0
        else:
            self.pi[s1] = 1

    # Gets a state in the original format that PyGame returns.
    # Both extracts the 4 keys in the problem description
    # And maps the value to the corresponding interval.
    # Returns a tuple of:
    # 1. the current y-position of the bird (player_y component of the game state)
    # 2. the top y position of the next gap (next_pipe_top_y)
    # 3. the horizontal distance between bird and next pipe (next_pipe_dist_to_player)
    # 4. the current velocity of the bird (player_vel)
    def parse_state(self, state):
        y_pos = min(self.y_pos_intervals, key=lambda x: abs(x - state['player_y']))
        top_y_gap = min(self.top_y_gap_intervals, key=lambda x: abs(x - state['next_pipe_top_y']))
        horizontal_distance_next_pipe = min(self.horizontal_distance_next_pipe,
                                            key=lambda x: abs(x - state['next_pipe_dist_to_player']))
        velocity = min(self.velocity_intervals, key=lambda x: abs(x - state['player_vel']))

        return y_pos, top_y_gap, horizontal_distance_next_pipe, velocity


class FlappyAgentMCAverage(FlappyAgent):
    def __init__(self, epsilon=0.1, discount=1):
        super(FlappyAgentMCAverage, self).__init__()

        self.discount = discount
        self.epsilon = epsilon
        self.method = "Monte_Carlo_Average"
        for state in self.states:
            for action in self.actions:
                self.Q[(state, action)] = (0, 0)

    def observe(self, s1, a, r, s2, end):
        self.observations.append((s1, a, r))
        if end:
            G = 0
            for (s, a, r) in reversed(self.observations):
                G = r + self.discount * G
                old_average = self.Q[(s, a)][0]
                old_count = self.Q[(s, a)][1]
                total = old_count * old_average

                new_count = old_count + 1
                new_average = (total+G) / new_count

                self.Q[(s, a)] = (new_average, new_count)

            for (s, a, r) in reversed(self.observations):
                self.update_policy_average(s)

            self.observations = []


# Flappy Agent that uses Monte-Carlo and a fixed learning rate.
class FlappyAgentMCLearningRate(FlappyAgent):
    def __init__(self, LearningRate=0.1, epsilon=0.1, discount=1):
        super(FlappyAgentMCLearningRate, self).__init__()

        self.discount = discount
        self.epsilon = epsilon
        self.learning_rate = LearningRate
        self.method = "Monte_Carlo_Learning_Rate"

    def observe(self, s1, a, r, s2, end):
        self.observations.append((s1, a, r))
        if end:
            G = 0
            for (s, a, r) in reversed(self.observations):
                G = r + self.discount * G
                self.Q[(s, a)] = self.Q[(s, a)] + self.learning_rate * (G - self.Q[(s, a)])

            for (s, a, r) in reversed(self.observations):
                self.update_policy_fixed(s)

            self.observations = []


class FlappyAgentQLearningLearningRate(FlappyAgent):
    def __init__(self, LearningRate=0.1, epsilon=0.1, discount=1):
        super(FlappyAgentQLearningLearningRate, self).__init__()

        self.discount = discount
        self.epsilon = epsilon
        self.learning_rate = LearningRate  # Todo: 0.11 er mjög gott, prófa kannski eitthvað í kringum það.
        self.method = "Q_Learning_Fixed"

    def observe(self, s1, a, r, s2, end):
        self.Q[(s1, a)] = self.Q[(s1, a)] + self.learning_rate * (r + self.discount * self.Q[(s2, self.pi[s2])] - self.Q[(s1, a)])

        self.update_policy_fixed(s1)


class FlappyAgentQLearningAverage(FlappyAgent):
    def __init__(self, epsilon=0.1, discount=1):
        super(FlappyAgentQLearningAverage, self).__init__()

        self.discount = discount
        self.epsilon = epsilon
        self.method = "Q_Learning_Average"
        for state in self.states:
            for action in self.actions:
                self.Q[(state, action)] = (0, 0)

    def observe(self, s1, a, r, s2, end):
        new_count = self.Q[(s1, a)][1]
        new_count += 1
        new_val = self.Q[(s1, a)][0]
        new_val = new_val + 1/new_count * (r + self.discount * self.Q[(s2, self.pi[s2])][0] - self.Q[(s1, a)][0])
        self.Q[(s1, a)] = (new_val, new_count)

        self.update_policy_average(s1)


class FlappyAgentQLearningElite(FlappyAgent):
    def __init__(self, LearningRate=0.11, epsilon=0.1, discount=1):
        super(FlappyAgentQLearningElite, self).__init__()

        self.discount = discount
        self.epsilon = epsilon
        self.learning_rate = LearningRate  # Todo: 0.11 er mjög gott, prófa kannski eitthvað í kringum það.
        self.method = "Q_Learning_Elite"

    def observe(self, s1, a, r, s2, end):
        self.Q[(s1, a)] = self.Q[(s1, a)] + self.learning_rate * (r + self.discount * self.Q[(s2, self.pi[s2])] - self.Q[(s1, a)])

        self.update_policy_fixed(s1)


def save_policy(agent, frames, folder=""):
    results = {"Frames": frames, "Policy": agent.pi}
    separator = ""
    if len(folder) > 0:
        separator = "/"
    if len(folder) > 0 and not os.path.exists(folder):
        os.makedirs(folder)
    numpy.save(folder + separator + agent.method + "_" + str(frames) + ".npy", results)
    print("Saved %s agents policy after training %d frames." % (agent.method, frames))


def run_game(agent, nb_episodes=0, frames_to_train=0, folder=""):
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
              reward_values=reward_values)
    env.init()

    elapsed_episodes = 0
    frames = 0
    score = 0
    while True:
        state = agent.parse_state(env.game.getGameState())
        action = agent.training_policy(state)
        reward = env.act(env.getActionSet()[action])
        state2 = agent.parse_state(env.game.getGameState())

        agent.observe(state, action, reward, state2, env.game_over())

        frames += 1
        if frames % 50000 == 0:
            save_policy(agent, frames, folder)
        score += reward
        if env.game_over():
            env.reset_game()
            elapsed_episodes += 1
            score = 0
            if frames_to_train != 0 and frames >= frames_to_train:
                save_policy(agent, frames, folder)
                break
            if nb_episodes != 0 and elapsed_episodes >= nb_episodes:
                save_policy(agent, frames, folder)
                break


def test_policy(nb_episodes, agent):
    reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
              reward_values=reward_values)
    env.init()
    print("Playing game as %s" % bird.method)

    scores = []
    score = 0
    score = 0
    while nb_episodes > 0:
        state = agent.parse_state(env.game.getGameState())
        action = agent.policy(state)

        reward = env.act(env.getActionSet()[action])
        score += reward

        if env.game_over():
            print("score for this episode: %d" % score)
            env.reset_game()
            scores.append(score)
            score = 0
            nb_episodes -= 1

    print("Best score: %d" % max(scores))
    print("Average: %f" % (sum(scores)/len(scores)))
    return max(scores), (sum(scores)/len(scores)), scores


def generate_learning_curve(folder, name, average_scores, max_scores, frames):
    plt.figure(figsize=(20, 10))
    plt.title(folder)
    plt.xlabel("Frames trained on")
    plt.ylabel("Score, 50 runs")
    average_line, = plt.plot(frames, average_scores, label="Average")
    max_line, = plt.plot(frames, max_scores, label="Max")
    plt.legend(handles=[average_line, max_line])
    plt.savefig(folder + name + ".png")


def generate_box_plot(folder, name, scores, frames):
    frames = [str(frame)[:-3] + "k" for frame in frames[1::2]]
    scores = scores[1::2]
    plt.figure(figsize=(20, 10))
    plt.title(folder)
    plt.xlabel("Frames trained on")
    plt.ylabel("Score, 50 runs")
    sns.boxplot(x=frames, y=scores)
    plt.savefig(folder + name + "box_plot" + ".png")


def evaluate_policies(agent_to_test, folder, name, total, step):
    frames = []
    average_scores = []
    max_scores = []
    scores = []
    for frame in range(step, total+1, step):
        file = folder + name + str(frame) + ".npy"
        try:
            agent_to_test.pi = numpy.load(file).item()["Policy"]
        except FileNotFoundError:
            print("Rakki")
            break
        max_score, average, score = test_policy(50, agent_to_test)
        frames.append(frame)
        scores.append(score)
        average_scores.append(average)
        max_scores.append(max_score)

    #generate_box_plot(folder, name, scores, frames)
    generate_learning_curve(folder, name, average_scores, max_scores, frames)


# for rate in [0.105]:
#     bird = FlappyAgentQLearningElite(LearningRate=rate)
#     run_game(bird, 0, 2000000, "rate_" + str(rate))

# for rate in [0.115]:
#     bird = FlappyAgentQLearningElite(LearningRate=rate)
#     run_game(bird, 0, 2000000, "rate_" + str(rate))
#
# for rate in [0.12]:
#     bird = FlappyAgentQLearningElite(LearningRate=rate)
#     run_game(bird, 0, 2000000, "rate_" + str(rate))

# for epsi in [0.15]:
#     bird = FlappyAgentQLearningElite(epsilon=epsi)
#     run_game(bird, 0, 2000000, "epsilon_" + str(epsi))
#
# for epsi in [0.3]:
#     bird = FlappyAgentQLearningElite(epsilon=epsi)
#     run_game(bird, 0, 2000000, "epsilon_" + str(epsi))
# #
# for epsi in [0.5, 1]:
#     bird = FlappyAgentQLearningElite(epsilon=epsi)
#     run_game(bird, 0, 2000000, "epsilon_" + str(epsi))


# run_game(200000, bird, 2000000, "Q_average")


# evaluate_policies(bird, "Q_average/", "Q_Learning_", 2000000, 50000)
#
#
bird = FlappyAgentQLearningElite()
evaluate_policies(bird, "rate_0.105/", "Q_Learning_Elite_", 2000000, 50000)
evaluate_policies(bird, "rate_0.115/", "Q_Learning_Elite_", 2000000, 50000)
evaluate_policies(bird, "rate_0.12/", "Q_Learning_Elite_", 2000000, 50000)
# evaluate_policies(bird, "epsilon_0.3/", "Q_Learning_Elite_", 2000000, 50000)
# evaluate_policies(bird, "epsilon_0.4/", "Q_Learning_Elite_", 2000000, 50000)
# evaluate_policies(bird, "epsilon_0.5/", "Q_Learning_Elite_", 2000000, 50000)
# evaluate_policies(bird, "epsilon_0.05/", "Q_Learning_Elite_", 2000000, 50000)
# evaluate_policies(bird, "epsilon_0.15/", "Q_Learning_Elite_", 2000000, 50000)
# evaluate_policies(bird, "epsilon_1/", "Q_Learning_Elite_", 2000000, 50000)

