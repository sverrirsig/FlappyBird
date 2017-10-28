from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import numpy
import itertools
import time
import os
import re
import matplotlib.pyplot as plt

class FlappyAgent:
    def __init__(self):
        # TODO: you may need to do some initialization for your agent here
        return
    
    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}
    
    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        # TODO: learn from the observation
        return

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        print("state: %s" % state)
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.
        return random.randint(0, 1) 

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        print("state: %s" % state)
        # TODO: change this to to policy the agent has learned
        # At the moment we just return an action uniformly at random.
        return random.randint(0, 1)


class FlappyAgentMCAverage(FlappyAgent):
    def __init__(self):
        super(FlappyAgentMCAverage, self).__init__()

        self.y_pos_intervals = [x[-1] for x in numpy.array_split(numpy.array(range(0, 388)), 15)]
        self.top_y_gap_intervals = [x[-1] for x in numpy.array_split(numpy.array(range(25, 193)), 15)]
        self.velocity_intervals = [x[-1] for x in numpy.array_split(numpy.array(range(-8, 11)), 15)]
        self.horizontal_distance_next_pipe = [x[-1] for x in numpy.array_split(numpy.array(range(3, 284)), 15)] # ToDo: Maybe refactor and make the first interval a bit bigger.

        self.states = list(itertools.product(*[
            self.y_pos_intervals,
            self.top_y_gap_intervals,
            self.horizontal_distance_next_pipe,
            self.velocity_intervals,
        ]))

        self.Q = {}
        self.pi = {}
        for state in self.states:
            self.pi[state] = random.randint(0, 1)
            for action in range(0, 2):
                self.Q[(state, action)] = (0, 0)
                self.pi[state] = 0 if random.randint(0, 2) == 0 else 1

        self.observations = []

        self.discount = 1
        self.epsilon = 0.1

    def reward_values(self):
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

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
                if self.Q[(s, 0)] > self.Q[(s, 1)]:
                    self.pi[s] = 0
                else:
                    self.pi[s] = 1

            self.observations = []

    def training_policy(self, state):
        actions = [0, 1]

        greedy_action = self.pi[state]

        if random.uniform(0, 1) < 0.95:
            return greedy_action
        else:
            return [x for x in actions if x != greedy_action][0]

    def policy(self, state):
        return self.pi[state]

    # Gets a state in the original format that PyGame returns.
    # Both extracts the 4 keys in the problem description
    # And maps the value to the corresponding interval.
    # Returns a tuple of:
    # 1. the current y-position of the bird (player_y component of the game state)
    # 2. the top y position of the next gap (next_pipe_top_y)
    # 3. the horizontal distance between bird and next pipe (next_pipe_dist_to_player)
    # 4. the current velocity of the bird (player_vel)
    def parse_state(self, state):
        y_pos = min(self.y_pos_intervals, key=lambda x:abs(x - state['player_y']))
        top_y_gap = min(self.top_y_gap_intervals, key=lambda x:abs(x - state['next_pipe_top_y']))
        horizontal_distance_next_pipe = min(self.horizontal_distance_next_pipe, key=lambda x:abs(x - state['next_pipe_dist_to_player']))
        velocity = min(self.velocity_intervals, key=lambda x:abs(x - state['player_vel']))

        return y_pos, top_y_gap, horizontal_distance_next_pipe, velocity

class FlappyAgentMCLearningRate(FlappyAgent):
    def __init__(self, LearningRate):
        super(FlappyAgentMCLearningRate, self).__init__()

        self.y_pos_intervals = [x[-1] for x in numpy.array_split(numpy.array(range(0, 388)), 15)]
        self.top_y_gap_intervals = [x[-1] for x in numpy.array_split(numpy.array(range(25, 193)), 15)]
        self.velocity_intervals = [x[-1] for x in numpy.array_split(numpy.array(range(-8, 11)), 15)]
        self.horizontal_distance_next_pipe = [x[-1] for x in numpy.array_split(numpy.array(range(3, 284)), 15)] # ToDo: Maybe refactor and make the first interval a bit bigger.

        self.states = list(itertools.product(*[
            self.y_pos_intervals,
            self.top_y_gap_intervals,
            self.horizontal_distance_next_pipe,
            self.velocity_intervals,
        ]))

        self.Q = {}
        self.pi = {}
        for state in self.states:
            self.pi[state] = random.randint(0, 1)
            for action in range(0, 2):
                self.Q[(state, action)] = 0
                self.pi[state] = 0 if random.randint(0, 2) == 0 else 1

        self.observations = []

        self.discount = 1
        self.epsilon = 0.1
        self.learning_rate = LearningRate

    def reward_values(self):
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

    def observe(self, s1, a, r, s2, end):
        self.observations.append((s1, a, r))
        if end:
            G = 0
            for (s, a, r) in reversed(self.observations):
                G = r + self.discount * G
                self.Q[(s, a)] = self.Q[(s, a)] + self.learning_rate * (G - self.Q[(s, a)])

            for (s, a, r) in reversed(self.observations):
                if self.Q[(s, 0)] > self.Q[(s, 1)]:
                    self.pi[s] = 0
                else:
                    self.pi[s] = 1

            self.observations = []

    def training_policy(self, state):
        actions = [0, 1]

        greedy_action = self.pi[state]

        if random.uniform(0, 1) < 0.95:
            return greedy_action
        else:
            return [x for x in actions if x != greedy_action][0]

    def policy(self, state):
        return self.pi[state]

    # Gets a state in the original format that PyGame returns.
    # Both extracts the 4 keys in the problem description
    # And maps the value to the corresponding interval.
    # Returns a tuple of:
    # 1. the current y-position of the bird (player_y component of the game state)
    # 2. the top y position of the next gap (next_pipe_top_y)
    # 3. the horizontal distance between bird and next pipe (next_pipe_dist_to_player)
    # 4. the current velocity of the bird (player_vel)
    def parse_state(self, state):
        y_pos = min(self.y_pos_intervals, key=lambda x:abs(x - state['player_y']))
        top_y_gap = min(self.top_y_gap_intervals, key=lambda x:abs(x - state['next_pipe_top_y']))
        horizontal_distance_next_pipe = min(self.horizontal_distance_next_pipe, key=lambda x:abs(x - state['next_pipe_dist_to_player']))
        velocity = min(self.velocity_intervals, key=lambda x:abs(x - state['player_vel']))

        return y_pos, top_y_gap, horizontal_distance_next_pipe, velocity


class FlappyAgentQLearningLearningRate(FlappyAgent):
    def __init__(self, LearningRate):
        super(FlappyAgentQLearningLearningRate, self).__init__()

        self.y_pos_intervals = [x[-1] for x in numpy.array_split(numpy.array(range(0, 388)), 15)]
        self.top_y_gap_intervals = [x[-1] for x in numpy.array_split(numpy.array(range(25, 193)), 15)]
        self.velocity_intervals = [x[-1] for x in numpy.array_split(numpy.array(range(-8, 11)), 15)]
        self.horizontal_distance_next_pipe = [x[-1] for x in numpy.array_split(numpy.array(range(3, 284)), 15)] # ToDo: Maybe refactor and make the first interval a bit bigger.

        self.states = list(itertools.product(*[
            self.y_pos_intervals,
            self.top_y_gap_intervals,
            self.horizontal_distance_next_pipe,
            self.velocity_intervals,
        ]))

        self.Q = {}
        self.pi = {}
        self.terminalState = (-1, -1, -1, -1)
        for state in self.states:
            self.pi[state] = random.randint(0, 1)
            for action in range(0, 2):
                self.Q[(state, action)] = 0
                self.pi[state] = 0 if random.randint(0, 2) == 0 else 1

        self.observations = []

        self.discount = 1
        self.epsilon = 0.1

        self.learning_rate = LearningRate

    def reward_values(self):
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

    def observe(self, s1, a, r, s2, end):
        self.Q[(s1, a)] = self.Q[(s1, a)] + self.learning_rate * (r + self.discount * self.Q[(s2, self.pi[s2])] - self.Q[(s1, a)])

        if self.Q[(s1, 0)] > self.Q[(s1, 1)]:
            self.pi[s1] = 0
        else:
            self.pi[s1] = 1


    def training_policy(self, state):
        actions = [0, 1]

        greedy_action = self.pi[state]

        if random.uniform(0, 1) < 0.95:
            return greedy_action
        else:
            return [x for x in actions if x != greedy_action][0]

    def policy(self, state):
        return self.pi[state]

    # Gets a state in the original format that PyGame returns.
    # Both extracts the 4 keys in the problem description
    # And maps the value to the corresponding interval.
    # Returns a tuple of:
    # 1. the current y-position of the bird (player_y component of the game state)
    # 2. the top y position of the next gap (next_pipe_top_y)
    # 3. the horizontal distance between bird and next pipe (next_pipe_dist_to_player)
    # 4. the current velocity of the bird (player_vel)
    def parse_state(self, state):
        y_pos = min(self.y_pos_intervals, key=lambda x:abs(x - state['player_y']))
        top_y_gap = min(self.top_y_gap_intervals, key=lambda x:abs(x - state['next_pipe_top_y']))
        horizontal_distance_next_pipe = min(self.horizontal_distance_next_pipe, key=lambda x:abs(x - state['next_pipe_dist_to_player']))
        velocity = min(self.velocity_intervals, key=lambda x:abs(x - state['player_vel']))

        return y_pos, top_y_gap, horizontal_distance_next_pipe, velocity


def run_game(nb_episodes, agent):
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
              reward_values=reward_values)
    env.init()

    elapsed_episodes = 0
    score = 0
    frames = 0
    while elapsed_episodes < nb_episodes:
        state = agent.parse_state(env.game.getGameState())
        frames += 1
        action = agent.training_policy(state)
        reward = env.act(env.getActionSet()[action])
        state2 = agent.parse_state(env.game.getGameState())

        agent.observe(state, action, reward, state2, env.game_over())

        score += reward
        scores = set()
        if env.game_over():
            #print("score for this episode: %d" % score)
            scores.add(score)
            env.reset_game()
            score = 0
            elapsed_episodes += 1
            print(elapsed_episodes)
            if elapsed_episodes % 1000 == 0:
                results = {"Frames": frames, "Policy": agent.pi}
                numpy.save("Q_Learning/LR_Frames_" + str(frames) + ".npy", results)


def test_policy(nb_episodes, agent):
    reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
              reward_values=reward_values)
    env.init()
    scores = []
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
    return max(scores), (sum(scores)/len(scores))


def iterate_policies(folder, name, total, step):
    directory = os.fsencode(folder)
    episodes = []
    scores = []
    files = []

    for episode in range(step, total+1, step):
        file = folder + name + str(episode) + ".npy"
        print(file)
        agent = FlappyAgentMCLearningRate(0.1)
        agent.pi = numpy.load(file).item()
        max_score, average = test_policy(30, agent)
        episodes.append(episode)
        scores.append(average)
    print(episodes)
    print(scores)
    plt.figure(figsize=(20, 10))
    plt.plot(episodes, scores)
    plt.savefig("Monte_Carlo/" + name + ".png")
    plt.show()



agent = FlappyAgentQLearningLearningRate(0.1)
run_game(50000, agent)
#pi = numpy.load("Q_Learning/LR_Frames_564.npy").item()
#agent.pi = pi
#test_policy(1, agent)



#iterate_policies("Monte_Carlo_Average/", "Average_Episodes_", 50000, 1000)

