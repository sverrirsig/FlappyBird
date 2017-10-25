from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import numpy
import itertools

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


class FlappyAgentMC(FlappyAgent):
    def __init__(self):
        super(FlappyAgentMC, self).__init__()

        self.y_pos_intervals = [x[-1] for x in numpy.array_split(numpy.array(range(0, 388)), 15)]
        self.top_y_gap_intervals = [x[-1] for x in numpy.array_split(numpy.array(range(25, 193)), 15)]
        self.velocity_intervals = [x[-1] for x in numpy.array_split(numpy.array(range(-8, 11)), 15)]
        self.horizontal_distance_next_pipe = [x[-1] for x in numpy.array_split(numpy.array(range(3, 284)), 15)] # ToDo: Maybe refactor and make the first interval a bit bigger.

        self.states = list(itertools.product(*[
            self.y_pos_intervals,
            self.top_y_gap_intervals,
            self.velocity_intervals,
            self.horizontal_distance_next_pipe
        ]))

        self.Q = {}
        for state in self.states:
            for action in range(0, 2):
                self.Q[(state, action)] = 0

        self.pi = {}
        for state in self.states:
            self.pi[state] = random.randint(0, 1)

        self.observations = []

        self.discount = 1

    # Q(s,a) := (Q, s, a) + 1/n(s,a) * [G - Q(s,a)]

    def reward_values(self):
        """ returns the reward values used for training

            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

    def observe(self, s1, a, r, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        self.observations.append((s1, a, r))
        if end:
            print("Rakki")
            # Todo Learn
            G = 0
            for (s, a, r) in reversed(self.observations):
                G = r * self.discount * G
                

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
        return self.pi[state]

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        print("state: %s" % state)
        # TODO: change this to to policy the agent has learned
        # At the moment we just return an action uniformly at random.
        return 1

    # Gets a state in the original format that PyGame returns.
    # Both extracts the 4 keys in the problem description
    # And maps the value to the corresponding interval.
    def parse_state(self, state):
        new_state = {}
        new_state['y_pos'] = min(self.y_pos_intervals, key=lambda x:abs(x - state['player_y']))
        new_state['top_y_gap'] = min(self.y_pos_intervals, key=lambda x:abs(x - state['next_pipe_top_y']))
        new_state['horizontal_distance_next_pipe'] = min(self.y_pos_intervals, key=lambda x:abs(x - state['next_pipe_dist_to_player']))
        new_state['velocity'] = min(self.y_pos_intervals, key=lambda x:abs(x - state['player_vel']))

        return new_state


def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    #reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    # TODO: when training use the following instead:
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
              reward_values=reward_values)
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True 
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        # TODO: for training using agent.training_policy instead
        state = agent.parse_state(env.game.getGameState())

        action = agent.training_policy(state)

        # step the environment
        reward = env.act(env.getActionSet()[action])
        print("reward=%d" % reward)

        # TODO: for training let the agent observe the current state transition

        score += reward

        agent.observe(state, action, reward, env.game_over())

        # reset the environment if the game is over
        if env.game_over():
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0


agent = FlappyAgentMC()
print(len(agent.states))
run_game(1, agent)
