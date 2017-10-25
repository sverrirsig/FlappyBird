
# An abstract class representing a Markov Decision Process (MDP).
class MDP:
    def __init__(self):
        self.computeStates()

    # discount factor
    discountFactor = 1

    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(self, state): raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', reward = r, prob = p(s', r | s, a)
    # If state is a terminal state, return the empty list.
    def succAndProbReward(self, state, action): raise NotImplementedError("Override me")

    # Compute set of states reachable from startState.  Helper function
    # to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)
        print "%d reachable states" % len(self.states)
        # print self.states
        
class BlackjackMDP(MDP):
    
    # the discount factor for future rewards
    discountFactor = 1

    # all possible card values and associated probabilities
    cardValueProbs = [
        (2, 4.0/52), (3, 4.0/52), (4, 4.0/52), (5, 4.0/52), (6, 4.0/52),
        (7, 4.0/52), (8, 4.0/52), (9, 4.0/52), (10, 16.0/52), (11, 4.0/52) ]

    # the one and only terminal state of this MDP
    terminalState = ('end', 0, False, 0, False)
    
    # Return the start state.
    def startState(self):
        # a state is a 5-tuple (phase, playerHand, usableAcePlayer, dealerHand, usableAceDealer)
        # there are 4  phases: start, player, dealer, end
        return ('start', 0, False, 0, False)

    # Return set of actions possible from |state|.
    def actions(self, state):
        # TODO: you may need to change this depending on your model
        #       not all actions may always be possible
        phase = state[0]
        if phase == 'player':
            return ['Hit', 'Stand']
        else:
            return ['Noop']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', reward = r, prob = p(s', r | s, a)
    # If state is a terminal state, return the empty list.
    def succAndProbReward(self, state, action):
        # TODO: implement this
        phase = state[0]
        results = []
        if phase == 'start':
            for (cardP1, probP1) in self.cardValueProbs:
                for (cardP2, probP2) in self.cardValueProbs:
                    playerHand = cardP1 + cardP2
                    usableAcePlayer = 11 in (cardP1, cardP2)
                    if playerHand > 21:
                        playerHand -= 10
                    for (cardD, probD) in self.cardValueProbs:
                        newState = ('player', playerHand, usableAcePlayer, cardD, (cardD == 11) )
                        results.append( (newState, probP1*probP2*probD, 0) )
        elif phase == 'end':
            results.append( (self.terminalState, 1, 0) )
        # else:
        
        return results

def valueIteration(mdp):
    # initialize the value estimates to 0
    v = {}
    for state in mdp.states:
        v[state] = 0
    
    # TODO: do actual value iteration

    pi = {}
    # TODO: extract policy
    
    return (v,pi)
    
    

mdp = BlackjackMDP()
(v, pi) = valueIteration(mdp)
