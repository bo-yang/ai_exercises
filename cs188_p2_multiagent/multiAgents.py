# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        WEIGHT_FOOD = 10.0
        WEIGHT_GHOST = 10.0

        score = successorGameState.getScore()
        distance = lambda x, y: ((x[0] - y[0])**2 + (x[1] - y[1])**2) ** 0.5
        dist_to_food = [distance(newPos, pos) for pos in newFood.asList()]
        if len(dist_to_food) > 0:
            score += WEIGHT_FOOD/min(dist_to_food)

        #dist_to_ghost = [distance(newPos, pos) for pos in successorGameState.getGhostPositions()]
        dist_to_ghost = distance(newPos, newGhostStates[0].getPosition())
        if dist_to_ghost != 0:
            score -= WEIGHT_GHOST/dist_to_ghost

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.INF_POS = float('inf')
        self.INF_NEG = float('-inf')

    # is this a terminal state
    def isTerminal(self, state, depth, agent):
        return depth == self.depth or \
               state.isWin() or \
               state.isLose() or \
               state.getLegalActions(agent) == 0

    # is this agent pacman
    def isPacman(self, state, agent):
        return agent % state.getNumAgents() == 0

    def _minimax(self, state, depth, agent, alpha = None, beta = None):
        """
        state - game state
        depth - search depth
        ismax - max player(true) or min player(false)
        """
        if agent == state.getNumAgents():  # is pacman
            return self._minimax(state, depth + 1, 0, alpha, beta)  # start next depth

        if self.isTerminal(state, depth, agent):
            return (self.evaluationFunction(state), None)

        bestScore = None
        bestAction = None
        if self.isPacman(state, agent):
            # max player
            func = max
            bestScore = self.INF_NEG
            is_pacman = True
        else:
            # min player
            func = min
            bestScore = self.INF_POS
            is_pacman = False

        for action in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent, action)
            (score,_) = self._minimax(successor, depth, agent + 1, alpha, beta)
            (bestScore, bestAction) = func((bestScore, bestAction), (score, action))
            if alpha != None and beta != None:
                if is_pacman:
                    if bestScore > beta:
                        #print "Pacman Pruning: (bestScore = %f, action = %s, alpha = %f, beta = %f)" %(bestScore, action, alpha, beta)
                        break
                    alpha = func(alpha, bestScore)
                else:
                    if bestScore < alpha:
                        #print "Ghost Pruning: bestScore = %f, action = %s, alpha = %f, beta = %f" %(bestScore, action, alpha, beta)
                        break
                    beta = func(beta, bestScore)
        return (bestScore, bestAction)

    def _expectimax(self, state, depth, agent):
        """
        state - game state
        depth - search depth
        ismax - max player(true) or min player(false)
        """
        if agent == state.getNumAgents():  # is pacman
            return self._expectimax(state, depth + 1, 0)  # start next depth

        if self.isTerminal(state, depth, agent):
            return (self.evaluationFunction(state), None)

        bestScore = None
        bestAction = None
        if self.isPacman(state, agent):
            # max player
            bestScore = self.INF_NEG
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                (score,_) = self._expectimax(successor, depth, agent + 1)
                (bestScore, bestAction) = max((bestScore, bestAction), (score, action))
        else:
            # min player
            bestScore = self.INF_POS
            totalScore = 0.0
            allActions = state.getLegalActions(agent)
            for action in allActions:
                successor = state.generateSuccessor(agent, action)
                (score,_) = self._expectimax(successor, depth, agent + 1)
                (bestScore, bestAction) = min((bestScore, bestAction), (score, action))
                totalScore += score
            bestScore = (totalScore / len(allActions))

        return (bestScore, bestAction)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        _, action = self._minimax(gameState, 0, 0)
        return action

#        # Collect legal moves and successor states
#        legalMoves = gameState.getLegalActions()
#
#        # Choose one of the best actions
#        scores = [self._minimax(gameState.generateSuccessor(0, action), 0, 1) for action in legalMoves]
#        bestScore = max(scores)
#        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
#        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
#        return legalMoves[chosenIndex]


#        def minimax(state, depth, agent):
#            if agent == state.getNumAgents():  # is pacman
#                return minimax(state, depth + 1, 0)  # start next depth
#
#            if self.isTerminal(state, depth, agent):
#                return self.evaluationFunction(state)  # return evaluation for bottom states
#
#            # find the "best" (min or max) state of the successors
#            successors = (
#                minimax(state.generateSuccessor(agent, action), depth, agent + 1)
#                for action in state.getLegalActions(agent)
#            )
#            return (max if self.isPacman(state, agent) else min)(successors)
#
#        # return the best of pacman's possible moves
#        return max(gameState.getLegalActions(0),
#                   key = lambda x: minimax(gameState.generateSuccessor(0, x), 0, 1)
#        )

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        _, action = self._minimax(gameState, 0, 0, alpha=self.INF_NEG, beta=self.INF_POS)
        return action

#        def dispatch(state, depth, agent, A=float("-inf"), B=float("inf")):
#            if agent == state.getNumAgents():  # next depth
#                depth += 1
#                agent = 0
#
#            if self.isTerminal(state, depth, agent):  # dead end
#                return self.evaluationFunction(state), None
#
#            if self.isPacman(state, agent):
#                return getValue(state, depth, agent, A, B, float('-inf'), max)
#            else:
#                return getValue(state, depth, agent, A, B, float('inf'), min)
#
#        def getValue(state, depth, agent, A, B, ms, mf):
#            bestScore = ms
#            bestAction = None
#
#            for action in state.getLegalActions(agent):
#                successor = state.generateSuccessor(agent, action)
#                score,_ = dispatch(successor, depth, agent + 1, A, B)
#                bestScore, bestAction = mf((bestScore, bestAction), (score, action))
#
#                if self.isPacman(state, agent):
#                    if bestScore > B:
#                        return bestScore, bestAction
#                    A = mf(A, bestScore)
#                else:
#                    if bestScore < A:
#                        return bestScore, bestAction
#                    B = mf(B, bestScore)
#
#            return bestScore, bestAction
#
#        _,action = dispatch(gameState, 0, 0)
#        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        _, action = self._expectimax(gameState, 0, 0)
        return action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    WEIGHT_FOOD = 10.0
    WEIGHT_GHOST = 10.0

    score = currentGameState.getScore()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    ghostPos = currentGameState.getGhostPositions()

    distance = lambda x, y: ((x[0] - y[0])**2 + (x[1] - y[1])**2) ** 0.5
    dist_to_food = [distance(newPos, pos) for pos in newFood.asList()]
    if len(dist_to_food) > 0:
        score += WEIGHT_FOOD/min(dist_to_food)

    #dist_to_ghost = [distance(newPos, pos) for pos in successorGameState.getGhostPositions()]
    dist_to_ghost = distance(newPos, ghostPos[0])
    if dist_to_ghost != 0:
        score -= WEIGHT_GHOST/dist_to_ghost

    return score

# Abbreviation
better = betterEvaluationFunction

