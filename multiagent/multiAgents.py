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
import sys

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
        foodList = newFood.asList()
        foodDistanceList = []
        ghostDistance = manhattanDistance(newPos, newGhostStates[0].getPosition())

        for food in foodList:
            foodDistance = 1.0/manhattanDistance(food, newPos)
            foodDistanceList.append(foodDistance)
        maxDistance = max(foodDistanceList + [0])

        if ghostDistance > 0:
            return maxDistance + successorGameState.getScore() - (4.5 / ghostDistance)
        return maxDistance + successorGameState.getScore()

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
        """
        "*** YOUR CODE HERE ***"
        return self.getMax(1, gameState)[1]

    def getMax(self, depth, gameState):
        legalActions = gameState.getLegalActions(0)
        if (legalActions == []) or (depth > self.depth):
            return self.evaluationFunction(gameState), None

        costs = []
        for action in legalActions:
            costs.append((self.getMin(depth, gameState.generateSuccessor(0, action), 1), action))

        return max(costs)

    def getMin(self, depth, gameState, agentIndex):
        legalActions = gameState.getLegalActions(agentIndex)

        if (legalActions == []):
            return self.evaluationFunction(gameState), None

        successors = []
        costs = []

        for action in legalActions:
            successors.append(gameState.generateSuccessor(agentIndex, action))

        if (agentIndex == gameState.getNumAgents() - 1):
            for successor in successors:
                costs.append(self.getMax(depth + 1, successor))
        else:
            for successor in successors:
                costs.append(self.getMin(depth, successor, agentIndex + 1))

        return min(costs)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        neg_infin = -sys.maxint
        infin = sys.maxint

        return self.getMax(1, gameState, neg_infin, infin)[1]

    def getMax(self, depth, gameState, alpha, beta):
        legalActions = gameState.getLegalActions(0)
        if (legalActions == []) or (depth > self.depth):
            return self.evaluationFunction(gameState), Directions.STOP

        currentValue = -sys.maxint
        bestOption = Directions.STOP

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            cost = self.getMin(depth, successor, 1, alpha, beta)[0]
            if (cost > currentValue):
                currentValue = cost
                bestOption = action

            # current value is a worst option
            if (currentValue >= beta):
                return currentValue, bestOption

            alpha = max(alpha, currentValue)

        return currentValue, bestOption

    def getMin(self, depth, gameState, agentIndex, alpha, beta):
        legalActions = gameState.getLegalActions(agentIndex)
        if (legalActions == []):
            return self.evaluationFunction(gameState), Directions.STOP

        currentValue = sys.maxint
        bestOption = Directions.STOP

        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if (agentIndex == gameState.getNumAgents() - 1):
                cost = self.getMax(depth + 1, successor, alpha, beta)[0]
            else:
                cost = self.getMin(depth, successor, agentIndex + 1, alpha, beta)[0]

            if (currentValue > cost):
                currentValue = cost
                bestOption = action

            # current value is a worst option
            if (currentValue <= alpha):
                return currentValue, bestOption

            beta = min(beta, currentValue)

        return currentValue, bestOption

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
        return self.getMax(1, gameState)[1]

    def getMax(self, depth, gameState):
        legalActions = gameState.getLegalActions(0)

        if (legalActions == []) or (depth > self.depth):
            return self.evaluationFunction(gameState), None

        costs = []
        for action in legalActions:
            costs.append((self.expectedValue(depth, gameState.generateSuccessor(0, action), 1)[0], action))

        return max(costs)

    def expectedValue(self, depth, gameState, agentIndex):
    	legalActions = gameState.getLegalActions(agentIndex)

    	if (legalActions == []):
            return self.evaluationFunction(gameState), None

        successors = []
        costs = []
        averages = []

    	for action in legalActions:
            successors.append(gameState.generateSuccessor(agentIndex, action))

    	for successor in successors:
            if (agentIndex == gameState.getNumAgents() - 1):
                costs.append(self.getMax(depth + 1, successor))
    	    else:
                costs.append(self.expectedValue(depth, successor, agentIndex + 1))

        for cost in costs:
            averages.append(float(cost[0]) / len(costs))

    	return sum(averages), None

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    foodList = currentGameState.getFood().asList()
    position = currentGameState.getPacmanPosition()

    foodDistance = []
    for food in foodList:
        foodDistance.append(1.0/manhattanDistance(food, position))

    return max(foodDistance + [0]) + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction
