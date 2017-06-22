'''
###Introduction

    An instance of the N-puzzle game consists of a board holding N = m^2 − 1 (m = 3, 4, 5, ...) distinct movable tiles, plus an empty space. The tiles are numbers from the set {1, …, m^2 − 1}. For any such board, the empty space may be legally swapped with any tile horizontally or vertically adjacent to it. In this assignment, we will represent the blank space with the number 0 and focus on the m = 3 case (8-puzzle).

Given an initial state of the board, the combinatorial search problem is to find a sequence of moves that transitions this state to the goal state; that is, the configuration with all tiles arranged in ascending order ⟨0, 1, …, m^2 − 1⟩. The search space is the set of all possible states reachable from the initial state.

The blank space may be swapped with a component in one of the four directions {‘Up’, ‘Down’, ‘Left’, ‘Right’}, one move at a time. The cost of moving from one configuration of the board to another is the same and equal to one. Thus, the total cost of path is equal to the number of moves made from the initial state to the goal state.

###Algorithm

First, we remove a node from the frontier set.

Second, we check the state against the goal state to determine if a solution has been found.

Finally, if the result of the check is negative, we then expand the node. To expand a given node, we generate successor nodes adjacent to the current node, and add them to the frontier set. Note that if these successor nodes are already in the frontier, or have already been pred, then they should not be added to the frontier again.

This describes the life-cycle of a visit, and is the basic order of operations for search agents in this assignment—(1) remove, (2) check, and (3) expand. In this assignment, we will implement algorithms as described here. Please refer to lecture notes for further details, and review the lecture pseudocode before you begin the assignment.

###Requirements

Your job in this assignment is to write driver.py, which solves any 8-puzzle board when given an arbitrary starting configuration. The program will be executed as follows:

    $ python driver.py <method> <board>

The method argument will be one of the following. You need to implement all three of them:

    bfs (Breadth-First Search)
    dfs (Depth-First Search)
    ast (A-Star Search)

The board argument will be a comma-separated list of integers containing no spaces. For example, to use the bread-first search strategy to solve the input board given by the starting configuration {0,8,7,6,5,4,3,2,1}, the program will be executed like so (with no spaces between commas):

    $ python driver.py bfs 0,8,7,6,5,4,3,2,1

When executed, your program will create / write to a file called output.txt, containing the following statistics:

    path_to_goal: the sequence of moves taken to reach the goal
    cost_of_path: the number of moves taken to reach the goal
    nodes_expanded: the number of nodes that have been expanded
    search_depth: the depth within the search tree when the goal node is found
    max_search_depth:  the maximum depth of the search tree in the lifetime of the algorithm
    running_time: the total running time of the search instance, reported in seconds
    max_ram_usage: the maximum RAM usage in the lifetime of the process as measured by the ru_maxrss attribute in the resource module, reported in megabytes

The output file (example) will contain exactly the following lines:

    path_to_goal: ['Up', 'Left', 'Left']
    cost_of_path: 3
    nodes_expanded: 10
    search_depth: 3
    max_search_depth: 4
    running_time: 0.00188088
    max_ram_usage: 0.07812500

Test Case #1
python driver.py bfs 3,1,2,0,4,5,6,7,8
python driver.py dfs 3,1,2,0,4,5,6,7,8
python driver.py ast 3,1,2,0,4,5,6,7,8

Test Case #2
python driver.py bfs 1,2,5,3,4,0,6,7,8
python driver.py dfs 1,2,5,3,4,0,6,7,8
python driver.py ast 1,2,5,3,4,0,6,7,8

Test Case #3
np.search(method="bfs", start=[1,2,3,4,5,0,6,7,8])
(1, 2, 3, 4, 5, 0, 6, 7, 8)
(1, 2, 3, 4, 0, 5, 6, 7, 8)
(1, 2, 3, 0, 4, 5, 6, 7, 8)
(0, 2, 3, 1, 4, 5, 6, 7, 8)
(2, 0, 3, 1, 4, 5, 6, 7, 8)
(2, 3, 0, 1, 4, 5, 6, 7, 8)
(2, 3, 5, 1, 4, 0, 6, 7, 8)
(2, 3, 5, 1, 0, 4, 6, 7, 8)
(2, 0, 5, 1, 3, 4, 6, 7, 8)
(0, 2, 5, 1, 3, 4, 6, 7, 8)
(1, 2, 5, 0, 3, 4, 6, 7, 8)
(1, 2, 5, 3, 0, 4, 6, 7, 8)
(1, 2, 5, 3, 4, 0, 6, 7, 8)
(1, 2, 0, 3, 4, 5, 6, 7, 8)
(1, 0, 2, 3, 4, 5, 6, 7, 8)
(0, 1, 2, 3, 4, 5, 6, 7, 8)

'''

import queue
from collections import deque
import numpy as np

class NPuzzle:
    '''
    N-puzzle game

    Board state is represented as list.
    '''
    def __init__(self, m=3, state=tuple()):
        self.reset(m, state)

    def reset(self,m=3,state=tuple()):
        self.N = m*m - 1 # max possible number
        self.m = m # number of rows/columns
        self.goal = tuple(range(0, m*m))
        self.start = tuple(state)  # start state
        self.explored = list() # states that have been explored
        self.pred = dict()  # preddecessor states, current_state : pred_state
        self.path_to_goal = list()
        self.nodes_expanded = 0
        self.max_search_depth = 0

    def _next_state(self, current, x, y):
        ''' swap grids x and y from state
            where, state is a list, x & y are indices of state.
        '''
        state = list(current) # convert tuple to list
        state[x], state[y] = state[y], state[x]
        return tuple(state)

    def get_successors(self, current):
        '''
        Find successors and insert them into frontier list
        '''
        if len(current) == 0:
            return list()
        idx = current.index(0) # get the index to item '0'
        i = int(idx / self.m)
        j = int(idx % self.m)
        successors = list()

        # up
        if i - 1 >= 0:
            state = self._next_state(current, idx, idx - self.m)
            if state not in self.explored:
                successors.append(state)

        # down
        if i + 1 < self.m:
            state = self._next_state(current, idx, idx + self.m)
            if state not in self.explored:
                successors.append(state)

        # left
        if j - 1 >= 0:
            state = self._next_state(current, idx, idx - 1)
            if state not in self.explored:
                successors.append(state)

        # right
        if j + 1 < self.m:
            state = self._next_state(current, idx, idx + 1)
            if state not in self.explored:
                successors.append(state)

        return successors

    def bfs(self):
        frontier = deque(tuple()) # frontier states, queue
        frontier.append(self.start)
        while len(frontier) > 0:
            current = frontier.popleft()
            self.nodes_expanded += 1
            if current not in self.explored:
                self.explored.append(current)

            if current == self.goal:
                break

            successors = self.get_successors(current)
            for state in successors:
                if state not in frontier: # avoid duplicates
                    frontier.append(state)
                    self.pred[state] = current

    def dfs(self):
        frontier = deque(tuple()) # frontier states, stack
        frontier.append(self.start)

        while len(frontier) > 0:
            current = frontier.pop()
            if current not in self.explored:
                self.explored.append(current)

            if current == self.goal:
                break

            successors = self.get_successors(current)
            for state in successors:
                frontier.append(state)

    def path_finding(self):
        '''
        Backtrack to find the path from start to goal.
        '''
        path = list()
        state = self.goal
        while state != self.start:
            path.append(state)
            state = self.pred[state]

        if self.start not in path:
            path.append(self.start)

        return path[::-1]

    def search(self, method='bfs', start=list()):
        '''
        search for the goal state.

        method: bfs, dfs, ucs, ast
        '''
        self.reset(state=start)
        if len(self.start) == 0:
            print('Error: invalid start state!')
            return

        if method == 'bfs':
            self.bfs()
        elif method == 'dfs':
            self.dfs()
        elif method == 'ucs':
            self.ucs()
        elif method == 'ast':
            self.ast()
        else:
            print('Error: unsupported search method %s' %method)
            return

        for state in self.path_finding():
            print(state)
