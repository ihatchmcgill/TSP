#!/usr/bin/python3
import copy

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
    from PyQt6.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

    def greedy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        visited = []
        not_visited = copy.copy(cities)

        city = cities[0]
        visited.append(city)
        not_visited.remove(city)

        start_time = time.time()
        while len(not_visited) > 0 and time.time() - start_time < time_allowance:
            next_city = min(not_visited, key=lambda x: city.costTo(x))

            visited.append(next_city)
            if city.costTo(next_city) == float('inf'):
                break
            not_visited.remove(next_city)
            city = next_city
        end_time = time.time()

        bssf = TSPSolution(visited)
        results['cost'] = bssf.cost if not bssf.cost == np.inf else math.inf
        results['time'] = end_time - start_time
        results['count'] = 1
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

        return results

    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

    def branchAndBound(self, time_allowance=60.0):
        num_pruned = 0
        num_solutions = 0
        num_states = 0
        max_num_states = 0

        cities = self._scenario.getCities()
        num_cities = len(cities)
        start_time = time.time()
        results = self.greedy()
        if results['cost'] == math.inf:
            # Greedy Algorithm was unable to find a valid solution, use random default solution instead.
            results = self.defaultRandomTour()

        pq = []
        path = [cities[0]]

        bssf = results['soln']

        if bssf.cost == math.inf:
            # No starting solution
            return results

        # Viable solution
        num_states += 1

        distance_matrix = self.generate_inital_matrix()
        reduced_matrix, lower_bound = self.reduce_matrix(distance_matrix)

        # Create tuple and push onto pq.
        # Ordered by depth and then lowerbound. Use id as well to prevent comparing city or matrix objects
        heapq.heappush(pq, (-len(path), lower_bound, id(reduced_matrix), reduced_matrix, path))
        max_num_states += 1

        while len(pq) != 0 and time.time() - start_time < time_allowance:
            # Pop off pq, prioritizes depth and then lowerbound
            curr_state = heapq.heappop(pq)
            if curr_state[0] < bssf.cost:
                sub_states = self.expand_state(curr_state)
                for state in sub_states:
                    num_states += 1
                    # If state is a leaf node and lowerbound < bssf. Bottom node must actually get back to starting node.
                    if len(state[4]) == num_cities and state[1] < bssf.cost \
                            and state[4][-1].costTo(cities[0]) != float('inf'):
                        bssf.cost = state[1]
                        bssf.route = state[4]
                        num_solutions += 1
                    # Not a leaf node, but partial solution. Add to pq
                    elif state[1] < bssf.cost and len(state[4]) != num_cities:
                        heapq.heappush(pq, state)
                        if len(pq) > max_num_states:
                            max_num_states = len(pq)
                    else:
                        num_pruned += 1
            else:
                num_pruned += 1

        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = num_solutions
        results['soln'] = bssf
        results['max'] = max_num_states
        results['total'] = num_states
        results['pruned'] = num_pruned
        return results


    def expand_state(self, state):
        sub_states = []
        cities = self._scenario.getCities()
        num_cities = len(cities)
        curr_city = state[4][-1]
        city_row = cities.index(curr_city)

        for j in range(num_cities):
            if state[3][city_row][j] != float('inf'):
                # City is a copy but different memory address
                copied_state = copy.deepcopy(state)
                starting_cost = state[3][city_row][j]

                copied_state[4].append(cities[j])

                # Set Col and Row to infinity
                copied_state[3][j][city_row] = float('inf')
                copied_state[3][city_row] = [float('inf') for _ in range(num_cities)]
                for k in range(num_cities):
                    copied_state[3][k][j] = float('inf')

                # Get new matrix and lower bound
                reduced_matrix, lower_bound = self.reduce_matrix(copied_state[3])

                # Update new state
                new_state = (-len(copied_state[4]), copied_state[1] + starting_cost + lower_bound, id(reduced_matrix), reduced_matrix, copied_state[4])
                sub_states.append(new_state)
        return sub_states

    def generate_inital_matrix(self):
        cities = self._scenario.getCities()
        ncities = len(cities)
        distance_matrix = [[float('inf') for j in range(ncities)] for i in range(ncities)]

        for i in range(ncities):
            for j in range(ncities):
                distance_matrix[i][j] = cities[i].costTo(cities[j])

        return distance_matrix

    def reduce_matrix(self, matrix):
        # Subtract the minimum value from each row and each column
        # Find the minimum value in each row
        row_min = [min(row) for row in matrix]
        # Subtract the minimum value from each element in the row
        for i in range(len(matrix)):
            if row_min[i] == float('inf') or row_min[i] == 0:
                continue
            for j in range(len(matrix[i])):
                matrix[i][j] -= row_min[i]
        # Find the minimum value in each column
        col_min = [min(col) for col in zip(*matrix)]
        # Subtract the minimum value from each element in the column
        for j in range(len(matrix)):
            if col_min[j] == float('inf') or col_min[j] == 0:
                continue
            for i in range(len(matrix[j])):
                matrix[i][j] -= col_min[j]

        row_sum = 0
        col_sum = 0
        for num in row_min:
            if not math.isinf(num):
                row_sum += num
        for num in col_min:
            if not math.isinf(num):
                col_sum += num
        lower_bound = row_sum + col_sum
        return matrix, lower_bound

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

    def fancy(self, time_allowance=60.0):
        pass
