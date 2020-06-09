import math
import numpy as np
np.random.seed(42)
from copy import deepcopy
import matplotlib.pyplot as plt


def get_data(path):
    with open(path, 'r') as file:
        data = file.readlines()
    return data


def parse_data(data):
    data_dict = {}
    for i in range(6):
        line = data[i].split(" : ")

        data_dict.update({line[0]: line[1][:-1]})
    routes = np.zeros((int(data_dict['CAPACITY']), int(data_dict['CAPACITY'])))
    for i in range(7, int(data_dict['DIMENSION'])):
        node = [int(value) for value in data[i][1:-1].split(' ')]
        routes[node[1], node[2]] = node[0]


class VPR:
    def __init__(self, benchmark, max_epoch):
        self.name = benchmark[0][7:benchmark[0].find('\n')]
        number_of_trucks = benchmark[1][benchmark[1].find('No of trucks: ') + len('No of trucks: '): ]
        self.number_of_trucks = int(number_of_trucks[:number_of_trucks.find(',')])
        self.vrp_value_validation = int(benchmark[1][benchmark[1].find('Optimal value: ') + len('Optimal value: '): benchmark[1].find(')')])
        self.dimension = int(benchmark[3][12:benchmark[3].find('\n')])
        self.capacity = int(benchmark[5][11:benchmark[5].find('\n')])
        self.coordinates = np.zeros(shape=(self.dimension, 2))
        for i in range(0, self.dimension):
            pxy = benchmark[7 + i].split(' ')
            p = int(pxy[1]) - 1
            x = int(pxy[2])
            y = int(pxy[-1][:pxy[-1].find('\n')])
            self.coordinates[i] = np.asarray([x, y])
        self.p_requests = np.zeros(shape=self.dimension, dtype=np.int16)
        for i in range(0, self.dimension):
            pr = benchmark[8 + self.dimension + i].split(' ')
            self.p_requests[i] = int(pr[1])

        self.adjacency_matrix = np.zeros(shape = (self.dimension, self.dimension), dtype = np.float32)
        for i in range(self.dimension):
            for j in range(self.dimension):
                self.adjacency_matrix[i,j] = math.sqrt(pow(self.coordinates[i][0] - self.coordinates[j][0], 2) + pow(self.coordinates[i][1] - self.coordinates[j][1], 2))
        self.average_weight = np.average(self.adjacency_matrix)
        self.init_pheramone_value = 100
        self.pheramone_map = np.full(shape=(self.dimension, self.dimension), fill_value=self.init_pheramone_value)
        np.fill_diagonal(self.pheramone_map, 0)
        self.alpha = 0.9
        self.beta = 0.1
        self.max_epoch = max_epoch
        self.tabu = np.ones(self.dimension)
        self.capacity_rest = self.capacity
        self.p = 100
        self.best_ant_sol = None
        self.raw_prob_matrix = (self.pheramone_map**self.alpha) * (self.adjacency_matrix**self.beta)
        self.K = 10
        self.potential_vertexes = np.ones(self.dimension)

    def get_probality(self, prob_list):
        # summa = prob_list.sum()

        if prob_list.sum() == 0:
            print("ON NO ZERO")
        prob_list = prob_list/prob_list.sum()

        return prob_list

    def get_next_vertex(self, pos):
        prob_list = deepcopy(self.raw_prob_matrix[pos]) * self.tabu * self.potential_vertexes
        idx = np.random.choice(np.arange(0, self.dimension), p=self.get_probality(prob_list))
        return idx

    def local_update(self, i, j):
        # self.pheramone_map[i, j] = (1-self.p)*self.pheramone_map[i, j] + \
        #                            self.p * self.init_pheramone_value/self.adjacency_matrix[i, j]
        self.pheramone_map[i, j] += self.p * self.init_pheramone_value / self.adjacency_matrix[i, j]
        self.pheramone_map[j, i] = self.pheramone_map[i, j]
        self.raw_prob_matrix = (self.pheramone_map**self.alpha) * (self.adjacency_matrix**self.beta)

    def global_update(self, best_solution, best_cost):
        for one_path in best_solution:
            for i in range(len(one_path)-1):
                # self.pheramone_map[one_path[i], one_path[i+1]] = \
                #     (1 - self.p) * self.pheramone_map[one_path[i], one_path[i+1]] \
                #     + self.p * self.capacity/best_cost
                self.pheramone_map[one_path[i], one_path[i+1]] += \
                    self.p * self.capacity/best_cost
                self.pheramone_map[one_path[i+1], one_path[i]] = \
                    self.pheramone_map[one_path[i], one_path[i + 1]]
        self.raw_prob_matrix = (self.pheramone_map**self.alpha) * (self.adjacency_matrix**self.beta)

    def get_cost(self, solution):
        current_cost = 0
        for i in range(len(solution) - 1):
            current_cost += self.adjacency_matrix[solution[i], solution[i + 1]]
        return current_cost

    def compute(self):
        show_epoch = []
        iam_the_best_of_the_best_cost = 1e+6
        iam_the_best_of_the_best_sol = None
        for epoch in range(self.max_epoch):
            current_state = 0
            all_solutions_by_ant = []
            all_costs_by_ant = []
            for ant_id in range(self.K):
                solutions = []
                one_path_solution = [0]
                capacity_left = self.capacity
                self.tabu = np.ones(self.dimension)
                self.tabu[0] = 0
                self.potential_vertexes = np.ones(self.dimension)
                self.potential_vertexes[0] = 0
                while self.tabu.sum() != 0:
                    potential_flag = 0
                    next_state = self.get_next_vertex(current_state)
                    store_capacity = self.p_requests[next_state]
                    if capacity_left - store_capacity < 0:
                        self.potential_vertexes[next_state] = 0
                        potential_flag = 1
                        if self.potential_vertexes.sum() == 0:
                            one_path_solution.append(0)
                            solutions.append(one_path_solution)
                            one_path_solution = [0]
                            self.potential_vertexes = deepcopy(self.tabu)
                            current_state = 0
                            capacity_left = self.capacity
                            continue
                    if potential_flag:
                        continue
                    one_path_solution.append(next_state)
                    capacity_left -= store_capacity
                    self.local_update(current_state, next_state)
                    current_state = deepcopy(next_state)
                    self.tabu[current_state] = 0
                    self.potential_vertexes[current_state] = 0

                one_path_solution.append(0)
                solutions.append(one_path_solution)

                full_cost = sum([self.get_cost(sol) for sol in solutions])
                assert all(np.unique(np.hstack(solutions)) == np.arange(self.dimension))
                # assert len(solutions) <= self.number_of_trucks
                # [sol for a in solutions if a ]
                all_solutions_by_ant.append(solutions)
                all_costs_by_ant.append(full_cost)

            all_solutions_by_ant = np.asarray(all_solutions_by_ant)

            all_costs_by_ant = np.asarray(all_costs_by_ant)
            minimum_idx = np.argmin(all_costs_by_ant)
            best_solution = all_solutions_by_ant[minimum_idx]
            best_cost = all_costs_by_ant[minimum_idx]
            self.global_update(best_solution, best_cost)
            show_epoch.append(best_cost)
            if iam_the_best_of_the_best_cost > best_cost:
                iam_the_best_of_the_best_cost = best_cost
                iam_the_best_of_the_best_sol = best_solution
            print(f'Epoch: {epoch} | best cost: {best_cost}')
        print(self.alpha, self.beta, self.p, self.K)
        plt.plot(np.arange(len(show_epoch)), np.array(show_epoch))
        plt.show()
        print(iam_the_best_of_the_best_sol)
        print(f'Cost: {iam_the_best_of_the_best_cost}')


test1 = VPR(get_data('benchmark/A/A-n32-k5.vrp'), 150)
test1.compute()
