import numba

import Reporter
import numpy as np
import math
import random
from random import sample
import matplotlib.pyplot as plt
import time
from numba import njit


# Modify the class name to match your student number.
def convert_adj_to_cycle(chromosome):
    breaks = 0
    longest_cycle = 1
    clens = list([0] * 50)
    clen = 1
    cycle = [0]
    curr_i = 0
    used = set()
    all = set(range(len(chromosome)))
    while len(used) < len(chromosome):
        next = chromosome[curr_i]
        # code to fix the subcycle issue, should not be necessary if there are no sub cycles
        if next in cycle:
            longest_cycle = max(longest_cycle, clen)
            clens[clen - 1] += 1
            clen = 0
            cycle.append('break')
            breaks += 1
            diff = all.difference(used)
            if len(diff) > 0:
                next = random.choice(list(all.difference(used)))
        cycle.append(next)
        clen += 1
        used.add(curr_i)
        curr_i = next
    return cycle, breaks, longest_cycle, clens


def convert_cycle_to_adj(chromosome):
    adj = [-1] * len(chromosome)
    for i in range(len(chromosome) - 1):
        adj[chromosome[i]] = chromosome[i + 1]
    adj[chromosome[-1]] = chromosome[0]
    return adj


# print(convert_cycle_to_adj([0,3,1,2]))
class r0735890:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.lambdaa = 150
        # self.beta = 0.5 # deprecated
        self.population = []
        # max number of optimization cycles
        self.k = 300
        self.dMatrix = None
        self.dMatrixAlt = None
        self.tournament_size = 5
        self.mu = 50

    def optimize(self, filename, plot_name='plot'):
        """The evolutionary algorithm's main loop"""
        # Create empty lists to be filled with values to report progress
        mean_fit_values = []
        best_fit_values = []

        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()
        self.dMatrix = distanceMatrix

        # appInfToZero = np.vectorize(infstozero)
        # appZeroToVal = np.vectorize(zerotoval)
        #
        # self.dMatrixAlt = appInfToZero(self.dMatrix)
        # self.dMatrixAlt = appZeroToVal(self.dMatrixAlt, self.dMatrixAlt.max()*2)

        self.dMatrixAlt = np.ma.masked_where(self.dMatrix == np.inf, self.dMatrix)
        max_ = self.dMatrixAlt.max() * 2
        self.dMatrixAlt = self.dMatrixAlt.filled(max_)
        self.dMatrixAlt = np.ma.masked_where(self.dMatrix == 0, self.dMatrix)
        self.dMatrixAlt = self.dMatrixAlt.filled(max_)



        print(self.dMatrixAlt)
        print(self.dMatrixAlt.argmin())

        # Your code here.
        if True:
            return initialize_NN(self.dMatrixAlt)
        self.initialize()
        self.population.sort(key=lambda x: x[1])

        # Loop:
        start = time.time()
        for _ in range(self.k):
            fitnesses = list(map(lambda x: x[1], self.population))
            meanObjective = sum(fitnesses) / self.lambdaa
            bestObjective = fitnesses[0]
            bestSolution = self.population[0][0]

            # Add to the list for later on:
            mean_fit_values.append(meanObjective)
            best_fit_values.append(bestObjective)

            # Your code here.
            offspring = []
            for _ in range(self.mu):
                p1, p2 = self.selection()
                child = self.recombination(p1[0], p2[0])
                child = self.mutation(child)
                offspring.append(child)
            self.elimination(offspring)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break
        end = time.time()
        time_diff = end - start
        number_of_minutes = time_diff // 60
        number_of_seconds = time_diff % 60
        print(f"The algorithm took {number_of_minutes} minutes and {number_of_seconds} seconds.")
        print(f"The final best fitness value was {bestObjective}")

        # Plot the results
        # plt.figure(figsize=(7, 5))
        # plt.plot(mean_fit_values, '--o', color='red', label="Mean")
        # plt.plot(best_fit_values, '--o', color='blue', label="Best")
        # plt.grid()
        # plt.legend()
        # plt.xlabel('Iteration step')
        # plt.ylabel('Fitness')
        # plt.title('TSP for ' + str(filename))
        # # Save the plots (as PNG and PDF)
        # plt.savefig('Plots/' + plot_name + '.png', bbox_inches='tight')
        # plt.savefig('Plots/' + plot_name + '.pdf', bbox_inches='tight')
        # # plt.show()
        # plt.close()

        # Your code here.
        # Return the best fitness value & best solution
        return bestObjective, bestSolution

    def initialize(self):
        size = np.shape(self.dMatrix)[0]
        for i in range(self.lambdaa):
            cycle = np.random.permutation(size)
            chromosome = convert_cycle_to_adj(cycle)
            fitness = self.fitness(chromosome)
            individual = (chromosome, fitness)
            self.population.append(individual)



    def recombination(self, parent1, parent2):
        """Parent1 and parent2 should be chromosomes"""

        # Save number of cities and parents for convenience
        number_of_cities = len(parent1)
        parents = [parent1, parent2]

        # Initialize a new child with appropriate length
        child = [0] * number_of_cities

        # Save the indices that were not assigned yet, and cities being assigned, throughout the process
        indices_not_assigned = [i for i in range(number_of_cities)]
        cities_not_assigned = [i for i in range(number_of_cities)]

        # Iterate over parents, save entries common in both parents
        # TODO: can this be achieved faster?
        for i in range(len(parent1)):
            if parent1[i] == parent2[i]:
                # Save value, and remove from remaining cities to be assigned
                child[i] = parent1[i]
                indices_not_assigned.remove(i)
                cities_not_assigned.remove(parent1[i])

        # If all indices were assigned (parents are identical chromosomes), return
        if len(indices_not_assigned) == 0:
            fit = self.fitness(child)
            return child, fit

        # # Otherwise, continue assigning the remaining indices
        # for i in indices_not_assigned:
        # 	# Randomly assign (with certain probability) one of parents' entries
        # 	dice_throw = np.random.uniform()
        # 	if dice_throw <= self.beta:
        # 		# Choose parent
        # 		parent_index = np.random.randint(2)
        # 		chosen_parent = parents[parent_index]
        # 		# Assign value and remove from cities to be assigned
        # 		child[i] = chosen_parent[i]
        # 		del indices_not_assigned[i]
        # 		cities_not_assigned.remove(chosen_parent[i])
        #
        # # Check again whether all indices were assigned
        # if len(indices_not_assigned) == 0:
        # 	# If child is ready, compute fitness
        # 	fit = self.fitness(child)
        # 	return child, fit

        # If not, randomly assign remaining cities
        while len(indices_not_assigned) > 0:
            random_city = random.choice(cities_not_assigned)
            next_index = indices_not_assigned[0]
            child[next_index] = random_city
            indices_not_assigned.remove(next_index)
            cities_not_assigned.remove(random_city)

        # Now, child should be completed - return it
        fit = self.fitness(child)
        return child, fit

    def mutation(self, child):
        # Get the cities of the tour first
        child = child[0]

        position_list = random.sample(range(0, len(child)), 2)
        temp = child[position_list[0]]
        child[position_list[0]] = child[position_list[1]]
        child[position_list[1]] = temp

        # Get fitness and return
        fit = self.fitness(child)
        return child, fit

    def selection(self):

        competitors_1 = sample(self.population, self.tournament_size)
        competitors_2 = sample(self.population, self.tournament_size)
        competitors_1.sort(key=lambda x: x[1])
        competitors_2.sort(key=lambda x: x[1])
        father, mother = competitors_1[0], competitors_2[0]
        return father, mother

    """lambda+mu elimination"""

    def elimination(self, offspring):
        self.population.extend(offspring)
        self.population.sort(key=lambda x: x[1])
        self.population = self.population[:self.lambdaa]

    def fitness(self, chromosome):
        # print(self.dMatrixAlt)
        fitness = 0
        for i, j in enumerate(chromosome):
            # print(i,j)
            fitness += self.dMatrixAlt[i, int(j)]
        return fitness


@njit
def infstozero(elm):
    return 0 if elm == np.inf else elm


@njit
def zerotoval(elm, val):
    return val if elm == 0 else elm


@njit
def delete(arr, i, a=0):
    mask = np.zeros(arr.shape[a], dtype=np.int64) == 0
    mask[i] = False
    if a == 0:
        return arr[mask]
    else:
        return arr.transpose()[mask].transpose()



def maskc(a, c):
    m = np.zeros_like(a)
    m[:, c] = 1
    return np.ma.masked_array(a, m)


def initialize_NN(mat):
    print(mat)
    print(mat[0,0])
    size = np.shape(mat)[0]

    chromosome = np.zeros(size)
    i = mat.argmin()
    print(i)
    r, c = i // size, i % size
    print(r, c)
    first = r
    mat = maskc(mat, [r, c])
    chromosome[r] = c
    print(chromosome)
    for i in range(size - 1):
        r, c = c, mat[c].argmin()
        chromosome[r] = c
        # print(chromosome)
        mat = maskc(mat, c)
        # print(r, c)
        # print(mat)

    chromosome[r] = first
    return chromosome

# # For hyperparameter testing: run the algorithm several times if desired
# number_of_runs = 1
# values = []
# for _ in range(number_of_runs):
# 	mytest = r0735890()
# 	fit, soln = mytest.optimize('./tour50.csv')
# 	values.append(fit)
#
# avg = np.mean(values)
# print(f"Ran {number_of_runs} times: average of best fitness was {avg}")
#
# # Plotting the results for parameter selection
# plt.figure(figsize=(15,10))
# plt.subplot(3,1,1)
# # tournament
# plt.plot([3, 5, 10, 15], [81998, 48530, 16883, 5117], '--o', color='red')
# plt.xlabel(r'$k$')
# plt.ylabel('Fitness')
# plt.grid()
# plt.subplot(3,1,2)
# # Offspring size
# plt.plot([20, 50, 100, 150], [126049, 78677, 51708, 36788], '--o', color='red')
# plt.xlabel(r'$\mu$')
# plt.ylabel('Fitness')
# plt.grid()
# plt.subplot(3,1,3)
# # Offspring size
# plt.plot([50, 100, 150, 200], [17382, 49639, 75251, 94189], '--o', color='red')
# plt.xlabel(r'$\lambda$')
# plt.ylabel('Fitness')
# plt.grid()
# plt.savefig('Plots/parameters_selection.png', bbox_inches='tight')
# plt.savefig('Plots/parameters_selection.pdf', bbox_inches='tight')
# plt.close()

if __name__ == "__main__":
    test = r0735890()
    res = test.optimize('./tour1000.csv')
    print(test.fitness(res))

