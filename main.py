import numba
from numpy import int64

import Reporter
import numpy as np
import math
import random
from random import sample
import matplotlib.pyplot as plt
import time
from numba import njit


# Modify the class name to match your student number.
def convert_adj_to_cycle_full(chromosome):
    breaks = 0
    longest_cycle = 1
    clens = list([0] * len(chromosome))
    clen = 1
    cycle = [0]
    curr_i = 0
    used = set()
    all = set(range(len(chromosome)))
    while len(used) < len(chromosome)-1:
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
    cycle.append(0)
    return cycle, breaks, longest_cycle, clens


def convert_adj_to_cycle(chromosome):
    cycle = [0]
    curr_i = 0
    used = set()
    while len(used) < len(chromosome):
        next = chromosome[curr_i]
        cycle.append(next)
        used.add(curr_i)
        curr_i = next
        # print(used)
        # print(len(used))
        # print(next)
    # print(cycle)
    return cycle


def convert_cycle_to_adj(chromosome):
    adj = np.ma.masked_all(len(chromosome)-1, dtype=np.int64)
    for i in range(len(chromosome) - 1):
        adj[chromosome[i]] = chromosome[i + 1]
    return adj


# print(convert_cycle_to_adj([0,3,1,2]))
class r0735890:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.lambdaa = 100
        # self.beta = 0.5 # deprecated
        self.population = []
        # max number of optimization cycles
        self.k = 1000
        self.dMatrix = None
        self.dMatrixAlt = None
        self.tournament_size = self.lambdaa // 20
        self.mu = self.lambdaa // 2
        self.nb_cities = None
        self.steps = None
        self.lsoprob = 1
        self.radius = None
        self.shape = 1
        self.ds = None
        self.multiObj = 5

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
        self.nb_cities = distanceMatrix.shape[0]
        self.radius = self.nb_cities//10
        self.ds = calc_shared_fitness_parameters(self.radius, self.shape, self.nb_cities)
        self.steps = self.nb_cities

        self.dMatrixAlt = np.ma.masked_where(self.dMatrix == np.inf, self.dMatrix)
        # print(self.dMatrixAlt)
        max_ = self.dMatrixAlt.max() * 2
        # print(max_)
        self.dMatrixAlt = self.dMatrixAlt.filled(max_)
        self.dMatrixAlt = np.ma.masked_where(self.dMatrixAlt == 0, self.dMatrixAlt)#.filled(max_)

        # print(self.dMatrixAlt)
        # print(self.dMatrixAlt.argmin())

        # Your code here.
        initialize(self)
        # print(self.population)
        # self.population = [greedy_stoch(self.dMatrixAlt, 1), initialize_NN_stoch_global(self.dMatrixAlt, 1)]
        # print(self.population)

        # Loop:
        start = time.time()
        # for _ in range(1):
        for gen in range(self.k):
            # print(f"generation {gen}")
            # Your code here.
            offspring = []
            # for _ in range(1):
            selected = []
            for c in range(self.mu):
                # print(f"child {c}")
                p1, p2 = selection(self, selected)
                selected += [p1, p2]
                # print(f"recombining with {list(p1[0])} and\n {list(p2[0])}")
                child = recombination(self, p1[0], p2[0])
                # print(f"child: {list(child)}")
                # child = convert_adj_to_cycle(child)
                # print(f"created {child} with fitness {self.fitness(convert_cycle_to_adj(child))}")
                # print(f"mutating {child}")
                child = mutation(self, child, self.lsoprob)
                # print(f"mutated {child}")
                adj = convert_cycle_to_adj(child)
                child = child, fitness(self, adj, self.dMatrixAlt), adj
                # print(f"created {child[0]} with fitness {child[1]}")
                offspring.append(child)
            # print(f"population size: {len(self.population)}")
            self.population = elimination(self, offspring, self.population, self.lambdaa)
            # print(f"population size: {len(self.population)}")
            # for i in range(self.lambdaa//10):
            #     lso(self, self.population[i][0], self.steps)
            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0

            self.population.sort(key=lambda x: x[1][0])
            fitnesses = list(map(lambda x: x[1][0], self.population))
            meanObjective = sum(fitnesses) / self.lambdaa
            bestObjective = fitnesses[0]
            bestSolution = self.population[0][0]

            # Add to the list for later on:
            mean_fit_values.append(meanObjective)
            best_fit_values.append(bestObjective)
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break
        end = time.time()
        time_diff = end - start
        number_of_minutes = time_diff // 60
        number_of_seconds = time_diff % 60
        print(f"The algorithm took {number_of_minutes} minutes and {number_of_seconds} seconds.")
        print(f"The final best fitness value was {bestObjective}")
        print(f"The final mean fitness value was {meanObjective}")


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
        return bestObjective, bestSolution, self.population


def initialize(problem):
    print("initializing...")
    print("---------------")
    print("greedy...")
    start = time.time()
    for i in range(problem.lambdaa // 20):
        print(i)
        chromosome, _ = greedy_stoch(problem.dMatrixAlt, 3)
        # fitness = self.fitness(chromosome)
        individual = (convert_adj_to_cycle(chromosome), fitness(problem, chromosome, problem.dMatrixAlt), chromosome)
        # print(individual)
        problem.population.append(individual)
    print(f"greedy done! (in {time.time()-start}s)")
    print("nearest neighbour...")
    for i in range(problem.lambdaa // 20):
        # print(i)
        chromosome = initialize_NN_stoch_global(problem.dMatrixAlt, 3)
        adj = convert_cycle_to_adj(chromosome)
        individual = (chromosome, fitness(problem, adj, problem.dMatrixAlt), adj)
        problem.population.append(individual)
    print("nearest neighbour done!")
    print("random...")
    for i in range(problem.lambdaa - len(problem.population)):
        # print(i)
        chromosome = init_random_cycle_no_infs(problem, problem.nb_cities)
        # chromosome = init_random_cycle(problem.nb_cities)
        # print(infcount(problem, chromosome))
        adj = convert_cycle_to_adj(chromosome)
        individual = (chromosome, fitness(problem, adj, problem.dMatrixAlt), adj)
        problem.population.append(individual)
    print("random done!")
    print("initialisation done!")
# the random additions here are slow because we constantly have to check that we don't introduce a cycle


def overlap_recombination(problem, p1, p2):
    ps = [p1, p2]
    # print("parents:", p1, p2)
    c = np.ma.masked_where(p1 != p2, p1)
    # print(c)
    for i, elm in enumerate(c):
        # print(i, elm)
        if elm is np.ma.masked:
            ch = np.random.randint(0, 2)
            chp = ps[ch]
            # if False:  # don't be influenced by the parents too much, less exploration
            if chp[i] not in c and not check_cycle(c, i, chp[i]):
                c[i] = chp[i]
                # print(f"orig choice {c}")
            elif ps[1-ch][i] not in c and not check_cycle(c, i, ps[1-ch][i]):
                c[i] = ps[1-ch][i]
                # print(f"2nd choice {c}")
            else:
                # print(f"rand {c}")
                # print(i)
                c = add_random(c, i)
                # print(f"added {c}")
    # print(list(c))
    return convert_adj_to_cycle(c)


def pmx(problem, p1, p2):
    problem.nb_cities = len(p1) - 1
    child = np.zeros(problem.nb_cities + 1, dtype=np.int64)
    child[1:problem.nb_cities] -= 1
    cp1 = np.random.randint(1, problem.nb_cities)
    cp2 = np.random.randint(cp1, problem.nb_cities)
    child[cp1:cp2] = p1[cp1:cp2]
    # print(child)
    for i in range(cp1, cp2):
        if p2[i] not in child:
            j = np.argwhere(p2 == p1[i])[0][0]
            while child[j] != -1:
                j = np.argwhere(p2 == p1[j])[0][0]
            child[j] = p2[i]
    # print(child)
    for i in range(1, problem.nb_cities):
            if child[i] == -1:
                child[i] = p2[i]

        # print(child)
    # print(child)
    return child


def recombination(problem, p1, p2):
    return pmx(problem, p1, p2)


def swap_mutation(problem, child):
    # Get the cities of the tour firs
    c1 = np.random.randint(1, len(child)-1)
    c2 = c1
    while c2 == c1:# or child[c2] == c1 or child[c1] == c2:
        c2 = np.random.randint(1, len(child)-1)
        # print(c1, c2, child[c1], child[c2]
    child[c1], child[c2] = child[c2], child[c1]
    return child


def two_opt_mutation(problem, child, c1=None, c2=None):
    # print(f"child: {child}")
    if c1 is None:
        c1 = np.random.randint(1, len(child) - 2)
        # print(c1)
    if c2 is None:
        c2 = np.random.randint(c1 + 1, len(child) - 1)
            # print(c2)
    child[c1:c2] = np.flip(child[c1:c2])
    # print(f"child after: {child}")
    return child


def shuffle_mutation(problem, child):
    c1 = np.random.randint(1, len(child) - 2)
    c2 = np.random.randint(c1 + 1, np.clip(c1 + 6, 1, len(child) - 1))
    np.random.shuffle(child[c1:c2])
    return child


def mutation(problem, c, beta):
    if np.random.rand() < 0.3:
        c = shuffle_mutation(problem, c)
    if np.random.random() < beta:
        c = lso(problem, c, problem.steps)
    # print(c)
    return c

# @njit
# @numba.int64[:]
def lso(problem, c, k):
    best = np.array(c)
    copy = np.array(c)
    best_fit = fitness(problem, convert_cycle_to_adj(c), problem.dMatrixAlt)
    # print(f"lso on {c} with fit: {best_fit}")
    for i in range(1, problem.nb_cities - 1):
        new = two_opt_mutation(problem, copy, i)
        new_fit = fitness(problem, convert_cycle_to_adj(new), problem.dMatrixAlt)
        if new_fit < best_fit:
            # print(f"new best found: {new}")
            best = np.array(new)
            best_fit = tuple(new_fit)
        # else:
            # print(f"no luck with {new}")
    # print(f"lso done! best: {best} with fit: {best_fit}")
    return best


def selection(problem, prev_selected=None):
    return ktournamentselection(problem, prev_selected)


def ranking_based_selection(problem, prev_selected=None):
    if prev_selected is None:
        prev_selected = []
    pass




def ktournamentselection(problem, prev_selected=None):

    if prev_selected is None:
        prev_selected = []
    competitors_1 = sample(problem.population, problem.tournament_size)
    # print(competitors_1)
    competitors_2 = sample(problem.population, problem.tournament_size)
    competitors_1.sort(key=lambda x: shared_fitness(x, prev_selected, problem.ds))
    competitors_2.sort(key=lambda x: shared_fitness(x, prev_selected, problem.ds))
    father, mother = competitors_1[0], competitors_2[0]
    # print(f"father: {father[0]} with fit: {father[1]}")
    # print(f"mother: {mother[0]} with fit: {mother[1]}")
    return father, mother


"""lambda+mu elimination"""
def elimination(problem, offspring, population, lambdaa):
    problem.population = population + offspring
    population = pareto_fitness(problem)
    # print(f"domcount: {population[0][1][2]}")
    population.sort(key=lambda x: (x[1][2], x[1][0]))
    population = population[:lambdaa]
    return population


def fitness(problem, chromosome, dmatrix):
    portion_fitnesses = []
    portion = len(chromosome) // problem.multiObj
    for k in range(problem.multiObj):
        portion_fitness = 0
        for i in range(portion):
            # print(i+portion*k)
            # print(chromosome[i+portion*k])
            portion_fitness += dmatrix[i + portion*k, chromosome[i + portion*k]]
        portion_fitnesses.append(portion_fitness)
    # print(f"chromosome {chromosome} with fitness: {fitness}")

    return sum(portion_fitnesses), portion_fitnesses


def pareto_fitness(problem):
    domcounts = np.zeros(len(problem.population), dtype=np.int64)
    portion_fitnesses = np.array(list(map(lambda x: x[1][1], problem.population)))
    # print(f"portion_fitnesses: {portion_fitnesses}")
    for i in range(len(problem.population)):
        domcount = domcounts[i]
        for j in range(i + 1, len(problem.population)):
            if np.all(portion_fitnesses[i] >= portion_fitnesses[j]):
                domcount += 1
            elif np.all(portion_fitnesses[i] <= portion_fitnesses[j]):
                domcounts[j] += 1
        problem.population[i] = (problem.population[i][0], (problem.population[i][1][0], problem.population[i][1][1], domcount), problem.population[i][2])
    # print(domcounts)
    return problem.population


# @njit
def shared_fitness(ind, pop, ds, sumInit=0):
    # print(pop)
    sum = sumInit
    dists = [np.count_nonzero(ind[2] != arr[2]) for arr in pop]
    mapped = np.take(ds, dists)
    sum += np.sum(mapped)
    # print(sum)

    return ind[1][0]*sum


def calc_shared_fitness_parameters(radius, shape, nb_cities):
        ds = np.arange(0, radius)
        ds = ds/radius
        ds = ds**shape
        ds = np.concatenate((ds, np.ones(nb_cities - radius + 1)))
        ds = 1 - ds
        return ds


def lb(mat):
    sum = 0
    for i in range(mat.shape[0]):
        sum += mat[i].min()
    return sum


def calcHamming(a, b):
    return np.count_nonzero(a != b)


def add_random(a, r):
    checked = set(a.compressed())
    checked.add(r)
    # print(checked)
    available = set(range(len(a))).difference(checked)

    while len(available) > 0:
        # print(available)
        i = np.random.choice(list(available))
        if not check_cycle(a, r, i):
            a[r] = i
            return a
        else:
            available.remove(i)


def edge_crossover(p1, p2):
    p1, p2 = np.array(p1), np.array(p2)
    # print(f"p1: {p1}")
    # print(f"p2: {p2}")
    child = np.full(p1.shape, -1, dtype=np.int64)
    child[[0, -1]] = 0
    # print(child)
    edge_table = create_edge_table([p1, p2])
    curr = 0
    for i in range(1, len(p1) - 1):
        # print(f"curr: {curr}")
        # print(f"child: {child}")
        # print(f"edge_table: {edge_table}")
        candidates = edge_table[curr]
        edge_table.pop(curr)
        # print(f"candidates: {candidates}")
        if len(candidates) == 0:
            # add a random edge to the child
            curr = np.random.choice(list(edge_table.keys()))
        else:
            if len(candidates) == 1:
                curr = candidates[0]
            else:
                if len(edge_table[candidates[0]]) < len(edge_table[candidates[1]]):
                    curr = candidates[0]
                else:
                    curr = candidates[1]

        child[i] = curr
        for key in edge_table.keys():
            if curr in edge_table[key]:
                edge_table[key].remove(curr)

    return child
        # print(f"curr: {curr}")





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


def maskr(a, r):
    m = np.zeros_like(a)
    m[r] = 1
    return np.ma.masked_array(a, m)


def maskelm(a, r, c):
    m = np.zeros_like(a)
    m[r, c] = 1
    return np.ma.masked_array(a, m)


def init_random_cycle(size):
    incycle = np.random.permutation(size-1) + 1
    cycle = np.ma.masked_all(size+1, dtype=np.int64)
    cycle[[0, -1]] = 0
    cycle[1:size] = incycle
    # print(cycle)
    return cycle


def init_random_cycle_no_infs(problem, size):
    cycle = np.zeros(size+1, dtype=np.int64)
    all_available = set(range(1, size))
    for i in range(1, size):
        available = all_available.difference(set(cycle))
        available_without_infs = set(available)
        # print(f"available: {available}")
        fit = np.inf
        while fit == np.inf:
            # print(f"available_without_infs: {available_without_infs}")
            id = np.random.choice(list(available_without_infs))
            fit = problem.dMatrix[cycle[i-1], id]
            available_without_infs.remove(id)
            if len(available_without_infs) == 0:
                id = np.random.choice(list(available))
                break
        cycle[i] = id
    #     print(f"cycle: {cycle}")
    # print(cycle)
    return cycle


def initialize_NN_stoch_global(mat, k, frm=0):
    # print(mat)
    # print(np.shape(mat))
    size = np.shape(mat)[0]

    chromosome = np.ma.masked_all(size + 1, dtype=np.int64)
    chromosome[[0, -1]] = frm
    mat = maskc(mat, frm)
    c = frm
    # print(chromosome)
    for i in range(size - 1):
        offset = size - i - 1
        offset = offset if offset < k else k
        # print(mat[c])
        idx = k_smallest_masked(mat[c], offset)
        # print(idx)
        c = np.random.choice(idx)
        chromosome[i+1] = c
        # print(chromosome)
        mat = maskc(mat, c)
        # print(r, c)
        # print(mat)
        # print(chromosome)
    # print(chromosome)
    return chromosome


def greedy_stoch(mat, k=1):
    size = mat.shape[0]
    order = np.argsort(mat.ravel())
    chromosome = np.ma.masked_all(size, dtype=np.int64)
    # print(chromosome)
    fit = 0
    for i in range(size):
        cycle = True
        offset = size - i
        offset = offset if offset < k else k
        while cycle:
            # print(mat)
            i = np.random.randint(0, offset)
            # print(i)
            # print(order[i])
            # print(order)
            r, c = np.unravel_index(order[i], (size, size))
            # print(r, c)
            # print(chromosome)
            if check_cycle(chromosome, r, c):
                order = delete(order, i)
            else:
                # print(r, c)
                fit += mat[r, c]
                # mat = maskc(mat, c)
                # mat = maskr(mat, r)
                # print(order)
                order = np.ma.masked_where(order % size == c, order)
                # print(order)
                order = np.ma.masked_inside(order, r*size, (r+1)*size-1)
                # print(order)
                order = order.compressed()
                # print(order)
                chromosome[r] = c
                cycle = False
                # print(chromosome)
    # print(chromosome)
    return chromosome, fit


def k_smallest_masked(a, k):
    res = []
    # print(a)
    for i in range(k):
        argmin = a.argmin()
        res.append(argmin)
        a = maskr(a, argmin)
    # print(res)
    return res


def check_cycle(chromosome, r, c):
    # print(f"checking for cycle in {chromosome} when adding {r}-{c}..")
    i = 1
    curr_i = c
    while i < len(chromosome) - 1:
        # print(curr_i)
        next = chromosome[curr_i]
        # print(next)
        if next is np.ma.masked:
            return False
        if next == r:
            # print(next, r)
            # print("cycle found")
            return True

        curr_i = next
        i += 1
    return False


def create_edge_table(parents):
    edges = {}
    for p in parents:
        for i in range(len(p) - 1):
            if p[i] not in edges.keys():
                if p[i+1] != 0:
                    # print(f"adding {p[i+1]} to {p[i]}")
                    edges[p[i]] = [p[i+1]]
                else:
                    edges[p[i]] = []
            else:
                next = p[i+1]
                if next not in edges[p[i]] and next != 0:
                    # print(f"adding {next} to {p[i]}")
                    edges[p[i]].append(next)
    # print(edges)
    return edges


def infcount(problem, chromosome):
    count = 0
    for i in range(len(chromosome) - 1):
        if problem.dMatrix[chromosome[i], chromosome[i+1]] == np.inf:
            count += 1
    return count


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

def calchammings(pop):
    hammings = []
    for i in range(len(pop)):
        currhammings = []
        for j in range(len(pop)):
            currhammings.append(np.count_nonzero(pop[i] != pop[j]))
        hammings.append(currhammings)
    return hammings

if __name__ == "__main__":
    test = r0735890()
    start = time.time()
    res = test.optimize('./tour50.csv')
    print(lb(test.dMatrixAlt))
    end = time.time()
    print(end-start)
    print(res[0:1])
    print(list(map(lambda x: x[0], res[2])))
    hammings = calchammings(list(map(lambda x: x[2], res[2])))
    print(hammings)
    for i in range(len(hammings)):
        print(np.mean(hammings[i][:20]))
        hammings[i] = np.sum(hammings[i])
    print(np.mean(hammings))
    # print(pmx(test, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0]), np.array([0, 2, 3, 6, 7, 4, 5, 1, 8, 0])))









