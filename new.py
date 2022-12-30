import numpy as np


def convert_cycle_to_adj(chromosome):
    adj = np.ma.masked_all(len(chromosome)-1, dtype=np.int64)
    for i in range(len(chromosome) - 1):
        adj[chromosome[i]] = chromosome[i + 1]
    return adj


def shuffle_mutation(problem, child):
    c1 = np.random.randint(1, len(child) - 2)
    c2 = np.random.randint(c1 + 1, np.clip(c1 + 6, 1, len(child) - 1))
    np.random.shuffle(child[c1:c2])
    return child


def lso(c, k):
    best = np.array(c)
    copy = np.array(c)
    best_fit = fitness(convert_cycle_to_adj(c), dMatrixAlt)
    # print(f"lso on {c} with fit: {best_fit}")
    for i in range(k):
        print(f"c before: {c}")
        new = two_opt_mutation(copy)
        print(f"c after: {c}")
        new_fit = fitness(convert_cycle_to_adj(new), dMatrixAlt)
        if new_fit < best_fit:
            print(f"new best found: {new} with fit: {new_fit}")
            best = np.array(new)
            best_fit = int(new_fit)
        # else:
            # print(f"no luck with {new}")
    # print(f"lso done! best: {best} with fit: {best_fit}")
    return best


def two_opt_mutation(child, c1=None, c2=None):
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


def fitness(chromosome, dmatrix):
    fitness = 0
    # print(f"chromosome: {chromosome}")
    for i in range(10):
        # print(chromosome[i])
        fitness += dmatrix[i, chromosome[i]]
    # print(f"chromosome {chromosome} with fitness: {fitness}")

    return fitness


# dMatrixAlt = [[i for i in range(10)] for j in range(10)]
# dMatrixAlt = np.array(dMatrixAlt)
dMatrixAlt = np.random.randint(1, 100, size=(10, 10))
test = [0, 4, 2, 5, 1, 8, 3, 6, 7, 9, 0]

print(dMatrixAlt)

print(fitness(convert_cycle_to_adj(test), dMatrixAlt))
for i in range(1000):
    # after = shuffle_mutation(None, test)
    # after = two_opt_mutation(test, 7)
    test = lso(test, 100)
    print(test)
    print(fitness(convert_cycle_to_adj(test), dMatrixAlt))
