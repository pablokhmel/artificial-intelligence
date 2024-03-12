from itertools import compress
import random
import time
import matplotlib.pyplot as plt
import numpy

from data import *

def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness

def roulette_wheel_selection(items, knapsack_max_capacity, n_selection, population):
    sum_f = sum([fitness(items, knapsack_max_capacity, individual) for individual in population])
    selection_probs = [fitness(items, knapsack_max_capacity, individual) / sum_f for individual in population]

    selection = [None] * n_selection
    for i in range(n_selection):
        selection[i] = population[numpy.random.choice(len(population), p=selection_probs)]

    return selection

def elitisme(n_elite, population, items, knapsack_max_capacity):
    population_fitness_indices = numpy.argsort([fitness(items, knapsack_max_capacity, individual) for individual in population])[::-1]
    return [population[population_fitness_indices[i]] for i in range(n_elite)]

def crossover(parent1, parent2):
    crossover_point = len(parent1) // 2
    child1 = parent1[crossover_point:] + parent2[:crossover_point]
    child2 = parent2[crossover_point:] + parent1[:crossover_point]
    return child1, child2

def mutate(population):
    for individual in population:
        i = random.randrange(len(individual))
        individual[i] = not individual[i]

items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 1

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
for _ in range(generations):
    population_history.append(population)
    # TODO: implement genetic algorithm
    selection = roulette_wheel_selection(items, knapsack_max_capacity, n_selection, population)
    elite = elitisme(n_elite, population, items, knapsack_max_capacity)
    next_generation = elite

    while (len(next_generation)) < population_size:
        parent1, parent2 = random.sample(selection, 2)
        child1, child2 = crossover(parent1, parent2)
        next_generation.extend([child1, child2])

    population = next_generation
    mutate(population)

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
