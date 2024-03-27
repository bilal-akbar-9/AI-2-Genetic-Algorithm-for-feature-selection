import numpy
import random
import pandas as pd
# Globals
population_size = 10
parent_crossovers = population_size // 2
total_generations = 10 #The number of generations
crossover_probability = 0.8
mutation_probability = 0.1
total_genes = 0


# Get the data from the CSV files
happy_data = pd.read_csv("data/happy.csv")
sad_data = pd.read_csv("data/sad.csv")

# Combine the happy and sad data
data = pd.concat([happy_data, sad_data])
total_genes = len(data.columns)

# The initial population function
def create_initial_population():
    return numpy.random.randint(2, size=(population_size, total_genes))

initial_population = create_initial_population()

#The fitness function
def fitness(individual):
    fitness = 0
    for i in range(4):
        fitness += individual[i] * data.iloc[i, 0]
    return fitness

#The selection function
def selection(population):
    selected_population = []
    for i in range(parent_crossovers):
        selected_population.append(max(population, key=fitness))
    return selected_population

#The crossover function
def crossover(parent1, parent2):
    crossover_point = random.randint(1, total_genes - 1)
    #child1 is the first part of parent1 and the second part of parent2
    child1 = numpy.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    #child2 is the first part of parent2 and the second part of parent1
    child2 = numpy.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

#The mutation function  
def mutation(individual):
    mutation_point = random.randint(0, total_genes - 1)
    #if the gene is 0, change it to 1 and vice versa
    individual[mutation_point] = 1 - individual[mutation_point]
    return individual

# The genetic algorithm function
def genetic_algorithm():
    population = initial_population
    for i in range(total_generations):
        selected_population = selection(population)
        new_population = []
        for j in range(parent_crossovers):
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            if random.random() < crossover_probability:
                child1, child2 = crossover(parent1, parent2)
                new_population.append(child1)
                new_population.append(child2)
            else:
                new_population.append(parent1)
                new_population.append(parent2)
        for k in range(population_size):
            if random.random() < mutation_probability:
                new_population[k] = mutation(new_population[k])
        population = new_population
    return max(population, key=fitness)

print(genetic_algorithm())
