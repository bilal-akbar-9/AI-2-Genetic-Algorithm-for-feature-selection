import numpy
import random
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Globals
population_size = 10
parent_crossovers = population_size // 2
total_generations = 10 #The number of generations
mutation_probability = 0.1
total_genes = 0

# Get the data from the CSV files
happy_data = pd.read_csv("data/happy.csv")
sad_data = pd.read_csv("data/sad.csv")

# Combine the happy and sad data
data = pd.concat([happy_data, sad_data])
labels = numpy.concatenate([numpy.zeros(len(happy_data)), numpy.ones(len(sad_data))])
total_genes = len(data.columns)

# Encode labels to one-hot vectors
# Encode labels to one-hot vectors
labels = LabelEncoder().fit_transform(labels)
labels = OneHotEncoder().fit_transform(labels.reshape(len(labels), 1))
labels = labels.toarray()


# Split the data into features and target
X = data.values
y = labels


# Define the neural network architecture
n_outputs = 2 #The number of emotion classes
inputd = X.shape[1]
def build_model(input_dim, n_outputs):
    model = Sequential()
    model.add(Dense(2, input_dim=input_dim, activation="relu"))
    model.add(Dense(2, activation="relu"))
    model.add(Dense(n_outputs, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def fitness(individual):
    # Select features based on the individual
    selected_features = numpy.where(individual == 1)[0]
    X_selected = X[:, selected_features]

    # Use a smaller random subset of the data
    subset_size = int(0.05 * len(X_selected))  # Reduced from 0.1 to 0.05
    subset_indices = numpy.random.choice(len(X_selected), subset_size, replace=False)
    X_subset = X_selected[subset_indices]
    y_subset = y[subset_indices]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset, test_size=0.2, random_state=42
    )

    # Build and train the model with fewer epochs
    model = build_model(input_dim=X_train.shape[1], n_outputs=n_outputs)
    model.fit(X_train, y_train, epochs=1, verbose=0)  # Reduced from 2 to 1

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy


# The initial population function
def create_initial_population():
    return numpy.random.randint(2, size=(population_size, total_genes))


initial_population = create_initial_population()

# The selection function
def selection(population):
    fitnesses = [fitness(individual) for individual in population]
    total_fitness = sum(fitnesses)
    probabilities = [fitness / total_fitness for fitness in fitnesses]

    selected_population = []
    for _ in range(parent_crossovers):
        selected_index = numpy.random.choice(len(population), p=probabilities)
        selected_population.append(population[selected_index])

    return selected_population

# The crossover function
def crossover(parent1, parent2):
    crossover_point = random.randint(1, total_genes - 1)
    #child1 is the first part of parent1 and the second part of parent2
    child1 = numpy.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    #child2 is the first part of parent2 and the second part of parent1
    child2 = numpy.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# The mutation function
def mutation(individual):
    mutation_point = random.randint(0, total_genes - 1)
    # if the gene is 0, change it to 1 and vice versa
    individual[mutation_point] = 1 - individual[mutation_point]
    return individual


# The genetic algorithm function
def genetic_algorithm():
    population = initial_population
    for i in range(total_generations):
        print(f"Generation {i + 1}:")
        selected_population = selection(population)
        new_population = []
        for _ in range(parent_crossovers):
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            child1, child2 = crossover(parent1, parent2)
            if random.random() < mutation_probability:
                child1 = mutation(child1)
            if random.random() < mutation_probability:
                child2 = mutation(child2)

            new_population.append(child1)
            new_population.append(child2)
        population = new_population

        # Print chromosome accuracy for each individual in the population
        count = 0
        for individual in population:
            accuracy = fitness(individual)
            print(f"\tChromosome {count+1} accuracy: {accuracy:.4f}")
            count += 1

    # Return the best chromosome and its accuracy
    best_chromosome = max(population, key=fitness)
    best_accuracy = fitness(best_chromosome)
    print(f"\nResulting chromosome of the GA function:")
    print(f"\tBest chromosome: {best_chromosome}")
    print(f"\tBest accuracy: {best_accuracy:.4f}")
    return best_chromosome, best_accuracy


genetic_algorithm()
