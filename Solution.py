import numpy
import random
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
import time

# Globals
population_size = 10
parent_crossovers = population_size // 2
total_generations = 2 #The number of generations
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
labels = LabelEncoder().fit_transform(labels)
labels = OneHotEncoder().fit_transform(labels.reshape(len(labels), 1))
labels = labels.toarray()


# Split the data into features and target
X = data.values
y = labels


# Define the neural network architecture
n_outputs = 2  # The number of emotion classes
inputd = X.shape[1]
def build_model(input_dim, n_outputs):
    model = Sequential()
    model.add(Dense(60, input_dim=input_dim, activation="relu"))
    model.add(Dense(40, activation="relu"))
    model.add(Dense(40, activation="relu"))
    model.add(Dense(40, activation="relu"))
    model.add(Dense(40, activation="relu"))
    model.add(Dense(40, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(n_outputs, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def fitness(individual):
    seed = 42
    numpy.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    # Select features based on the individual
    selected_features = numpy.where(individual == 1)[0]
    X_selected = X[:, selected_features]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    # Build and train the model
    model = build_model(input_dim=X_train.shape[1], n_outputs=n_outputs)
    model.fit(X_train, y_train, epochs=10, verbose=0)

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy


# The initial population function
def create_initial_population():
    return numpy.random.randint(2, size=(population_size, total_genes))

start_time = time.time()

initial_population = create_initial_population()


# The selection function
def selection(population, fitnesses, total_fitness):
    #Get a random value between 0 and the total fitness
    random_value = random.uniform(0, total_fitness)
    #Calculate the cumulative probability
    cumulative_probability = 0
    for index, fitness_value in enumerate(fitnesses):
        cumulative_probability += fitness_value
        #If the cumulative probability is greater than the random value, select the individual
        if cumulative_probability >= random_value:
            break
    return population[index]
        


# The crossover function
def crossover(parent1, parent2):
    # Get a random crossover point
    crossover_point = random.randint(1, total_genes - 1)
    # child is the first part of parent1 and the second part of parent2
    child = numpy.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child


# The mutation function
def mutation(individual):
    # Get a random mutation point
    mutation_point = random.randint(0, total_genes - 1)
    if (individual[mutation_point] == 0):
        individual[mutation_point] = 1
    else:
        individual[mutation_point] = 0
    return individual


def print_feature_names(best_chromosome):
    # Ask the user if they want to print the names of the features
    choice = input("\nDo you want to print the names of the features? (yes/no): ").lower()

    if choice == "yes":
        # Get the indices of selected features (where the gene is 1)
        selected_indices = [
            index for index, gene in enumerate(best_chromosome) if gene == 1
        ]

        feature_names = [
            f"Feature {index + 1} ({data.columns[index]})" for index in selected_indices
        ]

        # Print the names of the selected features
        print("Names of the selected features:")
        for i in range(0, len(feature_names), 4):
            print("\t".join(feature_names[i:i+4]))
    elif choice == "no":
        print("No feature names will be printed.")
    else:
        print("Invalid choice. No feature names will be printed.")


# The genetic algorithm function
def genetic_algorithm():
    population = initial_population
    print("Initial population:")
    count = 0
    
    for i in range(total_generations):
        # Calculate fitness for each individual in the population
        fitnesses = [fitness(individual) for individual in population]
        # Calculate total fitness
        total_fitness = sum(fitnesses)
        # Print chromosome accuracy for each individual in the population
        if i == 0:
            print(f"Inital Generation:")
        else:
            print(f"Generation {i+1}:")
        count = 0
        for individual in population:
            accuracy = fitnesses[count]
            print(f"\tChromosome {count+1} accuracy: {accuracy:.4f}")
            count += 1
        new_population = []
        for _ in range(population_size):
            parent1 = selection(population, fitnesses, total_fitness)
            parent2 = selection(population, fitnesses, total_fitness)
            child = crossover(parent1, parent2)
            if random.random() < mutation_probability:
                child = mutation(child)
            new_population.append(child)
        population = new_population

        if i + 1 == total_generations:
            print(f"Final Generation:")
            count = 0
            for individual in population:
                accuracy = fitness(individual)
                print(f"\tChromosome {count+1} accuracy: {accuracy:.4f}")
                count += 1
    # Return the best chromosome and its accuracy
    best_chromosome = max(population, key=fitness)
    best_accuracy = fitness(best_chromosome)
    print(f"\nResulting chromosome of the GA function:")
    print(f"Best chromosome: {best_chromosome}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    end_time = time.time()
    print(f"\nTime taken: {end_time - start_time:.2f} seconds")
    return best_chromosome, best_accuracy


# Run the genetic algorithm
best_chromosome, best_accuracy = genetic_algorithm()
print_feature_names(best_chromosome)
