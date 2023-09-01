'''
Problem statement:
    Y = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6

    given Y = 44
    x1 .... x6 = [4, -2, 3.5, 5, -11, -4.7]

    find the optimal weight vector [w1, ...., w6]

'''

import pygad as pg
import numpy as np

function_inputs = [4, -2, 3.5, 5, -11, -4.7]
desired_output  = 44

def fitness_func(instance, solution, solution_idx):
    output = np.sum(solution + function_inputs)
    fitness = 1.0 / np.abs(output-desired_output)
    return  fitness

fitness_function = fitness_func

sol_per_pop = 50
num_genes = 6  # as the number of elements is 6 in wieght vector

init_range_low = -2
init_range_high = 5
# Random values

pop_size = (sol_per_pop, num_genes)

initial_population = np.random.uniform(low=init_range_low, 
                                       high=init_range_high, 
                                       size = pop_size)

print(initial_population)



num_generation = 100

num_parents_mating = 8

#Roulette wheel selection method is used
parent_selection_type = 'rws'

keep_parents = 8

crossover_type = 'single_point'
mutation_type = 'random'

mutation_percent_genes = 10

# Create GA Constructor
ga_instance = pg.GA(num_generations= num_generation, 
                    num_parents_mating= num_parents_mating,
                    fitness_func= fitness_function,
                    sol_per_pop= sol_per_pop,
                    num_genes= num_genes,
                    init_range_low= init_range_low,
                    init_range_high= init_range_high,
                    parent_selection_type= parent_selection_type,
                    keep_parents= keep_parents,
                    crossover_type= crossover_type,
                    mutation_type= mutation_type,
                    mutation_percent_genes= mutation_percent_genes)

# ga_instance - constructor defined just above

# Runing the constructor
ga_instance.run()

ga_instance.plot_result()

solution, solution_fitness, _= ga_instance.best_solution()

print('\n Best solution vector: {solution}'.format(solution=solution))

prediction = np.sum(np.array(function_inputs)*solution)

print('\n Predicted value: {prediction}'.format(prediction = prediction))
