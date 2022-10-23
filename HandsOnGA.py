from __future__ import print_function
from cProfile import label
from turtle import color
import numpy as np
import copy
import time
from operator import attrgetter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.optimize as so


#Add restart when not imroving. tune local search if so. look in to muation role with this

#Minimizer

class Fitness:
    def __init__(self):
        self.name = 'Ackley function'
        self.anal_opt_x = [0]
        self.anal_opt_fit = 0
       
    # def function(self, x):
    #     s1 = -0.2 * np.sqrt(np.sum([i ** 2 for i in x]) / len(x))
    #     s2 = np.sum([np.cos(2 * np.pi * i) for i in x]) / len(x)
    #     return 10 + np.exp(1) - 20 * np.exp(s1) - np.exp(s2)
    def function(self, x):
        s1 = -0.2 * np.sqrt((x[0]**2+x[1]**2) / 2)
        s2 = (np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1])) / 2
        return 20 + np.exp(1) - 20 * np.exp(s1) - np.exp(s2)

# class Fitness:
#     name = 'Himmelblaus function'
#     anal_opt_x = '(3,2),(-2.805,3.12),(-3.779,-3.283),(3.584,-1.848)'
#     anal_opt_fit = 0
       
#     def function(x):

#         return (x[0]*2+x[1]-11)**2+(x[0]+x[1]**2-7)**2

# class Fitness:
#     name = 'Rastrigin function'
#     anal_opt_x = [0,0]
#     anal_opt_fit = 0
       
#     def function(x):
#         return 10*len(x)+np.sum([i**2-10*np.cos(2*np.pi*i) for i in x])


# class Fitness:
#     name = 'NLP Gekko'
#     anal_opt_x = 'x1: [1.000000057] x2: [4.74299963] x3: [3.8211500283] x4: [1.3794081795]'
#     anal_opt_fit = 17.0140171270735   
       
#     def function(x):
#         obj = x[0]*x[3]*(x[0]+x[1]+x[2])+x[2]
#         const1 = max(0,x[0]*x[1]*x[2]*x[3]-25)**2 #    
#         const2 = max(0,x[0]**2+x[1]**2+x[2]**2+x[3]**2-40)**2
#         const3 = max(0,-(x[0]**2+x[1]**2+x[2]**2+x[3]**2-40))**2
#         a = 10
#         return obj + a*const1 + a*const2 + a*const3




class Individual():  
    def __init__(self, gene):
        self.gene = gene
        self.fitness = Fitness().function(gene)

class MA():
    def __init__(self,
                  population_size,
                  fitness_vector_size,
                  generations,
                  prob_crossover,
                  prob_mutation,
                  mut_step,
                  prob_local,
                  max_local_gens,
                  upper_bound,
                  lower_bound):
        self.population_size = population_size
        self.fitness_vector_size = fitness_vector_size
        self.generations = generations
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
        self.mut_step = mut_step
        self.prob_local = prob_local
        self.max_local_gens = max_local_gens
        self.upper_bound = upper_bound #Ska vara en array för varje dimension kan ha sin egna bounds
        self.lower_bound = lower_bound
        self.population = None
        self.best_fitness = []
        self.mean_fitness = []
        self.best_gene = []


    def generate_population(self):
        self.population = [Individual(self.random_gene()) for _ in range(self.population_size)]

    def random_gene(self):
        return np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.fitness_vector_size))
    
    def bouderyInforcer(self, individual):
        for index, value in enumerate(individual.gene):
            if self.lower_bound == None:
                pass
            elif value < self.lower_bound:               
                individual.gene[index] = self.lower_bound
                
            if self.upper_bound == None:
                pass
            elif value > self.upper_bound:
                individual.gene[index] = self.upper_bound  
        return individual

    def selTournament(self, individuals, tournsize=2):
        individuals_copy = individuals.copy()
        chosen = []
        for i in range(int(self.population_size/tournsize)):
            aspirants = [np.random.choice(individuals_copy) for i in range(tournsize)]
            individuals_copy = [x for x in individuals_copy if x not in aspirants]
            chosen.append(min(aspirants, key=attrgetter('fitness')))
        return chosen
      
    def selElite(self, individuals):
        temp = sorted(individuals, key=lambda x: x.fitness)
        chosen = temp[:int(self.population_size/2)]
        return chosen

    def mutGaussian(self, individual): ##Lägg till så att man kan köra en lista med mu sigma
        size = self.fitness_vector_size

        for i in range(size):
            if np.random.random() < self.prob_mutation:
                individual[i] += np.random.normal(0, self.mut_step)
        return individual

    def cxOnePoint(self, ind1, ind2):
        if np.random.random() >= self.prob_crossover:
            return ind1, ind2
        else:
            size = self.fitness_vector_size
            if size > 2:                 
                cxpoint = np.random.randint(1, size - 1)
                ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
            if size == 2:
                if np.random.uniform()>0.5:
                    ind1[0], ind2[0] = ind2[0], ind1[0]
                else:
                    ind1[1], ind2[1] = ind2[1], ind1[1]
            if size == 1:
                raise
                print('1-D problems cant use this crossover')

            return ind1, ind2   
        
    def climber(self, individual):

        for _ in range(self.max_local_gens):
            candidate_gene = individual.gene + np.random.normal(0,self.mut_step/10)
            candidate = self.bouderyInforcer(Individual(candidate_gene)) 
            if candidate.fitness < individual.fitness:
                individual = candidate
        return individual        


    def reproduce(self, selected):
        children = []
        for a, b in zip(selected[::2], selected[1::2]):
            children.append(self.bouderyInforcer(Individual(self.mutGaussian(self.cxOnePoint(a.gene, b.gene)[0]))))
            children.append(self.bouderyInforcer(Individual(self.mutGaussian(self.cxOnePoint(a.gene, b.gene)[1]))))
        teenagers = []  
        for kid in children:
            if np.random.random() < self.prob_local:
                kid = self.climber(kid)
            teenagers.append(kid)
        
        return teenagers

    def run(self):
        self.generate_population()
        for gen in range(self.generations):
            selected = self.selElite(self.population) 
            # selected = self.selTournament(self.population)
            children = self.reproduce(selected)
            self.population = selected + children
            self.population = sorted(self.population, key=lambda x: x.fitness)
            self.mean_fitness.append(np.mean([i.fitness for i in self.population]))
            self.best_fitness.append(self.population[0].fitness)
            self.best_gene.append(self.population[0].gene)

population_size = 10
fitness_vector_size = 2
generations = 500
prob_crossover = 0.8
prob_mutation = 0.2
mut_step = 0.1
prob_local = 0.8
max_local_gens = 20
upper_bound = 5
lower_bound = -5

bnds = ((-5, 5), (-5, 5))

MA = MA( population_size,
    fitness_vector_size,
    generations,
    prob_crossover,
    prob_mutation,
    mut_step,
    prob_local,
    max_local_gens,
    upper_bound,
    lower_bound)

MA.run()

plt.plot(MA.best_fitness, label = 'Best')
plt.title(str(fitness_vector_size) +' dimensional')
plt.plot(MA.mean_fitness, label = 'Mean')
plt.legend()
plt.show()


print('This is the', Fitness().name )

print('The memetic algorithm gives:')

print('Best Fittness: ', round(MA.best_fitness[-1],5), ' Analytical Optimal Fitness: ', Fitness().anal_opt_fit)

print('Best Solution: ', MA.best_gene[-1], ' Analytical Optimal Solution: ', Fitness().anal_opt_x)

print('Using this as solution as a starting guess for Powell search we get:')

res = so.minimize(Fitness().function, MA.best_gene[-1], bounds = bnds , method='Powell')

print('Best Fittness: ', res.fun, ' Analytical Optimal Fitness: ', Fitness().anal_opt_fit)

print('Best Solution: ', res.x, ' Analytical Optimal Solution: ', Fitness().anal_opt_x)


r = 5
xlist = np.linspace(-r , r , 500)
ylist = np.linspace(-r , r , 500)

X, Y = np.meshgrid(xlist, ylist)

Z = Fitness().function([X,Y])


fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z,levels = np.arange(0,16,1))
fig.colorbar(cp) 

ax.scatter(res.x[0],res.x[1], color = 'r', label = "Found optimum")
plt.legend()
plt.show()