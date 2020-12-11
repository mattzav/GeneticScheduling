class Individual:
    def __init__(self, genotype = [], fitness = -1):
        if(fitness != -1):
            self.genotype = genotype
            self.fitness = fitness
        else:
            self.genotype = [0]*(nA+nB)


   


from random import seed
from random import randint
from random import uniform
from random import shuffle
from random import sample
import copy
import random as r


seed(2)

sizePopulation = 10
numIteration = 1
population = []


nA = 5
nB = 4
p = []
w = []


#init p and w
for i in range(nA+nB):
    p.append(randint(0,10))
    w.append(randint(0,10))
#end init p and w

#init population
totalInverseFitness = 0
for i in range (sizePopulation):
    value1_n = list(range(0,nA+nB))
    shuffle(value1_n)

    cumulativeTime = 0
    sumA = 0
    sumB = 0
    
    for j in range(nA+nB):

        cumulativeTime = cumulativeTime + p[value1_n[j]]
        
        if(value1_n[j]<nA):
            sumA = sumA + cumulativeTime*w[value1_n[j]]
        else:
            sumB = sumB + cumulativeTime*w[value1_n[j]]
    
    individual = Individual(value1_n,1/abs(sumA/nA-sumB/nB))
    population.append(individual)

    totalInverseFitness = totalInverseFitness + individual.fitness
#end init population

population_next = []
for i in range(numIteration):
    #select using roulette wheel
    random1 = uniform(0,totalInverseFitness)
    random2 = uniform(0,totalInverseFitness)
    cumulative = 0
    found1 = False
    found2 = False

    for j in range(sizePopulation):
        if cumulative<random1 and random1<cumulative+population[j].fitness and not found1:
            first = copy.deepcopy(population[j])
            found1 = True
        if cumulative<random2 and random2<cumulative+population[j].fitness and not found2:
            second = copy.deepcopy(population[j])
            found2 = True
        cumulative = cumulative + population[j].fitness
    #end selection using roulette wheel

    l=[1,2,3,4,5,6,7,8,9]
    l = [x - 1 for x in l]
    l1=[8,5,7,1,2,4,9,3,6]
    
    l1 = [x - 1 for x in l1]
    setattr(first,"genotype",l)
    setattr(second,"genotype",l1)

    print("FIRST = ",first.genotype, " , SECOND = ",second.genotype)

    #crossover ([New Variations of Order Crossover for Travelling Salesman Problem, O_X1])
    [first_point,second_point] = sample(range(1, nA+nB), 2)

    first_point = 2
    second_point = 5
    print("P1 = ",first_point, ", P2 = ",second_point)

    first_child = Individual()
    second_child = Individual()

    first_child_scheduled = [False]*(nA+nB)
    second_child_scheduled = [False]*(nA+nB)

    for j in range (min(first_point,second_point), max(first_point,second_point)):
        first_child.genotype[j] = first.genotype[j]
        first_child_scheduled[first.genotype[j]] = True

        second_child.genotype[j] = second.genotype[j]
        second_child_scheduled[second.genotype[j]] = True
    print("FIRST CHILD = ",first_child.genotype,",  SECOND CHILD = ",second_child.genotype)
        
    index1 = 0
    index2 = 0

    for j in range(0,nA+nB-max(first_point,second_point)+min(first_point,second_point)):
        print(first.genotype[(max(first_point,second_point)+j)%(nA+nB)])
        if(not second_child_scheduled[first.genotype[(max(first_point,second_point)+j)%(nA+nB)]]):
            second_child_scheduled[first.genotype[(max(first_point,second_point)+j)%(nA+nB)]] = True
            second_child.genotype[(max(first_point,second_point)+index2)%(nA+nB)] = first.genotype[(max(first_point,second_point)+j)%(nA+nB)]
            index2 = index2 + 1
       
        if(not first_child_scheduled[second.genotype[(max(first_point,second_point)+j)%(nA+nB)]]):
            first_child_scheduled[second.genotype[(max(first_point,second_point)+j)%(nA+nB)]] = True
            first_child.genotype[(max(first_point,second_point)+index1)%(nA+nB)] = second.genotype[(max(first_point,second_point)+j)%(nA+nB)]
            index1 = index1 + 1

    print("FIRST CHILD = ",first_child.genotype,",  SECOND CHILD = ",second_child.genotype)

    print(first_child_scheduled)
    for j in range (min(first_point,second_point), max(first_point,second_point)):
        print(second.genotype[j])
        if(not first_child_scheduled[second.genotype[j]]):
            first_child.genotype[(max(first_point,second_point)+index1)%(nA+nB)] = second.genotype[j]
            index1 = index1+1
        
        if(not second_child_scheduled[first.genotype[j]]):  
            second_child.genotype[(max(first_point,second_point)+index2)%(nA+nB)] = first.genotype[j]
            index2 = index2+1

    print("FIRST CHILD = ",first_child.genotype,",  SECOND CHILD = ",second_child.genotype)


    #mutation
    #postoptimization
    #replace worst solution with this new one


