class Individual:
    def __init__(self, genotype = [], fitness = -1, objF = -1,sumA=-1,sumB=-1):
        if(genotype != []):
            self.genotype = genotype
            self.fitness = fitness
            self.objF = objF
            self.sumA = sumA
            self.sumB = sumB
        else:
            self.genotype = [0]*(nA+nB)
   
def inverseMutation(individual):
    [first_point,second_point] = sample(range(0, nA+nB), 2)
    for j in range((max(first_point,second_point)-min(first_point,second_point))//2+1):
        swap = individual.genotype[max(first_point,second_point)-j]
        individual.genotype[max(first_point,second_point)-j] = individual.genotype[min(first_point,second_point)+j]
        individual.genotype[min(first_point,second_point)+j] = swap

def adjacentTwo_JobChange(individual):
    point = randint(0,nA+nB-2)
    #print("POINT",point)
    swap = individual.genotype[point]
    individual.genotype[point] = individual.genotype[point+1]
    individual.genotype[point+1] = swap
    return individual

def postOptimize(individual):
    global optimum,best
    for i in range(nA+nB-1):
        for j in range(i+1,nA+nB):
            newSumA = individual.sumA
            newSumB = individual.sumB
            for k in range(i,j+1):
                if(individual.genotype[i]<nA):
                    newSumA = newSumA+p[individual.genotype[k]]*w[individual.genotype[i]]
                else:
                    newSumB = newSumB+p[individual.genotype[k]]*w[individual.genotype[i]]

                if(individual.genotype[j]<nA):
                        newSumA = newSumA-p[individual.genotype[k]]*w[individual.genotype[j]]
                else:
                    newSumB = newSumB-p[individual.genotype[k]]*w[individual.genotype[j]]
                if(k>i and k<j):
                    if(individual.genotype[k]<nA):
                        newSumA = newSumA+(p[individual.genotype[j]]-p[individual.genotype[i]])*w[individual.genotype[k]]
                    else:
                        newSumB = newSumB+(p[individual.genotype[j]]-p[individual.genotype[i]])*w[individual.genotype[k]]
            if (individual.genotype[i]<nA):
                newSumA = newSumA-p[individual.genotype[i]]*w[individual.genotype[i]]
            else:
                newSumB = newSumB-p[individual.genotype[i]]*w[individual.genotype[i]]
            
            if (individual.genotype[j]<nA):
                newSumA = newSumA+p[individual.genotype[j]]*w[individual.genotype[j]]
            else:
                newSumB = newSumB+p[individual.genotype[j]]*w[individual.genotype[j]]
            
            objValue = abs(newSumA/nA-newSumB/nB)
            if(objValue<individual.objF):
                swap = individual.genotype[j]
                individual.genotype[j] = individual.genotype[i]
                individual.genotype[i] = swap
                setattr(individual,"objF",objValue)
                if(objValue <= 0):
                    optimum = True
                    best = individual
                    return
                setattr(individual,"fitness",1/objValue)
                setattr(individual,"sumA",newSumA)
                setattr(individual,"sumB",newSumB)

def evaluateFitness(individual):
    global optimum, best
    cumulativeTime = 0
    sumA = 0
    sumB = 0
    
    for j in range(nA+nB):

        cumulativeTime = cumulativeTime + p[individual.genotype[j]]
        
        if(individual.genotype[j]<nA):
            sumA = sumA + cumulativeTime*w[individual.genotype[j]]
        else:
            sumB = sumB + cumulativeTime*w[individual.genotype[j]]
    
    setattr(individual,"objF",abs(sumA/nA-sumB/nB))
    if(abs(sumA/nA-sumB/nB)<=0):
        optimum = True
        best = individual
        return
    setattr(individual,"fitness",1/abs(sumA/nA-sumB/nB))
    setattr(individual,"sumA",sumA)
    setattr(individual,"sumB",sumB)

def rouletteWheel():
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
        if(found1 and found2):
            break
    #end selection using roulette wheel

    return [first,second]

def twoPointCrossoverVerI(first,second):
    [first_point,second_point] = sample(range(1, nA+nB), 2)
    first_point = 2
    second_point = 6
    child = Individual()
    child_scheduled = [False]*(nA+nB)

    for j in range (0,min(first_point,second_point)):
        child.genotype[j] = first.genotype[j]
        child_scheduled[first.genotype[j]] = True

    for j in range(max(first_point,second_point),nA+nB):
        child.genotype[j] = first.genotype[j]
        child_scheduled[first.genotype[j]] = True
        
    
    index1 = min(first_point,second_point)

    #scan remaining element from point2 to point1 and eventually insert
    for j in range(0,nA+nB):
        if(not child_scheduled[second.genotype[j]]):
            child_scheduled[second.genotype[j]] = True
            child.genotype[index1] = second.genotype[j]
            index1 = index1 + 1
        
        if(index1 == max(first_point,second_point)):
            break
    return child

def twoPointCrossoverVerII(first,second):
    [first_point,second_point] = sample(range(1, nA+nB), 2)

    child = Individual()
    child_scheduled = [False]*(nA+nB)

    #copy element from point1 to point2
    for j in range (min(first_point,second_point), max(first_point,second_point)):
        child.genotype[j] = first.genotype[j]
        child_scheduled[first.genotype[j]] = True
        
    index1 = 0

    #scan remaining element from point2 to point1 and eventually insert
    for j in range(0,nA+nB):
        if(not child_scheduled[second.genotype[j]]):
            child_scheduled[second.genotype[j]] = True
            child.genotype[index1] = second.genotype[j]
            index1 = index1 + 1
            if(index1==min(first_point,second_point)):
                index1=max(first_point,second_point)
        
        if(index1 == nA+nB):
            break

    return child

def onePointCrossover(first,second):
    child = Individual()
    child_scheduled = [False]*(nA+nB)
    
    point = randint(1,nA+nB-1)
    #print("FIRST",first.genotype," SECODND",second.genotype)
    #print("POINT ",point)
    if uniform(0,1)<=0.5:
       # print("LEFT")
        start = 0
        end = point
        index = point
    else:
       # print("RIGHT")
        start = point
        end = nA+nB
        index = 0

    for i in range(start,end):
            child.genotype[i] = first.genotype[i]
            child_scheduled[first.genotype[i]] = True

    for i in range(nA+nB):
        if not child_scheduled[second.genotype[i]]:
            child.genotype[index] = second.genotype[i]
            index = index + 1
            if index == start or index == nA+nB:
                break
    return child

def positionBasedCrossover(first, second):
    child = Individual()
    child_scheduled = [False]*(nA+nB)    
    child_position_used = [False]*(nA+nB)

    numPos = randint(1,nA+nB)

    positions = sample(range(0, nA+nB), numPos)

    numPos = 4
    positions = [1,2,4,7]    
    for j in range (numPos):
        child.genotype[positions[j]] = first.genotype[positions[j]]
        child_scheduled[first.genotype[positions[j]]] = True
        child_position_used[positions[j]] = True

    index1 = 0
    inserted = 0
    for j in range (nA+nB):
        if(not child_scheduled[second.genotype[j]]):
            child_scheduled[second.genotype[j]] = True
            while child_position_used[index1]:
                index1 = index1+1
            child_position_used[index1] = True
            child.genotype[index1] = second.genotype[j]
            inserted = inserted+1
            if (inserted == nA+nB-numPos):
                break

    return child        
    
def initPopulation():
    global UB, totalInverseFitness, best, optimum

    #init population
    totalInverseFitness = 0
    UB = pwSum
    best = Individual()
    optimum = False


    #inizializzare random alternando pero nA e nB invece di farlo completamente random
    value1_n = list(range(0,nA+nB))
    shuffle(value1_n)
    individual = Individual(value1_n)
    evaluateFitness(individual)
    population.append(individual)

    #update UB
    if individual.objF <= UB:
        UB = individual.objF
        best = individual
        if individual.objF <=0:
            optimum = True
            return
    
    totalInverseFitness = totalInverseFitness + individual.fitness

def initParam():
    global sizePopulation, numIteration, population, crossoverProb,mutationProb,pwSum,nA,nB,p,w
    seed(scenario)

    sizePopulation = 10
    numIteration = 1000
    population = []
    crossoverProb = 1
    mutationProb = 1

    p = []
    w = []

    pwSum = 0

    #init p and w
    instance = f.readline()

    p = instance.split(";")[0].split(",")
    p = [int(numeric_string) for numeric_string in p]
    #print(p)
    w = instance.split(";")[1].split(",")
    w = [int(numeric_string) for numeric_string in w]
    #print(p)

    for i in range(nA+nB):
        pwSum = pwSum + p[i]*w[i]
    #end init p and w



    #end init param

if __name__ == "__main__":
    
    from random import seed
    from random import randint
    from random import uniform
    from random import shuffle
    from random import sample
    from random import choice
    import time
    import copy
    import random as r

    for nA in range(50,51,50):
        for nB in range(50,51,50):
            f = open("Dataset\\BigInterval\\"+str(nA)+"_"+str(nB)+".txt", "r")
            res = open("Result\\"+str(nA)+"_"+str(nB)+".txt", "a")
           
            totalTime = 0
            for scenario in range (50):
                initParam()

                start = time.time()

                for i in range (sizePopulation):
                    initPopulation() #(2) random - alternando uno di Ja e uno di Jb 
                
                if(optimum):
                    break
                newInverseFitness = 0  

                for i in range(numIteration):
                    population_next = []
                    
                    for j in range(sizePopulation):
                        [first,second] = rouletteWheel() #(3) roulette-binary tournment-ktournment 
                        
                        
                        nA = 4
                        nB = 4
                        first.genotype = [0,1,2,3,4,5,6,7]
                        second.genotype = [4,7,0,3,1,2,6,5]

                        if(uniform(0,1)<=1):
                            #child = onePointCrossover(first,second)
                            #child = twoPointCrossoverVerI(first,second)
                            #child = twoPointCrossoverVerII(first,second) #(5)  position based ver1 - extraggo k random e prendo i primi k di 1 poi k dell'altro controllando se non ci sono gia (ex 1,2,3,4,5,6 e 3,2,4,1,3,5,6 e  k = 2 diventa 1,2,4,3,5,6)
                            child = positionBasedCrossover(first,second)
                        else:
                            child = first
                        print("After cross, child = ",child.genotype)

                        #inverse mutation [A comparative study]
                        
                        if(uniform(0,1) <= mutationProb):
                            #inverseMutation(child) #(6-7)inverse mutation (tsp), adjacent two job, arbitrary two job, arbuitrary three job, shift change, "ARBITRARY LOT EXCHANGE (k job adiacenti e k altri job adiacenti e scambiamo)", "ADJACENT LOT EXCHANGE"
                            adjacentTwo_JobChange(child)
                        #end mutation
                        
                        #print("After mutation, child = ",child.genotype)

                        #compute objfunction
                        evaluateFitness(child)
                        
                        if(optimum):
                            break

                        #postoptimization two opt
                        postOptimize(child)
                        
                        if(optimum):
                            break

                        population_next.append(child)

                        newInverseFitness = newInverseFitness + child.fitness

                        if(child.objF <= UB):
                            UB = child.objF
                            indexBestChildren = j

                            if(child.objF <=0):
                                best = child
                                optimum = True
                                break

                    if optimum:
                        break

                    #choose which individual to delete in order to insert the previous best
                    toRemove = choice(list(range(indexBestChildren))+list(range(indexBestChildren+1,sizePopulation)))
                    
                    newInverseFitness = newInverseFitness - population_next[toRemove].fitness
                    newInverseFitness = newInverseFitness + best.fitness
                    totalInverseFitness = newInverseFitness
                    newInverseFitness = 0
                
                    population_next[toRemove] = best
                    population = population_next
                    
                    if(population_next[indexBestChildren].objF<best.objF):
                        best = population_next[indexBestChildren]
                
                totalTime = totalTime + (time.time()-start)
                res.write("UB,"+str(time.time()-start)+"\n")
            res.write("AVG TIME,"+str(totalTime/50))


