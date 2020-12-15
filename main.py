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

#################

#MUTATION OPERATORS

def adjacentTwo_JobChange(individual):
    point = randint(0,nA+nB-2)
    swap = individual.genotype[point]
    individual.genotype[point] = individual.genotype[point+1]
    individual.genotype[point+1] = swap

def arbitraryTwo_JobChange(individual):
    [first,second] = sample(range(0, nA+nB), 2)
    swap = individual.genotype[first]
    individual.genotype[first] = individual.genotype[second]
    individual.genotype[second] = swap

def arbitraryThree_JobChange(individual):
    [first,second,third] = sample(range(0, nA+nB), 3)
    swap = individual.genotype[third]
    individual.genotype[third] = individual.genotype[second]
    individual.genotype[second] = individual.genotype[first]
    individual.genotype[first] = swap

def shift(individual):
    [first,second] = sample(range(0, nA+nB), 2)

    #put element in second position in position first
    if(second>first):
        element = individual.genotype[second]
        for i in range(second,first,-1):
            individual.genotype[i] = individual.genotype[i-1]
        individual.genotype[first] = element
    else:
        element = individual.genotype[second]
        for i in range(second,first):
            individual.genotype[i] = individual.genotype[i+1]
        individual.genotype[first] = element

def adjacentBatchExchange(individual):
    first = randint(0,nA+nB-2)
    lotSize = randint(1,(nA+nB-first)//2)
    for i in range(lotSize):
        swap = individual.genotype[first+i]
        individual.genotype[first+i] = individual.genotype[first+lotSize+i]
        individual.genotype[first+lotSize+i] = swap

def arbitraryBatchExchange(individual):
    [first,second] = sample(range(0, nA+nB), 2)
    lotSize = randint(1,min(abs(first-second),nA+nB-max(first,second)))
    for i in range(lotSize):
        swap = individual.genotype[first+i]
        individual.genotype[first+i] = individual.genotype[second+i]
        individual.genotype[second+i] = swap

#################

#LOCAL SEARCH AND EVALUATION

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

#################

#SELECTION OPERATORS
def rouletteWheel():
    random1 = uniform(0,totalInverseFitness)
    random2 = uniform(0,totalInverseFitness)
    cumulative = 0
    found1 = False
    found2 = False

    for j in range(sizePopulation):
        if cumulative<random1 and random1<cumulative+population[j].fitness and not found1:
            first = population[j]
            found1 = True
        if cumulative<random2 and random2<cumulative+population[j].fitness and not found2:
            second = population[j]
            found2 = True
        cumulative = cumulative + population[j].fitness
        if(found1 and found2):
            break

    return [first,second]

def tournment(k):
    group1 = []
    group2 = []
    for i in range(k):
        group1.append(population[randint(0,sizePopulation-1)])
        group2.append(population[randint(0,sizePopulation-1)])
   
    win1 = group1[0]
    obj1 = group1[0].objF
    
    win2 = group2[0]
    obj2 = group2[0].objF

    for i in range(1,k):
        if group1[i].objF<obj1:
            obj1 = group1[i].objF
            win1 = group1[i]

        if group2[i].objF<obj2:
            obj2 = group2[i].objF
            win2 = group2[i]
        
    return [win1,win2]

#################

#CROSSOVER OPERATORS

def onePointCrossover(first,second):
    child = Individual()
    child_scheduled = [False]*(nA+nB)
    
    point = randint(1,nA+nB-1)
    
    if uniform(0,1)<=0.5:
        start = 0
        end = point
        index = point
    else:
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

def twoPointCrossoverVerI(first,second):
    [first_point,second_point] = sample(range(1, nA+nB), 2)
   
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

def positionBasedCrossover(first, second):
    child = Individual()
    child_scheduled = [False]*(nA+nB)    
    child_position_used = [False]*(nA+nB)

    numPos = randint(1,nA+nB)

    positions = sample(range(0, nA+nB), numPos)

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
    
def kStepSizeBasedCrossover(first, second):
    child = Individual()
    child_scheduled = [False]*(nA+nB)

    stepSize = randint(1,nA+nB-2) #se faccio nA+nB-1 mi rimane solo l'ultima posizione quindi ovviamente ci andrà l'elemento mancantes

    index1 = 0
    indexPar1 = 0
    indexPar2 = 0
    completed = False
    while(not completed):
        for i in range(stepSize):
            avoidOutOfRange1 = min(nA+nB-1,indexPar1+i)
            if(not child_scheduled[first.genotype[avoidOutOfRange1]]):
                child.genotype[index1] = first.genotype[avoidOutOfRange1]
                child_scheduled[first.genotype[avoidOutOfRange1]] = True
                index1 = index1 + 1

                if(index1 == nA + nB):
                    completed = True
                    break

        if(completed):
            break

        for i in range(stepSize):
            avoidOutOfRange2 = min(nA+nB-1,indexPar2+i)
            if(not child_scheduled[second.genotype[avoidOutOfRange2]]):
                child.genotype[index1] = second.genotype[avoidOutOfRange2]
                child_scheduled[second.genotype[avoidOutOfRange2]] = True
                index1 = index1 + 1
                if(index1 == nA + nB):
                    completed = True
                    break

        indexPar1 = indexPar1 + stepSize    
        indexPar2 = indexPar2 + stepSize
    return child

#################

#INIT OPERATORS
def initPopulationRandom():
    global UB, totalInverseFitness, best, optimum, sizePopulation

    #init population
    totalInverseFitness = 0
    UB = pwSum
    best = Individual()
    optimum = False

    for i in range (sizePopulation):
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

def initPopulationAlternating():
    global UB, totalInverseFitness, best, optimum, sizePopulation

    #init population
    totalInverseFitness = 0
    UB = pwSum
    best = Individual()
    optimum = False
    
    for i in range (sizePopulation):
        value_1_nA = list(range(0,nA))
        value_nA_n = list(range(nA,nA+nB))

        shuffle(value_1_nA)
        shuffle(value_nA_n)

        individual = Individual()

        index1 = 0
        index2 = 0
        for i in range (nA+nB):
            if(i%2==0):
                if(index1 < nA):
                    individual.genotype[i] = value_1_nA[index1]
                    index1 = index1 + 1
                else:
                    individual.genotype[i] = value_nA_n[index2]
                    index2 = index2 +1
            else:
                if(index2 < nB):
                    individual.genotype[i] = value_nA_n[index2]
                    index2 = index2 +1
                else:
                    individual.genotype[i] = value_1_nA[index1]
                    index1 = index1 + 1

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

    for nA in range(10,51,50):
        for nB in range(20,51,50):
            f = open("Dataset\\"+str(nA)+"_"+str(nB)+".txt", "r")
            res = open("Result\\"+str(nA)+"_"+str(nB)+".txt", "a")
           
            totalTime = 0

            for scenario in range (50):
                initParam() #initParameter
                start = time.time() #takeTime

                #####################################################
                #INIT POPULATION, DECOMMENT THE RIGHT FUNCTION
                #initPopulationRandom() 
                initPopulationAlternating()
                #####################################################

                if(optimum):
                    break

                newInverseFitness = 0 #used to update the total inverse fitness at the end of each iteration

                while(time.time()-start<=1800):
                    population_next = []
                    
                    for j in range(sizePopulation):

                        #####################################################
                        #SELECT PARENTS, DECOMMENT THE RIGHT FUNCTION
                        [first,second] = rouletteWheel() 
                        #[first,second] = tournment(2)
                        #[first,second] = tournment(randint(1,nA+nB-1))
                        #####################################################
                        
                        #####################################################
                        #CROSSOVER, DECOMMENT THE RIGHT FUNCTION
                        if(uniform(0,1)<=crossoverProb):
                            #child = onePointCrossover(first,second)
                            #child = twoPointCrossoverVerI(first,second)
                            #child = twoPointCrossoverVerII(first,second)
                            #child = positionBasedCrossover(first,second)
                            child = kStepSizeBasedCrossover(first,second)
                        else:
                            child = Individual()
                            child.genotype = first.genotype
                        #####################################################

                        #####################################################
                        #MUTATION, DECOMMENT THE RIGHT FUNCTION
                        if(uniform(0,1) <= mutationProb):
                            #inverseMutation(child) 
                            #adjacentTwo_JobChange(child)
                            #arbitraryTwo_JobChange(child)
                            #arbitraryThree_JobChange(child)
                            #shift(child)
                            #arbitraryBatchExchange(child)
                            adjacentBatchExchange(child)
                        #####################################################

                        #####################################################
                        #COMPUTE OBJECTIVE VALUE AND FITNESS
                        evaluateFitness(child)
                        if(optimum):
                            break

                        postOptimize(child)                        
                        if(optimum):
                            break
                        #####################################################

                        #####################################################
                        #APPEND NEW CHILD, UPDATE NEW INVERSE FITNESS, UPDATE UB, UPDATE BEST
                        population_next.append(child)
                        newInverseFitness = newInverseFitness + child.fitness

                        if(child.objF <= UB):
                            UB = child.objF
                            indexBestChildren = j

                            if(child.objF <=0):
                                best = child
                                optimum = True
                                break
                        #####################################################

                    #END ITERATION    
                    if optimum:
                        break

                    #####################################################
                    #REMOVE ONLY ONE CHILD OF THE NEW POPULATION TO INSERT THE PREVIOUS BEST (EXCLUDING, IF IT EXISTS, THE NEW BEST)
                    #UPDATE THE NEW INVERSE FITNESS AND ASSIGN IT TO THE TOTAL INVERSE FITNESS
                    try:
                        toRemove = choice(list(range(indexBestChildren))+list(range(indexBestChildren+1,sizePopulation)))
                    except NameError:
                        indexBestChildren = 0
                        toRemove = randint(0,sizePopulation)
                                            
                    newInverseFitness = newInverseFitness - population_next[toRemove].fitness
                    newInverseFitness = newInverseFitness + best.fitness
                    totalInverseFitness = newInverseFitness
                    newInverseFitness = 0
                
                    population_next[toRemove] = best
                    population = population_next
                    
                    if(population_next[indexBestChildren].objF<best.objF):
                        best = population_next[indexBestChildren]
                    #####################################################
                #END SCENARIO

                #####################################################
                #UPDATE TOTAL TIME AND WRITE THE RESULT
                totalTime = totalTime + (time.time()-start)
                res.write("UB,"+str(time.time()-start)+"\n")
                #####################################################

            #END SCENARIOS
            #####################################################
            #SAVE FINAL AVG RESULTS
            res.write("AVG TIME,"+str(totalTime/50))


