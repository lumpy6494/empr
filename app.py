import random
import math
import matplotlib.pyplot as plt
import numpy

NUMBER_GENERATION = 25


class GeneticAlgorithm:

    def __init__(self):
        self.best_Y = []
        self.best_X = []
        self.population_size = 150
        self.chromosome_length = 15
        self.probability = 0.5
        self.probability_mutation =0.001
        self.population = []


    # Функция приспособленности
    def fitness(self, x):
        return math.cos(3*x-15)*x

    # Создание начальной популяции
    def star_population(self):
        for i in range(self.population_size):
            chromosome = []
            for j in range(self.chromosome_length):
                chromosome.append(random.randint(0,1))
            self.population.append(chromosome)


    # Расчет значения функции приспособленности для каждого кандидата в популяции
    def evolution(self):
        result = []
        for chromosome in self.population:
            dec = self.decode(chromosome)
            result.append(self.fitness(dec))
        return result

    # Декодирование хромосомы в конкретное значение
    def decode(self, chromosome:list):
        decimal = 0
        for count, value in enumerate(chromosome):
            decimal += value * pow(2,count)
        x = -9.6 + decimal * (18.7/(pow(2,15)-1))
        return x

    def crossover(self,parent_one, parent_two):
        if random.random() < self.probability:
            point = random.randint(1, len(parent_one) - 1)
            child1 = parent_one[:point] + parent_two[point:]
            child2 = parent_one[:point] + parent_two[point:]
            return child1, child2
        else:
            return parent_one, parent_two


    #Оператор репродукции (Колесо рулетки)
    def roll_select(self):
        # calculate percentage
        winner_index = 0
        fitness_min = min(self.evolution())  # находим минимальное значение так как могут быть отрицательные f(x)
        offset_fitness_values = [x + abs(fitness_min * 2) for x in
                                 self.evolution()]  # к каждому значению добавляю минимальное значение умноженное на 2
        sum_fitness_offset = sum(offset_fitness_values)  # нахожу сумму значений
        p = [offset_fitness_values[i] / sum_fitness_offset for i in
             range(len(self.population))]  # делю каждое значение на сумму
        p_sum = 0
        m = random.triangular(0.3, 1)
        # Roulette
        for j in range(len(self.population)):
            p_sum += p[j]
            if p_sum >= m:
                winner_index = j
                break
        return self.population[winner_index]

    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < self.probability_mutation:
                chromosome[i] = 1 - chromosome[i]
        return chromosome


    def show_plot(self):
        max_value = max(self.best_Y)
        max_index = self.best_Y.index(max_value)
        print(f"Максимальное итоговое значение x = {self.best_X[max_index]}")
        print(f"Максимальное итоговое значение f(x) = {self.best_Y[max_index]}")
        X = numpy.arange(-9.6, 9.1, 0.01)
        X = numpy.delete(X, X.size - 1)
        Y = [round(self.fitness(x), 3) for x in X]
        plt.title("cos(3x-15)*x, [-9.6,9.1]")
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
        plt.plot(X, Y, "r")
        plt.plot(self.best_X, self.best_Y, "mo", markersize=3)
        # plt.plot(2.884, 0.984, "ks", markersize=3)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


    def setup(self):
        self.star_population()
        for i in range(NUMBER_GENERATION):
            parent = [self.roll_select() for i in range(self.population_size)]
            offspring = []
            def iterate():
                if (len(parent) > 0):
                    parent1 = random.choice(parent)
                    parent.remove(parent1)
                    parent2 = random.choice(parent)
                    parent.remove(parent2)
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    offspring.append(child1)
                    offspring.append(child2)
                    return iterate()
            iterate()
            print(f"Поколение {i + 1}:")
            best_fitness = max(self.evolution())
            best_chomosome = self.population[self.evolution().index(best_fitness)]
            best_x = self.decode(best_chomosome)
            print(f"Максимальное решение: x = {round(best_x, 3)}, f(x) = {round(best_fitness, 3)}")
            self.best_X.append(round(best_x, 3))
            self.best_Y.append(round(best_fitness, 3))
            print("Хромосомы:")
            for chromosome in self.population:
                 x = self.decode(chromosome)
                 fit = self.fitness(x)
                 print(f"{chromosome[::-1]} -> x = {round(x, 3)}, f(x) = {round(fit, 3)}")
            print("=" * 20)
            self.population = offspring
        self.show_plot()



if __name__=="__main__":
   algrm = GeneticAlgorithm()
   algrm.setup()






