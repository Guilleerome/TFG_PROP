import instances_reader as ir
import constructor as construct
import Solution as sol

plants = ir.read_instances()

for plant in plants:
    solution = construct.construct_random(plant)
    print(solution.cost)
    solution = construct.construct_greedy(plant)
    print(solution.cost)
    solution = construct.construct_greedy_2(plant, True)
    print(solution.cost)
    solution = construct.construct_greedy_2(plant, False)
    print(solution.cost)
    print("")