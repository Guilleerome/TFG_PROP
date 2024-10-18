import instances_reader as ir
import constructor as construct
import Solution as sol

plants = ir.read_instances()

for plant in plants:
    solution = construct.construct_greedy(plant)

    cost = solution.cost
    print(cost)
