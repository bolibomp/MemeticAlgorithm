# Memetic Algorithm

This is my Memetic Algorithm. It implements a simple hill climb as its local search and standard crossover and mutation as its "global" search. When the generations are done it's assumed that the soultion found will be in a quasi-/convex area so a local minimizer is applied to find the local optimum. This optimizer is known as Powell's method and is a derivative free method.

For the 2D Ackley function ![](https://user-images.githubusercontent.com/22666203/197384443-3e19d39b-a2c7-438c-8408-a108dd2adfce.svg)
the algorithm finds the optimal solution:

This is the Ackley function
The memetic algorithm gives:
Best Fittness:  0.00237  Analytical Optimal Fitness:  0
Best Solution:  [-0.14224275  0.22810971]  Analytical Optimal Solution:  [0]
Using this as solution as a starting guess for Powell search we get:
Best Fittness:  6.661338147750939e-15  Analytical Optimal Fitness:  0
Best Solution:  [1.22523451e-15 1.81596749e-15]  Analytical Optimal Solution:  [0]

The evolution of the best and mean solution can be seen here
![](https://github.com/bolibomp/MemeticAlgorithm/blob/main/Figure_1.png?raw=true)

The solution and solution space can be seen here
![](https://github.com/bolibomp/MemeticAlgorithm/blob/main/Figure_2.png?raw=true)
