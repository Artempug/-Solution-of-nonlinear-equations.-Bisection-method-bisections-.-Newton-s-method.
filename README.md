# -Solution-of-nonlinear-equations.-Bisection-method-bisections-.-Newton-s-method.
This code is designed to find roots of functions and visualize the results using both the Bisection and Newton-Raphson methods. It is divided into several tasks:

Bisection Method: This method iteratively divides an interval and selects subintervals where the function changes sign to approximate the root. The result is displayed along with a simple graph of the function.
Newton-Raphson Method: This method uses the function's derivative to iteratively improve the estimate of the root. It involves functions for calculating the root and plotting the function's graph.
Bisection with Table: Extends the bisection method to output intermediate values in a formatted table for easier understanding of the iteration process.
Error Analysis in Bisection: Visualizes the error reduction over iterations by plotting the logarithm of the interval size versus iteration number.
Newton's Method for Multiple Roots: Applies Newton's method to find and refine multiple roots of the function.
Newton's Method with Iteration Details: Enhances the Newton's method function to print detailed iteration information, showing the root and error at each step.
Root Finding with Iteration Limit: Uses the Newton's method to find a root with a set maximum number of iterations and a specified tolerance.
Error Analysis in Newton's Method: Plots the error convergence in Newton's method by displaying the logarithm of the difference between consecutive iterations.
The code employs the matplotlib library for graph plotting and PrettyTable for tabular data presentation, making it a versatile tool for both numerical analysis and result visualization.
