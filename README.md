# globalopt-project

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)

## General info
Project for the Global Optimization course at Sapienza University of Rome in 2015. 
The project involves the multiobjective optimization for the optimal design of a disk brake based on this [paper](https://pdfs.semanticscholar.org/fe65/868d0aa4c2bf714410e7f093dce40d6aaa1b.pdf). 
The objective was to find a strategy to improve the performance of the NSGA-2 algorithm. 
This has been achieved by ibridizing the NSGA-2 with a global mixed integer optimization solver (MIDACO) once the multiobjective problem has been transformed as single objective one with the epsilon-constraint method.
	
## Technologies
* Python version: 2.7
* Pygmo 
* MIDACO
