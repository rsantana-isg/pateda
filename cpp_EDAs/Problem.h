// Problem.h: definition of the functions specific to the problem.
//
// Author:	Ramon Etxeberria
// Date:	1999-09-25
//
// Note:
//		You must implement the following three functions if
//		you want to solve a specific problem.

#ifndef _PROBLEM_
#define _PROBLEM_

#include "Solution.h"
#include "IsingModel.h"
#include "CNF.h" 
#include "ProteinClass.h" 
 


// The evaluation function of the individuals.
double Metric(int * genes);

// It gets the information related to the problem being solved.
// At least, this function must initaliaze the size of the
// individuals and the number of states of their genes.
int GetProblemInfo();

// This functions is used to free all the memory required
// for the problem's information.
void RemoveProblemInfo();

// This function can be used to process any computation relative
// to the recently generated generation. This function is executed every time
// a generation is created. It can be used to record the best 
// individual of each generation.
void AnalyzeGenerationProblem(CSolution solution);

// This function prepares the final solution and writes it
// in the screen. It can also be modified in order to
// write the answer also in a file.
void WriteSolutionProblem(CSolution solution);

// Chain of symbols for prediction 
extern int* chain; 
 
// Length of the chain of symbols. 
extern int chainlength; 


extern int* STATE;
extern int FUNC;

#endif



