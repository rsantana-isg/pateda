#ifndef _EDA_
#define _EDA_

#include <fstream>

// The size of the population.
extern int POP_SIZE;

// The number of selected individuals.
extern int SEL_SIZE;

// The selection type.
extern int SELECTION;

#define RANGE_BASED 0
#define TRUNCATION 1


// The offspring size.
extern int OFFSPRING_SIZE;

// Whether elitism is used or not.
extern int ELITISM;

// The learning type (i.e. the EDA type).
extern int LEARNING;

#define UMDA 0
#define EBNA_B 1
#define EBNA_LOCAL 2
#define PBIL 3
#define BSC 4
#define TREE 5
#define MIMIC 6
#define EBNA_K2 7
#define EBNA_PC 8
//Needed when PBIL or EBNAPC are executed

extern double ALPHA_PBIL;
extern double ALPHA_EBNAPC;

// Scores for the EBNA-local learning types
extern int SCORE;

#define BIC_SCORE 0
#define K2_SCORE 1

// The simulation type (i.e. PLS is the simplest).
extern int SIMULATION;
#define PLS 0
#define PLS_ALL_VALUES_1 1
#define PLS_ALL_VALUES_2 2
#define PLS_CORRECT		 3
#define PENALIZATION	 4


// The individual size.
extern int IND_SIZE;

// The number of states the genes can take.
extern int * STATES;

// Whether caching of individual's values is used or not.
extern int CACHING;

// Number of evaluations performed.
extern int EVALUATIONS;

// Maximum number of generations (used by the stopping criterium).
// When MAX_GENERATIONS=0 -> no limit
extern int MAX_GENERATIONS;

// Name of the file where the output will be stored (optional).
extern char OUTPUTFILE[50];
extern std::ofstream foutput;

// Variables to control the generation's correctness
extern int	POPCORRECT;
extern int	POPMISS1;
extern int	POPMISS2;
extern int	POPMISS3;
extern int	POPMISSMORE;

// Whether the structures have to be saved or not (in LEDA graph format)
extern int LEDAGRAPHS;

// Global variable to be used as an ending criterion
extern bool ENDING_CRITERION;

// Whether the execution is only done in part-time
extern int PARTTIME;
extern char OUTPOPFILE[];

//The value of the optimum when known (Roberto 15/4/2003)
extern double OPTVAL; 

//The function to be optimized (Roberto 15/4/2003)
extern int FUNC;


//Every how many generations do we have to print the information
#define GENERATIONSPRINTRESULT 1


bool ReadParametersEBNA(int, double, double, int , int, int,  int, int, int, double, int );
int RunEBNA(double*, int*);

#endif




