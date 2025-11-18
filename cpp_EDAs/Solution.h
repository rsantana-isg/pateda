// Solution.h: interface for the CSolution class.
//
// Description:
//      The class which solves the problem. It consists of
//      a population which evolves according to EDA.
//
// Author:  Ramon Etxeberria
// Date:    1999-09-25
//
// Notes: 
//      There will be only one object of this class in the
//      program. All the global variables that define the
//      problem and the optimization parameters must be 
//      initialized before that object is created.

#ifndef _SOLUTION_
#define _SOLUTION_

#include <time.h>
#include "BayesianNetwork.h"
#include "Individual.h"
#include "Population.h"
#include "Timer.h"
#ifdef PARALLEL
    #include "Parallel.h"
#endif

class CSolution 
{
public:

    // It performs a generation of the EDA.
    void Improve();

    // It returns true if the solution has converged, false
    // otherwise.
    bool Last();

    // The constructor. It creates the initial population.
    CSolution();

    // The destructor. It frees all the used memory.
    virtual ~CSolution();

    // It displays the best solution found so far, together
    // with the information of the optimization process. 
    // Usually used when the algorithm has converged.
    friend ostream & operator<<(ostream & os,CSolution & soluzioa);

    // It returns the value of the private attribute m_generation.
    int GetGenerationNumber();

    // It returns the best individual for the actual generation, 
    // as well as its fitness value.
    CIndividual  * & GetBestIndividual();

    // The Bayesian network estimated from the selected
    // individuals.
    CBayesianNetwork m_bayesian_network;

    // It inserts the given individual in the population.
    // The individual is inserted according to its value.
    void AddToPopulation(CIndividual * individual);

    // This function can be used to process any computation relative
    // to the recently generated generation. This function is executed every time
    // a generation is created. It can be used to record the best 
    // individual of each generation.
    void AnalyzeGeneration();

    // This function prepares the final solution and writes it
    // in the screen. It can also be modified in order to
    // write the answer also in a file.
    void WriteSolution();

    // Sets the timer to 0 and start it
    void StartTimer();

    // Stops the timer
    void StopTimer();

#ifdef PARALLEL
    //Generation of the object CParallel to create all the 
    //required structures
    CParallel m_paralelo;
#endif
    
private:

    // The population of individuals. It will be an ordered
    // list. This way, the best individual will be allways
    // at the head of the list.
    CPopulation m_population;

    // It will store the selected individuals.
    int ** m_cases;

    // Generation counter.
    int m_generation;
    int m_generation_init; //only for part-time executions

    // The sum of all individual's values. 
    double m_total;

    // The m_total of the previous generation. It is required
    // for convergence detection.
    double m_old_total;

    // The m_values will store the evaluation function values
    // of all positions to make BSC
    double ** m_values;

    // The m_sel_total stores the sum of all the evaluation
    // function values of selected individuals
    double m_sel_total;

    //This timer is used to calculate the execution time of the whole
    //program.store the time. This is used to 
    //display at the end of the program the time needed by the
    //algorithm to reach the final solution
    CTimer timer;

    // It randomly selects an individual from the population.
    // The selection of the individual is performed according
    // to their range.
    CIndividual * & RangeBasedSelection();
};

#endif 
