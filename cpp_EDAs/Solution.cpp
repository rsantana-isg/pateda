// Solution.cpp: implementation of the CSolution class.
//

#include "EDA.h"
#include "Solution.h"
#include "Problem.h"
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <fstream.h>

CSolution::CSolution()
: m_generation(0),m_generation_init(0),m_total(0),m_old_total(INT_MIN)
{int i;

    // Seed for random numbers.
    srand(time(NULL));

    // Creation of the initial population.
    if (PARTTIME>0) {
      ifstream popinput;
      char TEXT[50];
//      popinput.open(OUTPOPFILE);
//      if (popinput) {
      popinput.open(OUTPOPFILE,ios::out);
      if (!popinput.fail()) {
        cout << "Importing population..." << endl;
        popinput >> TEXT >> m_generation;
        m_generation_init = m_generation;
        popinput >> TEXT >> EVALUATIONS;
        popinput >> TEXT >> POP_SIZE;
        popinput >> TEXT >> IND_SIZE;
        
        for(i=0; i<POP_SIZE; i++) {
          CIndividual * individual = new CIndividual;
          popinput >> individual;
          AddToPopulation(individual);
          m_total += individual->Value();
        }
        popinput.close();
        cout << " done!" << endl;
      }
      else {
/*
        //Parallel Simulation
        m_paralelo.ParallelSimulation(this, POP_SIZE);

        // Calculate the new total.
        m_old_total = 0;
        m_total = 0;
        for(POSITION pos = m_population.GetHeadPosition();
            pos!=NULL;pos = m_population.GetNext(pos))
            m_total += m_population.GetAt(pos)->Value();   
            /
*/
        for(i=0;i<POP_SIZE;i++)
          {
        CIndividual * individual = m_bayesian_network.Simulate();
        AddToPopulation(individual);
        m_total += individual->Value();
          }
      }
    }
    else {
/*
        //Parallel Simulation
        m_paralelo.ParallelSimulation(this, POP_SIZE);

        // Calculate the new total.
        m_old_total = 0;
        m_total = 0;
        for(POSITION pos = m_population.GetHeadPosition();
            pos!=NULL;pos = m_population.GetNext(pos))
            m_total += m_population.GetAt(pos)->Value(); 
*/
      //Initial population created randomly as usual
      for(i=0;i<POP_SIZE;i++)
        {
          CIndividual * individual = m_bayesian_network.Simulate();
          AddToPopulation(individual);
          m_total += individual->Value();
 #ifdef VERBOSE
    cout << "indiv " << i << " egina! " << endl;
 #endif
        }
    }

    // Memory allocation for the structure storing the
    // selected individuals.
    m_cases = new int*[SEL_SIZE];

    // Memory allocation for the structure storing the
    // evaluation function values of selected individuals (when BSC)
    m_values = new double*[IND_SIZE];
    for (i=0;i<IND_SIZE;i++)
    {
        m_values[i] = new double[STATES[i]-1];
        for (int j=0;j<(STATES[i]-1);j++)
            m_values[i][j] = 0;
    }
    m_sel_total = (double)0;

    //starts the timer from 0.
    StartTimer();

    //Display information of the 0th generation
    //AnalyzeGeneration();  //Modification by ROBERTO

}

CSolution::~CSolution()
{int i;

    // Destruction of the population.
    for(i=0;i<POP_SIZE;i++)
    {
        delete m_population.GetTail();
        m_population.RemoveTail();
    }

    // Destruction of the structure storing the selected
    // individuals.
    delete [] m_cases;

    for (i=0;i<IND_SIZE;i++) 
        delete [] m_values[i];
    delete [] m_values;
}

void CSolution::Improve()
{int i,r;

    // Select SEL_SIZE individual from the population.
    // Initialize eval_func structures for BSC
    m_sel_total = 0;
    for (r=0;r<IND_SIZE;r++)
        for (int s=0;s<(STATES[0]-1);s++)
            m_values[r][s] = (double)0;

    if(SELECTION == TRUNCATION)
    {
        POSITION pos = m_population.GetHeadPosition();

        for(i=0;i<SEL_SIZE;i++)
        {
            m_cases[i] = m_population.GetAt(pos)->Genes();
            for (int j=0;j<IND_SIZE;j++)
            {
                if (m_cases[i][j]!=(STATES[j]-1))
                    m_values[j][m_cases[i][j]] += m_population.GetAt(pos)->Value();
            }
            m_sel_total += m_population.GetAt(pos)->Value();
            pos = m_population.GetNext(pos);            
        }

    }
    else if(SELECTION == RANGE_BASED)
    {
        for(i=0;i<SEL_SIZE;i++)
        {
            m_cases[i] = RangeBasedSelection()->Genes();
        }
    }

    // Learn the Bayesian network which best fits the selected
    // individuals.
    m_bayesian_network.Learn(m_cases,m_values,m_sel_total);

    //If the structure is to be saved, do it now
    if (LEDAGRAPHS) {
        char str[20], str1[7];
        strcpy(str, "graph");
        switch (LEARNING) {
            case UMDA : strcpy(str1, "UMDA"); break;
            case EBNA_B : strcpy(str1, "EBNA_B"); break;
            case EBNA_LOCAL : strcpy(str1, "EBNA_L"); break;
            case EBNA_K2 : strcpy(str1, "EBNA_K2"); break;
            case EBNA_PC : strcpy(str1, "EBNA_PC"); break;
            case PBIL : strcpy(str1, "PBIL"); break;
            case BSC : strcpy(str1, "BSC"); break;
            case TREE : strcpy(str1, "TREE"); break;
            case MIMIC : strcpy(str1, "MIMIC"); break;
            default : strcpy(str1, "UNKNOWN"); break;
        }
        strcat(str, str1);
        sprintf(str1, "%03d", GetGenerationNumber());
        strcat(str, str1);
        strcat(str, ".gw");
        m_bayesian_network.SaveBayesianNetworkLEDA(str);
    }

    // Create the new population.
    if(ELITISM)
    {
        for(i=0;i<OFFSPRING_SIZE;i++)
            AddToPopulation(m_bayesian_network.Simulate());

        for(i=0;i<OFFSPRING_SIZE;i++)
        {
            delete m_population.GetTail();
            m_population.RemoveTail();
        }
    }
    else
    {
        for(i=0;i<OFFSPRING_SIZE;i++)
        {
            delete m_population.GetTail();
            m_population.RemoveTail();
        }
/*
        //Parallel Simulation
        m_paralelo.ParallelSimulation(this, OFFSPRING_SIZE);
*/
        
        //Sequential Simulation
        for(i=0;i<OFFSPRING_SIZE;i++) {
            AddToPopulation(m_bayesian_network.Simulate());
 #ifdef VERBOSE
    cout << "indiv " << i << " egina! " << endl;
 #endif
        }
    }

    // Calculate the new total.
    m_old_total = m_total;
    m_total = 0;
    for(POSITION pos = m_population.GetHeadPosition();
        pos!=NULL;pos = m_population.GetNext(pos))
        m_total += m_population.GetAt(pos)->Value();

    // Update the generation counter.
    m_generation++;

    // Analyze the generation
    // AnalyzeGeneration(); //QUITADO EL ANALYZE GENERATION ROBERTO

}

bool CSolution::Last()
{
  if ((ENDING_CRITERION)
     ||((MAX_GENERATIONS>0)&&(m_generation==MAX_GENERATIONS))
     ||((PARTTIME>0)&&(m_generation-m_generation_init==PARTTIME)))
       return true;

  else return m_total==m_old_total;
}

ostream & operator<<(ostream & os,CSolution & solution)
{

    os 
        << "Population size: " << POP_SIZE << endl
        << "Selected individuals: " << SEL_SIZE << endl
        << "Selection type: " << SELECTION << endl
        << "Offspring size: " << OFFSPRING_SIZE << endl
        << "Elitism: " << ELITISM << endl
        << "Learning: " << LEARNING << endl
        << "Caching: " << CACHING << endl
        << "Simulation: " << SIMULATION << endl
        << "Generation: " << solution.m_generation << endl
        << "Evaluations: " << EVALUATIONS << endl
        << "Total time: " << solution.timer.TimeString() << endl

        << "Execution time: " << solution.timer.ExecutionTimeString() << endl

        << "Best individual: " << solution.m_population.GetHead();

    return os;
}

CIndividual * & CSolution::RangeBasedSelection()
{
    double bound = (1+POP_SIZE)*POP_SIZE/2;
    double which = rand()*bound/RAND_MAX;

    POSITION pos = m_population.GetHeadPosition();
    for(int i=POP_SIZE;which>i;i--) 
    {
        pos = m_population.GetNext(pos);
        which -= i;
    }

    return m_population.GetAt(pos);
}

void CSolution::AddToPopulation(CIndividual * individual)
{
  m_population.AddToPopulation(individual);
  /*    POSITION pos;

    for(pos=m_population.GetHeadPosition();
        pos!=NULL && individual->Value()<m_population.GetAt(pos)->Value();
        pos = m_population.GetNext(pos));

    if(pos==NULL) m_population.AddTail(individual);
    else m_population.InsertBefore(pos,individual);
  */}

void CSolution::StartTimer()
{
    //set the time at the beginning
    timer.Reset();
}



void CSolution::StopTimer()
{
    //Stop the timer.
    timer.End();
}


int CSolution::GetGenerationNumber()
{
    return m_generation;
}


CIndividual  * & CSolution::GetBestIndividual()
{
    return m_population.GetHead();
}


void CSolution::AnalyzeGeneration()
{
//  AnalyzeGenerationProblem(*this);
      if (GetGenerationNumber() % GENERATIONSPRINTRESULT == 0) {

    cout << "Generation "<< GetGenerationNumber() << ": "
                     << GetBestIndividual() << endl;
    if (foutput) 
       foutput << "Generation "<< GetGenerationNumber() << ": "
                   << GetBestIndividual() << endl;
      }
    POPCORRECT=0;
    POPMISS1=0;
    POPMISS2=0;
    POPMISS3=0;
    POPMISSMORE=0;
}

void CSolution::WriteSolution()
{
//  WriteSolutionProblem(*this);

    // Stop the timer
    StopTimer();

    //Write the information of the solution in the screen
    cout << *this << endl;

    //Write the information of the solution in the file (if exists)
    if (foutput) {
        if (GetGenerationNumber() % GENERATIONSPRINTRESULT != 0) {
            foutput << "Generation "<< GetGenerationNumber() << ": "
                    << GetBestIndividual() << endl;
            cout << "Generation "<< GetGenerationNumber() << ": "
                    << GetBestIndividual() << endl;
        }
        foutput << "== final RESULTS =="<< endl
                << *this << endl;
        foutput.close();
    }

    //if this is a part-time execution, write a file with the final partial results
        if ((PARTTIME>0)&&(MAX_GENERATIONS>GetGenerationNumber())) {
      if (OUTPOPFILE) {
        cout << "Exporting population..." << endl;
        ofstream popoutput;
        popoutput.open(OUTPOPFILE);
        if (!popoutput) {
          cerr << "Could not create file " << OUTPOPFILE << ". Ignoring this file." << endl;
          OUTPOPFILE[0]='\0';
        }
        else {
          popoutput << " generation: "<< GetGenerationNumber() << endl
            << " evaluations: "<< EVALUATIONS << endl
            << m_population << endl;

        }
        popoutput.close();
        cout << "done!" << endl;
      }
    }

}
