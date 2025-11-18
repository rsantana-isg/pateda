/*  ===============================================================================
 *  Program:        EDA (Estimation Distribution Algorithms)
 *  Module:         EDA.cpp : Defines the entry point for the console application.
 *  Date:           26-Nov-1999
 *  Author:         Ramon Etxeberria
 *  Description:    
 *      Calculates the optimum solution of a problem following an EDA approach. 
 *      All the problem specification is defined in Problem.cpp. The rest of the
 *        files should remain unaltered.
 *      Output is displayed only in the screen. The function WriteSolution can 
 *        be modified in order to generate the desired aoutput format.
 *
 *  History: 
 *      27-Sep-1999 EDA is created and compiled by Ramon Etxeberria,
 *                  as part of his PhD.
 *
 *      18-Oct-1999 PBIL algorithm added to the program,
 *                  changes made by Iñaki Inza.
 *
 *      24-Oct-1999 BSC algorithm added to the program,
 *                  changes made by Iñaki Inza.
 *
 *      26-Dec-1999 - Information about the time required to obtain the solution
 *                    is added to the output (module CTimer),
 *                  - Mofification in the simulation step: the program accepts
 *                    a new parameter to specify which simulation type has to be used
 *                    (default is PLS).
 *                  changes made by Endika Bengoetxea.
 *
 *      27-Dec-1999 - Chow & Liu algorithm to learn TREE structures added to the program, 
 *                    chamges made by Inaki Inza. 
 * 
 *      30-Dec-1999 - A new stopping criterium is added, so that a maximum of 100 
 *                    generations are allowed. This is controlled by the global
 *                    variable MAX_GENERATIONS in EDA.cpp 
 *                  - Minor changes in the general structure of the program, so that
 *                    the programmer can have a higher flexibility from the file
 *                    Program.cpp. Two new functions added, one that executes every
 *                    generation, and another after the solution is found, to prepare 
 *                    result and write it.
 *                    changes made by Endika Bengoetxea.
 *
 *      30-Oct-2000 - EBNA K2 + penalization and EBNA PC algorithms added, 
 *                    as well as the K2 score.
 *                    changes made by iñaki Inza and Endika Bengoetxea
 *
 *      09-Nov-2000 - Changed the way to read the parameters for the program.
 *                    changes made by Iñaki Inza and Endika Bengoetxea
 *
 *      03-Jan-2001 - Added an option to store the structures for each 
 *                    generation in the form of LEDA graphs, able to be read with
 *                    the LEDA package.
 *                    changes made by Endika Bengoetxea.
 *
 *      28-Dec-2001 - This new version has two main improvements:
 *                    * A new module to allow Parallel execution of EBNA. 
 *                      It requires all the files to be compiled with the -D PARALLEL option.
 *                      A file Makefile_Parallel is provided for this purpose.
 *                      NOTE: this only works in UNIX systems.
 *                    * The computation of the BIC and K2 scores have been improved in time,
 *                      even for the sequential case.
 *                    changes made by Endika Bengoetxea.
 *
 *      03-Jan-2002 - Added a new parameter to set the max. number of Generations.
 *                    changes made by Endika Bengoetxea
 *
 *      25-Feb-2003 - Added a new parameter to set the ALPHA for PBIL and EBNA PC.
 *                    changes made by Endika Bengoetxea
 *
 *      02-Mar-2003 - A new limitation added to limit the number of parents that a variable can have.
 *                    This aims at limiting the amount of memory requested by the program.
 *                    changes made by Endika Bengoetxea
 *
 *  =============================================================================== 
 */ 


#include "EDA.h"
#include "Solution.h"
#include "Problem.h"
#include <stdlib.h>

//#ifdef WIN32
   #include "Getopt.h"
   #include "string.h"
//#endif


int POP_SIZE=2000;
int SEL_SIZE=1000;
int SELECTION=1;
int OFFSPRING_SIZE=1999;
int ELITISM=1;
int LEARNING=1;
int CACHING=1;
int IND_SIZE;
int * STATES;
int EVALUATIONS = 0;
int SIMULATION=PLS;
int SCORE=BIC_SCORE;
int MAX_GENERATIONS=100;
int LEDAGRAPHS=0; 
int PARTTIME=0;
char OUTPOPFILE[] = "pop.out";

char OUTPUTFILE[50];
ofstream foutput;
bool ENDING_CRITERION=false;

//Needed when PBIL or EBNAPC are executed
double ALPHA_PBIL=0.5;
double ALPHA_EBNAPC=0.05;


int	POPCORRECT=0;
int	POPMISS1=0;
int	POPMISS2=0;
int	POPMISS3=0;
int	POPMISSMORE=0;


double OPTVAL;
int FUNC;



void usage(char *progname)
{
   cerr
        << "Usage: " << progname << " -N pop_size -S sel_size -f offspring_size -e elitism -s selection -a caching" << endl
        << "        -l learning_type -c score -A alpha_learning -i simulation_type -G max_generations -P part_time" << endl
        << "        -g save_structures -o output_file" << endl
        << endl
        << "-N pop_size: number of individuals in the population. (def:2000)" << endl
        << "-S sel_size: number of selected individuals. (def:1000)" << endl
        << "-f offspring_size: how many individuals are created in" << endl
        << "       each generation. (def:1999)" << endl
        << "-e elitism: how the next population is created: " << endl
        << "       elitism (1) or no-elitism (0). (def:1)" << endl
        << "-s selection: how the individuals are selected:" << endl
        << "       range based (0) or truncation (1). (def:1)" << endl
        << "-a caching: whether the values of the evaluated individuals.  (def:1)" << endl	      
        << "-l learning: how the Bayesian network is learned:" << endl
        << "       UMDA (0), EBNA with B algorithm (1), " << endl
        << "       EBNA with local search (2), PBIL with alpha = 0.5 (3)," << endl
        << "       BSC (4), TREE(5), MIMIC(6), EBNA K2 + penalization (7)," << endl
        << "       or EBNA PC (8)." << endl
        << "-c score: type of score (only for EBNA): " << endl
        << "       BIC (0) or K2 (1). (def:0)" << endl
        << "-A alpha_learning: alpha value for PBIL or EBNA-PC" << endl
        << "       (def: ALPHA_PBIL: 0.5, ALPHA_EBNA-PC: 0.05)" << endl
        << "-i simulation: type of simulation: PLS (0), " << endl
        << "       PLS forcing all values LTM (1) and ATM (2)," << endl
        << "       correction after generation (3) and penalization (4).(def:0)" << endl	
        << "-P part_time: work in parts, execute only until generation 'part_time'  " << endl	      
        << "       and store partial population in file 'pop.out'. (def:0,no output)" << endl
        << "-G generations: max. munber of generations before stopping" << endl
        << "       the program. (def:100)" << endl
        << "-g save_structure: whether the structures of each generation " << endl	      
        << "       have to be saved or not.  (def:0)" << endl
        << "-o output_file: name of output_file to store the results" << endl
        << "       (optional parameter). (def:-)" << endl;
}

/*
bool GetParameters(int argc,char * argv[])
{
 char c;

     if(argc==1) {
             usage(argv[0]);
             return false;
     }

    char** optarg;
    optarg = new char*[argc];
    while ((c = GetOption (argc, argv, "hN:S:f:e:s:a:l:c:A:i:P:G:g:o:",optarg)) != '\0') {
		 
        switch (c) {
             case 'h' :
                          usage(argv[0]);
                          return false;
                          break;
             case 'N' :
                          POP_SIZE = atoi(*optarg);
                          break;
             case 'S' :
                          SEL_SIZE = atoi(*optarg);
                          break;
             case 'f' :
                          OFFSPRING_SIZE = atoi(*optarg);
                          break; 
             case 'e' : 
                          ELITISM = atoi(*optarg);
                          break; 
             case 's' :
                          SELECTION = atoi(*optarg);
                          break;
             case 'a' : 
                          CACHING= atoi(*optarg);
                          break; 
             case 'l' :
                          LEARNING = atoi(*optarg);
                          break;
             case 'c' :
                          SCORE = atoi(*optarg);
                          break;
             case 'A' :
                          ALPHA_PBIL = atof(*optarg);
                          ALPHA_EBNAPC = atof(*optarg);
                          break;
             case 'i' : 
                          SIMULATION = atoi(*optarg);
                          break; 
             case 'P' : 
                          PARTTIME= atoi(*optarg);
                          //If the file 'pop.out exists, this is a next part-time execution
                          //This is controlled in CSolution.
                          break; 
             case 'G' : 
                          MAX_GENERATIONS= atoi(*optarg);
                          break; 
             case 'g' : 
                          LEDAGRAPHS= atoi(*optarg);
                          break; 
             case 'o' : 
                          strcpy(OUTPUTFILE, *optarg);
                          //OUTPUTFILE=optarg;
                          //open the output file
                          if (OUTPUTFILE) {
                             foutput.open(OUTPUTFILE);
                             if (!foutput) {
                                cerr << "Could not open file " << OUTPUTFILE << ". Ignoring this file." << endl;
                                OUTPUTFILE[0]='\0';
                             }
                          }
                          break; 
	     }
     }

     delete [] optarg;

	if(!ELITISM && OFFSPRING_SIZE>POP_SIZE)
	{
		cerr
			<< "Warning: If no elitism is used when creating the new population" << endl
			<< "         offspring_size cannot be higher than pop_size. " << endl
			<< "         offspring_size truncated to " << POP_SIZE << "." << endl
			<< endl;

		OFFSPRING_SIZE = POP_SIZE;
	}

	if(SELECTION==TRUNCATION && SEL_SIZE>POP_SIZE)
	{
		cerr
			<< "Oharra: If truncation selection is used, sel_size cannot be" << endl
			<< "        higher than pop_size. sel_size truncated to" << POP_SIZE << "." << endl
			<< endl;

		SEL_SIZE = POP_SIZE;
	}

	return true;
}

*/

bool ReadParametersEBNA(int vars, double Max, double Trunc, int psize, int func, int Elit, int Maxgen, int LEARNEBNA, int EBNASCORE, double EBNA_ALPHA, int EBNA_SIMUL)
{

LEDAGRAPHS=0; 
PARTTIME=0;
ENDING_CRITERION=false;
ALPHA_EBNAPC=0.05;
POPCORRECT=0;
POPMISS1=0;
POPMISS2=0;
POPMISS3=0;
POPMISSMORE=0;
EVALUATIONS = 0;

    IND_SIZE = vars;
    OPTVAL = Max;
    SEL_SIZE = int(Trunc*psize);
    POP_SIZE = psize;
    FUNC = func;
    ELITISM = Elit;
    MAX_GENERATIONS= Maxgen;
    LEARNING = LEARNEBNA;
    SCORE = EBNASCORE; 
    ALPHA_PBIL = EBNA_ALPHA;               
    ALPHA_EBNAPC = EBNA_ALPHA; 
    SIMULATION  = EBNA_SIMUL;
    OFFSPRING_SIZE = psize - ELITISM;
    SELECTION = 1;
    CACHING =0;

    return true;
}

int RunEBNA(double* bestval, int *succ)
{
if (GetProblemInfo() < 0) return -2;

	CSolution solution;


	while(!solution.Last() && solution.GetBestIndividual()->Value() != OPTVAL) 
	{
	 solution.Improve();
	 // solution.WriteSolution();
        }
	if(solution.GetBestIndividual()->Value() == OPTVAL) *succ = solution.GetGenerationNumber();
        *bestval = solution.GetBestIndividual()->Value();  
 
	solution.WriteSolution();
	
	RemoveProblemInfo();
	
	return 0;
}



/*
int main(int argc, char* argv[])
{
	if(!GetParameters(argc,argv)) return -1;

	if (GetProblemInfo() < 0) return -2;

	CSolution solution;
	
	while(!solution.Last()) 
		
		solution.Improve();
	
	solution.WriteSolution();
	
	RemoveProblemInfo();
	
	return 0;
}

*/



