#ifndef __PROTEIN 
#define __PROTEIN
 
#include <stdio.h> 
#include <stdlib.h>
#include <iostream> 
#include <fstream>
#include "auxfunc.h"  

class HPProtein { 
public: 
  
  
   int sizeProtein;
   int* IntConf;
   int** Pos;
   int** NewPos;
   double* gains;
   int* svals;
   int* bestallpos;
   unsigned int* newvector;
   unsigned int* auxnewvector;
   int* statvector;
   int* Memory;
   unsigned int** moves; 
   int nmoves;
   int combmoves;
   unsigned int grid[200][200];
   double moveprob[10]; //Probabilities associated to movements for GenPop
   int theone;    // Unique movement when applied
   double alpha;  //Coefficient for determining weight of local search in GenPop
   double* contact_weights; // Assign a weight to every possible contact of the protein
   int TotContacts; //Bound for the maximum of contacts in the protein

   //Auxiliary arrays
   //int totnumboptint;
   // int optmatrix[23][23];
   // double energymatrix[10][10];
   //int freqmatrix[10][10];    
 
    HPProtein(int,int*,int,int);
    HPProtein(int,int*);

virtual   ~HPProtein();
virtual void  InitClass(int,int*,int,int);
virtual   void   SetInitPos(int**);                      // Init the first positions of the grid
virtual   void   CreatePos();                            // Creates the first positions of the grid
virtual   void   DeletePos();                       // Deletes the grid
virtual   void   FindPos(int, unsigned int*);               // Translates the vector to the  positions in the grid     
virtual   void   TranslatePos(int, unsigned int*);         // Translate an array of grid positions to a vector of movements
virtual   int    CheckValidity(int, int**);              // Checks if the chain is connected at position at_i_Pos
virtual   void   AssignPos(int,int**,int**);             // Copy the position of one grid into another
virtual   int    PullMoves(int,int,int);     // Given a chain of molecules, and a position makes a pull move
virtual   double EvalChain(int,int**);         // Evaluates the folding of a given protein using the score
virtual   double EvalChainWithCO(int,int**);     
virtual   double EvalChainModel(int,int**);         // Evaluates the folding of a given protein using the score
virtual   double EvalChainLong(int,int**);         // Evaluates the folding of a given protein using the score
virtual   double EvalChain(int,unsigned int*,int**);                                         // Evaluates the chain but using the grid positions and not the vector
virtual   double EvalOnlyVector(int,unsigned int*);                // Evaluates the chain receiving only the  vector
virtual   double EvalWithCO(int,unsigned int*); 
virtual   double ContactOrderVector(int,unsigned int*);   
virtual   double ContactOrderChain(int, int**);   
virtual   double EvalOnlyVectorLong(int,unsigned int*);                // Evaluates the chain receiving only the  vector (long interaction between residues are priviledge
virtual   double EvalOnlyVectorModel(int,unsigned int*);                // Evaluates the chain receiving only the  vector
          int    CheckFree(int,int**,int,int);                                      // Checks if coordinate at_i_Pos is occuppied in the grid
   double ProteinLocalOptimizer(int,unsigned int*,double,int*,int);                  // Given a chain of molecules, and a position makes a pull move 
virtual   double TabuOptimizer(int,unsigned int*,double,int*,int,int);                  // Given a chain of molecules, and a position makes a pull move 
   double ProteinLocalOptimizerSimple(int,unsigned int*,double,int*);                  // Given a chain of molecules, and a position makes a mutation
   void   SetGainsAndVals(int,double,double,double);                          // Sets the perturbation best value
virtual   int    TabuSetNewBestPos(int,int*,double*,int,int);                                   // Modifies the best current grid in local optimizer
   int    SetNewBestPos(int,int*,double*,int);                                   // Modifies the best current grid in local optimizer
   int    SetNewBestPosSimple(int,int*,double*,unsigned int*);                                   // Modifies the best current grid in local optimizer
virtual   double TabuFindNewEval(int,int,int,int*);                                     // Evaluates a vector perturbed at a given position
   double FindNewEval(int,int,int,int*);                                     // Evaluates a vector perturbed at a given position
   double FindNewEvalSimple(int,int,int,int*,unsigned int*);                                     // Evaluates a vector perturbed at a given position
virtual   void   SetInitMoves();                                                    // Init all possible ordering of legal move   void   SetInitPosAt_i(int**,int);
virtual   void   SetInitPosAt_i(int**,int);
virtual   int    Feasible(unsigned int*,int);                                            // Checks if there are not overlappings in the grid
virtual   void   PutMoveAtPos(int,int);                                             // Sets a given move in the grid
virtual   int    Repair(int,int,unsigned int*,unsigned int*);                                     // Recursive function for repairing a protein
virtual   int    PartialRepair(int,int,unsigned int*,unsigned int*);                                     // Recursive function for repairing a protein
virtual   void   CallRepair(unsigned int*,int);                                        // Calls to the recursive Repair procedure
virtual   int    DownFeasible(int,int, int);                                            // Checks if there are not overlappings in the grid
virtual   void   DownPutMoveAtPos(int,int);                                             // Sets a given move in the grid
virtual   int    DownRepair(int,int,unsigned int*,unsigned int*);                                     // Recursive function for repairing a protein
virtual   void   DownCallRepair(int,unsigned int*,int);                                        // Calls to the recursive Repair procedure
virtual   void   DownCallRepair(int,int*,int);                                        // Calls to the recursive Repair procedure
virtual    void   CallRepair(int*,int);                                                 // Calls to the recursive Repair procedure
virtual   int    BackTracking(int,int,unsigned int*);                               // Recursive function for creating protein by backtracking 
virtual   void   CallBackTracking(unsigned int*);                                       // Calls to the recursive BackTracking procedure
virtual   void   CallBackTracking(int*);                                       // Calls to the recursive BackTracking procedure
virtual   void   PrintPos(int,int**);                                                     // Prints all or part of the protein's grid
virtual   double   FindIsLegal(int,int, int,int*,double);
   double ProteinLocalPerturbation(int,unsigned int*,double,int*,int);       
   void CleanGridFromPos(int);
void CalculatePos(int,int,int*,int*);  //Sets a given move in the grid
 void SetAlpha(double);
 int  SetPosInGrid(int,int);
int FindGridProbabilities(int);
 void init_contact_weights(double);
 void delete_contact_weights();
void create_contact_weights(); 
void update_contact_weights(double, unsigned int* ); 
double  EvalVectorWithWeights(int,unsigned int*,double*); 
 double  EvalChainWithWeights(int, int**,double*);
};



class HPProtein3D:public HPProtein{ 
public: 

    HPProtein3D(int,int*);
virtual   ~HPProtein3D();

  
virtual   void   SetInitPos(int**);                      // Init the first positions of the grid
virtual   void   CreatePos();                            // Creates the first positions of the grid
virtual   void   FindPos(int, unsigned int*);               // Translates the vector to the  positions in the grid     
virtual   void   TranslatePos(int, unsigned int*);         // Translate an array of grid positions to a vector of movements
virtual   int    CheckValidity(int, int**);              // Checks if the chain is connected at position at_i_Pos
virtual   void   AssignPos(int,int**,int**);             // Copy the position of one grid into another
virtual   int    PullMoves(int,int,int);     // Given a chain of molecules, and a position makes a pull move
virtual   double EvalChain(int,int**);         // Evaluates the folding of a given protein using the score
virtual   double EvalChainModel(int,int**);         // Evaluates the folding of a given protein using the score
virtual   double EvalChainLong(int,int**);         // Evaluates the folding of a given protein using the score
virtual   double EvalChain(int,unsigned int*,int**);                                         // Evaluates the chain but using the grid positions and not the vector

  
virtual   int    CheckFree(int,int**,int,int,int);                                      // Checks if coordinate at_i_Pos is occuppied in the grid
virtual   double TabuOptimizer(int,unsigned int*,double,int*,int,int);                  // Given a chain of molecules, and a position makes a pull move 
virtual   int    TabuSetNewBestPos(int,int*,double*,int,int);                                   // Modifies the best current grid in local optimizer
virtual   double TabuFindNewEval(int,int,int,int*);                                     // Evaluates a vector perturbed at a given position
virtual   void   SetInitMoves();                                                    // Init all possible ordering of legal move   void   SetInitPosAt_i(int**,int);
virtual   void   SetInitPosAt_i(int**,int);
virtual   int    Feasible(unsigned int*,int);                                            // Checks if there are not overlappings in the grid
virtual   void   PutMoveAtPos(int,int);                                             // Sets a given 
virtual   void   PrintPos(int,int**);                                                     // Prints all or part of the protein's grid
 int SetPosInGrid(int,int);
};



class HPProtein3Diamond:public HPProtein{ 
public: 

           HPProtein3Diamond(int,int*);
virtual   ~HPProtein3Diamond();

  
virtual   void   SetInitPos(int**);                      // Init the first positions of the grid
virtual   void   CreatePos();                            // Creates the first positions of the grid
virtual   void   FindPos(int, unsigned int*);               // Translates the vector to the  positions in the grid     
virtual   void   TranslatePos(int, unsigned int*);         // Translate an array of grid positions to a vector of movements
virtual   int    CheckValidity(int, int**);              // Checks if the chain is connected at position at_i_Pos
virtual   int    CheckFree(int,int**,int,int,int);                                      // Checks if coordinate at_i_Pos is occuppied in the grid
virtual   void   AssignPos(int,int**,int**);             // Copy the position of one grid into another
virtual   double EvalChain(int,int**);         // Evaluates the folding of a given protein using the score
virtual   double EvalChainModel(int,int**);         // Evaluates the folding of a given protein using the score
virtual   double EvalChain(int,unsigned int*,int**);                                         // Evaluates the chain but using the grid positions and not the vector
virtual   void   SetInitPosAt_i(int**,int);
virtual   int    Feasible(unsigned int*,int);                                            // Checks if there are not overlappings in the grid
virtual   void   PrintPos(int,int**);                                                     // Prints all or part of the protein's grid
};
#endif
