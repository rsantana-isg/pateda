#include "InferenceAlgorithm.h"

#ifndef __LOOPY__
#define __LOOPY__

class Loopy : public InferenceAlgorithm {

  /**
     This class makes inference using Loopy Belief Propagation algorithm
   
     Part of the c_inference package
     @version November 2004
     @author Talya Meltzer
  */
  
 public:

  // ctor
  Loopy(MRF const* mrf, SumOrMax m = MAX, Strategy s = SEQUENTIAL, int maxIter = 2000) :
    InferenceAlgorithm(mrf),
    l_strategy(s), l_sumOrMax(m), l_maxIter(maxIter)
    {
          l_messages = 0; l_pairBeliefs = 0; initMessages(); initPairBeliefs(); 
    }

  virtual ~Loopy(); // dtor

  virtual double** inference(int* converged);
  void initMessages();
  void initPairBeliefs();
  double**** calcPairBeliefs();
  
 private:
  
  Strategy l_strategy; // strategy of updating
  double*** l_messages; // the messages from node to node
  double**** l_pairBeliefs; // the pairwise beliefs
  SumOrMax l_sumOrMax; // use sum or max 
  int l_maxIter; // maximum number of iterations in inference

  void freeMessages();
  void freePairBeliefs();
};

#endif
