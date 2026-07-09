#include "InferenceAlgorithm.h"

#ifndef __MEAN_FIELD__
#define __MEAN_FIELD__

class MeanField : public InferenceAlgorithm {

  /**
     This class makes inference using mean-field approximation
   
     Part of the c_inference package
     @version November 2004
     @author Talya Meltzer
  */
  
 public:

  // ctor
  MeanField(MRF const* mrf, int maxIter = 2000) :
    InferenceAlgorithm(mrf), mf_maxIter(maxIter) {}
  
  virtual ~MeanField() {} // dtor

  virtual double** inference(int* converged);
  
 private:
  
  int mf_maxIter; // maximum number of iterations in inference
  
};

#endif
