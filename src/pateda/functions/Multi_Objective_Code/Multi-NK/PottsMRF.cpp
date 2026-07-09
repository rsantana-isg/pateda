#include "PottsMRF.h"
#include <iostream> 
#include <fstream>
#include <math.h>
//#include "mex.h"

PottsMRF::~PottsMRF() {
  for (int i=0; i<N; i++) {
    delete[] lambdaValues[i];
  }
  delete[] lambdaValues;
  lambdaValues = 0;
}

double PottsMRF::pairPotential (int i, int n, int xi, int xj) const {

  double lambda_ij = lambdaValues[i][n];
  if (xi==xj) {
    return exp(lambda_ij / T);
  }
  else {
    return 1.0;
  }
}

void PottsMRF::initLambdaValues() {
  lambdaValues = new double*[N];
  for (int i=0; i<N; i++) {
    lambdaValues[i] = new double[neighbNum(i)];
    for (int n=0; n<neighbNum(i); n++) {
      lambdaValues[i][n] = 0.0;
    }
  }
}

double PottsMRF::getEnergy(int const* assignment) const {
  double energy = 0.0;
  for (int i=0; i<N; i++) {
    int xi = assignment[i];
    energy -= (log(localMat[i][xi]) * T);
    for (int n=0; n<neighbNum(i); n++) {
      int j = adjMat[i][n];
      if (i<j) {
	int xj = assignment[j];
	if (xi == xj) {
	  double lambda_ij = lambdaValues[i][n];
	  energy -= lambda_ij;
	}
      }
    }
  }
  return energy;
}

void PottsMRF::setTemperature(double sT) {
  if (localMat != 0) {
    if (sT < 0.01) {
      //mexWarnMsgTxt("temperature must be >= 0.01 for numeric stability. updated temperature to 0.01 instead of the given one\n");
      cout<<"temperature must be >= 0.01 for numeric stability. updated temperature to 0.01 instead of the given one"<<endl;
      sT = 0.01;
    }
    if (T != sT) {
      // the power in which the potentials should be raised
      double power = T / sT;
      
      for (int i=0; i<N; i++) {
	// raise local potentials in power
	for (int xi=0; xi<V[i]; xi++) {
	  localMat[i][xi] = pow(localMat[i][xi],power);
	}
      }
      T = sT;
    }
  }
  else {
    //mexWarnMsgTxt("local potentials must be given before updating temperature, cannot update temperature yet\n");
    cout<<"local potentials must be given before updating temperature, cannot update temperature yet"<<endl;
  }
}
