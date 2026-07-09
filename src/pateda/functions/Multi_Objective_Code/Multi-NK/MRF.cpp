#include "MRF.h"
#include <iostream> 
#include <fstream>
using namespace std;
#include <math.h>
//#include "mex.h"


MRF::~MRF() {
  if (lambdaMat != 0) {
    for (int i=0; i<N; i++) {
      for (int n=0; n<neighbNum(i); n++) {
	int j = adjMat[i][n];
	if (i<j) {
	  for (int xi=0; xi<V[i]; xi++) {
	    delete[] lambdaMat[i][n][xi];
	  }
	  delete[] lambdaMat[i][n];
	}
      }
      delete[] lambdaMat[i];
    }
    delete[] lambdaMat;
    lambdaMat = 0;
  }
  if (localMat != 0) {
    for (int i=0; i<N; i++) {
      delete[] localMat[i];
    }
    delete[] localMat;
    localMat = 0;
  }
  delete[] V;
  V = 0;
  
  for (int i=0; i<N; i++) {    
    adjMat[i].clear();
  }
  adjMat.clear();
  
}

void MRF::assignPairPotential(int i, int nj, PairPotentials& pairPot) {
  int j = adjMat[i][nj]; // assuming i<j !!!

  PairPotentials lambda_ij = lambdaMat[i][nj];
  for (int xi=0; xi<V[i]; xi++) {
    for (int xj=0; xj<V[j]; xj++) {
      lambda_ij[xi][xj] = pairPot[xi][xj];
     
    }
  }
}

double MRF::pairPotential(int i, int n, int xi, int xj) const {
  int j = adjMat[i][n];
  PairPotentials lambda_ij = lambdaMat[i][n];
  if (i<j) {
    return lambda_ij[xi][xj];
  }
  else {
    return lambda_ij[xj][xi];
  }
}

void MRF::initPairPotentials() {
  lambdaMat = new PairPotentials*[N];
  for (int i=0; i<N; i++) {
    int nn = adjMat[i].size();
    lambdaMat[i] = new PairPotentials[nn];
    for (int nj=0; nj<nn; nj++) {
      int j = adjMat[i][nj];
      if (i<j) {
	PairPotentials pairPot = new Potentials[V[i]];
	for (int xi=0; xi<V[i]; xi++) {
	  pairPot[xi] = new Potential[V[j]];
	  for (int xj=0; xj<V[j]; xj++) {
	    pairPot[xi][xj] = 1.0;
	  }
	}
	lambdaMat[i][nj] = pairPot;
      }
      else {
	int ni = 0;
	while (adjMat[j][ni] != i) {
	  ni++;
	}
	lambdaMat[i][nj] = lambdaMat[j][ni];
      }
    }
  }
}

void MRF::initLocalPotentials() {
  localMat = new Potentials[N];
  for (int i=0; i<N; i++) {
    localMat[i] = new Potential[V[i]];
    for (int xi=0; xi<V[i]; xi++) {
      localMat[i][xi] = 1.0;
    }
  }
}

//MODIFIED SANTANA 7/11/2005

double MRF::getEnergy(unsigned const* assignment) const {
  double exp_minus_beta_energy = 1.0;
  // for (int xi=0; xi<20; xi++) 
  //    for (int xj=0; xj<20; xj++) 
  //       cout<<"0 - 1  "<<xi<<" "<<xj<<" "<<pairPotential(0,1,xi,xj)<<endl;

  for (int i=0; i<N; i++) {
    int xi = assignment[i];
    exp_minus_beta_energy *= localMat[i][xi];
    for (int n=0; n<neighbNum(i); n++) {
      int j = adjMat[i][n];
      if (i<j) {
	int xj = assignment[j];
	exp_minus_beta_energy *= pairPotential(i,n,xi,xj);
        //cout<<i<<" "<<j<<" "<<xi<<" "<<xj<<" "<<pairPotential(i,n,xi,xj)<<endl;
      }
    }
  }
  double energy = 0.0;
  if (exp_minus_beta_energy > 0.0) {
    energy = - log(exp_minus_beta_energy) * T;
  }
  else {
    //mexWarnMsgTxt("potential for assignment is 0, cannot calculate energy\n");
    cout<<"potential for assignment is 0, cannot calculate energy"<<endl;
  }
  return energy;
}


double MRF::getEnergy(int const* assignment) const {
  double exp_minus_beta_energy = 1.0;
  for (int i=0; i<N; i++) {
    int xi = assignment[i];
    exp_minus_beta_energy *= localMat[i][xi];
    for (int n=0; n<neighbNum(i); n++) {
      int j = adjMat[i][n];
      if (i<j) {
	int xj = assignment[j];
	exp_minus_beta_energy *= pairPotential(i,n,xi,xj);
      }
    }
  }
  double energy = 0.0;
  if (exp_minus_beta_energy > 0.0) {
    energy = - log(exp_minus_beta_energy) * T;
  }
  else {
    //mexWarnMsgTxt("potential for assignment is 0, cannot calculate energy\n");
    cout<<"potential for assignment is 0, cannot calculate energy"<<endl;
  }
  return energy;
}

void MRF::setTemperature(double sT) {
  if (lambdaMat != 0 && localMat != 0) {
    if (sT < 0.01) {      
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
	
	// raise pairwise potentials in power
	int nn = adjMat[i].size();
	for (int nj=0; nj<nn; nj++) {
	  int j = adjMat[i][nj];
	  if (i<j) {
	    PairPotentials lambda_ij = lambdaMat[i][nj];
	    for (int xi=0; xi<V[i]; xi++) {
	      for (int xj=0; xj<V[j]; xj++) {
		lambda_ij[xi][xj] = pow(lambda_ij[xi][xj],power);
	      }
	    }
	  }
	}
      }
	
      T = sT;
    }
  }
  else {
    // mexWarnMsgTxt("local and pair potentials must be given before updating temperature, cannot update temperature yet\n");
    cout<<"local and pair potentials must be given before updating temperature, cannot update temperature yet"<<endl;
  }
}
