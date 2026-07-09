#include "Loopy.h"
#include <math.h>
#include <iostream>
//#include "mex.h"

using namespace std;
Loopy::~Loopy() {

  freeMessages();
  freePairBeliefs();
}


void Loopy::initMessages() {

  freeMessages();
  
  // init the messages matrix
  l_messages = new double**[ia_mrf->N];
  for (int i=0; i<ia_mrf->N; i++) {
    l_messages[i] = new double*[ia_mrf->N];
    for (int j=0; j<ia_mrf->N; j++) {
      l_messages[i][j] = 0;
    }
    for (int n=0; n<ia_mrf->neighbNum(i); n++) {
      int j = ia_mrf->adjMat[i][n];
      l_messages[i][j] = new double[ia_mrf->V[j]];
      for (int xj=0; xj<ia_mrf->V[j]; xj++) {
	l_messages[i][j][xj] = 1.0 / ia_mrf->V[j];
      }
    }
  }
}

void Loopy::freeMessages() {
 

if (l_messages != 0) {
    // free the messages matrix
    for (int i=0; i<ia_mrf->N; i++) {
      for (int j=0; j<ia_mrf->N; j++) {
	if (l_messages[i][j] != 0) {
	  delete[] l_messages[i][j];
	}
      }
      delete[] l_messages[i];
    }
    delete[] l_messages;
    l_messages = 0;
  }

}

void Loopy::initPairBeliefs() {

  freePairBeliefs();
  // init the pair beliefs defultive to p(Xi=xi, Xj=xj) = Psi(xi,i)*Psi(xj,j)*Psi(xi,i,xj,j)
  l_pairBeliefs = new double***[ia_mrf->N];
  for (int i=0; i<ia_mrf->N; i++) {
    l_pairBeliefs[i] = new double**[ia_mrf->neighbNum(i)];
    for (int n=0; n<ia_mrf->neighbNum(i); n++) {
      l_pairBeliefs[i][n] = 0;
      int j = ia_mrf->adjMat[i][n];
      if (i<j) {
	l_pairBeliefs[i][n] = new double*[ia_mrf->V[i]];
	for (int xi=0; xi<ia_mrf->V[i]; xi++) {
	  l_pairBeliefs[i][n][xi] = new double[ia_mrf->V[j]];
	  for (int xj=0; xj<ia_mrf->V[j]; xj++) {
	    l_pairBeliefs[i][n][xi][xj] = (ia_mrf->localMat[i][xi] *
					   ia_mrf->localMat[j][xj] *
					   ia_mrf->pairPotential(i,n,xi,xj));
	  }
	}
      }
    }
  }
}

void Loopy::freePairBeliefs() {
 
  if (l_pairBeliefs != 0) {
    for (int i=0; i<ia_mrf->N; i++) {
      for (int n=0; n<ia_mrf->neighbNum(i); n++) {
	if (l_pairBeliefs[i][n] != 0) {
	  for (int xi=0; xi<ia_mrf->V[i]; xi++) {
	    delete[] l_pairBeliefs[i][n][xi];
	  }
	  delete[] l_pairBeliefs[i][n];
	}
      }
      delete[] l_pairBeliefs[i];
    }
    delete[] l_pairBeliefs;
    l_pairBeliefs = 0;
  }

}

double**** Loopy::calcPairBeliefs() {
  
  double**** new_pairBeliefs = new double***[ia_mrf->N];

  for (int i=0; i<ia_mrf->N; i++) {
    new_pairBeliefs[i] = new double**[ia_mrf->neighbNum(i)];
    for (int n=0; n<ia_mrf->neighbNum(i); n++) {
      new_pairBeliefs[i][n] = 0;
      int j = ia_mrf->adjMat[i][n];
      if (i<j) {
	double sum_beliefs_ij = 0.0;
	new_pairBeliefs[i][n] = new double*[ia_mrf->V[i]];
	for (int xi=0; xi<ia_mrf->V[i]; xi++) {
	  new_pairBeliefs[i][n][xi] = new double[ia_mrf->V[j]];
	  for (int xj=0; xj<ia_mrf->V[j]; xj++) {
	    new_pairBeliefs[i][n][xi][xj] = (ia_mrf->localMat[i][xi] *
					     ia_mrf->localMat[j][xj] *
					     ia_mrf->pairPotential(i,n,xi,xj));
	    for (int ni=0; ni<ia_mrf->neighbNum(i); ni++) {
	      int k = ia_mrf->adjMat[i][ni];
	      if (k!=j) {
		new_pairBeliefs[i][n][xi][xj] *= l_messages[k][i][xi];
	      }
	    }
	    for (int nj=0; nj<ia_mrf->neighbNum(j); nj++) {
	      int k = ia_mrf->adjMat[j][nj];
	      if (k!=i) {
		new_pairBeliefs[i][n][xi][xj] *= l_messages[k][j][xj];
	      }
	    }

	    sum_beliefs_ij += new_pairBeliefs[i][n][xi][xj];
	  }
	}

	// normalize the ij-beliefs
	if (sum_beliefs_ij > 0.0) {
	  for (int xi=0; xi<ia_mrf->V[i]; xi++) {
	    for (int xj=0; xj<ia_mrf->V[j]; xj++) {
	      new_pairBeliefs[i][n][xi][xj] /= sum_beliefs_ij;
	      //              cout<<i<<" "<<j<<" "<<n<<" "<<xi<<" "<<xj<<" "<<  new_pairBeliefs[i][n][xi][xj] <<endl;  
	    }
	  }
	}
      }
    }
  }
  freePairBeliefs();
  l_pairBeliefs = new_pairBeliefs;
  new_pairBeliefs = 0;

  return l_pairBeliefs;
}

double** Loopy::inference(int* converged) {

  double th = pow(10.,-8);

  double dBel = th+1.0;
  int nIter = 0;
  /*
   cout<<ia_mrf->N<<endl; 
  for (int i=0; i<ia_mrf->N; i++)
  { 
     cout<<"i="<<i<<": ";
     for (int n=0; n<ia_mrf->neighbNum(i); n++) cout<<ia_mrf->adjMat[i][n]<<" ";
    cout<<endl;
   }
  */
  double*** new_messages = 0;
  if (l_strategy == PARALLEL) {
    new_messages = new double**[ia_mrf->N];
    for (int i=0; i<ia_mrf->N; i++) {
      new_messages[i] = new double*[ia_mrf->N];
      for (int j=0; j<ia_mrf->N; j++) {
	new_messages[i][j] = 0;
      }
      for (int n=0; n<ia_mrf->neighbNum(i); n++) {
	int j = ia_mrf->adjMat[i][n];
	new_messages[i][j] = new double[ia_mrf->V[j]];
	for (int xj=0; xj<ia_mrf->V[j]; xj++) {
	  new_messages[i][j][xj] = l_messages[i][j][xj];
	}
      }
    }
  }
 
  while (dBel>th && nIter<l_maxIter) {
    nIter++;
    
    for (int i=0; i<ia_mrf->N; i++) {

      // init the incoming messages to 1
      double* incoming = new double[ia_mrf->V[i]];
      double* factor = new double[ia_mrf->neighbNum(i)];
      for (int xi=0; xi<ia_mrf->V[i]; xi++) {
	incoming[xi] = 1.0;
      }
      // get incoming messages
      for (int n=0; n<ia_mrf->neighbNum(i); n++) {
	int j = ia_mrf->adjMat[i][n];
	factor[n] = 0.0;
	for (int xi=0; xi<ia_mrf->V[i]; xi++) {
	  incoming[xi] *= l_messages[j][i][xi];
	  factor[n] += incoming[xi];
	}
	for (int xi=0; xi<ia_mrf->V[i]; xi++) {
	  incoming[xi] /= factor[n];
	}
      }
    
      // calculate outgoing messages
      for (int n=0; n<ia_mrf->neighbNum(i); n++) {
	int j = ia_mrf->adjMat[i][n];

	double sum_outgoing_to_j = 0.0;
	double* outgoing = new double[ia_mrf->V[j]];
	
	for (int xj=0; xj<ia_mrf->V[j]; xj++) {
	  
	  switch (l_sumOrMax) {
	    case SUM:
	      outgoing[xj] = 0.0;
	      break;
	    case MAX:
	      outgoing[xj] = -1.0;
	      break;
	    default:
	      break;
	  }
	    
	  for (int xi=0; xi<ia_mrf->V[i]; xi++) {
	    double outM = ia_mrf->pairPotential(i,n,xi,xj) * ia_mrf->localMat[i][xi] *
	      incoming[xi] * factor[n] / l_messages[j][i][xi];

	    switch (l_sumOrMax) {
	      case SUM:
		outgoing[xj] += outM;
		break;
	      case MAX:
		if (outM > outgoing[xj]) {
		  outgoing[xj] = outM;
		}
		break;
	      default:
		break;
	    }
	  }
	    
	  sum_outgoing_to_j += outgoing[xj];
	}
	double epsilon = pow(10.,-16);
	for (int xj=0; xj<ia_mrf->V[j]; xj++) {
	  if (sum_outgoing_to_j > 0.0) {
	    outgoing[xj] /= sum_outgoing_to_j;
	    if (outgoing[xj] < epsilon)
	      outgoing[xj] = epsilon;
	  }
	    
	  switch (l_strategy) {

	    case SEQUENTIAL:
	      l_messages[i][j][xj] = outgoing[xj];
	      break;

	    case PARALLEL:
	      new_messages[i][j][xj] = outgoing[xj];
	      break;

	    default:
	      break;
	  }
	}
	delete[] outgoing;
	outgoing = 0;
      }
      delete[] incoming;
      incoming = 0;
      delete[] factor;
      factor = 0;
    }

    if (l_strategy == PARALLEL) {
      for (int i=0; i<ia_mrf->N; i++) {
	for (int n=0; n<ia_mrf->neighbNum(i); n++) {
	  int j = ia_mrf->adjMat[i][n];
	  for (int xj=0; xj<ia_mrf->V[j]; xj++) {
	    l_messages[i][j][xj] = new_messages[i][j][xj];
	  }
	}
      }
    }

    // update beliefs and check for convergence
    
    dBel = 0.0;
    
    double** new_beliefs = new double*[ia_mrf->N];
    
    for (int i=0; i<ia_mrf->N; i++) {
      new_beliefs[i] = new double[ia_mrf->V[i]];
      double sum_beliefs_i = 0.0;
      for (int xi=0; xi<ia_mrf->V[i]; xi++) {
	new_beliefs[i][xi] = ia_mrf->localMat[i][xi];
	for (int n=0; n<ia_mrf->neighbNum(i); n++) {
	  int j = ia_mrf->adjMat[i][n];
	  new_beliefs[i][xi] *= l_messages[j][i][xi];
	}
	sum_beliefs_i += new_beliefs[i][xi];
      }

      double norm_dBel_i = 0.0;
      for (int xi=0; xi<ia_mrf->V[i]; xi++) {
	if (sum_beliefs_i > 0.0) {
	  new_beliefs[i][xi] /= sum_beliefs_i;
	}
	norm_dBel_i += pow((new_beliefs[i][xi] - ia_beliefs[i][xi]), 2.0);
      }
      norm_dBel_i = pow(norm_dBel_i, 0.5);

      dBel += norm_dBel_i;
    }
    freeBeliefs();
    ia_beliefs = new_beliefs;
    new_beliefs = 0;

    //   cout<<"nIter loopy "<<nIter<<" dBel "<<dBel<<" th "<<th<<" l_maxIter "<<l_maxIter<<endl;
  }

  if (l_strategy == PARALLEL) {
    for (int i=0; i<ia_mrf->N; i++) {
      for (int n=0; n<ia_mrf->neighbNum(i); n++) {
	int j = ia_mrf->adjMat[i][n];
	delete[] new_messages[i][j];
      }
      delete[] new_messages[i];
    }
    delete[] new_messages;
    new_messages = 0;
  }

  if (dBel<=th) {
    (*converged) = nIter;
    //mexPrintf("c-Loopy: converged in %d iterations\n",nIter);
    //  cout << "c-Loopy: converged in " << nIter << " iterations " << endl;
  }
  else {
    (*converged) = -1;
    //mexPrintf("c-Loopy: did not converge\n");
    //  cout << "c-Loopy: did not converge" << endl;
  }
  
  return ia_beliefs;
  
}
