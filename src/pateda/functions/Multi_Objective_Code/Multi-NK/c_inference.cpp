#include "mex.h"
#include "fillMethods.h"
#include "Loopy.h"
#include "GBP.h"
#include "Gibbs.h"
#include "Wolff.h"
#include "SwendsenWang.h"
#include "Metropolis.h"
#include "MeanField.h"
#include "GBPPreProcessor.h"

// *****************************************************************************
// enumerators
// *****************************************************************************

enum algorithmType {AT_LOOPY,AT_GBP,AT_GIBBS,AT_WOLFF,AT_SWENDSEN_WANG,
		    AT_METROPOLIS,AT_MEAN_FIELD};

// *****************************************************************************
// mexFunction
// *****************************************************************************

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

  // **************************************************************************
  // variables declaration
  // **************************************************************************

  // variables for Loopy
  Strategy strategy;
  SumOrMax sumOrMax;
  double gbp_alpha;

  // variables for GBP
  int*** assignInd = 0;
  double* bethe = 0;
  GBPPreProcessor* processor = 0;
  MRF* reg_mrf = 0;
  RegionLevel* regions = 0;
  vector<RegionLevel>* allRegions = 0;
  bool allLevels = false;
  bool trw = false;
  
  // variables for Monte-Carlo
  int burningTime, samplingInterval, num_samples;
  int* startX = 0;

  // for Loopy,GBP,Mean-Field
  int maxIter;

  
  // **************************************************************************
  // reading input arguments
  // **************************************************************************

  // check number of input arguments.
  // arguments should be:
  //
  // adjMat - 1xN cell array, each cell {i} is a row vector with the indices of
  //          i's neighbours
  //
  // lambda - there are 2 forms for lambda:
  //          1. in general MRF algorithms (loopy, gbp, gibbs, mean-field) :
  //             lambda should be a cell array of 1xN, each cell {i} is a cell
  //             array of 1xneighbNum(i). each cell {i}{n} is a VixVj matrix,
  //             where j is the n-th neighbour of i
  //          2. in PottsMRF alhorithms (monte-carlo algorithms which are planned
  //             for Potts model, i.e. metropolis and the cluster algorithms
  //             wolff and swendsen-wang) :
  //             here lambda should be 1xN cell array, each cell {i} is a row
  //             vector with the strength of interaction of i with each of its
  //             neighbours
  // note: Psi{i,j} = exp( [lambda(i,j), 0; 0, lambda(i,j)] )
  //
  // local - cell array of Nx1, each cell {i} is a row vector of length Vi
  //
  // algorithm - integer representing the inference algorithm to use, see the
  //             enumerator algorithmType at the top of this page
  //
  // temperature - double scalar, the temperature of the system
  //
  // model - integeger representing the model, see the enumerator in "definitions.h"
  //
  // trw - use Tree-Reweighted
  //
  // for other parameters required for each algorithm see the header of "inference.m"
  //
  //
  // note: N = number of nodes, V = number of possible values
  
  if (nrhs < 7 || nrhs > 12) {
    mexErrMsgTxt("Incorrect number of inputs.");
  }

  // get algorithm-type
  algorithmType algo_type = (algorithmType)((int)(mxGetScalar(prhs[3])));
  Model model = (Model)((int)(mxGetScalar(prhs[5])));
  bool potts_model = ((model==POTTS) ||
		      (algo_type==AT_WOLFF) ||
		      (algo_type==AT_SWENDSEN_WANG));
  bool monte_carlo = ((algo_type==AT_GIBBS) ||
		      (algo_type==AT_WOLFF) ||
		      (algo_type==AT_SWENDSEN_WANG) ||
		      (algo_type==AT_METROPOLIS));
  
  // check number of output arguments
  if ((nlhs > 3) || ((nlhs > 2) && (algo_type != AT_GBP) && (algo_type != AT_LOOPY))) {
    mexErrMsgTxt("Too many output arguments.");
  }

  // get number of nodes and adjMat
  vector<Nodes>* adjMat = new vector<Nodes>();
  fillAdjMat(prhs[0],*adjMat);
  int num_nodes = adjMat->size();

  // define the MRF
  MRF* mrf = 0;
  if (potts_model) {
    mrf = new PottsMRF(*adjMat);
  }
  else {
    mrf = new MRF(*adjMat);
  }
  
  // get local potentials
  fillLocalMat(prhs[2],mrf);
  
  // get pairwise potentials
  if (potts_model) {
    fillLambdaMat(prhs[1],(PottsMRF*)mrf);
  }
  else {
    fillPsiMat(prhs[1],mrf);
  }

  // get tepmerature
  double temperature = mxGetScalar(prhs[4]);
  mrf->setTemperature(temperature);  
  
  // For monte-carlo algorithms (gibbs, wolff, swendsen-wang), get
  // the initial state and the sampling parameters
  if (monte_carlo) {
    if (nrhs != 10) {
      mexErrMsgTxt("incorrect number of inputs");
    }

    startX = new int[num_nodes];
    fillInitialAssignment(prhs[6], startX, num_nodes);

    // get burningTime, samplingInterval, num_samples
    burningTime = (int)(mxGetScalar(prhs[7]));
    samplingInterval = (int)(mxGetScalar(prhs[8]));
    num_samples = (int)(mxGetScalar(prhs[9]));
    
  }
  else {
    // for all non-monte-carlo-algorithms (Mean-Field, BP & GBP)
    maxIter = (int)(mxGetScalar(prhs[6]));

    if (algo_type == AT_LOOPY) {

	// for loopy belief propagation:
	// get sum-or-max-flag and strategy

	if (nrhs != 9) {
	  mexErrMsgTxt("incorrect number of inputs");
	}
	
	sumOrMax = (SumOrMax)((int)(mxGetScalar(prhs[7])));
	strategy = (Strategy)((int)(mxGetScalar(prhs[8])));
	
    }
    if (algo_type == AT_GBP) {
	// for generalized belief propagation:
	// get regions, regions-adj (if given), sum-or-max-flag
	// and alpha

	if (nrhs != 12) {
	  mexErrMsgTxt("incorrect number of inputs");
	}

	allLevels = (int)(mxGetScalar(prhs[8])) > 0;
	if (allLevels) {
	  allRegions = new vector<RegionLevel>();
	  allRegions->clear();
	  fillRegionLevels(prhs[7],*allRegions);
	}
	else {
	  regions = new RegionLevel();
	  fillRegions(prhs[7],*regions);
	}

	sumOrMax = (SumOrMax)((int)(mxGetScalar(prhs[9])));
	gbp_alpha = mxGetScalar(prhs[10]);

	trw = ((int)(mxGetScalar(prhs[11]))) > 0;
    }
  }

  // **************************************************************************
  // create the algorithm
  // **************************************************************************

  InferenceAlgorithm* algorithm = 0;
  switch (algo_type) {

    case AT_LOOPY:

      algorithm = new Loopy(mrf,sumOrMax,strategy,maxIter);
      
      break;

    case AT_GBP:
      if (allLevels) {
	processor = new GBPPreProcessor(allRegions, mrf, trw);
      }
      else {
	processor = new GBPPreProcessor(*regions, mrf, trw);
	regions->clear();
	delete regions;
	regions = 0;
      }
      
      reg_mrf = processor->getRegionMRF();
      assignInd = processor->getAssignTable();
      bethe = processor->getBethe();

      algorithm = new GBP(reg_mrf,assignInd,bethe,sumOrMax,gbp_alpha,maxIter);
      
      break;

    case AT_GIBBS:

      algorithm = new Gibbs(mrf,startX,burningTime,samplingInterval,num_samples);

      delete[] startX;
      startX = 0;
      
      break;

    case AT_WOLFF:

      algorithm = new Wolff((PottsMRF*)mrf,startX,burningTime,samplingInterval,num_samples);

      delete[] startX;
      startX = 0;
      
      break;

    case AT_SWENDSEN_WANG:

      algorithm = new SwendsenWang((PottsMRF*)mrf,startX,burningTime,samplingInterval,num_samples);

      delete[] startX;
      startX = 0;
      
      break;

    case AT_METROPOLIS:

      algorithm = new Metropolis(mrf,startX,burningTime,samplingInterval,num_samples);

      delete[] startX;
      startX = 0;
      
      break;

    case AT_MEAN_FIELD:

      algorithm = new MeanField(mrf,maxIter);

      break;
      
    default:

      mexErrMsgTxt("invalid algorithm type. possible values are: 0-loopy, 1-gbp, 2-gibbs, 3-wolff, 4-swendswen-wang, 5-metropolis, 6-mean-field");
      break;
  }
  

  // **************************************************************************
  // make inference
  // **************************************************************************
  int converged;
  double** beliefs = algorithm->inference(&converged);

  double** singleBeliefs = 0;
  double**** pairBeliefs = 0;

  switch (algo_type) {
    
    case AT_LOOPY:
      
      if (nlhs > 2) {
	pairBeliefs = ((Loopy*)algorithm)->calcPairBeliefs();
      }
      
      break;
      
    case AT_GBP:
      
      singleBeliefs = new double*[num_nodes];
      for (int i=0; i<num_nodes; i++) {
	singleBeliefs[i] = new double[mrf->V[i]];      
      }
      processor->extractSingle(beliefs,singleBeliefs,sumOrMax);

      if (nlhs > 2) {
	pairBeliefs = new double***[num_nodes];
	for (int i=0; i<num_nodes; i++) {
	  pairBeliefs[i] = new double**[mrf->neighbNum(i)];
	  for (int n=0; n<mrf->neighbNum(i); n++) {
	    pairBeliefs[i][n] = 0;
	    int j = mrf->adjMat[i][n];
	    if (i<j) {
	      pairBeliefs[i][n] = new double*[mrf->V[i]];
	      for (int xi=0; xi<mrf->V[i]; xi++) {
		pairBeliefs[i][n][xi] = new double[mrf->V[j]];
	      }
	    }
	  }
	}
	processor->extractPairs(beliefs,pairBeliefs,sumOrMax);
      }
      beliefs = singleBeliefs;
      
      break;
      
    default:
      
      break;
  }
  
  // **************************************************************************
  // assign results to output argument (if given)
  // **************************************************************************

  if (nlhs > 0) {
    int bel_dims[2] = {1,num_nodes};
    plhs[0] = mxCreateCellArray(2,bel_dims);
    for (int i=0; i<num_nodes; i++) {
      int val_dims[2] = {mrf->V[i],1};
      mxArray* bel_i = mxCreateNumericArray(2,val_dims,mxDOUBLE_CLASS, mxREAL);
      double* resBelPtr = mxGetPr(bel_i);
      for (int xi=0; xi<mrf->V[i]; xi++) {
	resBelPtr[xi] = beliefs[i][xi];
      }
      mxSetCell(plhs[0],i,bel_i);
    }
    if (nlhs > 1) {
      plhs[1] = mxCreateDoubleScalar(converged); // For matlab6.5
      //plhs[1] = mxCreateScalarDouble(converged);
      if (nlhs > 2) {
	int pair_dims[2] = {1,num_nodes};
	plhs[2] = mxCreateCellArray(2,pair_dims);

	for (int i=0; i<num_nodes; i++) {
	  int pair_i_dims[2] = {1, mrf->neighbNum(i)};
	  mxArray* bel_i = mxCreateCellArray(2,pair_i_dims);
	  
	  for (int n=0; n<mrf->neighbNum(i); n++) {
	    int j = mrf->adjMat[i][n];
	    if (i<j) {
	      int pval_dims[2] = {mrf->V[i], mrf->V[j]};
	      mxArray* bel_ij = mxCreateNumericArray(2,pval_dims,mxDOUBLE_CLASS,mxREAL);
	      
	      double* resBelPtr = mxGetPr(bel_ij);
	      for (int xi=0; xi<mrf->V[i]; xi++) {
		for (int xj=0; xj<mrf->V[j]; xj++) {
		  resBelPtr[xi + xj*mrf->V[i]] = pairBeliefs[i][n][xi][xj];
		}
	      }
	      mxSetCell(bel_i, n, bel_ij);	      
	    }
	  }
	  mxSetCell(plhs[2], i, bel_i);
	}	
      }
    }
  }


  // **************************************************************************
  // free memory
  // **************************************************************************

  delete algorithm;
  algorithm = 0;
  
  if (singleBeliefs != 0) {
    for (int i=0; i<num_nodes; i++) {
      delete[] singleBeliefs[i];
    }
    delete[] singleBeliefs;    
    singleBeliefs = 0;
  }
  if (pairBeliefs != 0 && algo_type == AT_GBP) {
    for (int i=0; i<num_nodes; i++) {
      for (int n=0; n<mrf->neighbNum(i); n++) {
	if (pairBeliefs[i][n] != 0) {
	  for (int xi=0; xi<mrf->V[i]; xi++) {
	    delete[] pairBeliefs[i][n][xi];
	  }
	  delete[] pairBeliefs[i][n];
	}
      }
      delete[] pairBeliefs[i];
    }
    delete[] pairBeliefs;
    pairBeliefs = 0;
  }
  if (processor != 0) {
    delete processor;
    processor = 0;
  }

  delete mrf;
  mrf = 0;
  delete adjMat;
  adjMat = 0;
  
}


