#ifndef __FACTOR_H   
#define __FACTOR_H   
   
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <math.h> 
#include <time.h> 
#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 

#include <map>
#include <dai/alldai.h>  // Include main libDAI header file
#include <dai/jtree.h>
#include <dai/bp.h>
#include <dai/decmap.h>
#include <dai/fbp.h>
#include <dai/bbp.h>
#include <dai/hak.h>
#include <dai/trwbp.h>
#include <dai/treeep.h>
#include <dai/lc.h>   

#include "Popul.h"   
#include "TriangSubgraph.h"   
#include "AbstractTree.h" 
#include "FDA.h" 

using namespace dai;
using namespace std;

int FindMAP(FactorGraph, int, unsigned int*);
FactorGraph CreateFactorGraph(DynFDA*,unsigned int*,int);

#endif
