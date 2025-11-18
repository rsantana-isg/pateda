#include "FactorGraphMethods.h" 


FactorGraph CreateFactorGraph(DynFDA* MyMarkovNet,unsigned int* Cardinalities,int nclust) 
{
     vector<Factor> facs;
     size_t nr_Factors;
     double val;
     FactorGraph fg;
     int k,j,l;

       for(k=0; k<nclust; k++)
        {       
	  vector<Var> Ivars;         
          for(j=0; j<MyMarkovNet->SetOfCliques->ListCliques[MyMarkovNet->OrderCliques[k]]->NumberVars; j++)
	    {
	      Ivars.push_back(Var(MyMarkovNet->SetOfCliques->ListCliques[MyMarkovNet->OrderCliques[k]]->vars[j], Cardinalities[MyMarkovNet->SetOfCliques->ListCliques[MyMarkovNet->OrderCliques[k]]->vars[j]]));          
            }
	 
         facs.push_back( Factor( VarSet( Ivars.begin(), Ivars.end(), Ivars.size() ), (Real)0 ) );
        
         // calculate permutation object
           Permute permindex( Ivars );          
           //cout<<"Values "<<endl;
         for(l=0; l<MyMarkovNet->SetOfCliques->ListCliques[MyMarkovNet->OrderCliques[k]]->r_NumberCases; l++)
	   {
	    val = MyMarkovNet->SetOfCliques->ListCliques[MyMarkovNet->OrderCliques[k]]->marg[l];          
            facs.back().set(permindex.convertLinearIndex(l), val ); // These are marginal tables of the last factor in facs
           }    
	}
       
       fg = FactorGraph(facs);
      
       return fg;
}		




int FindMAP(FactorGraph fg, int typeinfmethod, unsigned int* bestconf) {
       
  //cout << "Reads factor graph <alarm.fg> and runs" << endl;
  //      cout << "Belief Propagation, Max-Product and JunctionTree on it." << endl;
  //      cout << "JunctionTree is only run if a junction tree is found with" << endl;
  //     cout << "total number of states less than <maxstates> (where 0 means unlimited)." << endl << endl;
      
        size_t maxstates = 1000000;
      
        // Set some constants
        size_t maxiter = 1000; //  maxiter = 10000; 
        Real   tol = 1e-7; //  1e-9;
        size_t verb = 0;

        // Store the constants in a PropertySet object
        PropertySet opts;
        opts.set("maxiter",maxiter);  // Maximum number of iterations
        opts.set("tol",tol);          // Tolerance for convergence
        opts.set("verbose",verb);     // Verbosity (amount of output generated)

	if(typeinfmethod==1)
	  {        
            // Bound treewidth for junctiontree
             bool do_jt = true;
             try {
                   boundTreewidth(fg, &eliminationCost_MinFill, maxstates );
                 } catch( Exception &e ) {
                 if( e.getCode() == Exception::OUT_OF_MEMORY ) {
                   do_jt = false;
                   cout << "Skipping junction tree (need more than " << maxstates << " states)." << endl;
                   return 0;
                 }
               else
                throw;
             }
          

        JTree jt, jtmap;
        vector<size_t> jtmapstate;
        if( do_jt ) {
             // Construct another JTree (junction tree) object that is used to calculate
            // the joint configuration of variables that has maximum probability (MAP state)
            jtmap = JTree( fg, opts("updates",string("HUGIN"))("inference",string("MAXPROD")) );
            // Initialize junction tree algorithm
            jtmap.init();
            // Run junction tree algorithm
            jtmap.run();
            // Calculate joint state of all variables that has maximum probability
            jtmapstate = jtmap.findMaximum();
            }
          if( do_jt ) {
	    for( size_t i = 0; i < jtmapstate.size(); i++ ) bestconf[i] = jtmapstate[i];
            // Report exact MAP joint state
            //cout << "Exact MAP state (log score = " << fg.logScore( jtmapstate ) << "):" << endl;
	    // cout<<"MAPSTATE: ";
	    //  for( size_t i = 0; i < jtmapstate.size(); i++ )
	    // cout << jtmapstate[i] <<" ";
	      //cout << fg.var(i) << ": " << jtmapstate[i] << endl;
	    //cout<<endl;
           }
	  jtmapstate.clear();
	  }
        else if(typeinfmethod==2)
	  {
        // Construct a BP (belief propagation) object from the FactorGraph fg
        // using the parameters specified by opts and two additional properties,
        // specifying the type of updates the BP algorithm should perform and
        // whether they should be done in the real or in the logdomain
        //
        // Note that inference is set to MAXPROD, which means that the object
        // will perform the max-product algorithm instead of the sum-product algorithm
        BP mp(fg, opts("updates",string("SEQRND"))("logdomain",false)("inference",string("MAXPROD"))("damping",string("0.1")));
        // Initialize max-product algorithm
        mp.init();
        // Run max-product algorithm
        mp.run();
        // Calculate joint state of all variables that has maximum probability
        // based on the max-product result
        vector<size_t> mpstate = mp.findMaximum();

	// Report max-product MAP joint state
        //cout << "Approximate (max-product) MAP state (log score = " << fg.logScore( mpstate ) << "):" << endl;
	for( size_t i = 0; i < mpstate.size(); i++ ) bestconf[i] = mpstate[i];
        mpstate.clear();
        vector<size_t>( mpstate ).swap( mpstate );
        // cout<<"MAPSTATE: ";
 	// for( size_t i = 0; i < mpstate.size(); i++ ) cout<< mpstate[i] << " ";;
	//cout << fg.var(i) << ": " << mpstate[i] << endl;
	// cout<<endl;
        

        }
        else if(typeinfmethod>2 & typeinfmethod<=10)
       {
        // Construct a decimation algorithm object from the FactorGraph fg
        // using the parameters specified by opts and three additional properties,
        // specifying that the decimation algorithm should use the max-product
        // algorithm and should completely reinitalize its state at every step
      
       	 DAIAlgFG* ptrdecmap;

        if(typeinfmethod==3)    	    
	 ptrdecmap = new DecMAP(fg, opts("reinit",true)("ianame",string("BP"))("iaopts",string("[damping=0.1,inference=MAXPROD,logdomain=0,maxiter=1000,tol=1e-9,updates=SEQRND,verbose=0]")) );       
	else if(typeinfmethod==4)
	 ptrdecmap = new  TRWBP(fg, opts("updates",string("SEQRND"))("logdomain",false)("inference",string("MAXPROD"))("damping",string("0.1")));
	else if(typeinfmethod==5)
	  ptrdecmap = new FBP(fg, opts("updates",string("SEQRND"))("logdomain",false)("inference",string("MAXPROD"))("damping",string("0.1")));
        else if(typeinfmethod==6)
        ptrdecmap = new DecMAP(fg, opts("reinit",true)("ianame",string("FBP"))("iaopts",string("[damping=0.1,inference=MAXPROD,logdomain=0,maxiter=1000,tol=1e-3,updates=SEQRND,verbose=0]")) );
        else if(typeinfmethod==7)
        ptrdecmap = new DecMAP(fg, opts("reinit",true)("ianame",string("TRWBP"))("iaopts",string("[damping=0.1,inference=MAXPROD,logdomain=0,maxiter=1000,tol=1e-3,updates=SEQRND,verbose=0]")) );
          
        ptrdecmap->init();
        ptrdecmap->run();        
        vector<size_t> decmapstate;
        decmapstate = *&ptrdecmap->findMaximum(); 
        
         // Report DecMAP joint state
        //cout << "Approximate DecMAP state (log score = " << fg.logScore( decmapstate ) << "):" << endl;
	for( size_t i = 0; i < decmapstate.size(); i++ ) bestconf[i] = decmapstate.at(i);
        //for( size_t i = 0; i < 512; i++ ) bestconf[i] = 1;
       
	decmapstate.clear();        
	//cout << "Capacity after clear() is " << decmapstate.capacity() << '\n';
        vector<size_t>( decmapstate ).swap( decmapstate );
     	//cout << "Capacity after swap() is " << decmapstate.capacity() << '\n';
	
         //for( size_t i = 0; i < decmapstate.size(); i++ )
	 //cout << decmapstate[i] <<" ";
	//cout << fg.var(i) << ": " << decmapstate[i] << endl;
        //cout<<endl;
	 delete ptrdecmap;
        
        //cout<<"MAPSTATE: ";
        //for( size_t i = 0; i < decmapstate.size(); i++ ) cout<< bestconf[i] << " ";
        //cout<<endl;
       }
	else if(typeinfmethod>10) //HAK implementation
       {            
	 //DAIAlgRG* ptrdecmap;
         //if(typeinfmethod==11)
	 //  ptrdecmap = new HAK(fg,opts("clusters",string("BETHE"))("doubleloop",true)("InitType",string("UNIFORM"))("inference",string("MAXPROD")));
	 // ptrdecmap->init();
	 //ptrdecmap->run();
	 // vector<size_t> decmapstate = ptrdecmap->findMaximum(); 
         // for( size_t i = 0; i < decmapstate.size(); i++ ) bestconf[i] = decmapstate[i];
 
       }
	
   return 1;
}

/*

int FindMAP(FactorGraph fg, int typeinfmethod, unsigned int* bestconf) {
       
  //cout << "Reads factor graph <alarm.fg> and runs" << endl;
  //      cout << "Belief Propagation, Max-Product and JunctionTree on it." << endl;
  //      cout << "JunctionTree is only run if a junction tree is found with" << endl;
  //     cout << "total number of states less than <maxstates> (where 0 means unlimited)." << endl << endl;
      
         size_t maxstates = 1000000;
      
        // Set some constants
        size_t maxiter = 10000;
        Real   tol = 1e-9;
        size_t verb = 0;

        // Store the constants in a PropertySet object
        PropertySet opts;
        opts.set("maxiter",maxiter);  // Maximum number of iterations
        opts.set("tol",tol);          // Tolerance for convergence
        opts.set("verbose",verb);     // Verbosity (amount of output generated)

	if(typeinfmethod==1)
	  {        
            // Bound treewidth for junctiontree
             bool do_jt = true;
             try {
                   boundTreewidth(fg, &eliminationCost_MinFill, maxstates );
                 } catch( Exception &e ) {
                 if( e.getCode() == Exception::OUT_OF_MEMORY ) {
                   do_jt = false;
                   cout << "Skipping junction tree (need more than " << maxstates << " states)." << endl;
                 }
               else
                throw;
             }
          

        JTree jt, jtmap;
        vector<size_t> jtmapstate;
        if( do_jt ) {
             // Construct another JTree (junction tree) object that is used to calculate
            // the joint configuration of variables that has maximum probability (MAP state)
            jtmap = JTree( fg, opts("updates",string("HUGIN"))("inference",string("MAXPROD")) );
            // Initialize junction tree algorithm
            jtmap.init();
            // Run junction tree algorithm
            jtmap.run();
            // Calculate joint state of all variables that has maximum probability
            jtmapstate = jtmap.findMaximum();
            }
          if( do_jt ) {
	    for( size_t i = 0; i < jtmapstate.size(); i++ ) bestconf[i] = jtmapstate[i];
            // Report exact MAP joint state
            cout << "Exact MAP state (log score = " << fg.logScore( jtmapstate ) << "):" << endl;
	    // for( size_t i = 0; i < jtmapstate.size(); i++ )
	    // cout << jtmapstate[i] <<" ";
	      //cout << fg.var(i) << ": " << jtmapstate[i] << endl;
	    //cout<<endl;
           }
        }
        else if(typeinfmethod==2)
	  {
        // Construct a BP (belief propagation) object from the FactorGraph fg
        // using the parameters specified by opts and two additional properties,
        // specifying the type of updates the BP algorithm should perform and
        // whether they should be done in the real or in the logdomain
        //
        // Note that inference is set to MAXPROD, which means that the object
        // will perform the max-product algorithm instead of the sum-product algorithm
        BP mp(fg, opts("updates",string("SEQRND"))("logdomain",false)("inference",string("MAXPROD"))("damping",string("0.1")));
        // Initialize max-product algorithm
        mp.init();
        // Run max-product algorithm
        mp.run();
        // Calculate joint state of all variables that has maximum probability
        // based on the max-product result
        vector<size_t> mpstate = mp.findMaximum();

	// Report max-product MAP joint state
        cout << "Approximate (max-product) MAP state (log score = " << fg.logScore( mpstate ) << "):" << endl;
         for( size_t i = 0; i < mpstate.size(); i++ ) bestconf[i] = mpstate[i];
	 for( size_t i = 0; i < mpstate.size(); i++ ) cout<< mpstate[i] << " ";;
	//cout << fg.var(i) << ": " << mpstate[i] << endl;
           cout<<endl;
        }
        else if(typeinfmethod>2 & typeinfmethod<=10)
       {
        // Construct a decimation algorithm object from the FactorGraph fg
        // using the parameters specified by opts and three additional properties,
        // specifying that the decimation algorithm should use the max-product
        // algorithm and should completely reinitalize its state at every step
      
       	 DAIAlgFG* ptrdecmap;

        if(typeinfmethod==3)    	    
	 ptrdecmap = new DecMAP(fg, opts("reinit",true)("ianame",string("BP"))("iaopts",string("[damping=0.1,inference=MAXPROD,logdomain=0,maxiter=1000,tol=1e-9,updates=SEQRND,verbose=0]")) );       
	else if(typeinfmethod==4)
	 ptrdecmap = new  TRWBP(fg, opts("updates",string("SEQRND"))("logdomain",false)("inference",string("MAXPROD"))("damping",string("0.1")));
	else if(typeinfmethod==5)
	  ptrdecmap = new FBP(fg, opts("updates",string("SEQRND"))("logdomain",false)("inference",string("MAXPROD"))("damping",string("0.1")));
        else if(typeinfmethod==6)
        ptrdecmap = new DecMAP(fg, opts("reinit",true)("ianame",string("FBP"))("iaopts",string("[damping=0.1,inference=MAXPROD,logdomain=0,maxiter=1000,tol=1e-9,updates=SEQRND,verbose=0]")) );
        else if(typeinfmethod==7)
        ptrdecmap = new DecMAP(fg, opts("reinit",true)("ianame",string("TRWBP"))("iaopts",string("[damping=0.1,inference=MAXPROD,logdomain=0,maxiter=1000,tol=1e-9,updates=SEQRND,verbose=0]")) );
          
        ptrdecmap->init();
        ptrdecmap->run();
        vector<size_t> decmapstate = ptrdecmap->findMaximum(); 
         // Report DecMAP joint state
        cout << "Approximate DecMAP state (log score = " << fg.logScore( decmapstate ) << "):" << endl;
         for( size_t i = 0; i < decmapstate.size(); i++ ) bestconf[i] = decmapstate[i];
	 //for( size_t i = 0; i < decmapstate.size(); i++ )
	 //  cout << decmapstate[i] <<" ";
	//cout << fg.var(i) << ": " << decmapstate[i] << endl;
        //cout<<endl;
	// delete ptrdecmap;
for( size_t i = 0; i < decmapstate.size(); i++ ) cout<< bestconf[i] << " ";
 cout<<endl;
       }
	else if(typeinfmethod>10) //HAK implementation
       {            
	 //DAIAlgRG* ptrdecmap;
         //if(typeinfmethod==11)
	 //  ptrdecmap = new HAK(fg,opts("clusters",string("BETHE"))("doubleloop",true)("InitType",string("UNIFORM"))("inference",string("MAXPROD")));
	 // ptrdecmap->init();
	 //ptrdecmap->run();
	 // vector<size_t> decmapstate = ptrdecmap->findMaximum(); 
         // for( size_t i = 0; i < decmapstate.size(); i++ ) bestconf[i] = decmapstate[i];
 
       }

	
           
   return 0;
}
*/
