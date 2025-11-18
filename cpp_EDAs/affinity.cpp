#include <cstring> 
#include "auxfunc.h"
#include "affinity.h"
   
using namespace std;


/* C-implementation of the affinity propagation clustering algorithm. See */
/* BJ Frey and D Dueck, Science 315, 972-976, Feb 16, 2007, for a         */
/* description of the algorithm.                                          */
/*                                                                        */
/* Copyright 2007, BJ Frey and Delbert Dueck. This software may be freely */
/* used and distributed for non-commercial purposes.                      */


// n: Number of points
// m: Pairs of points for which there is a distance
// lam: Damping coefficient
// maxits: Number of iterations
// convits: Error before convergence


//AffPropagation::AffPropagation(){};


AffPropagation::AffPropagation(unsigned long length)
{
  Abs_n = length;  // Dimension of the matrix
} 


int AffPropagation::affinity_prop_clust(double lam, int maxits, int convits, unsigned long n, unsigned long m, unsigned long** points, double* pref, double* sim, unsigned long *idx, unsigned long *n_affclusters)
{
  int flag, dn, it, conv, decit;
  unsigned long i1, i2, j, *i, *k, l, o, **dec, *decsum,  K;
  double tmp, *s, *a, *r, *p, *mx1, *mx2, *srp, netsim, dpsim, expref;
  unsigned long** index;

  //lam=0.5; maxits=500; convits=50;

  if (MINDOUBLE==0.0) {
	  printf("There are numerical precision problems on this architecture.  Please recompile after adjusting MIN_DOUBLE and MAX_DOUBLE\n\n");
  }


  /* Allocate memory for similarities, preferences, messages, etc */
  i=(unsigned long *)calloc(m+n,sizeof(unsigned long));
  k=(unsigned long *)calloc(m+n,sizeof(unsigned long));
  s=(double *)calloc(m+n,sizeof(double));
  a=(double *)calloc(m+n,sizeof(double));
  r=(double *)calloc(m+n,sizeof(double));
  mx1=(double *)calloc(n,sizeof(double));
  mx2=(double *)calloc(n,sizeof(double));
  srp=(double *)calloc(n,sizeof(double));
  dec=(unsigned long **)calloc(convits,sizeof(unsigned long *));
  for(j=0;j<convits;j++)
    dec[j]=(unsigned long *)calloc(n,sizeof(unsigned long));
  decsum=(unsigned long *)calloc(n,sizeof(unsigned long));
  //idx=(unsigned long *)calloc(n,sizeof(unsigned long));


  index = new unsigned long* [n];
  for(j=0;j<n;j++)   
   {
      index[j] = new unsigned long[n];
      memset(index[j], 0, n*sizeof(unsigned long));
   }

  //cout<<" Reading similarities and preferences "<<endl;

  /* Read similarities and preferences */
    for(j=0;j<m;j++){
      i[j] = points[0][j];  k[j] = points[1][j]; s[j] =sim[j]; 
      index[points[0][j]][points[1][j]] = j+1;        
      //cout<<j<<" "<<points[0][j]<<" "<<points[1][j]<<" "<<sim[j]<<endl; 
    }
 
 /* Read preferences */
  for(j=0;j<n;j++){
    i[m+j]=j; k[m+j]=j;
    s[m+j] = pref[j];
    //cout<<j<<" "<<pref[j]<<endl; 
   }
 
 
  m=m+n;

  //cout<<" Finished similarities and preferences "<<endl;

  /* Include a tiny amount of noise in similarities to avoid degeneracies */
  // for(j=0;j<m;j++) s[j]=s[j]+(1e-16*s[j]+MINDOUBLE*100)*(rand()/((double)RAND_MAX+1));

  
  //     for(j=0;j<m;j++){
  //    cout<<j<<" "<<i[j]<<" "<<k[j]<<" "<<s[j]<<endl; 
  //  }
  

  // cout<<" Initializing availabilities "<<endl;

  /* Initialize availabilities to 0 and run affinity propagation */
  for(j=0;j<m;j++) a[j]=0.0;
  for(j=0;j<convits;j++) for(i1=0;i1<n;i1++) dec[j][i1]=0;
  for(j=0;j<n;j++) decsum[j]=0;
  dn=0; it=0; decit=convits;
  while(dn==0){
    it++; /* Increase iteration index */

    //  cout<<" Computing responsibilities "<<endl;

    /* Compute responsibilities */
    for(j=0;j<n;j++){ mx1[j]=-MAXDOUBLE; mx2[j]=-MAXDOUBLE; }
    for(j=0;j<m;j++){
      tmp=a[j]+s[j];
      if(tmp>mx1[i[j]]){
	mx2[i[j]]=mx1[i[j]]; //Maximum of all availabilities that go to i[j] 
	mx1[i[j]]=tmp;
      } else if(tmp>mx2[i[j]]) mx2[i[j]]=tmp; 
    }
    for(j=0;j<m;j++){
      tmp=a[j]+s[j];
      if(tmp==mx1[i[j]]) r[j]=lam*r[j]+(1-lam)*(s[j]-mx2[i[j]]);
      else r[j]=lam*r[j]+(1-lam)*(s[j]-mx1[i[j]]);
       cout<<j<<"  "<<i[j]<<" "<<k[j]<<"  "<<s[j]<<"  "<<r[j]<<"   "<<a[j]<<endl;
    }

    //cout<<" Computing availabilities "<<endl;

    /* Compute availabilities */
    for(j=0;j<n;j++) srp[j]=0.0;
    //for(j=0;j<m-n;j++) if(r[j]>0.0) srp[k[j]]=srp[k[j]]+r[j]; 
    // for(j=m-n;j<m;j++) srp[k[j]]=srp[k[j]]+r[j];


   for(j=0;j<m-n;j++) // All edges
   {
     tmp= 0;
     for(l=0;l<n;l++) //Possible values of k   
	{
         
          if( (l != i[j])  && (l != k[j]) )
            {
	        cout<<j<<"  "<<l<<"  "<<i[j]<<" "<<k[j]<<"  "<<s[j]<<"  "<<r[j]<<"   "<<a[j]<<"  "<<r[index[l][k[j]]-1]<<" "<<s[index[i[j]][l]-1]<<" "<<tmp<<endl;
              if(index[l][k[j]]>0 && index[i[j]][l]>0)
                {
                  if(( r[index[l][k[j]]-1] * s[index[i[j]][l]-1]) > 0)  tmp = tmp + r[index[l][k[j]]-1] * s[index[i[j]][l]-1];
                }                
             }     

        }
     tmp += s[m-n+k[j]]*r[m-n+k[j]];
     if( tmp < 0)  
          {
            a[j]=lam*a[j]+(1-lam)*tmp; 
          }
     else a[j]=lam*a[j];

     cout<<j<<"  "<<i[j]<<" "<<k[j]<<"  "<<s[j]<<"  "<<r[j]<<"   "<<a[j]<<"   "<<a[j]+r[j]<<"  "<<tmp<<endl;
  }  

   for(j=0;j<n;j++) // All edges
   {
     tmp= 0;
     for(l=0;l<n;l++) //Possible values of k   
	{         
          if( l != j )
            {
              if(index[l][j]>0)
                {
                  if(( r[index[l][j]-1] * s[index[l][j]-1]) > 0)  tmp = tmp + r[index[l][j]-1] * s[index[l][j]-1];
                }                
             }     
        }
     if( tmp > 0)  
          {
            a[m-n+j]=lam*a[m-n+j]+(1-lam)*tmp; 
          }
     else a[m-n+j]=lam*a[m-n+j];
        cout<<j<<"  "<<i[m-n+j]<<" "<<k[m-n+j]<<"  "<<s[m-n+j]<<"  "<<r[m-n+j]<<"   "<<a[m-n+j]<<"   "<<a[m-n+j]+r[m-n+j]<<" "<<tmp<<endl;
   }  


   /*   
  for(j=0;j<m-n;j++) 
      {
       if(r[j]>0.0) srp[k[j]]=srp[k[j]]+r[j];  //CHANGE ROBERTO
	//cout<<j<<"  "<<i[j]<<" "<<k[j]<<"  "<<r[j]<<"  "<<srp[k[j]]<<"   "<<s[j]<<"  "<<s[k[j]]<<endl;
      }
   
    for(j=m-n;j<m;j++) 
      {
       srp[k[j]]=srp[k[j]]+r[j];               //CHANGE ROBERTO
       //cout<<j<<"  "<<i[j]<<" "<<k[j]<<"  "<<r[j]<<"  "<<srp[k[j]]<<endl;
      }
   
    for(j=0;j<m-n;j++){
      // if(r[j]>0.0) tmp=srp[k[j]]-r[j]; else tmp=srp[k[j]];
      if(r[j]>0.0) tmp=srp[k[j]]-r[j]; else tmp=srp[j];  //CHANGE ROBERTO
      //cout<<j<<"  "<<i[j]<<" "<<k[j]<<"  "<<r[j]<<"  "<<srp[k[j]]<<"   "<<s[j]<<"  "<<s[k[j]]<<endl;     
      if(tmp<0.0) a[j]=lam*a[j]+(1-lam)*tmp; else a[j]=lam*a[j];
    }
        //for(j=m-n;j<m;j++) a[j]=lam*a[j]+(1-lam)*(srp[k[j]]-r[j]);
     for(j=m-n;j<m;j++) a[j]=lam*a[j]+(1-lam)*(srp[k[j]]-r[j]);   //CHANGE ROBERTO
   
   */



    //cout<<" Identifying exemplars "<<endl;

    /* Identify exemplars and check to see if finished */
    decit++; if(decit>=convits) decit=0;
    for(j=0;j<n;j++) decsum[j]=decsum[j]-dec[decit][j];
    for(j=0;j<n;j++)
      if(a[m-n+j]+r[m-n+j]>=0.0) dec[decit][j]=1; else dec[decit][j]=0;
    //if(a[m-n+j]+r[m-n+j]>0.0) dec[decit][j]=1; else dec[decit][j]=0;
 
   K=0; for(j=0;j<n;j++) K=K+dec[decit][j];
    for(j=0;j<n;j++) decsum[j]=decsum[j]+dec[decit][j];
    for(j=0;j<n;j++) printf("%lu ",decsum[j]);
    cout<<endl;
    if((it>=convits)||(it>=maxits)){
      /* Check convergence */
      conv=1; for(j=0;j<n;j++) if((decsum[j]!=0)&&(decsum[j]!=convits)) conv=0;
    
      /* Check to see if done */
      if(((conv==1)&&(K>0))||(it==maxits)) dn=1;
    }

    cout<<" Iter "<<it<<" K  "<<K<<" conv  "<<conv<<endl;
  }
 
   /* If clusters were identified, find the assignments and output them */
  if(K>0){
    for(j=0;j<m;j++)
      if(dec[decit][k[j]]==1) a[j]=0.0; else a[j]=-MAXDOUBLE;
    for(j=0;j<n;j++) mx1[j]=-MAXDOUBLE;
    for(j=0;j<m;j++){
      tmp=a[j]+s[j];
      if(tmp>mx1[i[j]]){
	mx1[i[j]]=tmp;
	idx[i[j]]=k[j];
      }
      
    }

    //for(j=0;j<m;j++) printf("%lu %lu %lu %e\n",j, i[j], k[j], mx1[i[j]]);
    
    //for(j=0;j<m;j++) printf("%f ",a[j]+s[j]);
    //cout<<endl;   

    for(j=0;j<n;j++) if(idx[j]>=n)
       {
         // idx[j]=j; //Roberto
         dec[decit][j] = 1;
       }

    for(j=0;j<n;j++) if(dec[decit][j]) idx[j]=j;
    for(j=0;j<n;j++) srp[j]=0.0;
    for(j=0;j<m;j++) if(idx[i[j]]==idx[k[j]]) srp[k[j]]=srp[k[j]]+s[j];
    for(j=0;j<n;j++) mx1[j]=-MAXDOUBLE;
    for(j=0;j<n;j++) if(srp[j]>mx1[idx[j]]) mx1[idx[j]]=srp[j];
    for(j=0;j<n;j++)
      if(srp[j]==mx1[idx[j]]) dec[decit][j]=1; else dec[decit][j]=0;
    for(j=0;j<m;j++)
      if(dec[decit][k[j]]==1) a[j]=0.0; else a[j]=-MAXDOUBLE;
    for(j=0;j<n;j++) mx1[j]=-MAXDOUBLE;
    for(j=0;j<m;j++){
      tmp=a[j]+s[j];
      if(tmp>mx1[i[j]]){
	mx1[i[j]]=tmp;
	idx[i[j]]=k[j];
      }
    }
    
    for(j=0;j<n;j++) if(dec[decit][j]) idx[j]=j;
  
   

    dpsim=0.0; expref=0.0;
    for(j=0;j<m;j++){
      if(idx[i[j]]==k[j]){
	if(i[j]==k[j]) expref=expref+s[j];
	else dpsim=dpsim+s[j];
      }
    }
    netsim=dpsim+expref;


    //Vectors are checked for consistency ROBERTO 

     for(j=0;j<n;j++) 
       if(idx[idx[j]] != idx[j])  
	 {
           for(l=0;l<n;l++) if(idx[l] == idx[j]) idx[l] = j;               
         }

 

    (*n_affclusters) = K;
  
 
 
    
    printf("\nNumber of identified clusters: %d\n",K);
    printf("Fitness (net similarity): %f\n",netsim);
    printf("  Similarities of data points to exemplars: %f\n",dpsim);
    printf("  Preferences of selected exemplars: %f\n",expref);
    printf("Number of iterations: %d\n\n",it);
    
  }
    else printf("\nDid not identify any clusters\n");

    
freememory:
	free(i);
	free(k);
	free(s);
	free(a);
	free(r);
	free(mx1);
	free(mx2);
	free(srp);
	for(j=0;j<convits;j++) free(dec[j]); free(dec);
	free(decsum);
	//free(idx);
        for(j=0;j<n;i++)   delete[] index[j];
        cout<<"Pass 1"<<endl;
        delete[] index;
        cout<<"Pass 2"<<endl;

	return conv;

}


int AffPropagation::affinity_prop(double lam, int maxits, int convits, unsigned long n, unsigned long m, unsigned long** points, double* pref, double* sim, unsigned long *idx, unsigned long *n_affclusters)
{
  int flag, dn, it, conv, decit;
  unsigned long i1, i2, j, *i, *k, l, **dec, *decsum,  K;
  double tmp, *s, *a, *r, *p, *mx1, *mx2, *srp, netsim, dpsim, expref;
  
  //lam=0.5; maxits=500; convits=50;

  if (MINDOUBLE==0.0) {
	  printf("There are numerical precision problems on this architecture.  Please recompile after adjusting MIN_DOUBLE and MAX_DOUBLE\n\n");
  }


  /* Allocate memory for similarities, preferences, messages, etc */
  i=(unsigned long *)calloc(m+n,sizeof(unsigned long));
  k=(unsigned long *)calloc(m+n,sizeof(unsigned long));
  s=(double *)calloc(m+n,sizeof(double));
  a=(double *)calloc(m+n,sizeof(double));
  r=(double *)calloc(m+n,sizeof(double));
  mx1=(double *)calloc(n,sizeof(double));
  mx2=(double *)calloc(n,sizeof(double));
  srp=(double *)calloc(n,sizeof(double));
  dec=(unsigned long **)calloc(convits,sizeof(unsigned long *));
  for(j=0;j<convits;j++)
    dec[j]=(unsigned long *)calloc(n,sizeof(unsigned long));
  decsum=(unsigned long *)calloc(n,sizeof(unsigned long));
  //idx=(unsigned long *)calloc(n,sizeof(unsigned long));


  //cout<<" Reading similarities and preferences "<<endl;

  /* Read similarities and preferences */
    for(j=0;j<m;j++){
      i[j] = points[0][j];  k[j] = points[1][j]; s[j] =sim[j]; 
      //cout<<j<<" "<<points[0][j]<<" "<<points[1][j]<<" "<<sim[j]<<endl; 
    }
 
 /* Read preferences */
  for(j=0;j<n;j++){
    i[m+j]=j; k[m+j]=j;
    s[m+j] = pref[j];
    //cout<<j<<" "<<pref[j]<<endl; 
   }
 
 
  m=m+n;

  //cout<<" Finished similarities and preferences "<<endl;

  /* Include a tiny amount of noise in similarities to avoid degeneracies */
     for(j=0;j<m;j++) s[j]=s[j]+(1e-16*s[j]+MINDOUBLE*100)*(rand()/((double)RAND_MAX+1));

  
  //     for(j=0;j<m;j++){
  //    cout<<j<<" "<<i[j]<<" "<<k[j]<<" "<<s[j]<<endl; 
  //  }
  

  // cout<<" Initializing availabilities "<<endl;

  /* Initialize availabilities to 0 and run affinity propagation */
  for(j=0;j<m;j++) a[j]=0.0;
  for(j=0;j<convits;j++) for(i1=0;i1<n;i1++) dec[j][i1]=0;
  for(j=0;j<n;j++) decsum[j]=0;
  dn=0; it=0; decit=convits;
  while(dn==0){
    it++; /* Increase iteration index */

    //  cout<<" Computing responsibilities "<<endl;

    /* Compute responsibilities */
    for(j=0;j<n;j++){ mx1[j]=-MAXDOUBLE; mx2[j]=-MAXDOUBLE; }
    for(j=0;j<m;j++){
      tmp=a[j]+s[j];
      if(tmp>mx1[i[j]]){
	mx2[i[j]]=mx1[i[j]];
	mx1[i[j]]=tmp;
      } else if(tmp>mx2[i[j]]) mx2[i[j]]=tmp;
    }
    for(j=0;j<m;j++){
      tmp=a[j]+s[j];
      if(tmp==mx1[i[j]]) r[j]=lam*r[j]+(1-lam)*(s[j]-mx2[i[j]]);
      else r[j]=lam*r[j]+(1-lam)*(s[j]-mx1[i[j]]);
    }

    //cout<<" Computing availabilities "<<endl;

    /* Compute availabilities */
    for(j=0;j<n;j++) srp[j]=0.0;
    for(j=0;j<m-n;j++) if(r[j]>0.0) srp[k[j]]=srp[k[j]]+r[j]; 
    for(j=m-n;j<m;j++) srp[k[j]]=srp[k[j]]+r[j];
   

    for(j=0;j<m-n;j++){
      if(r[j]>0.0) tmp=srp[k[j]]-r[j]; else tmp=srp[k[j]];
      if(tmp<0.0) a[j]=lam*a[j]+(1-lam)*tmp; else a[j]=lam*a[j];
    }
     for(j=m-n;j<m;j++) a[j]=lam*a[j]+(1-lam)*(srp[k[j]]-r[j]);
    

    //cout<<" Identifying exemplars "<<endl;

    /* Identify exemplars and check to see if finished */
    decit++; if(decit>=convits) decit=0;
    for(j=0;j<n;j++) decsum[j]=decsum[j]-dec[decit][j];
    for(j=0;j<n;j++)
      if(a[m-n+j]+r[m-n+j]>0.0) dec[decit][j]=1; else dec[decit][j]=0;
    K=0; for(j=0;j<n;j++) K=K+dec[decit][j];
    for(j=0;j<n;j++) decsum[j]=decsum[j]+dec[decit][j];
    //for(j=0;j<n;j++) printf("%lu ",decsum[j]);
    //cout<<endl;
    if((it>=convits)||(it>=maxits)){
      /* Check convergence */
      conv=1; for(j=0;j<n;j++) if((decsum[j]!=0)&&(decsum[j]!=convits)) conv=0;
    
      /* Check to see if done */
      if(((conv==1)&&(K>0))||(it==maxits)) dn=1;
    }

    //    cout<<" Iter "<<it<<" K  "<<K<<" conv  "<<conv<<endl;
  }
 
   /* If clusters were identified, find the assignments and output them */
  if(K>0){
    for(j=0;j<m;j++)
      if(dec[decit][k[j]]==1) a[j]=0.0; else a[j]=-MAXDOUBLE;
    for(j=0;j<n;j++) mx1[j]=-MAXDOUBLE;
    for(j=0;j<m;j++){
      tmp=a[j]+s[j];
      if(tmp>mx1[i[j]]){
	mx1[i[j]]=tmp;
	idx[i[j]]=k[j];
      }
      
    }

    // for(j=0;j<m;j++) printf("%lu %lu %lu %e\n",j, i[j], k[j], mx1[i[j]]);
    
    // for(j=0;j<n;j++) printf("%lu ",idx[j]);
    //cout<<endl;   

    for(j=0;j<n;j++)
    if(idx[j]>=n)
     {
       //cout<<j<<"----"<<idx[j]<<endl;   
      // idx[j]=j; //Roberto
       dec[decit][j] = 1;
     }


    for(j=0;j<n;j++) if(dec[decit][j]) idx[j]=j;
    for(j=0;j<n;j++) srp[j]=0.0;
    for(j=0;j<m;j++) if(idx[i[j]]==idx[k[j]]) srp[k[j]]=srp[k[j]]+s[j];
    for(j=0;j<n;j++) mx1[j]=-MAXDOUBLE;
    for(j=0;j<n;j++) if(srp[j]>mx1[idx[j]]) mx1[idx[j]]=srp[j];
    for(j=0;j<n;j++)
      if(srp[j]==mx1[idx[j]]) dec[decit][j]=1; else dec[decit][j]=0;
    for(j=0;j<m;j++)
      if(dec[decit][k[j]]==1) a[j]=0.0; else a[j]=-MAXDOUBLE;
    for(j=0;j<n;j++) mx1[j]=-MAXDOUBLE;
    for(j=0;j<m;j++){
      tmp=a[j]+s[j];
      if(tmp>mx1[i[j]]){
	mx1[i[j]]=tmp;
	idx[i[j]]=k[j];
      }
    }
    
    for(j=0;j<n;j++) if(dec[decit][j]) idx[j]=j;
  

    dpsim=0.0; expref=0.0;
    for(j=0;j<m;j++){
      if(idx[i[j]]==k[j]){
	if(i[j]==k[j]) expref=expref+s[j];
	else dpsim=dpsim+s[j];
      }
    }
    netsim=dpsim+expref;


    //  printf("\nNumber of identified clusters BEFORE: %d\n",K);
    //for(j=0;j<n;j++) printf("%lu ",idx[j]);
    //cout<<endl;   


    //Vectors are checked for consistency ROBERTO 
    // The value of K is recalculated
    K=0;     
     for(j=0;j<n;j++) 
       {
        if(idx[idx[j]] != idx[j])  
	  {
            for(l=0;l<n;l++) if(idx[l] == idx[j]) idx[l] = j;               
          }
        if(idx[j] == j) K++;
       } 
 

    (*n_affclusters) = K;


    //printf("\nNumber of identified clusters: %d\n",K);
    //for(j=0;j<n;j++) printf("%lu ",idx[j]);
    //cout<<endl;  


// At this point the cluster solution is improved by exchaging elements

 if(K>1)
   {   

    

double** meanclusters;
unsigned long* elements;
unsigned long* exemplars;
unsigned long* clusters;
double maxdist;

unsigned long o,first,current,bestcluster;
int init;

 int a=0;
while( a<3 && K>1 && K!=n)
       {
         //cout<<"Pass "<<a<<endl;
	 //printf("\nNumber of identified clusters: %d\n",K);
	 //for(j=0;j<n;j++) printf("%lu ",idx[j]);
	 //cout<<endl;  


 meanclusters = new double* [K];  // Mean distance of every point to every cluster
  for(j=0;j<K;j++) 
   {
    meanclusters[j] = new double[n];  
    memset(meanclusters[j], 0, sizeof(double)*n);  
   }
 
 clusters = new unsigned long[n];
 elements = new unsigned long[K]; // Number of elements in every cluster
 exemplars = new unsigned long[K]; // Number of elements in every cluster


 for(j=0;j<K;j++) 
 first = 0;
 l = 0;
 init = 0;
 current = 0;

 while(l<n)
  { 
    if(init==0) 
     {
      l = 0;
      init =1;
     }
    else l=first+1;
    while(l<n && idx[l] != l) l++;  //Exemplars are identified
    if(l<n)
     {  
      first = l;
      exemplars[current] = first;  // The exemplar corresponding to each cluster is saved
      elements[current] = 0; 
      for(j=0;j<n;j++)
      { 
       if(idx[j] == first)
         {    
             elements[current]++;  // Points in each cluster are counted
             clusters[j] = current;  // The index of all points of the cluster are saved                
         }  
       }      
       current++;
      }
    } 

 /* 
 cout<<"Ordering of the clusters "<<endl;
 for(j=0;j<n;j++) printf("%lu ",clusters[j]);
 cout<<endl;    
 cout<<"Exemplar of each cluster "<<endl;
 for(j=0;j<K;j++) printf("%lu ",exemplars[j]);
 cout<<endl;   
 */
    for(l=0;l<m-n;l++)
       { 
        meanclusters[clusters[i[l]]][k[l]] += s[l]; // Distance from cluster[i[l]] to each point
        meanclusters[clusters[k[l]]][i[l]] += s[l];
       }

 
    for(j=0;j<K;j++)
    {
     for(l=0;l<n;l++)   
       if(elements[j]>0)
            {
              meanclusters[j][l] /= (elements[j]); //Distances are normalized
              //cout<<meanclusters[j][l]<<" ";
            }
     //else cout<<"0 ";
     //cout<<endl;
    }
      
     for(l=0;l<n;l++)
      {
         maxdist = -MAXDOUBLE;
         for(j=0;j<K;j++)
         {  
           if(maxdist < meanclusters[j][l] && elements[j]>1) 
            { 
             maxdist = meanclusters[j][l];
             bestcluster = j;
            }
         }
	 //cout<<" var "<<l<<" bestdist "<<maxdist<<" current cluster "<<clusters[l]<<" to cluster "<<bestcluster<<endl; 
	  //   cout<<" var "<<l<<" bestdist "<<maxdist<<" current idx  "<<idx[l]<<" to idx "<<exemplars[bestcluster]<<endl; 
         if(idx[l]==l  && bestcluster!=clusters[l])  
            {
              for(o=0;o<K;o++) if(exemplars[o]==exemplars[clusters[l]])    exemplars[o] = exemplars[bestcluster]; 
		
	      // cout<<"Now exemplar of "<<l<<" is "<<exemplars[bestcluster]<<endl;
             for(o=0;o<n;o++)
		{
                  if(idx[o]==l)
                    {
                      //clusters[o] = bestcluster;
                      idx[o] = exemplars[bestcluster];
                      //clusters[o] = bestcluster;
                     }
                }
	     /*
             for(o=0;o<n;o++) printf("%lu ",idx[o]);
             cout<<endl;  
             for(o=0;o<n;o++) printf("%lu ",clusters[o]);
             cout<<endl; 
             for(o=0;o<K;o++) printf("%lu ",exemplars[o]);
             cout<<endl;
             */ 
            }
         idx[l] = exemplars[bestcluster]; //Point is reassigned to the cluster with shortest distance     
         //clusters[l] = bestcluster;   
      }
 
     //for(j=0;j<n;j++) printf("%lu ",idx[j]);
     //cout<<endl;  

  for(j=0;j<K;j++) delete[] meanclusters[j];
  delete[] meanclusters;

  K=0;     
  for(j=0;j<n;j++)  if(idx[j] == j) K++;
  

  delete[] clusters;
  delete[] exemplars;
  delete[] elements;

  a++;

     }
   }
 
   
    /*
    printf("\nNumber of identified clusters: %d\n",K);
    printf("Fitness (net similarity): %f\n",netsim);
    printf("  Similarities of data points to exemplars: %f\n",dpsim);
    printf("  Preferences of selected exemplars: %f\n",expref);
    printf("Number of iterations: %d\n\n",it);
    */
  }
  //  else printf("\nDid not identify any clusters\n");

   (*n_affclusters) = K;
freememory:
	free(i);
	free(k);
	free(s);
	free(a);
	free(r);
	free(mx1);
	free(mx2);
	free(srp);
	for(j=0;j<convits;j++) free(dec[j]); free(dec);
	free(decsum);
	//free(idx);

	return conv;

} 





// Find the connected components of the similarity matrix
// Those components with less than three elements are considered as single cliques. 


unsigned long AffPropagation::FindConnComponents( unsigned long n, unsigned long* allvars, unsigned long** conn_components, int maxsize, double SimThreshold)
{

 unsigned long *components;
 unsigned long i,j,a,ni,aux,nconn,node,auxj;
 int Finish;
 //unsigned long** conn_components;

  components = new unsigned long[n];
  
  for(i=0;i<n;i++)  components[i] = i;

nconn = 0;

// cout<<"Number of vars "<<n<<endl;

for(i=0;i<n;i++)  
 {    
   if(components[i] != n) // n is the value taken when it is already occupied.
     {
       nconn++;
       a = 1;
       conn_components[nconn-1] = new unsigned long[n+1];
       ni = 1;
       conn_components[nconn-1][ni] = allvars[i]; 
       components[i] = n;    //Here n is taken a the value that indicates already occupied
       Finish = 0;
       while(!Finish)
	 {
	   node = conn_components[nconn-1][a];
           for(auxj=0;auxj<n;auxj++)  
            {   
	       j = allvars[auxj];         //Absolute reference with respect to the matrix 
               if(j<node) aux = j*(2*Abs_n-j+1)/2 +node-2*j-1;
               else    aux = node*(2*Abs_n-node+1)/2 +j-2*node-1;
              
               if(Matrix[aux] > SimThreshold && components[auxj]!=n)    
                {
		  components[auxj] = n;   //Here n is taken a the value that indicates already occupied
                   ni++;
                   conn_components[nconn-1][ni] = j; 
	        }             
             }
	   a++;
           Finish = (a>ni);
	 }
       conn_components[nconn-1][0] = ni; 
      } 
 }
 
/*
for(i=0;i<nconn;i++)  
  { 
    for(j=0;j<conn_components[i][0]+1;j++)  cout<<conn_components[i][j]<<" "; 
    cout<<endl; 
 }

*/


 delete[] components;
 return nconn;
 
} 
  




// Call Affinity Propagation in every single connected component of the MI 
// Those components with less than three elements are considered as 
// factors of the factorization (i.e. No affinity is called for them) . 



unsigned long  AffPropagation::CallAffinity(double lam, int maxits, int convits, unsigned long n, unsigned long* allvars, int deph, unsigned long** listclusters, int maxsize, double SimThreshold, unsigned long* nclust, int* ncalls)
{

  unsigned long i,j,k,aux,m,remaining,nconn,clustsize;
  unsigned long** points;
  double meansim;
  double *pref, *sim,MatrixEntriesValues;
  unsigned long* idx;
  unsigned long* posvars; 
  int conv; 
  unsigned long** conn_components;
  int AllMatrixEntriesEqual,Finish;
  int* auxperm;


  posvars = new unsigned long[n];
  idx     = new unsigned long[n];


 
  conn_components = new unsigned long*[n];
  nconn = FindConnComponents(n,allvars,conn_components,maxsize,SimThreshold);


for(i=0;i<nconn;i++)
 {
   clustsize = conn_components[i][0];   
    // Check whether all the values in the similarities are equal

   //cout<<"Component "<<i<<" component  size "<<clustsize<<endl;
    if(clustsize<=maxsize && clustsize<=3) 
    {   
  	 (*nclust)++;
         remaining--;
         listclusters[(*nclust)-1] = new unsigned long[clustsize+1];
         for(j=1;j<clustsize+1;j++)  listclusters[(*nclust)-1][j] =conn_components[i][j];
         listclusters[(*nclust)-1][0] = clustsize;   
    }  
    else
    {   
     for(j=0;j<clustsize;j++) posvars[j] = conn_components[i][j+1];
     j=0;
     AllMatrixEntriesEqual = 1;
     MatrixEntriesValues = -1;
     Finish = 0;
     while(AllMatrixEntriesEqual && j<clustsize) // It stops if there are two equal values in the matrix (apart from zero)
      {
       k = j+1;
       while(AllMatrixEntriesEqual && k<clustsize)
        {
    	  if(posvars[j]<posvars[k]) aux = posvars[j]*(2*Abs_n-posvars[j]+1)/2 +posvars[k]-2*posvars[j]-1;
          else   aux = posvars[k]*(2*Abs_n-posvars[k]+1)/2 +posvars[j]-2*posvars[k]-1;
	  // cout << Matrix[aux]<<" ";
           if(MatrixEntriesValues == -1 && Matrix[aux] != 0) MatrixEntriesValues = Matrix[aux] ;  
          if(MatrixEntriesValues != -1 || Matrix[aux] != 0) AllMatrixEntriesEqual =  (MatrixEntriesValues == Matrix[aux]);      
          k++; 
        }
       j++;
      } 
     //cout<<endl;
     //cout<<"Are all the entries of the matrix equal? "<<AllMatrixEntriesEqual<<" "<<clustsize<<endl;

      if(AllMatrixEntriesEqual)
       {
        RandomPerm(clustsize,2*clustsize,posvars); 
        for(k=0;k<clustsize;k+=(maxsize-1))
        {
         (*nclust)++;
         if(clustsize-k<(maxsize-1))
	   {
              listclusters[(*nclust)-1] = new unsigned long[clustsize-k+1];
              listclusters[(*nclust)-1][0] = clustsize-k;              //Number of variables in the cluster
              for(j=k;j<clustsize;j++)  listclusters[(*nclust)-1][j-k+1] = posvars[j];      // Variables j       
           }
         else 
	   {
             listclusters[(*nclust)-1] = new unsigned long[maxsize];
             listclusters[(*nclust)-1][0] = maxsize-1;               //Number of variables in the cluster
             for(j=k;j<k+(maxsize-1);j++)  listclusters[(*nclust)-1][j-k+1] = posvars[j];      // Variables j       
           }    
         }
        
       }
    else //Construct similarity matrix and call affinity
    { 
     m = 0;
     meansim = 0;    
     // The list of similarity entries is constructed from the reduced matrix

     unsigned long dimmatrix = clustsize*clustsize;
      points = new unsigned long*[2];
      points[0] = new unsigned long[dimmatrix];  //This is the maximal number of entries it can have
      points[1] = new unsigned long[dimmatrix];
      sim =  new double[dimmatrix];
      pref =  new double[clustsize];
     unsigned long* allnpoints = new unsigned long[clustsize];
       
         for(j=0;j<clustsize;j++) 
	   {
             allnpoints[j] = 0; pref[j] = 0;
	   }

      for(j=0;j<clustsize;j++)         //Only entries that include the clustsize variables
       {
         for(k=0;k<clustsize;k++)
          {
	   if(posvars[k] != posvars[j])
	    {
       	      if(posvars[j]<posvars[k]) aux = posvars[j]*(2*Abs_n-posvars[j]+1)/2 +posvars[k]-2*posvars[j]-1;
              else             aux = posvars[k]*(2*Abs_n-posvars[k]+1)/2 +posvars[j]-2*posvars[k]-1;
              if(Matrix[aux] > SimThreshold)  
               {
	        points[0][m] = j;
                points[1][m] = k;                
                sim[m] = Matrix[aux];               //There is a link between points
                pref[j] += sim[m];
                pref[k] += sim[m];
                allnpoints[k]++;
                allnpoints[j]++;
                meansim += sim[m];
                m++;
	       }
	     }
           }  
        }

      

      meansim = meansim/m;       //Average of the matrix entries 
      //cout<<"meansim "<<meansim<<endl;
 
       for(j=0;j<clustsize;j++)    pref[j] = meansim; // + myrand()/100.0; 

       /*
       for(j=0;j<clustsize;j++) 
       {
	 if(allnpoints[j]>0) pref[j] =  (pref[j])/(3*allnpoints[j]);
				 else pref[j] = 0;
	 cout<<pref[j]<<" ";
       }
       cout<<endl;
       */
       // for(j=0;j<clustsize;j++) cout<<posvars[j]<<"  ";
       // cout<<endl;
      //cout<<"lam "<<lam<<" maxits "<<maxits<<" convits "<<convits<<" clustsize "<<clustsize<<" m "<<m<<" deph "<<deph<<" maxsize "<<maxsize<<endl;
      // Constrained affinity propagation is called

           
      conv =   constr_affinity_prop(lam, maxits, convits, clustsize, m, points, pref, sim, idx, ncalls,  deph, posvars, listclusters, nclust, maxsize,SimThreshold);
  
    
     delete[] points[0];    
     delete[] points[1];         
     delete[] points;        
     delete[] sim;    
     delete[] allnpoints; 
     delete[] pref;
         

 
    }
  }
 }
  delete[] posvars;
  delete[] idx;
  for(i=0;i<nconn;i++) delete[] conn_components[i];
  delete[] conn_components;

return (*nclust);
  
 }




int AffPropagation::constr_affinity_prop(double lam, int maxits, int convits, unsigned long n, unsigned long m, unsigned long** points, double* pref, double* sim, unsigned long *idx, int* ncalls, int deph, unsigned long *posvars, unsigned long **listclusters, unsigned long *nclust, int maxsize, double SimThreshold)
{
  
  unsigned long i,j,auxconv,nremaining,ent_count,first,nmembers,reduced_m;
  unsigned long  *auxmembers, *remainingvars, *newposvars ;
  double auxlam;
  int conv,Finish;
  unsigned long n_affclusters;
  int Pass;
 

  i = 0;
  auxconv = 0;

 n_affclusters = 0;

 while( (!auxconv || n_affclusters<=1) && i<10) //While not converged increase the damping coefficient
  {
    //auxlam = 0.5 + 0.05*i;
    auxlam = 0.90 + 0.01*i;
    auxconv = affinity_prop(auxlam, maxits, convits,  n,  m, points, pref, sim, idx, &n_affclusters);
    //cout<<" Number or clusters "<<n_affclusters<<"  "<<i<<" auxconv "<<auxconv<<endl;
    i++;    
  }


 (*ncalls)++;

 
 if( (n_affclusters<=1))  // && (*ncalls>=deph)  //If not converged or deph reached  add each point as a single cluster
   {
    
     RandomPerm(n,2*n,posvars); 
         for(i=0;i<n;i+=(maxsize-1))
       {
         (*nclust)++;
         if(n-i<(maxsize-1))
	   {
              listclusters[(*nclust)-1] = new unsigned long[n-i+1];
              listclusters[(*nclust)-1][0] = n-i;              //Number of variables in the cluster
              for(j=i;j<n;j++)  listclusters[(*nclust)-1][j-i+1] = posvars[j];      // Variables j       
           }
         else
	   {
             listclusters[(*nclust)-1] = new unsigned long[maxsize];
             listclusters[(*nclust)-1][0] = maxsize-1;               //Number of variables in the cluster
             for(j=i;j<i+(maxsize-1);j++)  listclusters[(*nclust)-1][j-i+1] = posvars[j];      // Variables j       
           }    
      }
    //  if (*ncalls>=deph) *ncalls = 0;
  
    return 0;
 }



  auxmembers = new unsigned long[n];
  remainingvars = new unsigned long[n];
  newposvars =  new unsigned long[n]; 

Pass = 0;

while(Pass<=1)
 {
 first = 0;
 Finish = 0;
 i = 0;
 int init = 0;
 nremaining = 0;
 
 while(!Finish && i<n)
  { 
   if(init==0) 
   {
      i = 0;
      init = 1;
   }
   else i=first+1;
   
   //cout<<"Identifying exemplars  i "<<i<<" first  "<<first<<" nremaining "<<nremaining<<endl;
   while(i<n && idx[i] != i) i++;  //Exemplars are identified
   if(i<n)
    {  
      first = i;
      nmembers = 0;   
      
   for(j=0;j<n;j++)  //Members of the cluster first are counted and identified
     {
       //cout<<first<<"----"<<idx[j]<<endl;
       if(idx[j]==first)
	 {
           auxmembers[nmembers] = j;  
	   nmembers++;
         }
     }
    
   if(nmembers< maxsize) //The cluster satisfies the constraint on size and it is added to list
     {
       (*nclust)++;
       // cout<<" Cluster "<<(*nclust)<<" "<<nmembers<<endl;   
 
       listclusters[(*nclust)-1] = new unsigned long[nmembers+1];
       listclusters[(*nclust)-1][0] = nmembers; //The first element is the number of points in the cluster
       for(j=0;j<nmembers;j++)  //The points are added
         {
          listclusters[(*nclust)-1][j+1] = posvars[auxmembers[j]];
	  // cout<<posvars[auxmembers[j]]<<" ";
          remainingvars[auxmembers[j]] = 0;
         }
       //cout<<endl;  
     } 
   else if(Pass==0)  //These vars will be joined to call again Affinity propagation  unless n == nremaining
     {
       nremaining = nremaining + nmembers;
       //cout<<" remaining "<<nremaining<<endl;
        for(j=0;j<nmembers;j++) 
         {
	   //cout<<posvars[auxmembers[j]]<<" ";
          remainingvars[auxmembers[j]] = 1;
         } 
      }
   else  // At the second Pass, all the clusters are recursively decomposed
      {
         for(j=0;j<nmembers;j++)  
         {
          newposvars[j] = posvars[auxmembers[j]];
	  //cout<<posvars[auxmembers[j]]<<" ";
          remainingvars[auxmembers[j]] = 0;
         } 
         CallAffinity(lam,maxits,convits,nmembers,newposvars,deph,listclusters,maxsize,SimThreshold,nclust,ncalls);   

      }
 
   }
   else  Finish = 1;
  }

  if(Pass==0  && nremaining==n) Pass = 1;
  else Pass=2; 
}
//cout<<endl;
// cout<<nremaining<<" variables have not been added "<<endl;
 //At this point clusters have been added to the list or marked as remaining            
 
 if(nremaining==0)  // The programs finishes, constrained clustering was successful
   {
     delete[] auxmembers;
     delete[] remainingvars;
     delete[] newposvars;
     return 1;
   }
 else //Construct reduced similarity matrix and find parameters
   {   
     //cout<<" Vars that were not clustered : "<<endl;
     
     ent_count = 0;
      for(i=0;i<n;i++) 
      { 
        if(remainingvars[i] == 1)
	  {
            newposvars[ent_count] = posvars[i];
            //cout<<posvars[i]<<" ";
            ent_count++;
          }                
      } 
      //cout<<endl;

      // Call the program recursively 
       CallAffinity(lam,maxits,convits,nremaining,newposvars,deph,listclusters,maxsize,SimThreshold,nclust,ncalls);     
    
      
   } 
 
      delete[] auxmembers;
      delete[] remainingvars;
      delete[] newposvars; 
   return conv; 
 
}


/*



int AffPropagation::constr_affinity_prop(double lam, int maxits, int convits, unsigned long n, unsigned long m, unsigned long** points, double* pref, double* sim, unsigned long *idx, int* ncalls, int deph, unsigned long *posvars, unsigned long **listclusters, unsigned long *nclust, int maxsize, double SimThreshold)
{
  
  unsigned long i,j,auxconv,nremaining,ent_count,first,nmembers,reduced_m;
  unsigned long  *auxmembers, *remainingvars, *newposvars;
  double auxlam;
  int conv,Finish;
  unsigned long n_affclusters;

  nremaining = 0;
  i = 0;
  auxconv = 0;

 n_affclusters = 0;

 while(n_affclusters<=1 && i<10) //While not converged increase the damping coefficient
  {
    auxlam = 0.5 + 0.05*i;
    auxconv = affinity_prop(auxlam, maxits, convits,  n,  m, points, pref, sim, idx, &n_affclusters);
    //cout<<n_affclusters<<"  "<<i<<" ncalls "<<*ncalls<<endl;
    i++;    
  }


 (*ncalls)++;

 
 if( (n_affclusters<=1) ||  (*ncalls>=deph) )  // && (*ncalls>=deph)  //If not converged or deph reached  add each point as a single cluster
   {
      for(i=0;i<n;i+=(maxsize-1))
       {
         (*nclust)++;
         if(n-i<(maxsize-1))
	   {
              listclusters[(*nclust)-1] = new unsigned long[n-i+1];
              listclusters[(*nclust)-1][0] = n-i;              //Number of variables in the cluster
              for(j=i;j<n;j++)  listclusters[(*nclust)-1][j-i+1] = posvars[j];      // Variables j       
           }
         else
	   {
             listclusters[(*nclust)-1] = new unsigned long[maxsize];
             listclusters[(*nclust)-1][0] = maxsize-1;               //Number of variables in the cluster
             for(j=i;j<i+(maxsize-1);j++)  listclusters[(*nclust)-1][j-i+1] = posvars[j];      // Variables j       
           }    
      }
      if (*ncalls>=deph) *ncalls = 0;
    return 0;
 }



  auxmembers = new unsigned long[n];
  remainingvars = new unsigned long[n];


 first = 0;
 Finish = 0;
 i = 0;
 int init = 0;

 while(!Finish && i<n)
  { 
   if(init==0) 
   {
      i = 0;
      init = 1;
   }
   else i=first+1;
   
   //cout<<"Identifying exemplars  i "<<i<<" first  "<<first<<" nremaining "<<nremaining<<endl;
   while(i<n && idx[i] != i) i++;  //Exemplars are identified
   if(i<n)
    {  
      first = i;
      nmembers = 0;   
      
   for(j=0;j<n;j++)  //Members of the cluster first are counted and identified
     {
       //cout<<first<<"----"<<idx[j]<<endl;
       if(idx[j]==first)
	 {
           auxmembers[nmembers] = j;  
	   nmembers++;
         }
     }
    
   if(nmembers< maxsize) //The cluster satisfies the constraint on size and it is added to list
     {
       (*nclust)++;
       //cout<<" Cluster "<<(*nclust)<<" "<<nmembers<<endl;   
 
       listclusters[(*nclust)-1] = new unsigned long[nmembers+1];
       listclusters[(*nclust)-1][0] = nmembers; //The first element is the number of points in the cluster
       for(j=0;j<nmembers;j++)  //The points are added
         {
          listclusters[(*nclust)-1][j+1] = posvars[auxmembers[j]];
	  //cout<<posvars[auxmembers[j]]<<" ";
          remainingvars[auxmembers[j]] = 0;
         }
       //cout<<endl;  
     }    
   else   //These vars will remain in the similarity matrix
     {
       nremaining = nremaining + nmembers;
       //cout<<nremaining<<endl;
        for(j=0;j<nmembers;j++) 
         {
          remainingvars[auxmembers[j]] = 1;
         } 
      }
    }
   else Finish = 1;
  }

 //cout<<nremaining<<" variables have not been added "<<endl;
 //At this point clusters have been added to the list or marked as remaining            
 
 if(nremaining==0)  // The programs finishes, constrained clustering was successful
   {
     delete[] auxmembers;
     delete[] remainingvars;
     return 1;
   }
 else //Construct reduced similarity matrix and find parameters
   {   
     newposvars =  new unsigned long[nremaining]; 
     //cout<<" Vars that were not clustered : "<<endl;
     
     ent_count = 0;
      for(i=0;i<n;i++) 
      { 
        if(remainingvars[i] == 1)
	  {
            newposvars[ent_count] = posvars[i];
            //cout<<posvars[i]<<" ";
            ent_count++;
          }                
      } 
      //cout<<endl;

      // Call the program recursively 
      CallAffinity(lam,maxits,convits,nremaining,newposvars,deph,listclusters,maxsize,SimThreshold,nclust,ncalls);    
      delete[] auxmembers;
      delete[] remainingvars;
      delete[] newposvars;   
   } 
 return conv; 
 
}




int AffPropagation::constr_affinity_prop(double lam, int maxits, int convits, unsigned long n , unsigned long Abs_n, unsigned long m, unsigned long** points, double* pref, double* sim, unsigned long *idx, int* ncalls, int deph, unsigned long *posvars, unsigned long **listclusters, unsigned long *nclust, int maxsize)
{
  
  unsigned long i,j,auxconv,nremaining,ent_count,first,nmembers,reduced_m;
  unsigned long  *auxmembers, *remainingvars, *newposvars, *auxposvars;
  double auxlam;
  int conv,Finish;
  unsigned long** newpoints;
  double* newpref;
  double* newsim;
  unsigned long n_affclusters;

  nremaining = 0;
  i = 0;
  auxconv = 0;
  n_affclusters = 0;



 // cout<<"Number of calls is "<<*ncalls<<endl;

 while( (n_affclusters<=1 || auxconv==0) && i<10) //While not converged increase the damping coefficient
  {
    //cout<<n_affclusters<<"  "<<i<<endl;
    auxlam = 0.5 + 0.05*i;
    auxconv = affinity_prop(auxlam, maxits, convits,  n,  m, points, pref, sim, idx, &n_affclusters);
    // cout<<i<<"  "<<auxlam<<" "<<auxconv<<" "<<n_affclusters<<endl;
    i++;    
  }

 (*ncalls)++;


 // cout<<"Number of calls is "<<*ncalls<<" clusters identified "<<n_affclusters<<endl;

  
 //if( (n_affclusters<=1) ||  (*ncalls>=deph) )  // && (*ncalls>=deph)  //If not converged or deph reached  add each point as a single cluster
 // {
 //   for(i=0;i<n;i++)
 //    {
 //      (*nclust)++;
 //     
 //      listclusters[(*nclust)-1] = new unsigned long[2];
 //      listclusters[(*nclust)-1][0] = 1;               //Number of variables in the cluster
 //      listclusters[(*nclust)-1][1] = posvars[i];      // Only one variable, j       
 //    }    
 //  return 0;
 // }
 //

if( (n_affclusters<=1) ||  (*ncalls>=deph) )  // && (*ncalls>=deph)  //If not converged or deph reached  add each point as a single cluster
   {
      for(i=0;i<n;i+=(maxsize-1))
       {
         (*nclust)++;
         if(n-i<(maxsize-1))
	   {
              listclusters[(*nclust)-1] = new unsigned long[n-i+1];
              listclusters[(*nclust)-1][0] = n-i;              //Number of variables in the cluster
              for(j=i;j<n;j++)  listclusters[(*nclust)-1][j-i+1] = posvars[j];      // Variables j       
           }
         else
	   {
             listclusters[(*nclust)-1] = new unsigned long[maxsize];
             listclusters[(*nclust)-1][0] = maxsize-1;               //Number of variables in the cluster
             for(j=i;j<i+(maxsize-1);j++)  listclusters[(*nclust)-1][j-i+1] = posvars[j];      // Variables j       
           }    
      }
    (*ncalls) = 0;
    return 0;
 }



  auxmembers = new unsigned long[n];
  remainingvars = new unsigned long[n];


 first = 0;
 Finish = 0;
 i = 0;
 int init = 0;

 while(!Finish && i<n)
  { 
   if(init==0) 
   {
      i = 0;
      init = 1;
   }
   else i=first+1;

   //cout<<"Identifying exemplars "<<i<<" "<<n<<endl;
   while(i<n && idx[i] != i) i++;  //Exemplars are identified
   if(i<n)
    {  
      first = i;
      nmembers = 0;   
      
   for(j=0;j<n;j++)  //Members of the cluster first are counted and identified
     {
       //cout<<first<<"----"<<posvars[idx[j]]<<endl;
       if(idx[j]==first)
	 {
           auxmembers[nmembers] = j;  
	   nmembers++;
         }
     }
    
   if(nmembers< maxsize) //The cluster satisfies the constraint on size and it is added to list
     {
       (*nclust)++;
       //cout<<" Cluster "<<(*nclust)<<" "<<nmembers<<endl;   
 
       listclusters[(*nclust)-1] = new unsigned long[nmembers+1];
       listclusters[(*nclust)-1][0] = nmembers; //The first element is the number of points in the cluster
       for(j=0;j<nmembers;j++)  //The points are added
         {
          listclusters[(*nclust)-1][j+1] = posvars[auxmembers[j]];
	  //cout<<posvars[auxmembers[j]]<<" ";
          remainingvars[auxmembers[j]] = 0;
         }
       //cout<<endl;  
     }    
   else   //These vars will remain in the similarity matrix
     {
       nremaining = nremaining + nmembers;
       //cout<<nremaining<<endl;
        for(j=0;j<nmembers;j++) 
         {
          remainingvars[auxmembers[j]] = 1;
         } 

     }

    }   else Finish = 1;
  }

 //cout<<nremaining<<" variables have not been added "<<endl;
 //At this point clusters have been added to the list or marked as remaining            
 
 if(nremaining==0)  // The programs finishes, constrained clustering was successful
   {
     delete[] auxmembers;
     delete[] remainingvars;
     return 1;
   }
 else //Construct reduced similarity matrix and find parameters
   {
     // Remaining similarity entries are counted 
     reduced_m = 0;
     for(j=0;j<m;j++) 
       if(remainingvars[points[0][j]]==1 && remainingvars[points[1][j]]==1)  reduced_m++;
     
     //cout<<"The number of reduced entries is "<<reduced_m<<endl;

      newpoints = new unsigned long*[2];
      newpoints[0] = new unsigned long[reduced_m];
      newpoints[1] = new unsigned long[reduced_m];
      newsim =  new double[reduced_m];
      newpref =  new double[nremaining];
      newposvars =  new unsigned long[nremaining];
      auxposvars = new unsigned long[Abs_n];


     // The reduced preferences and variables vectors are computed
      ent_count = 0;
      // cout<<" Vars that were not clustered : "<<endl;
      for(i=0;i<n;i++) 
      { 
        if(remainingvars[i] == 1)
	  {
            newposvars[ent_count] = posvars[i];
            auxposvars[i] = ent_count;  //New positions of the old variables
	    //    cout<<posvars[i]<<" ";
            newpref[ent_count] = pref[i];
            ent_count++;
          }
                
      } 
      //cout<<endl;
   


      // The reduced matrix of similarity is computed
     ent_count = 0;
     for(j=0;j<m;j++) 
      {
	if(remainingvars[points[0][j]]==1 && remainingvars[points[1][j]]==1)
	  {
            newpoints[0][ent_count] =   auxposvars[points[0][j]]; 
            newpoints[1][ent_count] =   auxposvars[points[1][j]];
            newsim[ent_count] = sim[j];
	    ent_count++;
          }
      }
       
      // Call the program recursively

      conv = constr_affinity_prop(lam, maxits, convits, nremaining, Abs_n, reduced_m, newpoints, newpref, newsim, idx, ncalls,  deph, newposvars, listclusters, nclust, maxsize);

      delete[] auxmembers;
      delete[] remainingvars;
      delete[] newpoints[0];
      delete[] newpoints[1];
      delete[] newpoints;
      delete[] newsim;
      delete[] newpref;
      delete[] newposvars;
      delete[] auxposvars;
   } 
 return conv; 
 
}



unsigned long  AffPropagation::CallAffinity(double lam, int maxits, int convits, unsigned long n, double* Matrix, int deph, unsigned long **listclusters, int maxsize)
{
  unsigned long j,k,totlinks,aux,m,remaining,auxm,nclust;
  unsigned long** points;
  double meansim;
  double *pref, *sim;
  unsigned long *idx;
  unsigned long *posvars; 
  int conv, ncalls; 

  posvars = new unsigned long[n];
  idx     = new unsigned long[n];

  // For each point it is analyzed whether it is linked to at least another point in the matrix
  // If not, it is added as a node of the list of factors

  m = 0;
  remaining = n;
  meansim = 0;
  nclust = 0;

  for(j=0;j<n;j++)
   {
     totlinks = 0;
     for(k=0;k<n;k++)
      {
	if(k!=j)
	  {
       	   if(j<k) aux = j*(2*n-j+1)/2 +k-2*j-1;
           else    aux = k*(2*n-k+1)/2 +j-2*k-1;
           //cout<<Matrix[aux] <<" ";
           if(Matrix[aux] > 1e-5)    
             {
                 totlinks++ ;               //There is a link between points
                 m++;
                 meansim += Matrix[aux];
	     }
	  }
      }
    
     // cout<<endl<<j<<" "<<remaining<<" m  "<<m<<" "<<meansim<<" totlinks "<<totlinks<<endl;
     if (totlinks == 0)  // This point is not linked to any and it forms a cluster
       { 
	 nclust++;
         remaining--;
         listclusters[nclust-1] = new unsigned long[2];
         listclusters[nclust-1][0] = 1;      //Number of variables in the cluster
         listclusters[nclust-1][1] = j;      // Only one variable, j
         //cout<<nclust-1<<" (  ) " <<remaining<<endl;        
       }
     else
       {  
          posvars[remaining-n+j] = j;    
       }    
    }

  meansim = meansim/m;       //Average of the matrix entries 
        
  // The list of similarity entries is constructed from the reduced matrix
      points = new unsigned long*[2];
      points[0] = new unsigned long[m];
      points[1] = new unsigned long[m];
      sim =  new double[m];
      pref =  new double[remaining];
       
      
   auxm = 0;
   for(j=0;j<remaining;j++)         //Only entries that include the remaining variables
   {
     pref[j] = meansim; 
     for(k=0;k<remaining;k++)
      {
	if(posvars[k]!=posvars[j])
	  {
       	   if(posvars[j]<posvars[k]) aux = posvars[j]*(2*n-posvars[j]+1)/2 +posvars[k]-2*posvars[j]-1;
           else             aux = posvars[k]*(2*n-posvars[k]+1)/2 +posvars[j]-2*posvars[k]-1;
           if(Matrix[aux] > 1e-5)  
             {
	        points[0][auxm] = j;
                points[1][auxm] = k;                
                sim[auxm] = Matrix[aux];               //There is a link between points
                auxm++;
	     }
	  }
      }  
   }
   //for(j=0;j<remaining;j++) cout<<posvars[j]<<"  ";
   //cout<<endl;
    
   //cout<<"lam "<<lam<<" maxits "<<maxits<<" convits "<<convits<<" remaining "<<remaining<<" auxm "<<auxm<<" deph "<<deph<<" maxsize "<<maxsize<<endl;
   // Constrained affinity propagation is called

     ncalls = 0; 
     
     conv =   constr_affinity_prop(lam, maxits, convits, remaining, n, auxm, points, pref, sim, idx, &ncalls,  deph, posvars, listclusters, &nclust,maxsize);


     delete[] points[0];   
     delete[] points[1];       
     delete[] points;          
     delete[] sim;     
     delete[] pref;   
     delete[] posvars;        
     delete[] idx;    
     return nclust;
}


*/
