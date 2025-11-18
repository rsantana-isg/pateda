#include "GUMDA.h" 



Gaussian::Gaussian(int vars)
{
  mean = 0;
  std = 0;
  trunc = 0;
  t1 = 0;
  t2 = 0;
  s2 = 0;
  c = 0;
  xbar = 0;
  s = 0;
} 


Gaussian::~Gaussian(int vars)
{
}




void Gaussian::truncstatistics(double lt,double* samples)
{ 
 // Calculate the statitistics from a sample of the truncated normal
  // lt is the left truncated value
  
  
  t1 = mean(samples) - lt; 
  //t2 = -sum(samples.*samples);
  //s2 = -(t2+t1^2);
  //s = -sqrt(t2+t1^2); 
  s =  std(samples);  
  c = s/t1;
  xbar = t1;
}


double Gaussian::FindInTable(double cc)
{
 double  XC[80] = {7.07107, 6.42824, 5.89256, 5.43928, 5.05076, 4.71405, 4.41942, 4.15945, 3.92837, 3.72161, 3.53552, 3.36713, 3.21402, 3.07415, 2.94581, 2.82757, 2.71818, 2.61656, 2.52178, 2.43304, 2.34963, 2.27092, 2.19639, 2.12556, 2.05802, 1.99340, 1.93138, 1.87169, 1.81406, 1.75829,  1.70416, 1.65151, 1.60018, 1.55001, 1.50088, 1.45268, 1.40528, 1.35860, 1.31253, 1.26700, 1.22191, 1.17720, 1.13278, 1.08859, 1.04456, 1.00063, 0.95673, 0.91278, 0.86875, 0.82454, 0.78012, 0.73539, 0.69031, 0.64479, 0.59878, 0.55217, 0.50491, 0.45690, 0.40805, 0.35825, 0.30741, 0.25540, 0.20209, 0.14735, 0.09100, 0.03287, -0.02724, -0.08957, -0.15436, -0.22192,  -0.29261, -0.36682, -0.44504, -0.52784, -0.61592, -0.71010, -0.81144, -0.92123, -1.04114, -1.17331, -1.32064, -1.48704, -1.67807, -1.90192, -2.17124, -2.50701, -2.94728, -3.57088, -4.58023, -6.77255};
 return XC[(int(100*cc)-10];
}

double Gaussian::mean(int N, double* f)
  {
    int i;
    double resp=0;

    for (i=0;i<N;i++) resp += f[i];
    return (resp/N); 
  }

double Gaussian::std(int N, double* f, double meanm)
  {
    int i;
    double resp=0;
    double auxmean = meanm*meanm;

    for (i=0;i<N;i++) resp += (auxmean - f[i]*f[i]);
    return  sqrt(resp); 
  }



double Gaussian::SampleLeftTrunc(double ld )
  // randomsampletrunc  Sample one point from a left truncated normal distribution 

  double inva, invb;

inva = normcdf(t,mu,sigma);
invb = 1;

x = (Random normal)*sigma+mu;
y = norminv(normcdf(x,mu,sigma)*(invb-inva)+inva,mu,sigma);

double Gaussian::EstTruncParam(double lt)
 {
   // Calculate the statitistics from a sample of the truncated normal
   xc = FindInTable(c);
   fc = (xc+sqrt(xc*xc+2*(1+c*c)))/(1+c*c);
   sigma = sqrt(2)*xbar/fc;
   mu = sqrt(2)*sigma*xc + lt;
 }


G_UnivariateModel::G_UnivariateModel(vars):AbstractProbModel(vars)
 {
   int i;
   tval = 0;
   
       AllCondGaussians = new (Gaussian**)[2];
     
    for(j=0;j<=1;j++);
     { 
       AllCondGaussians[j] = new (Gaussian*)[vars];
       for (i=0;i<N;i++) 
       {
         AllCondGaussians[j][i] = (Gaussian*)0;  
       }
     }
   FullGaussian = (Gaussian*)0;
 } 


G_UnivariateModel::~G_UnivariateModel()
 {
   int i;
   tval = 0;
   
       AllCondGaussians = new (Gaussian**)[2];

    for(j=0;j<=1;j++);
     { 
       for (i=0;i<N;i++) 
       {
         if( AllCondGaussians[j][i] != (Gaussian*)0) delete AllCondGaussians[j][i];  
       }
       delete[] AllCondGaussians[j];     
     }   
     if(FullGaussian != (Gaussian*)0) delete FullGaussian;
 } 


void G_UnivariateModel::FindPopIndices(int n_ind, double tval, int N, Popul* pop, int xi,int valxi,double*f, double* auxf)
  {
    int i,j;
    n_ind = 0; 
     for (i=0;i<N;i++) 
      {
        if(f[i]>=tval && P[i][xi]==valxi) 
          {
            auxf[n_ind] = f[i];
            n_ind++;
          }

      }
  }



void G_UnivariateModel::FindPopIndices(int n_ind, double tval, int N, double*f, double* auxf)
  {
    int i,j;
    n_ind = 0; 
     for (i=0;i<N;i++) 
      {
        if(f[i]>=tval) 
          {
            auxf[n_ind] = f[i];
            n_ind++;
          }

      }
  }

 double G_UnivariateModel::Find_T(int N, double*f)
   {
     return f[N-1]; //Minimo valor del conjunto truncado
   }

double G_UnivariateModel::FindTruncParams(int N, double*f, Popul* X)
  // Se calculan las Gaussianas truncadas para cada uno de los valores de cada una de las variables
  // Para cada valor de cada variable se devuelven los parametros de la Gaussiana truncada (t,mu,sigma) en una matriz

 int i,j,n_ind;
 double aux_f;
 Gaussian* Gvar;


tval = Find_T(f,N); //  En la primera aproximacion hay un solo valor de truncamiento para todas las normales condicionadas

 aux_f = new double(N);

 for (i=0;i<vars;i++),
    for(j=0;j<=1;j++);
     {        
       FindPopIndices(n_ind,tval,N,pop,i,j,f,auxf);

       if (n_ind !=  0)
	 {
           Gvar = new Gaussian;
           Gvar->truncstatistics(t,aux_f);
           if (c>=0.1 && c<1)
           { 
            Gvar->EstTruncParam(t);
            Gvar->FindIntTrunc(t,mu,sigma);
           }
           else
           {
             delete Gvar;
             Gvar = (Gaussian*)0;
           } 
         }
       else  Gvar = (Gaussian*)0;       
       AllCondGaussians[j][i] = Gvar;
     }       
        
     // At the end the parameters the full Truncated Gaussian are also calculated
        
       FindPopIndices(n_ind,pop,f,auxf);

       if (n_ind !=  0)
	 {
           Gvar = new Gaussian;
           Gvar->truncstatistics(t,aux_f);
           if (c>=0.1 && c<1)
           { 
            Gvar->EstTruncParam(t);
            Gvar->FindIntTrunc(t,mu,sigma);
           }
           else
           {
             delete Gvar;
             Gvar = (Gaussian*)0;
           } 
         }
       else  Gvar = (Gaussian*)0;
       FullGaussian = Gvar;     

       delete[] aux_f;
 }
