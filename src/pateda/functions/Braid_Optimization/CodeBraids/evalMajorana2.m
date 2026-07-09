function[val] = evalMajorana2(vector)
% [val] = evalMajorana(vector)
% evalMajorana: Product of Majorana Fermions
%
% INPUTS
% vector: Vector of variables with cardinality 10
% OUTPUTS
% val: Evaluation of the deceptive function
%
% Last version 10/25/2013. Roberto Santana (rsantana@si.ehu.es)

global Sigma
global Target
global LAMBDA

NumbVar = size(vector,2);
thesize = 0;

AuxMat=Sigma{vector(1)+1};
flag=0;
for i=2:NumbVar,
   AuxMat = AuxMat*Sigma{vector(i)+1};
   if (vector(i)==vector(i-1)+2 & flag==0)
     thesize = thesize+1;
     flag=1;
   else
     flag=0;
   end
end

PAuxMat = Target-AuxMat;
val = norm(PAuxMat)
thesize

if val > 0.0000001
 val = (1.0-LAMBDA)/(1+val) + LAMBDA/(NumbVar-thesize);
else
 val=  -10000000.0;		%return -INFINITY;  
end
