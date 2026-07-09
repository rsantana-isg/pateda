function[val] = evalMajoranaPlus(vector)
% [val] = evalMajoranaPlus2(vector)
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
val = 10^20;
AuxMat=Sigma{vector(1)+1};

for j=2:NumbVar,
   AuxMat = AuxMat*Sigma{vector(j)+1};
   PAuxMat = Target-AuxMat;
   auxval = norm(PAuxMat);
   if auxval<val,     
      val = auxval;
      thesize = j;
   end
end,

%val
%log(val)
if val > 0.0000001
 val = (1.0-LAMBDA)/(1+val) + LAMBDA/NumbVar;
else
 val=  -10000000.0;		%return -INFINITY;  
end
