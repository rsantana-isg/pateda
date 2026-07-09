function[val] = evalMajorana(vector)
% [val] = evalMajorana(vector)
% evalMajorana: Product of Majorana Fermions
%
% INPUTS
% vector: Vector of variables with cardinality 10
% OUTPUTS
% val: Evaluation of the deceptive function
%
% Last version 10/25/2013. Roberto Santana (rsantana@si.ehu.es)

global B
global Target
global LAMBDA

NumbVar = size(vector,2);
thesize = 0;

AuxMat=B{vector(1)+1};
flag = 0;
for j=2:NumbVar,
   AuxMat = AuxMat*B{vector(j)+1};
   if ((vector(j)==vector(j-1)+5 || vector(j)==vector(j-1)-5) & flag==0)
     thesize = thesize+1;
     flag=1;
   else
     flag=0;
   end
end

PAuxMat = Target-AuxMat;
val = norm(PAuxMat);
val
thesize

if val > 0.0000001
 val = (1.0-LAMBDA)/(1+val) + LAMBDA/(NumbVar-thesize);
else
 val=  -10000000.0;		%return -INFINITY;  
end
