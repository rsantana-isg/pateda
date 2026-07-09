function[val] = fitness_braid_best(vector)
% [val] = fitness_braid_best(vector)
% fitness_braid: Product of generator matrices taking as output the best
% approximation achieved by the product of the first j matrices where
% j is in (1,n). In fact it compares n possible products of matrices and
% takes the best approximation. The length is the number of matrices
% involved in the product
%
% INPUTS
% vector: Vector of variables with cardinality equal to the number of
% generators
% OUTPUTS
% val: Distance between the product of generators and the target matrix
%
% Last version 10/25/2013. Roberto Santana (rsantana@si.ehu.es)

global Sigma
global Target
global LAMBDA

NumbVar = size(vector,2);
AuxMat=Sigma{vector(1)+1};

for i=2:NumbVar,
   AuxMat = AuxMat*Sigma{vector(i)+1}   
   PAuxMat = Target-AuxMat;
   val = norm(PAuxMat);
   [i,val]
end

PAuxMat = Target-AuxMat;
val = norm(PAuxMat);

%if val > 0.0000001
% val = (1.0-LAMBDA)/(1+val) + LAMBDA/(NumbVar);
%else
% val=  -10000000.0;		%return -INFINITY;  
%end

