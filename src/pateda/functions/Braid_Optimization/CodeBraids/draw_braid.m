function[thebraid,compact_vector,var_times,current] = draw_braid(vector)
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
PAuxMat = Target-AuxMat;
bestval = norm(PAuxMat);
BestMat = AuxMat;
current = 1;

for i=2:NumbVar,
   AuxMat = AuxMat*Sigma{vector(i)+1};   
   PAuxMat = Target-AuxMat;
   val = norm(PAuxMat);
   if(val<bestval)
     current = i;
     bestval = val;
     BestMat = AuxMat;
   end      
end

vector = vector + 1; % Indices currently begin with 0, for visualization change to 1
a = 1;
compact_vector(a) = vector(1);
var_times(a) = 1;
for i=2:current,
 if vector(i)==vector(i-1)
   var_times(a) = var_times(a)+1;  
 else
   a = a +1;
   compact_vector(a) = vector(i);   
   var_times(a) = 1;
 end
end    

thebraid = [];
for i=1:size(var_times,2),
 if compact_vector(i)>2 
  compact_vector(i) = compact_vector(i)-2;   
  var_times(i) = -1*var_times(i);
 end
 if var_times(i)~=1   
  thebraid = [thebraid,'\sigma_',num2str(compact_vector(i)),'^{',num2str(var_times(i)),'}'];       
 else
  thebraid = [thebraid,'\sigma_',num2str(compact_vector(i))];          
 end 
end


