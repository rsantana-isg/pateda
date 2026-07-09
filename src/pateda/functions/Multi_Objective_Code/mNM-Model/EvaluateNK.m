function [val] =  EvaluateNK(vector)

 global ListFactors
 global Tables
 global PE
 
 n = size(ListFactors,2);
 k = size(ListFactors{1},2);
 val = 0;
 for i=1:n,
   ind = 0;  
   for j=1:k,    
    %[i,j,ListFactors{i}(j)]
    ind = ind + vector(ListFactors{i}(j))*PE(j);     
   end    

   val = val + Tables{i}(ind+1);  
   %[i,ind,val]  
 end
 
 return