function [fv] =  EvaluatePopNK(Pop,LF)
 
 global Tables
 global PE
 
 M = size(LF);
 n = size(Pop,2);
 k = size(LF{1},2);

 for t=1:size(Pop,1),
  val = 0;   
  for i=1:n,     
   ind = 0;  
   for j=1:k,    
    %[i,j,ListFactors{i}(j)]
    ind = ind + Pop(t,LF{i}(j)) *PE(j);     
   end  
   val = val + Tables{i}(ind+1);  
   %[i,ind,val]  
  end
  fv(t) = val/n;
 end
 
 
 return