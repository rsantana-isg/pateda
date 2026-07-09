function [fv] =  EvaluatePop_nNM(Pop,mNM)
 
 ncomp = size(mNM.betas,1);
 for t=1:size(Pop,1),
  val = mNM.betas(1);  
  for i=2:ncomp,     
   ind = 1;  
   for j=1:size(mNM.components{i},2),    
     ind = ind * Pop(t,mNM.components{i}(j)) ;     
   end  
   val = val + mNM.betas(i)*ind;   
  end
  fv(t) = val;
 end
 aux_fv = fv/ncomp;
 %min(aux_fv) 
 %max(aux_fv)
 fv = (aux_fv-min(aux_fv))/(max(aux_fv)-min(aux_fv));
 
      
 
 return