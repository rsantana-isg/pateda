function [mNM] =  Create_mNM_model(sigma,M,N)

 nvect = [1:N];
 count = 1;            % Uk0 does not depend on any variable 
 components{count} = [];
 
 for i=1:M,
   aux_vect = combnk(nvect,i);   % Combinations of N in i for i betwenn 1 and M
   for j=1:size(aux_vect,1),     
       count = count+1;
       components{count} = aux_vect(j,:);   % Each combination is saved      
   end    
 end

 mNM.betas = exp(-1*abs(sigma*randn(count,1)));   % Betas are sampled from normal distribution (0,sigma)  
 %mNM.betas = abs(sigma*randn(count,1));   % Betas are sampled from normal distribution (0,sigma)  
 mNM.components = components;
 return