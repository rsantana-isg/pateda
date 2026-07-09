function [other_mNM] =  Save_mNM_model(mNM,max_M)

 
 ncomp = size(mNM.betas,1);
 count = 1;
 while(count< size(mNM.betas,1) & size(mNM.components{count},2)<=max_M)
    other_mNM.components{count} = mNM.components{count};
    other_mNM.betas(count,1) = mNM.betas(count);
    count = count + 1;
 end   
 
 return