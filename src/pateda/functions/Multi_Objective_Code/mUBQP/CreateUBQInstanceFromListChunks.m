function[] = CreateUBQInstanceFromListChunks(k,n,nobj,SelectedParams,perm,fname,seed)

 desp = size(SelectedParams{1,1},1);
 nblocks = (n/k); 
 nedges = [desp*nblocks,desp*nblocks];
 
for i=1:nblocks,
 mat1 = SelectedParams{perm(i),1};
 mat2 = SelectedParams{perm(i),2};
 all_obj{1}((i-1)*desp+1:i*desp,1:2) = mat1(:,1:2) + (i-1)*k;
 all_obj{2}((i-1)*desp+1:i*desp,1:2) = mat2(:,1:2) + (i-1)*k;
 all_obj{1}((i-1)*desp+1:i*desp,3) = mat1(:,3);
 all_obj{2}((i-1)*desp+1:i*desp,3) = mat2(:,3);
end      
  
    
newfname = [fname,'_n_',num2str(n),'.dat']; 
fid = fopen(newfname,'w');
fprintf(fid,'%d  \n',seed);
for j=1:nobj,
  fprintf(fid,'%d %d \n',n,nedges(j));   
  for i=1:nedges(j),
     fprintf(fid,'%d %d %d \n', all_obj{j}(i,:));           
  end     
end
fclose(fid);