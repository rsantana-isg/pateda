function[] = CreateHeavyUBQInstanceFromListChunks(k,n,nobj,SelectedParams,perm,fname,seed)

 sizeblock =  size(perm,2);
 nblocks = size(SelectedParams{1,1},1);
 %nblocks = (n/k); 
 %nedges = [desp*nblocks,desp*nblocks];
 
 fullmat{1} = zeros(n);
 fullmat{2} = zeros(n);
for i=1:sizeblock,
 ordr = randperm(n);
 ordr = ordr(1:k); 
 mat1 = SelectedParams{perm(i),1};
 mat2 = SelectedParams{perm(i),2};
 indices = ordr(mat1(:,1:2));
 for j=1:nblocks,  
    fullmat{1}(indices(j,1),indices(j,2)) =  mat1(j,3);  
    fullmat{2}(indices(j,1),indices(j,2)) =  mat2(j,3);
 end
end      
  
[r,c] = find(fullmat{1}~=0);
nedges(1)= size(r,1);
nedges(2) = nedges(1);
newfname = [fname,'_n_',num2str(n),'.dat']; 
fid = fopen(newfname,'w');
fprintf(fid,'%d  \n',seed);
for j=1:nobj,
  fprintf(fid,'%d %d \n',n,nedges(j));   
  for i=1:nedges(j),
     fprintf(fid,'%d %d %d \n', c(i),r(i),fullmat{j}(r(i),c(i)));           
  end     
end
fclose(fid);