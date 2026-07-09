function[] = CreateUBQInstance(n,nobj,itype,fname,seed)

if(itype == 1)
  all_obj{1} = zeros(n,3);
  all_obj{2} = zeros(n,3);
  nedges(1) = n; 
  nedges(2) = n;
  for i=1:n,
    all_obj{1}(i,:) = [i,i,1];
    all_obj{2}(i,:) = [i,i,-1];
  end      
elseif(itype == 2)
  all_obj{1} = zeros(n,3);
  all_obj{2} = zeros(n,3);
  nedges(1) = n; 
  nedges(2) = n;
  for i=1:n,
    all_obj{1}(i,:) = [i,i,i];
    all_obj{2}(i,:) = [i,i,-i];
  end     
elseif(itype == 3)
  all_obj{1} = zeros(n,3);
  all_obj{2} = zeros(n,3);
  nedges(1) = 2*n -1; 
  nedges(2) = 2*n -1;   
  all_obj{1}(1,:) = [1,1,-1];    
  all_obj{2}(1,:) = [1,1, 1];
  a = 2;
  for i=2:n,
    all_obj{1}(a,:) = [i-1,i,3];    
    all_obj{2}(a,:) = [i-1,i,-3];      
    a = a + 1;
    all_obj{1}(a,:) = [i,i,-1];    
    all_obj{2}(a,:) = [i,i,1];
    a = a + 1;
  end     
elseif(itype == 4)
  all_obj{1} = zeros(n,3);
  all_obj{2} = zeros(n,3);
  nedges(1) = 2*n -1; 
  nedges(2) = 2*n -1;   
  all_obj{1}(1,:) = [1,1,-1];    
  all_obj{2}(1,:) = [1,1, 1];
  a = 2;
  for i=2:n,
    all_obj{1}(a,:) = [i-1,i,3*i-1];    
    all_obj{2}(a,:) = [i-1,i,-3*i+1];      
    a = a + 1;
    all_obj{1}(a,:) = [i,i,-i];    
    all_obj{2}(a,:) = [i,i,i];
    a = a + 1;
  end     
elseif(itype == 5)
  all_obj{1} = zeros(n,3);
  all_obj{2} = zeros(n,3);
  nedges(1) = 5*(n/4); 
  nedges(2) = 5*(n/4);   
  
  all_obj{1}(1,:) = [1,2,1];    
  all_obj{1}(2,:) = [1,4,1];    
  all_obj{1}(3,:) = [2,3,1];    
  all_obj{1}(4,:) = [2,4,-5];    
  all_obj{1}(5,:) = [3,4,1];    
  
  all_obj{2}(1:5,1:2) = all_obj{1}(1:5,1:2);
  all_obj{2}(1:5,3) =  -1*all_obj{1}(1:5,3);
  
  for i=1:(n/4)-1,
    all_obj{1}(i*5+1:(i+1)*5,1:2) = all_obj{1}(1:5,1:2)+4*i;
    all_obj{1}(i*5+1:(i+1)*5,3) = all_obj{1}(1:5,3);
    all_obj{2}(i*5+1:(i+1)*5,1:2) = all_obj{2}(1:5,1:2)+4*i;
    all_obj{2}(i*5+1:(i+1)*5,3) = all_obj{2}(1:5,3);    
  end     
end  
    
    
newfname = [fname,'_type_',num2str(itype),'_n_',num2str(n),'.dat']; 
fid = fopen(newfname,'w');
fprintf(fid,'%d  \n',seed);
for j=1:nobj,
  fprintf(fid,'%d %d \n',n,nedges(j));   
  for i=1:nedges(j),
     fprintf(fid,'%d %d %d \n', all_obj{j}(i,:));           
  end     
end
fclose(fid);
        
        
        
        