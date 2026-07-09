for i=1:5,
   fname =['Art_type_',num2str(i),'_n_100.dat'];
   mat = load(fname);
   spars(i) = sum(mat==0)/size(mat,1)*size(mat,1)
end