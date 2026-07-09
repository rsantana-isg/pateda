
for ext=1:2,

  for typ=1:5,
   TheMat{ext,mod,typ} = zeros(100);   
   for exp=1:30,   
    extreme = (ext-1)*200;  
    model = 8;        
    fname = ['type_',num2str(typ),'_model_',num2str(model),'_',num2str(exp-1),'_',num2str(extreme),'.dat.txt'];
    mat = load(fname);
    TheMat{ext,mod,typ} =TheMat{ext,mod,typ} + mat;
   end
  end

end

for ext=1:2,
 extreme = (ext-1)*200;        

 close all
  for typ=1:10,
    H = figure
    readfname = ['30_runs_type_',num2str(typ),'_Tree_0.dat'];
    mat = load(readfname);
    imagesc(mat);
    colorbar
    fname = ['Fig_30_runs_type_',num2str(typ),'_Tree.eps'];
    saveas(H, fname, 'eps')
  end
 


%%%%%%%%%%%%%%%%%%%%


for ext=1:2,

  for typ=1:5,
   TheMat{ext,mod,typ} = zeros(100);   
   for exp=1:30,   
    extreme = (ext-1)*200;  
    model = 8;        
    fname = ['type_',num2str(typ),'_model_',num2str(model),'_',num2str(exp-1),'_',num2str(extreme),'.dat.txt'];
    mat = load(fname);
    TheMat{ext,mod,typ} =TheMat{ext,mod,typ} + mat;
   end
  end

end



 close all
  for typ=1:9,
    H = figure    
    readfname = ['Freq_Matrix_type_',num2str(typ),'model_8_50_0.dat.txt'];
    auxmat = load(readfname);
    readfname = ['Freq_Matrix_type_',num2str(typ),'model_8_50_200.dat.txt'];
    %fname = ['Fig_30_runs_type_',num2str(typ),'_Tree.eps'];
    mat = load(readfname) + auxmat;
    imagesc(mat);
    colorbar
    %saveas(H, fname, 'eps')
  end
 
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%

  

 close all
 mat = zeros(100);
  typ = 5;
  for run=1:30,
    H = figure        
    readfname = ['type_',num2str(typ),'_model_8_50_',num2str(run-1),'_0.dat.txt'];
    auxmat = load(readfname);
    mat = auxmat + mat;
    imagesc(mat);
    colorbar
    %readfname = ['Freq_Matrix_type_',num2str(typ),'model8_50_0.dat'];
    %fname = ['Fig_30_runs_type_',num2str(typ),'_Tree.eps'];
    %saveas(H, fname, 'eps')
  end

