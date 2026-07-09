%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = 500;
nexp =30;
for i=1:10,
  fname = ['HardInst',num2str(i),'__n_',num2str(n),'.dat'];
  [AllMat] = TestTreeEDA_Instances_uBQP(fname,n,nexp);
  eval(['save HardAllMat',num2str(i),'.mat AllMat']);
end


%%%%%%%%%%%%%%%%%%%%%%%%%
n = 500;
nexp =30;
for i=1:9,
  fname = ['LessPopInst',num2str(i),'__n_',num2str(n),'.dat'];
  [AllMat] = TestTreeEDA_Instances_uBQP(fname,n,nexp);
  eval(['save LessPopAllMat',num2str(i),'.mat AllMat']);
end

%%%%%%%%%%%%%%%%%%%%%%%

close all
n = 100;
for i=1:6,
  fname = ['HardAllMat',num2str(i),'.mat'];
  load(fname);
  figure
  imagesc(AllMat);
  colorbar
end

close all
n = 100;
for i=1:9,
  fname = ['LessPopAllMat',num2str(i),'.mat'];
  load(fname);
  H = figure
  imagesc(AllMat);
  colorbar
  outname = ['ParRankingStructure_type_',num2str(i),'_Tree.eps'];
  saveas(H, outname, 'psc2')
end




%%%%%%%%%%%%%%%%%%%%%%%%%
n = 100;
nexp =10;
for i=1:10,
  fname = ['ComposeInstHard10P',num2str(i),'__n_',num2str(n),'.dat'];
  [AllMat] = TestTreeEDA_Instances_uBQP(fname,n,nexp);
  eval(['save ComposeInstHard10PAllMat',num2str(i),'.mat AllMat']);
end

