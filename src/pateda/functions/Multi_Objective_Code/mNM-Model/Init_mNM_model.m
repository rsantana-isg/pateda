
N = 12;

AccCard = 2.^[N-1:-1:0];
Card = 2*ones(1,N);

clear AllVectors;
for i=1:2^N,
  AllVectors(i,:) = IndexconvertCard(i-1,N,AccCard);    
end  

M = 11;

for sigma=1:5:40,
%sigma = 10;
[mNM] =  Create_mNM_model(sigma,M,N);
clear AllMulti_NM
for k=1:11, 
 [sigma,k]   
 other_mNM1 =  Save_mNM_model(mNM,k);
  for m=1:2,     
   if m==1,
    [fv] =  EvaluatePop_nNM(2*AllVectors-1,other_mNM1);
   elseif m==2
    [fv] =  EvaluatePop_nNM(-2*AllVectors+1,other_mNM1);  
   end
   Multi_NM(m,:) = fv;
  end
  AllMulti_NM{k} = Multi_NM; 
end  

eval(['save expAllMulti_sigma',num2str(sigma),'.mat AllMulti_NM;']);
end



close all
for i=1:10,
 figure
 plot(AllMulti_NM{i}(1,:),AllMulti_NM{i}(2,:),'k.')
end


close all
for i=1:3,
 figure
 plot(AllMulti_NM{i}(1,:),AllMulti_NM{4-i}(2,:),'k.')
end


close all
for i=1:10,
 figure
 plot(AllMulti_NM{i}(1,:),AllMulti_NM{11-i}(2,:),'k.')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing the size of the Pareto sets and number of fronts 
% for all bi-objective problems

nlamb = 0;
for sigma=1:5:40,
  nlamb = nlamb + 1
  clear AllMulti_NM
  for k=1:11, 
   eval(['load expAllMulti_sigma',num2str(sigma),'.mat;']);
   for l=1:11,
    AA = [AllMulti_NM{k}(1,:)',AllMulti_NM{l}(2,:)'];   
    [Index]=FindParetoSet(AA,AA);
    NSiz{nlamb}(k,l)= size(Index,2);
   end
  end
end

% Computing the correlations between objectives
nlamb = 0;
for sigma=1:5:40,
  nlamb = nlamb + 1
  clear AllMulti_NM
  for k=1:11, 
   eval(['load expAllMulti_sigma',num2str(sigma),'.mat;']);
   for l=1:11,
    AA = [AllMulti_NM{k}(1,:)',AllMulti_NM{l}(2,:)'];   
    dd = corrcoef(AA);    
    NCorr{nlamb}(k,l)= dd(1,2);
   end
  end
end

for count=1:8,
  dad(count,:) = diag(NSiz{count});
end


% Figures and tables of the paper
  fid = fopen('exp.txt','w');
   for i=1:8,
     for k=1:11,
        fprintf(fid,'& %d ',dad(i,k));
     end
     fprintf(fid,' \\\\ \\hline  \n');
   end  
   fclose(fid);
   
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pareto fronts
sigma = 1;
eval(['load expAllMulti_sigma',num2str(sigma),'.mat;']);

close all

fa = 16;

for k=1:11,
 h = figure
 fvals = AllMulti_NM{k}'; 
 plot(fvals(:,1),fvals(:,2),'.b','MarkerSize',10)
 [Index]=FindParetoSet(fvals,fvals);
 hold on
 plot(fvals(Index,1),fvals(Index,2),'r*','MarkerSize',10)
 set(gca,'fontsize',fa)

 fname = ['PF','_',num2str(sigma),'int','_',num2str(k),'.eps']; 
 xlabel('f_1','FontSize',fa);
 ylabel('f_2','FontSize',fa);
 saveas(gcf,fname,'psc2');

end

%%%
sigma = 36;
eval(['load expAllMulti_sigma',num2str(sigma),'.mat;']);

close all
fa = 16;

for k=1:11,
 h = figure
 fvals = AllMulti_NM{k}'; 
 plot(fvals(:,1),fvals(:,2),'.b','MarkerSize',10)
 [Index]=FindParetoSet(fvals,fvals);
 hold on
 plot(fvals(Index,1),fvals(Index,2),'r*','MarkerSize',10)
 set(gca,'fontsize',fa)

 fname = ['PF','_',num2str(sigma),'int','_',num2str(k),'.eps']; 
 xlabel('f_1','FontSize',fa);
 ylabel('f_2','FontSize',fa);
 saveas(gcf,fname,'psc2');

end


% Pareto fronts. Different interactions involved
sigma = 36;
eval(['load expAllMulti_sigma',num2str(sigma),'.mat;']);

close all

fa = 16;
for k=1:11,
 h = figure
 fvals(:,1) = AllMulti_NM{k}(1,:)'; 
 fvals(:,2) = AllMulti_NM{12-k}(2,:)'; 
 plot(fvals(:,1),fvals(:,2),'.b','MarkerSize',10)
 [Index]=FindParetoSet(fvals,fvals);
 hold on
 plot(fvals(Index,1),fvals(Index,2),'r*','MarkerSize',10)
 set(gca,'fontsize',fa)

 fname = ['difPF','_',num2str(sigma),'int','_',num2str(k),'.eps']; 
 xlabel('f_1','FontSize',fa);
 ylabel('f_2','FontSize',fa);
 saveas(gcf,fname,'psc2');

end


% Pareto fronts. Consecutive interactions involved
sigma = 36;
eval(['load expAllMulti_sigma',num2str(sigma),'.mat;']);

close all

fa = 16;
for k=1:11,
 h = figure
 fvals(:,1) = AllMulti_NM{k}(1,:)'; 
 fvals(:,2) = AllMulti_NM{k+1}(2,:)'; 
 plot(fvals(:,1),fvals(:,2),'.b','MarkerSize',10)
 [Index]=FindParetoSet(fvals,fvals);
 hold on
 plot(fvals(Index,1),fvals(Index,2),'r*','MarkerSize',10)
 set(gca,'fontsize',fa)

 fname = ['ConPF','_',num2str(sigma),'int','_',num2str(k),'.eps']; 
 xlabel('f_1','FontSize',fa);
 ylabel('f_2','FontSize',fa);
 saveas(gcf,fname,'psc2');

end


%%% Correlations
load PFSizeCorrs



close all
fa = 16;
 sigma = 1;
 h = figure
 imagesc(NCorr{1})
 set(gca,'fontsize',fa)

 fname = ['Corrs','_',num2str(sigma),'.eps']; 
 xlabel('Max. order of interactions f_1','FontSize',fa);
 ylabel('Max. order of interactions f_2','FontSize',fa);
 colorbar
 saveas(gcf,fname,'psc2');



close all
fa = 16;
 sigma = 16;
 h = figure
 imagesc(NCorr{8})
 set(gca,'fontsize',fa)

 fname = ['Corrs','_',num2str(sigma),'.eps']; 
 xlabel('Max. order of interactions f_1','FontSize',fa);
 ylabel('Max. order of interactions f_2','FontSize',fa);
 colorbar
 saveas(gcf,fname,'psc2');
 
 
 
 %%%%%%%%%%%%%%%%%%%%
 
 % Figures and tables of the paper
  fid = fopen('exp.txt','w');
   for i=1:11,
     for k=1:11,
        if i==k 
          fprintf(fid,'& %d ',0);
        elseif i<k
          fprintf(fid,'& %d ',NSiz{1}(i,k));           
        else 
          fprintf(fid,'& %d ',NSiz{8}(k,i));           
        end
     end
     fprintf(fid,' \\\\ \\hline  \n');
   end  
   fclose(fid);
   
  