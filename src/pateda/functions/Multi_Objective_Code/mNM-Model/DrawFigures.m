
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Maximization of the correlation results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 10;
for in=1:2,
 if in==1,
   n=10;
   beg = 1;
   en = 30;
 else
   n=12;
   beg = 1;
   en = 10;
 end
 for k=1:4,
  for i=beg:en,
     if n==10,
       eval(['load pos_corr_results_',num2str(n),'_',num2str(k),'_',num2str(i),'.mat']);
     else 
        eval(['load pos_corr_results_',num2str(n),'_',num2str(k),'_',num2str(i+2),'.mat']); 
     end
     aux_corr = corrcoef(org_fvals);
     thecorrs = [aux_corr(1,2),TheMax(2,:)];
     all_poscorrs{in,k}(i,:) = thecorrs;
     [org_Index]=FindParetoSet(org_fvals,org_fvals);    
     org_aux_siz = size(org_Index,2);
     [Index]=FindParetoSet(fvals,fvals);    
     aux_siz = size(Index,2);
     InitSize{in}(k,i) =  org_aux_siz;
     LastSize{in}(k,i) =  aux_siz;
     
     InitNeighbors = zeros(n);
     LastNeighbors = zeros(n);
     FirstObjNeighbors = zeros(n);
     for l=1:n,
      for m=2:k+1          
         nn = ListFactors{1}{l}(m); 
         FirstObjNeighbors(l,nn) = 1;            
         nn = ListFactors{2}{l}(m); 
         InitNeighbors(l,nn) = 1;          
         nn = ListOfLists{2}{l}(m); 
         LastNeighbors(l,nn) = 1;               
      end
     end
     CoindInit{in}(k,i) = sum(sum(InitNeighbors+FirstObjNeighbors)==2);
     CoindLast{in}(k,i) = sum(sum(LastNeighbors+FirstObjNeighbors)==2);
  end
 end
end


fa  = 14;
close all
thelines = '*os+dx<p'
thecolors = 'gmrkcbyg'
a = 0;
h = figure
for in=1:2, 
 if in==1,
   n=10;
 else
   n=12;
 end
 for k=1:4,
  a = a + 1;
  plot(mean(all_poscorrs{in,k}),['-',thelines(a),thecolors(a)],'MarkerSize',8,'LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor',thecolors(a));
  hold on     
 end
end 
set(gca,'fontsize',fa)
legend(['n=10, k=1';'n=10, k=2';'n=10, k=3';'n=10, k=4';'n=12, k=1';'n=12, k=2';'n=12, k=3';'n=12, k=4'],'Location','SouthEast')
set(gca,'fontsize',fa)
xlabel('Iterations','FontSize',fa);
ylabel('Objectives correlation','FontSize',fa);
     

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Minimization of the correlation results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for in=1:2,
 if in==1,
   n=10;
   beg = 1;
   en = 30;
 else
   n=12;
   beg = 1;
   en = 10;
 end
 for k=1:4,
  for i=beg:en,
     if n==10,
       eval(['load neg_corr_results_',num2str(n),'_',num2str(k),'_',num2str(i),'.mat']);
     else 
        eval(['load neg_corr_results_',num2str(n),'_',num2str(k),'_',num2str(i+2),'.mat']); 
     end
     aux_corr = corrcoef(org_fvals);
     thecorrs = [aux_corr(1,2),TheMax(2,:)];
     all_negcorrs{in,k}(i,:) = thecorrs;
     [org_Index]=FindParetoSet(org_fvals,org_fvals);    
     org_aux_siz = size(org_Index,2);
     [Index]=FindParetoSet(fvals,fvals);    
     aux_siz = size(Index,2);
     InitSize{in}(k,i) =  org_aux_siz;
     LastSize{in}(k,i) =  aux_siz;
  end
 end
end


fa  = 14;
close all
thelines = '*os+dx<p'
thecolors = 'gmrkcbyg'
a = 0;
h = figure
for in=1:2, 
 if in==1,
   n=10;
 else
   n=12;
 end
 for k=1:4,
  a = a + 1;
  plot(mean(all_negcorrs{in,k}),['-',thelines(a),thecolors(a)],'MarkerSize',8,'LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor',thecolors(a));
  hold on     
 end
end 
set(gca,'fontsize',fa)
legend(['n=10, k=1';'n=10, k=2';'n=10, k=3';'n=10, k=4';'n=12, k=1';'n=12, k=2';'n=12, k=3';'n=12, k=4'],'Location','NorthEast')
set(gca,'fontsize',fa)
xlabel('Iterations','FontSize',fa);
ylabel('Objectives correlation','FontSize',fa);
     

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PF results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for in=1:2,
 if in==1,
   n=10;
   beg = 1;
   en = 30;
 else
   n=12;
   beg = 1;
   en = 10;
 end
 for k=1:4,
  for i=beg:en,
     if n==10,
       eval(['load pos_PF_results_',num2str(n),'_',num2str(k),'_',num2str(i),'.mat']);
     else 
       eval(['load pos_PF_results_',num2str(n),'_',num2str(k),'_',num2str(i+2),'.mat']); 
     end
     [Index]=FindParetoSet(org_fvals,org_fvals);    
     aux_siz = size(Index,2);
     thecorrs = [aux_siz,TheMax(2,:)];
     all_posPF{in,k}(i,:) = thecorrs;
     [org_Index]=FindParetoSet(org_fvals,org_fvals);    
     org_aux_siz = size(org_Index,2);
     [Index]=FindParetoSet(fvals,fvals);    
     aux_siz = size(Index,2);
     InitSize{in}(k,i) =  org_aux_siz;
     LastSize{in}(k,i) =  aux_siz;
  end
 end
end


fa  = 14;
close all
thelines = '*os+dx<p'
thecolors = 'gmrkcbyg'
a = 0;
h = figure
for in=1:2, 
 if in==1,
   n=10;
 else
   n=12;
 end
 for k=1:4,
  a = a + 1;
  plot(mean(all_posPF{in,k}),['-',thelines(a),thecolors(a)],'MarkerSize',8,'LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor',thecolors(a));
  hold on     
 end
end 
set(gca,'fontsize',fa)
legend(['n=10, k=1';'n=10, k=2';'n=10, k=3';'n=10, k=4';'n=12, k=1';'n=12, k=2';'n=12, k=3';'n=12, k=4'],'Location','SouthEast')
set(gca,'fontsize',fa)
xlabel('Iterations','FontSize',fa);
ylabel('Objectives correlation','FontSize',fa);
     

%save results_MNK_toplot.mat all_negcorrs all_poscorrs all_posPF

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Saving the Pareto Sets of the first and last instances
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 12, k=1, i=8
eval(['load pos_PF_results_',num2str(n),'_',num2str(k),'_',num2str(i),'.mat']);


% The Pareto front is computed
[Index]=FindParetoSet(fvals,fvals);
fvals(Index,:)

k = 1;
close all
fa = 14;

h = figure
plot(fvals(:,1),fvals(:,2),'.b')
hold on
plot(fvals(Index,1),fvals(Index,2),'r*')

%title(['Problem seed ',num2str(69+inst),' number neighbors ',num2str(nhei(l))],'FontSize',fx)
fname = ['First_n12_k','_',num2str(k),'pos_corr.eps']; 
xlabel('f_1','FontSize',fa);
ylabel('f_2','FontSize',fa);
saveas(gcf,fname,'psc2');


[org_Index]=FindParetoSet(org_fvals,org_fvals);
org_fvals(org_Index,:)

h = figure
plot(org_fvals(:,1),org_fvals(:,2),'.b')
hold on
plot(org_fvals(org_Index,1),org_fvals(org_Index,2),'r*')

%title(['Problem seed ',num2str(69+inst),' number neighbors ',num2str(nhei(l))],'FontSize',fx)
fname = ['Last_n12_k','_',num2str(k),'pos_corr.eps']; 
xlabel('f_1','FontSize',fa);
ylabel('f_2','FontSize',fa);
saveas(gcf,fname,'psc2');

figure
plot(fvals(:,2),org_fvals(:,2),'.b')
hold on
plot(fvals(Index,2),org_fvals(Index,2),'r*')

save PFFirst_Last.mat fvals org_fvals Index org_Index

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Minimization of the correlation results APPROX n=50
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 n = 50;
 for k=1:4,
  for i=1:10,     
     eval(['load approx_neg_corr_results_',num2str(n),'_',num2str(k),'_',num2str(i),'.mat']);     
     aux_corr = corrcoef(org_fvals);
     thecorrs = [aux_corr(1,2),TheMax(2,:)];
     [org_Index]=FindParetoSet(org_fvals,org_fvals);    
     org_aux_siz = size(org_Index,2);
     [Index]=FindParetoSet(fvals,fvals);    
     aux_siz = size(Index,2);
     InitSize(k,i) =  org_aux_siz;
     LastSize(k,i) =  aux_siz;
     all_approx_negcorrs{k}(i,:) = thecorrs;
  end
 end



fa  = 14;
close all
thelines = '*os+dx'
thecolors = 'gmrkcb'
a = 0;
h = figure
 for k=1:4,
  a = a + 1;
  plot(mean(all_approx_negcorrs{k}),['-',thelines(a),thecolors(a)],'MarkerSize',8,'LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor',thecolors(a));
  hold on     
 end

set(gca,'fontsize',fa)
legend(['k=1';'k=2';'k=3';'k=4'],'Location','NorthEast')
set(gca,'fontsize',fa)
xlabel('Iterations','FontSize',fa);
ylabel('Objectives correlation','FontSize',fa);
     
save ApproxMinCorrn50.mat all_approx_negcorrs


