
N = 10;
M = 9;
AccCard = 2.^[N-1:-1:0];
Card = 2*ones(1,N);
sigma = 1;

% The complete population is created
clear AllVectors;
for i=1:2^N,
  AllVectors(i,:) = IndexconvertCard(i-1,N,AccCard);    
end  
n = N
Pop = AllVectors;


% FIGURES
%load BoltzmannModels_1-19.mat
%Models = AllSig{1}{3};

close all
n = 10
pos = 2
for mm=1:9,
Models = AllSig{mm}{3};    
t = 1;
themod = [Models{3}{pos}(1,:);Models{3}{pos}(2,:)];
thebol =  exp(themod/t)';
thebol =  [thebol./repmat(sum(thebol),1024,1)]';

 clear Univ
  for i=1:n,
    for j=1:2;
     ind = find(Pop(:,i)==(j-1));
     aUniv(j,i) = sum(thebol(1,ind));
     bUniv(j,i) = sum(thebol(2,ind));
    end
  end
  for i=1:2^N,
    theproba =1;
    theprobb =1;
    for j=1:n,
      if  AllVectors(i,j) ==0
       theproba = theproba * aUniv(1,j);
       theprobb = theprobb * bUniv(1,j);
      else
       theproba = theproba * aUniv(2,j);
       theprobb = theprobb * bUniv(2,j);   
      end 
    end
    newbol(:,i) = [theproba;theprobb];
  end 

%figure
%plot(thebol(1,:),thebol(2,:),'r.')

 fa = 16;
 h = figure
 plot(themod(1,:),themod(2,:),'r.')
 set(gca,'fontsize',fa)
 fname = ['newOrigF_n10_I_',num2str(pos),'_s_',num2str(mm),'.eps']; 
 xlabel('f_1','FontSize',fa);
 ylabel('f_2','FontSize',fa); 
 saveas(gcf,fname,'psc2');

 
 fa = 16;
 h = figure
 plot(thebol(1,:),thebol(2,:),'b.')
 set(gca,'fontsize',fa)
 fname = ['newBolt_n10_I_',num2str(pos),'_s_',num2str(mm),'.eps']; 
 xlabel('p^1_{B_i}','FontSize',fa);
 ylabel('p^2_{B_i}','FontSize',fa); 
 saveas(gcf,fname,'psc2');
 
 fa = 16;
 h = figure
 plot(newbol(1,:),newbol(2,:),'g.')
 set(gca,'fontsize',fa)
 fname = ['newUMDA_n10_I_',num2str(pos),'_s_',num2str(mm),'.eps']; 
 xlabel('q^1_{B_i}','FontSize',fa);
 ylabel('q^2_{B_i}','FontSize',fa); 
 saveas(gcf,fname,'psc2'); 
end 