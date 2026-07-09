load ResultsFunc7_ICO60_120.txt
AllEDAs = ResultsFunc7_ICO60_120;

for i=1:3,
  BraidModels{i} = AllEDAs((i-1)*6000+1:i*6000,:);
end

infix = [1:9];
postfix = [130:135];
pos_sols = [10:129];

pos_error = 132;
pos_size = 131;

for i=1:3,
  for j=1:60,  
   theruns = BraidModels{i}((j-1)*100+1:j*100,pos_error);
   [minval,thepos] = min(theruns);   
   themins(i,j) = minval;
   theminslength(i,j) = BraidModels{i}((j-1)*100+thepos,pos_size);
   thesols{i}(j,:) = BraidModels{i}((j-1)*100+thepos,pos_sols);
   themeans(i,j) = mean(theruns);
  end   
end


for i=1:3,
  for j=1:60,  
   theruns = BraidModels{i}((j-1)*100+1:j*100,pos_size);   
   themeanslength(i,j) = mean(theruns);
  end   
end




the24vector = 24*ones(1,60);
load ResultsBurrello.mat

figure
plot(themins(1,:),'-+')
hold on
plot(themins(2,:),'r-o')
hold on
plot(themins(3,:),'g-*')
hold on
plot(ResultsBurrello(2:2:end,3),'k-v')




*******************************************************

figure
plot(themeans(1,:),'-+')
hold on
plot(themeans(2,:),'r-o')
hold on
plot(themeans(3,:),'g-*')


*******************************************************

figure
plot(theminslength(1,:),'-+')
hold on
plot(theminslength(2,:),'r-o')
hold on
plot(theminslength(3,:),'g-*')


*******************************************************


figure
plot(themeanslength(1,:),'-+')
hold on
plot(themeanslength(2,:),'r-o')
hold on
plot(themeanslength(3,:),'g-*')


*******************************************************

figure
plot(themeanslength(1,:),themeans(1,:),'+')
hold on
plot(themeanslength(2,:),themeans(2,:),'ro')
hold on
plot(themeanslength(3,:),themeans(3,:),'g*')



*******************************************************

the24vector = 24*ones(1,60);
the44vector = 44*ones(1,60);
load ResultsBurrello.mat


f= figure
plot(theminslength(1,:),log10(themins(1,:)),'+','MarkerSize',10);
hold on
plot(theminslength(2,:),log10(themins(2,:)),'ro','MarkerSize',10)
hold on
plot(theminslength(3,:),log10(themins(3,:)),'g*','MarkerSize',10) 
%hold on
%plot(the44vector,log10(ResultsBurrello(2:2:end,3)),'kv','MarkerSize',10)

fs=14;  
legend('  UMDA  ',' Mk-EDA','Tree-EDA',' Brute ','Location','SouthWest');
H = xlabel('Braid length') 
set(H, 'Fontsize',fs);
H = ylabel('log_{10}\epsilon') 
set(H,'Fontsize',fs);
saveas(f,'ComparisonEDAs.eps','psc2');

*******************************************************

thelogmins = log10(themins);

f= figure
plot(thelogmins(1,:),'-+','MarkerSize',10)
hold on
plot(thelogmins(2,:),'r-o','MarkerSize',10)
hold on
plot(thelogmins(3,:),'g-*','MarkerSize',10)
hold on
plot(log10(ResultsBurrello(2:2:end,3)),'k-v','MarkerSize',10)
legend('  UMDA  ',' Mk-EDA','Tree-EDA',' Brute ','Location','SouthEast');
H = xlabel('Instances') 
set(H, 'Fontsize',fs);
H = ylabel('log_{10}\epsilon') 
set(H,'Fontsize',fs);
saveas(f,'ComparisonInstBurrello.eps','psc2');


***************************************************************


brute = ResultsBurrello(2:2:end,3)';
umda = themins(1,:);
mkeda = themins(2,:);
treeda = themins(3,:);

sum(umda<brute)
sum(mkeda<brute)
sum(treeda<brute)
