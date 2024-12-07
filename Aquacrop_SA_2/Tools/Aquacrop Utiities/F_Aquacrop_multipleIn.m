function [CC_m,Yield_m]=F_Aquacrop_multipleIn(N,Paths) 

cd(cell2mat(Paths(5)))
n=1;
 file_sim=sprintf('Pro_e%dPROday.OUT', n);
 data_sim=dlmread(file_sim,'',4,0 );
 d=length(data_sim(:,1));
 
CC_m=zeros(N,d);
Yield_m=zeros(N,d);
    for n=1:N
        file_sim=sprintf('Pro_e%dPROday.OUT', n);
        data_sim=dlmread(file_sim,'',4,0 );
        CC_m(n,:)=data_sim(:,12); %all values for plotting 
        Yield_m(n,:)=data_sim(:,21);
    end
    cd(cell2mat(Paths(6)))
    