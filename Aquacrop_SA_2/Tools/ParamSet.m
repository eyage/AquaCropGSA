% Settaggio parametri
%Calcolo di TEW, TAW 

% Input
FC=0.46;
Rd_tab_end=1.1;
ProfHum=40;

if FC<0.18
    WP=FC-0.007 ;
elseif FC>=0.18 && FC<0.3
    WP=(0.38*FC)+0.0416 ;
elseif FC>=0.3                       % impostare il limite massimo di FC (inferiore a 1) nel file Param.m
    WP=(1.127*FC)-0.1766 ;
end

% Output
TEW=1000*(FC-0.5*WP)*0.05;
TAW=1000*(FC-WP)*Rd_tab_end;
TAWr=100*(FC-WP)*ProfHum;