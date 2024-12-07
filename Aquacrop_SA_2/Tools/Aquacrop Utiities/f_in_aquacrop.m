function [Paths]=f_in_aquacrop(fileinput,endDay,endMonth, Param, Law)     
%% Creates input files and file .PRO to be used in the AquaCrop PlugIn
% needs to be launched in a directory having as subdirectories
% AquaCropFolder64bit and Aquacrop_plugin (with all their subdirectories)
% and directory MAT4AC (containing scripts and functions to create input
% files)
% .....
% INPUT:
%   fileinput   struttura con dentro tuti gli input di Aquacrop
%   endDay      giorno del mese del termine della simulazione
%   endMonth    mese del termine della simulazione
%   Param       file con lista parametri da ottimizzare e loro ranges ecc...
%   Law         p x N matrix of parameter realisations
% 
%  ASSUMPTIONS
%    1. all parameters except those in Param are fixed to values defined in the fileinput structure
%    2. all soil layers have the same fc and pwp
%    3. intial WC in the first depth (20 cm) is according to Param realisation, bottom depth (50 cm) is fixed at fc 
%       N.B. soil depths not layers !
% 
%Paolo C. Silvestro, March 2014
% modif Raffaele May 2014

     
%% Percorsi                
path0=pwd;                                                    %percorso di partenza
path1=fullfile(path0,'AquaCropFolder64bit','DATA\');          %Dati richiamati dal plugin AquaCrop
path2=fullfile(path0,'AquaCropFolder64bit','SIMUL\');         %Dati richiamati dal plugin Aquacrop
path3=fullfile(path0,'Aquacrop_plugin','LIST\');              %Cartella contenente il file .PRO da cui il Plugin AquaCrop carica le info.
path4=fullfile(path0,'Aquacrop_plugin\');                     %Cartella contenente il file eseguibile del plug-in di AquaCrop
% path5=fullfile(path0,'MAT4AC\');                            %Cartella di file utili a Matlab for AquaCrop (funzioni, utilit� e output)
% path_obs=fullfile(path0,'File observed\');                  %Cartella contenente il file .xls con le misure osesrvate
path_sim=fullfile(path4,'OUTP\');

Paths={};
Paths(1,1)={path1};
Paths(2,1)={path2};
Paths(3,1)={path3};
Paths(4,1)={path4};
Paths(5,1)={path_sim};
Paths(6,1)={path0};
% addpath(fullfile(path5,'Tool\'));                             
addpath(genpath(path0));                                        % Aggiunge i percorsi necessari
% addpath(path1);  
% addpath(path2);
% addpath(path3);
% addpath(path4);
% addpath(path_obs);
% addpath(path_sim);
%% cancella tutto quel che c'e' nella cartella LIST
cd(path3)
delete('*.*')
cd(path0)
%% number of realisations (simulations)
N= size(Law,2);

%% Dichiarazione Input 
%Input Variabili  !!!!! N.B. alcune variabili, es. var1, var2... le usa dentro lo script writerPRO
ni=1;               %numero di iterazioni
                  
% First day of simulation period        
spidd=cell2mat(fileinput.simdate.first(1));          %giorno (dd) formattato come richiesto dalla funzione dataconv.m 
spimm=cell2mat(fileinput.simdate.first(2));          %mese (mm) da scrivere come richiesto in dataconv (jan,feb,mar,etc.)
spiyy=cell2mat(fileinput.simdate.first(3));          %anno (yy)     
Dates(1)= dataconv(spidd,spimm,spiyy); %First day of simulation period
     
% Last day of simulation period        
% spfdd=cell2mat(fileinput.simdate.last(1));          %giorno (dd) formattato come richiesto dalla funzione dataconv.m 
% spfmm=cell2mat(fileinput.simdate.last(2));          %mese (mm) da scrivere come richiesto in dataconv (jan,feb,mar,etc.) 
spfdd=endDay;                                       %prende dall'input la data di fine simulazione
spfmm=endMonth;                                     %prende dall'input il mese di fine simulazione
spfyy=cell2mat(fileinput.simdate.last(3));          %anno (yy)     
Dates(2)= dataconv(spfdd,spfmm,spfyy);                  %Last day of simulation period  
     
% First day of cropping period         
cpidd=cell2mat(fileinput.cropdate.first(1));        %giorno (dd) formattato come richiesto dalla funzione dataconv.m 
cpimm=cell2mat(fileinput.cropdate.first(2));        %mese (mm) da scrivere come richiesto in dataconv (jan,feb,mar,etc.) 
cpiyy=cell2mat(fileinput.cropdate.first(3));        %anno (yy)     
Dates(3)= dataconv(cpidd,cpimm,cpiyy); %First day of cropping period
     
% Last day of cropping period         
cpfdd=cell2mat(fileinput.cropdate.last(1));        %giorno (dd) formattato come richiesto dalla funzione dataconv.m 
cpfmm=cell2mat(fileinput.cropdate.last(2));        %mese (mm) da scrivere come richiesto in dataconv (jan,feb,mar,etc.) 
cpfyy=cell2mat(fileinput.cropdate.last(3));        %anno (yy)     
Dates(4)= dataconv(cpfdd,cpfmm,cpfyy); %data dell'ultimo giorno di crescita convertita  

%% Creation input files
% Creation Climate files
fileinput.climate=fileinput.CLIMA.txt;

cli={};
cli(1)= {fileinput.climate};             %info sul clima
fCLI=climatex(cell2mat(cli(1)));
cli(1)=fCLI(1);                        % Climate File
cli(2)= fCLI(2);                       %info Temperatura 
cli(3)= fCLI(3);                       % Evapotraspirazione 
cli(4)= fCLI(4);                       % Piovosit�
cli(5)= fCLI(5);                       % Anidride Carbonica Atmosferica

%% constraints
% loop for each realisation
names=Param.text(2:end,1); %cell array of parameter names
fc_row = strmatch('fc', names, 'exact');
pwp_row = strmatch('pwp', names, 'exact');
WC_row = strmatch('WC', names, 'exact');
pexlw_row=strmatch('pexlw', names, 'exact');
pexup_row=strmatch('pexup', names, 'exact');

%% Aggiunta Paolo, soluzione problema eme-flo-mat-sen part 1/2
eme_row = strmatch('eme', names, 'exact');
flo_row = strmatch('flo', names, 'exact');
mat_row = strmatch('mat', names, 'exact');
sen_row = strmatch('sen', names, 'exact');
    


for e=1:N  %loop for each ensemble
  %% Aggiunta Paolo, soluzione problema eme-flo-mat-sen, part 2/2
  eme=Law(eme_row,e);
  flo=Law(flo_row,e);
  mat=Law(mat_row,e);
  sen=Law(sen_row,e);
  if sen>mat
      sen=mat-0.3*mat;
      Law(sen_row,e)=sen;
  end
     
  if flo>sen
      flo=sen-0.4*sen;
      Law(flo_row,e)=flo;
  end
    
  if eme>flo
      eme=flo-0.6*flo;
      Law(eme_row,e)=eme;
  end
 %%   

  

  
  %%
    % constraints
   fc=Law(fc_row,e);
   pwp=Law(pwp_row,e);
   WC=Law(WC_row,e);
   pexup=Law(pexup_row,e);
   pexlw=Law(pexlw_row,e);

   % saturation is function of fc
   sat=0.4764*fc+30.38;
   fileinput.soil.sat(:)=sat;
   
   % pwp should be less than fc by at least 7 %vol
   if fc-pwp<7
       pwp=fc-7;
       Law(pwp_row,e)=pwp;
   end
   
   % intial soil moisture WC should be between fc and pwp
   if WC>fc 
       WC=fc;
       Law(WC_row,e)=WC;
   elseif WC<pwp
       WC=pwp;
       Law(WC_row,e)=WC;
   end
   
   % Soil water depletion factor for canopy expansion - Lower threshold (pexlw) should be larger than Upper threshold (pexup) 
    if pexlw<pexup
       pexup=max(0, pexlw-0.2);
       Law(pexup_row,e)=pexup;
    end

    
    for p=1:size(Param.num,1)               %each variable parameter
        par_name=cell2mat(Param.text(p+1,1));
        val=Law(p,e);                           % parameter value random realisation
        loc=cell2mat(Param.text(p+1,2));        % its location (field) in the fileinput structure
        loc=[loc '.' par_name];                 % full path in structure fileinput
        if ~ isempty(strmatch(par_name,'fc')) || ~ isempty(strmatch(par_name,'pwp'))  %assumes equal fc or pwp for all soil layers
            nhor=fileinput.soil.nhor;
            for n=1:nhor
                stringa=sprintf('%s(n)=%f',loc, val); 
                eval(stringa)
            end
        elseif  ~ isempty(strmatch(par_name,'WC'))
            % uses the random realisation only for initial conditions of the top soil layer (30 cm)
            stringa=sprintf('%s(1)=%f',loc, val); 
            eval(stringa)
            % sets all the other bottom layers to fc
            fc=fileinput.soil.fc(1);
            stringa=sprintf('%s(2)=%f',loc, fc); 
            eval(stringa)
        else
            stringa=sprintf('%s=%f',loc, val); 
            eval(stringa)                           % writes the value from Law into the fileinput structure
        end
    end

%% Trasformazione in numero intero di alcune variabili
fileinput.soil.cn=int8(fileinput.soil.cn);
%% Variazione Day of sowing (dos, in JD) 
% start simulation, stop simulation, sowing day, end crop cycle day


% Input Variabili  !!!!! N.B. alcune variabili, es. var1, var2... le usa dentro lo script writerPRO
% ni=1;               %numero di iterazioni
JD=fileinput.cropdate.dos;
yy=fileinput.cropdate.yearsow;
[dd,mm]=JD2data(JD,yy);
fileinput.cropdate.first(1)={dd};
fileinput.cropdate.first(2)={mm};


% First day of simulation period        
spidd=cell2mat(fileinput.simdate.first(1));          %giorno (dd) formattato come richiesto dalla funzione dataconv.m 
spimm=cell2mat(fileinput.simdate.first(2));          %mese (mm) da scrivere come richiesto in dataconv (jan,feb,mar,etc.)
spiyy=cell2mat(fileinput.simdate.first(3));          %anno (yy)     
Dates(1)= dataconv(spidd,spimm,spiyy); %First day of simulation period
      
% Last day of simulation period        
% spfdd=cell2mat(fileinput.simdate.last(1));          %giorno (dd) formattato come richiesto dalla funzione dataconv.m 
% spfmm=cell2mat(fileinput.simdate.last(2));          %mese (mm) da scrivere come richiesto in dataconv (jan,feb,mar,etc.) 
spfdd=endDay;                                       %prende dall'input la data di fine simulazione
spfmm=endMonth;                                     %prende dall'input il mese di fine simulazione
spfyy=cell2mat(fileinput.simdate.last(3));          %anno (yy)     
Dates(2)= dataconv(spfdd,spfmm,spfyy);                  %Last day of simulation period  
     
% First day of cropping period         
cpidd=cell2mat(fileinput.cropdate.first(1));        %giorno (dd) formattato come richiesto dalla funzione dataconv.m 
cpimm=cell2mat(fileinput.cropdate.first(2));        %mese (mm) da scrivere come richiesto in dataconv (jan,feb,mar,etc.) 
cpiyy=cell2mat(fileinput.cropdate.first(3));        %anno (yy)     
Dates(3)= dataconv(cpidd,cpimm,cpiyy); %First day of cropping period
     
% Last day of cropping period         
cpfdd=cell2mat(fileinput.cropdate.last(1));        %giorno (dd) formattato come richiesto dalla funzione dataconv.m 
cpfmm=cell2mat(fileinput.cropdate.last(2));        %mese (mm) da scrivere come richiesto in dataconv (jan,feb,mar,etc.) 
cpfyy=cell2mat(fileinput.cropdate.last(3));        %anno (yy)     
Dates(4)= dataconv(cpfdd,cpfmm,cpfyy); %data dell'ultimo giorno di crescita convertita 
 
    % Creation Soil text file
    soilname=sprintf('Soil_e%d.SOL', e);        % File Name incremental
    f_soil(path1,soilname, fileinput)                   % writes soil text file  
    fileinput.soilname=soilname;
    % Creation Crop text File
    cropname=sprintf('Crop_e%d.CRO', e);        % File Name incremental    
    f_crop(path1,cropname,fileinput);                 % writes crop text file  
    fileinput.cropname=cropname;
    % Creation Initial soil condition text File
    inconname=sprintf('InCon_e%d.SW0', e);
    f_InCon(path1,inconname,fileinput);               %writes initial soil conditions text file     
    fileinput.inconname=inconname;

    namepro=sprintf('Pro_e%d.PRO', e);
    Names={};
    Names(1,1)={soilname};
    Names(2,1)={cropname};
    Names(3,1)={inconname};
    Names(4,1)={namepro};
    Names(5,1)={fileinput.Irrname};
    Names(6,1)={fileinput.Manname};
    Names(7,1)={fileinput.Gwtname};
    f_writerPRO(Paths,Dates,Names,fileinput,cli)                              %Script che crea il file di testo .PRO e lo mette nella cartella LIST del Plug in
end