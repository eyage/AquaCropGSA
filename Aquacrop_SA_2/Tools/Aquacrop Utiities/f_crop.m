%%%Crop model Creator for AquaCrop Plugin
 %% Paolo C. Silvestro March 2014
 
 %%Percorsi
 function f_crop(path1,cropname,fileinput)                 
%% Aquacrop parameters adjustment
dos=fileinput.cropdate.dos; % day of sowing
dof=fileinput.cropdate.dof; % day of end crop cycle
TMP_mean= fileinput.CLIMA.TMP;

[eme_cd,root_cd,sen_cd,mat_cd,flo_cd,flolen_cd]=GDD2Cal(fileinput,TMP_mean,dos,dof);

fileinput.crop.other.eme_cd=eme_cd;
fileinput.crop.other.root_cd=root_cd;
fileinput.crop.other.sen_cd=sen_cd;
fileinput.crop.other.mat_cd=mat_cd;
fileinput.crop.other.flo_cd=flo_cd;
fileinput.crop.other.flolen_cd=flolen_cd;

%% INPUT 

%header
type='Default Wheat'; % Crop Type
time='GDD'; % day or GDD (growth degree day)
loc='Viterbo'; %Place
datesel='4Dec06'; %Date

%Variables 
 var1=fileinput.crop.other.version;                  %AquaCrop Version (June 2012)
 var2=0;                                             %File protected
 var3=fileinput.crop.other.type;                    %fruit/grain producing crop
 var4=fileinput.crop.other.t_s;                    %Crop is sown
 var5=fileinput.crop.other.cycle;                    %Determination of crop cycle : by growing degree-days
 var6=fileinput.crop.other.p_soil;                    %Soil water depletion factors (p) are adjusted by ETo
 var7=fileinput.crop.other.To_crop;                  %Base temperature (�C) below which crop development does not progress
 var8=fileinput.crop.other.Tmax_crop;                 %Upper temperature (�C) above which crop development no longer increases with an increase in temperature
 var9=fileinput.crop.canopy.mat;                 %Total length of crop cycle in growing degree-days
 var10=fileinput.crop.biomass.pexup;                %Soil water depletion factor for canopy expansion (p-exp) - Upper threshold
 var11=fileinput.crop.biomass.pexlw;                %Soil water depletion factor for canopy expansion (p-exp) - Lower threshold
 var12=fileinput.crop.biomass.pexshp;                 %Shape factor for water stress coefficient for canopy expansion (0.0 = straight line)
 var13=fileinput.crop.biomass.psto;                %Soil water depletion fraction for stomatal control (p - sto) - Upper threshold
 var14=fileinput.crop.biomass.pstoshp;                 %Shape factor for water stress coefficient for stomatal control (0.0 = straight line)
 var15=fileinput.crop.biomass.psen;                %Soil water depletion factor for canopy senescence (p - sen) - Upper threshold
 var16=fileinput.crop.biomass.psenshp;                 %Shape factor for water stress coefficient for canopy senescence (0.0 = straight line)
 var17=fileinput.crop.other.SumET0;                   %Sum(ETo) during stress period to be exceeded before senescence is triggered
 var18=fileinput.crop.biomass.ppol;                %Soil water depletion factor for pollination (p - pol) - Upper threshold
 var19=fileinput.crop.biomass.anaer;                   %Vol% for Anaerobiotic point (* (SAT - [vol%]) at which deficient aeration occurs *)
 var20=fileinput.crop.other.Ssf;                  %Considered soil fertility/salinity stress for calibration of stress response (%)
 var21=fileinput.crop.other.Rcan_exp;               %Response of canopy expansion is not considered
 var22=fileinput.crop.other.Rmax_CC;               %Response of maximum canopy cover is not considered
 var23=fileinput.crop.other.RWP;               %Response of crop Water Productivity is not considered
 var24=fileinput.crop.other.RdCC;               %Response of decline of canopy cover is not considered
 var25=fileinput.crop.other.Rsc;               %Response of stomatal closure is not considered
 var26=fileinput.crop.biomass.polmn;                   %Minimum air temperature below which pollination starts to fail (cold stress) (�C)
 var27=fileinput.crop.biomass.polmx;                  %Maximum air temperature above which pollination starts to fail (heat stress) (�C)
 var28=fileinput.crop.biomass.stbio;                %Minimum growing degrees required for full biomass production (�C - day)
 var29=fileinput.crop.other.ecsss;                   %Electrical Conductivity of soil saturation extract at which crop starts to be affected by soil salinity (dS/m)
 var30=fileinput.crop.other.ecss;                  %Electrical Conductivity of soil saturation extract at which crop can no longer grow (dS/m)
 var31=fileinput.crop.other.sfsssc;                   %Shape factor for soil salinity stress coefficient (0 : linear response)
 var32=fileinput.crop.trasp.kc;                %Crop coefficient when canopy is complete but prior to senescence (Kcb,x)
 var33=fileinput.crop.trasp.kcdcl;               %Decline of crop coefficient (%/day) as a result of ageing, nitrogen deficiency, etc.
 var34=fileinput.crop.root.rtmin;                %Minimum effective rooting depth (m)
 var35=fileinput.crop.root.rtx;                %Maximum effective rooting depth (m)
 var36=fileinput.crop.root.rtshp;                  %Shape factor describing root zone expansion
 var37=fileinput.crop.root.rtexup;               %Maximum root water extraction (m3water/m3soil.day) in top quarter of root zone
 var38=fileinput.crop.root.rtexlw;               %Maximum root water extraction (m3water/m3soil.day) in bottom quarter of root zone
 var39=fileinput.crop.trasp.evardc;                  %Effect of canopy cover in reducing soil evaporation in late season stage
 var40=fileinput.crop.canopy.ccs;                %Soil surface covered by an individual seedling at 90 % emergence (cm2)
 var41=fileinput.crop.canopy.den;             %Number of plants per hectare
 var42=fileinput.crop.canopy.cgc;             %Canopy growth coefficient (CGC): Increase in canopy cover (fraction soil cover per day)
 var43=fileinput.crop.other.MdCCbs;                  %decrease of Canopy Growth Coefficient in and between seasons - Not Applicable
 var44=fileinput.crop.other.nsCC;                  %Number of seasons at which maximum decrease of Canopy Growth Coefficient is reached - Not Applicable
 var45=fileinput.crop.other.sfdCCc;                %Shape factor for decrease Canopy Growth Coefficient - Not Applicable
 var46=fileinput.crop.canopy.ccx;                %Maximum canopy cover (CCx) in fraction soil cover
 var47=fileinput.crop.canopy.cdc;             %Canopy decline coefficient (CDC): Decrease in canopy cover (in fraction per day)
 var48=fileinput.crop.other.eme_cd;                  %Calendar Days: from sowing to emergence
 var49=fileinput.crop.other.root_cd;                  %Calendar Days: from sowing to maximum rooting depth
 var50=fileinput.crop.other.sen_cd;                 %Calendar Days: from sowing to start senescence
 var51=fileinput.crop.other.mat_cd;                 %Calendar Days: from sowing to maturity (length of crop cycle)
 var52=fileinput.crop.other.flo_cd;                 %Calendar Days: from sowing to flowering
 var53=fileinput.crop.other.flolen_cd;                  %Length of the flowering stage (days)
 var54=fileinput.crop.other.cdlf;                   %Crop determinancy linked with flowering
 var55=fileinput.crop.biomass.exc;                 %Excess of potential fruits (%)
 var56=fileinput.crop.other.hilen_cd;                  %Building up of Harvest Index starting at flowering (days)
 var57=fileinput.crop.biomass.wp ;               %Water Productivity normalized for ETo and CO2 (WP*) (gram/m2)
 var58=fileinput.crop.other.wp_yfp;                 %Water Productivity normalized for ETo and CO2 during yield formation (as % WP*)
 var59=fileinput.crop.other.cpco2;                  %Crop performance under elevated atmospheric CO2 concentration (%)
 var60=fileinput.crop.biomass.hi;                  %Reference Harvest Index (HIo) (%)
 var61=fileinput.crop.biomass.hipsflo;                   %Possible increase (%) of HI due to water stress before flowering
 var62=fileinput.crop.biomass.hipsveg;                %Coefficient describing positive impact on HI of restricted vegetative growth during yield formation
 var63=fileinput.crop.biomass.hingsto;                 %Coefficient describing negative impact on HI of stomatal closure during yield formation
 var64=fileinput.crop.biomass.hinc;                  %Allowable maximum increase (%) of specified HI
 var65=fileinput.crop.canopy.eme;                 %GDDays: from sowing to emergence
 var66=fileinput.crop.root.root;                 %GDDays: from sowing to maximum rooting depth
 var67=fileinput.crop.canopy.sen;                %GDDays: from sowing to start senescence
 var68=fileinput.crop.canopy.mat;                %GDDays: from sowing to maturity (length of crop cycle)
 var69=fileinput.crop.canopy.flo;                %GDDays: from sowing to flowering
 var70=fileinput.crop.canopy.flolen;                 %Length of the flowering stage (growing degree days)
 var71=fileinput.crop.other.cgc4ggd;            %CGC for GGDays: Increase in canopy cover (in fraction soil cover per growing-degree day)
 var72=fileinput.crop.other.cdc4ggd;            %CDC for GGDays: Decrease in canopy cover (in fraction per growing-degree day)
 var73=fileinput.crop.canopy.hilen;                %GDDays: building-up of Harvest Index during yield formation
%% Text File .CRO Creation
 
%  cd(path1)
filename=[path1 cropname];
f1 = fopen(filename,'wt');

fprintf(f1,'%s, %s (%s, %s )\n',type,time,loc,datesel);
fprintf(f1,'      %1.1f\n',var1);
fprintf(f1,'      %1.0f \n',var2);
fprintf(f1,'      %1.0f \n',var3);
fprintf(f1,'      %1.0f \n',var4);
fprintf(f1,'      %1.0f \n',var5);
fprintf(f1,'      %1.0f\n',var6);
fprintf(f1,'      %1.1f\n',var7);
fprintf(f1,'     %2.1f\n',var8);
fprintf(f1,'   %4.0f\n',var9);
fprintf(f1,'      %1.2f\n',var10);
fprintf(f1,'      %1.2f\n',var11);
fprintf(f1,'      %1.1f\n',var12);
fprintf(f1,'      %1.2f\n',var13);
fprintf(f1,'      %1.1f\n',var14);
fprintf(f1,'      %1.2f\n',var15);
fprintf(f1,'      %1.1f\n',var16);
fprintf(f1,'      %1.0f\n',var17);
fprintf(f1,'      %1.2f\n',var18);
fprintf(f1,'      %1.0f\n',var19);
fprintf(f1,'     %2.0f\n',var20);
fprintf(f1,'     %2.2f\n',var21);
fprintf(f1,'     %2.2f\n',var22);
fprintf(f1,'     %2.2f\n',var23);
fprintf(f1,'     %2.2f\n',var24);
fprintf(f1,'     %2.2f\n',var25);
fprintf(f1,'      %1.0f\n',var26);
fprintf(f1,'     %2.0f\n',var27);
fprintf(f1,'     %2.1f \n',var28);
fprintf(f1,'     %1.0f\n',var29);
fprintf(f1,'     %1.0f\n',var30);
fprintf(f1,'      %1.0f \n',var31);
fprintf(f1,'      %1.2f\n',var32);
fprintf(f1,'      %1.3f\n',var33);
fprintf(f1,'      %1.2f\n',var34);
fprintf(f1,'      %1.2f\n',var35);
fprintf(f1,'     %2.0f\n',var36);
fprintf(f1,'      %1.3f\n',var37);
fprintf(f1,'      %1.3f\n',var38);
fprintf(f1,'     %2.0f\n',var39);
fprintf(f1,'      %1.2f\n',var40);
fprintf(f1,'   %7.0f\n',var41);
fprintf(f1,'      %1.5f\n',var42);
fprintf(f1,'     %1.0f\n',var43);
fprintf(f1,'     %1.0f\n',var44);
fprintf(f1,'     %1.1f\n',var45);
fprintf(f1,'      %1.2f\n',var46);
fprintf(f1,'      %1.5f\n',var47);
fprintf(f1,'     %2.0f\n',var48);
fprintf(f1,'     %2.0f\n',var49);
fprintf(f1,'    %3.0f\n',var50);
fprintf(f1,'    %3.0f\n',var51);
fprintf(f1,'    %3.0f\n',var52);
fprintf(f1,'     %2.0f\n',var53);
fprintf(f1,'      %1.0f \n',var54);
fprintf(f1,'    %3.0f \n',var55);
fprintf(f1,'     %2.0f \n',var56);
fprintf(f1,'     %2.1f \n',var57);
fprintf(f1,'    %3.0f\n',var58);
fprintf(f1,'     %2.0f\n',var59);
fprintf(f1,'     %2.0f\n',var60);
fprintf(f1,'      %1.0f\n',var61);
fprintf(f1,'     %2.1f\n',var62);
fprintf(f1,'      %1.1f\n',var63);
fprintf(f1,'     %2.0f\n',var64);
fprintf(f1,'     %2.0f\n',var65);
fprintf(f1,'    %3.0f\n',var66);
fprintf(f1,'   %4.0f\n',var67);
fprintf(f1,'   %4.0f\n',var68);
fprintf(f1,'   %4.0f\n',var69);
fprintf(f1,'    %3.0f\n',var70);
fprintf(f1,'      %1.6f\n',var71);
fprintf(f1,'      %1.6f\n',var72);
fprintf(f1,'   %4.0f\n',var73);
   
fclose(f1);
    