% Confronta i valori degli indici v_S e v_ST restituendo i vettori
% results_p e results_T contenenenti i paramentri Influenti ricavati
% dall'analisi di sensibilitÓ EFAST

v_S=vs_indices.v_S(1,:,1);
v_ST=vs_indices.v_ST(1,:,1);

results_p=Param.text;
ind=find(v_S>v_S(1));
results_p(length(ind)+1:end,:)=[];
ind=ind-1;

for i=1:length(ind)
    results_p(i,:)=Param.text(ind(i)-1,:);
end

results_T=Param.text;
results_T(length(ind)+1:end,:)=[];
ind=find(v_ST>v_ST(1));
ind=ind-1;

for i=1:length(ind)
    results_T(i,:)=Param.text(ind(i)-1,:);
end