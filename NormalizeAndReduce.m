function [ data_norm ] = NormalizeAndReduce(data, U_mat)
%UNTITLED12 Summary of this function goes here
%   Detailed explanation goes here

 nombre_person=size(data,1);
 
 for i=1:nombre_person
 
     nombre_seq=size(data{i},1);
     for j=1:nombre_seq
     g=data{i}{j};
     % normalize features 
     [X_norm, mu, sigma] = featureNormalize(g');
     
     %appliquer le PCA
     data_norm=X_norm*U_mat';
     data{i}{j}=data_norm;
     
     
     end
 end
 
data_norm=data;



end

