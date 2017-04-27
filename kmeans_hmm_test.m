% function to check clusters and probability
load('testing_norm.mat');  
nombre_person=size(testing_norm,1);
K=7;
data=[];
for j=1:nombre_person % for each person
        P2_test=testing_norm{j}{1}; %test
        data=[data; P2_test];
        
end
opts = statset('Display','final');
[idx,C] = kmeans(data, K,  'Replicates',5,'Options',opts); %'Distance','sqeuclidean',


for j=1:nombre_person % for each person
        P2_test=testing_norm{j}{1}; %test
        idxCL=zeros(size(P2_test,1),1);
    for m=1:size(P2_test,1);  
           
            D=zeros(K,1);
            for n=1:K %equal to number of clusters
            %find the neares cluster
            %euclidean distance
            D(n) = sqrt(sum((C(n,:) -P2_test(m,:)) .^ 2));
%             obs=[C; P2_test(m)];
%             D=pdist(obs, 'sqeuclidean');
            end
        [val, pos]=min(D);
    idxCL(m)=pos; % find the neares cluster
    end
    
end