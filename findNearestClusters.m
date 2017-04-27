function [ idx ] = findNearestClusters( seq_train, C, nb_states )

%UNTITLED14 Summary of this function goes here
%   Function takes cluster centers and training sequence and finds the
%   correpondences between them
% output is 1d vector of neares clusters for each frame

idx=zeros(size(seq_train,1),1);
    for m=1:size(seq_train,1);   
        D=zeros(nb_states,1);
        for n=1:nb_states %equal to number of clusters
            %find the neares cluster
            %euclidean distance
            D(n) = sqrt(sum((C(n) - seq_train(m)) .^ 2));
        end
        [val, pos]=min(D);
        idx(m)=pos; % find the neares cluster
            
    
    end

end

