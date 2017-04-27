%calcul le K-means du nouveau vecteur
opts = statset('Display','final');
[cluster_idx, cluster_center] = kmeans(Data_fin,10,'Distance','sqeuclidean','Replicates',5,'Options',opts);

% %affichage
% for j=1:10
%     subplot(5, 2, j); 
%     imshow(cluster_idx==j);
%     title(['Régions ' num2str(j)]);
% end
% figure, imshow(cluster_idx/10); title('Les 10 régions en dégradé');
