function [ trans ] = Transission( matrice )

% cluster indexes 1D 
%     [l]=size(matrice);
%     k=max(matrice);
%     trans=ones(k);
%     for i=2:l
%         trans(matrice(i-1), matrice(i))=trans(matrice(i-1), matrice(i))+1;
%     end
%     for i=1:k
%        s=sum(trans(i, :));
%        trans(i,:)=trans(i, :)/s;
%     end
% end
% 
% 
    [l]=size(matrice);
    k=max(matrice);
    trans=ones(k);
    for i=2:l
        trans(matrice(i-1), matrice(i))=trans(matrice(i-1), matrice(i))+1;
    end
    for i=1:k
        trans(i,:)=trans(i,:)/sum(trans(i,:));
    end
end