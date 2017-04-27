function [ cpt, vide, adresse] = listFolder( adresse, cpt)
    d=dir(adresse);
    [lg, r]=size(d);
    vide=0;
    for i=3:lg
        if d(i).isdir
            vide=vide+1;
            [cpt, v, name]=listFolder([adresse '/' d(i).name], cpt);
            if v==0 
                notEmpty (name);
            end
        end
    end
end

