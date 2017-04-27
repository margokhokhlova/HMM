function [ b ] = notEmpty ( name )
    global l;
    sd=dir([name '/pdflow_cloud*']); % sous dossier
    [b, r]=size(sd);
    if b
        disp(name);
        l{end+1,1}=name;
    end
end

