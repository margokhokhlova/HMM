% load all the training data 
function trainCell=load_data(folder)
filenames = dir(fullfile(folder, '*.mat'));  % read all images with specified extention, its jpg in our case
 total_images = numel(filenames); 
 trainCell=cell(numel(filenames),1); %matrix to store the data
 for n = 1:total_images
     sequence=load(strcat(folder, '\', filenames(n).name));
     sequence=sequence.data_g;
     names = fieldnames(sequence);
     name_P=names{1};
     sequence=sequence.(name_P); % because I always have just one structure
     names = fieldnames(sequence); % get all the sequences for the walk of a person
     testData_s=cell(size(names,1),1);
     for j=1:numel(names)
        testData_s{j} =sequence.(names{j}).V;
     end
     trainCell{n}=testData_s;
   
 end
end