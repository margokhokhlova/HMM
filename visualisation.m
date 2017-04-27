%visualization of the features
load('testing_norm.mat');
load('training_norm.mat');
load('G:\GitHub\HMM\data\data_abnormal.mat');
test=testing_norm;
train=training_norm;

nombre_person=size(train,1);

 for i=1:nombre_person
      seq_train=[];
      seq_test=[];
      nombre_seq=size(train{i},1);
     for j=1:nombre_seq
        seq_train=[seq_train; train{i}{j}];    
     end
    seq_test =[seq_test;   test{i}{1}];
     
 end
%% abnormal data
data_a=[];
for i=size(data,2)
    data_a=[data_a, data{i}];
    
end
     
     alldata=[seq_train; seq_test; data_a];
     plot(alldata(:,1:10));
     hold on
%# vertical line
 
SP=size(seq_train,1); %your point goes here 
line([SP SP], [-6;6],'Color',[1 0 0])
SP=SP+size(seq_test,1); %all the data size
line([SP SP], [-6; 6],'Color',[1 0 0]);