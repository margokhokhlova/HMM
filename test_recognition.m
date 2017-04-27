%
%
% Unit test with GMM's and human gate data
%


function test_recognition(train, test, C)
% train = training data, test= test data, C= cluster centers


close all;

%parameters for HMM
nb_feature = 20; % change for an automatic calc of the size
nb_states = 7;
nb_gmm = 1;
cov_type = 'diag';
% Also,forthecontinuous models, we havefound that it is preferableto use
% diagonal covariance matriceswith several mixtures, rather than fewer
% mixtures with full covariance matrices.

%Get the data for traning
nombre_person=size(train,1);
HMM_mod=cell(nombre_person,1);
 for i=1:nombre_person
      seq_train=[];
      
      nombre_seq=size(train{i},1);
     for j=1:nombre_seq
        seq_train=[seq_train; train{i}{j}];    
     end
     %Here we have trining data for a person -> train the model
     % data_train{i}=seq_train;
 
% Cluster settings
prior0 		= normalize(ones(nb_states,1));%normalize(rand(nb_states, 1)); % prior 
transmat0 	= mk_stochastic(rand(nb_states, nb_states));

% Inital cluster guess using kmeans -> performed only if cluster centers
% are provided

if nargin==3 % if cluster centers given, change the transmat
idx=zeros(size(seq_train,1),1);
    for m=1:size(seq_train,1);   
        D=zeros(nb_states,1);
        for n=1:nb_states %equal to number of clusters
            %find the neares cluster
            %euclidean distance
           D(n) = sqrt(sum((C(n,:) -P2_test(m,:)) .^ 2));
        end
        [val, pos]=min(D);
        idx(m)=pos; % find the neares cluster
            
    
    end

%replace transmat0 by a cluster probability for a sequence

[ transmat0 ] = Transission( idx );

% calculate mu and sigma for each cluster
mu=[];
sigma=zeros(20:20);
Sigma0=zeros(20:20:7);
for j=1:nb_states
mu=[mu;mean(seq_train(find(idx==j), :))];
diagS=std(seq_train(find(idx==j), :));
for t=1:20
    sigma(t,t)=diagS(t);
end
Sigma0(:,:,j)=sigma;
end
end

%HMM parameters
 %[mu0, Sigma0, weights] = mixgauss_init(nb_states*nb_gmm, seq_train', cov_type); % change

%  change to learn all this from initial cluster parameters
%mu0 	= reshape(mu0, [nb_feature nb_states nb_gmm]);
%Sigma0 	= reshape(Sigma0, [nb_feature nb_feature nb_states nb_gmm]);
mu0=mu';
Sigma0= reshape(Sigma0, [nb_feature nb_feature nb_states nb_gmm]);

mixmat0 = mk_stochastic(rand(nb_states, nb_gmm)); %check this matrix

%mixmat0=reshape(weights, nb_states, nb_gmm);
%Train
[LL, prior1, transmat1, mu1, Sigma1, mixmat1] = ...
    mhmm_em(seq_train', prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 500);
 
HMM_mod{i}=struct('LL', LL, 'prior1', prior1, 'transmat1', transmat1, 'mu1', mu1, 'Sigma1', Sigma1, 'mixmat1', mixmat1);
 
 
 
 end
 
 
%% TEST the model on data
 
Matrix_results=zeros(nombre_person, nombre_person); % matrix to save the recognition
Matrix_results_test=zeros(nombre_person, nombre_person); % matrix to save the recognition
% rows are sequence and cols are model
 % test on the given models for all the testing data
 for i=1:nombre_person %for each model
    prior1=HMM_mod{i}.prior1;
    transmat1=HMM_mod{i}.transmat1;
    mu1=HMM_mod{i}.mu1;
    Sigma1=HMM_mod{i}.Sigma1;
    mixmat1= HMM_mod{i}.mixmat1;   
  
  
     
     for j=1:nombre_person % for each person
    
         
  

        P1_test=train{j}{1}; %train
        P2_test=test{j}{1}; %test
        
        ll_P1_test = mhmm_logprob(P1_test', prior1, transmat1, mu1, Sigma1, mixmat1);% train data

        ll_P2_test = mhmm_logprob(P2_test', prior1, transmat1, mu1, Sigma1, mixmat1); %test
        
        %fprintf('Training data log prob %2.4f\n', ll_P1_test);
        Matrix_results(i,j)=(ll_P1_test);
        Matrix_results_test(i,j)=(ll_P2_test);
       
     
 
     end
 end

 
 % display the results
 [val, ind]=max(Matrix_results); %If A is a matrix, then min(A) is a row vector containing the minimum value of each column
 % -> then I have for each person the model which fits the most
 
 [val2, ind2]=max(Matrix_results_test);
  fprintf('Person\n');
 1:8
 fprintf('Results on training data\n');
 ind
 
 fprintf('Results on testing data\n');
 ind2
 
 Matrix_results_test
 
 

% 
% ll_P1_test 		= mhmm_logprob(P1_test', prior1, transmat1, mu1, Sigma1, mixmat1);
% ll_P2_test 		= mhmm_logprob(P2_test', prior2, transmat2, mu2, Sigma2, mixmat2);
% 
% ll_P1_cross_test 	= mhmm_logprob(P2_test', prior2, transmat2, mu2, Sigma2, mixmat2);
% ll_P2_cross_test 	= mhmm_logprob(P1_test', prior1, transmat1, mu1, Sigma1, mixmat1);








