%% Unit test with GMM's and human gate data
%


function test_recognition(train, test, C)
% train = training data, test= test data, C= cluster centers
close all;
%parameters for HMM
nb_feature = 20; % change for an automatic calc of the size
nb_states = 7;
nb_gmm = 1;
cov_type = 'diag';
prior0 		= normalize(ones(nb_states,1));%normalize(rand(nb_states, 1)); % prior 
prior0= [0.0666666666666667 0.466666666666667 0.0666666666666667 0.200000000000000 0.0666666666666667 0.0666666666666667 0.0666666666666667];
transmat0 	= mk_stochastic(rand(nb_states, nb_states));
nombre_person=size(train,1);
% Also,forthecontinuous models, we havefound that it is preferableto use
% diagonal covariance matriceswith several mixtures, rather than fewer
% mixtures with full covariance matrices.

% if there are no cluster centers provided, recalculate them
if nargin==2
    K=nb_states;
   
   data=[];
    for i=1:nombre_person
       nombre_seq=size(train{i},1);
       for j=1:nombre_seq
         data=[data; train{i}{j}];    
       end
    end
    opts = statset('Display','final');
    [idx,C] = kmeans(data, K,  'Replicates', 15,'Options',opts); %'Distance','sqeuclidean',
 
    %calculate the cluster centers and variance and mu within the clusters
    mu_Cluster=[]; sigma_Cluster=[];
    for j=1:K
        mu_Cluster=[mu_Cluster;mean(data(find(idx==j), :))];
        sigma_Cluster=[ sigma_Cluster; std(data(find(idx==j),:))]; % varinance var  (?)
        for t=1:20
             sigma(t,t)=sigma_Cluster(j,t);  % CHANGE THIS PART FOR SOMETHING correct
        end
    Sigma0(:,:,j)=sigma;
    end
end


%Get the data for traning

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
% Inital cluster guess using kmeans -> performed only if cluster centers
% are provided
% if cluster centers given, change the transmat

    idx_s=zeros(size(seq_train,1),1);   % transition matrix is unique for each model
    for m=1:size(seq_train,1);   
        D=zeros(nb_states,1);
        for n=1:nb_states %equal to number of clusters
            %find the neares cluster
            %euclidean distance
            D(n) = sqrt(sum((C(n) - seq_train(m)) .^ 2));
        end
        [val, pos]=min(D);
        idx_s(m)=pos; % find the neares cluster
    end

%replace transmat0 by a cluster probability for a sequence

    [ transmat0 ] = Transission( idx_s ); % train sequence 
%HMM parameters
%[mu_p0, Sigma0, weights] = mixgauss_init(nb_states*nb_gmm, seq_train', cov_type); % change
%  change to learn all this from initial cluster parameters
%mu0 	= reshape(mu0, [nb_feature nb_states nb_gmm]);
%Sigma0 	= reshape(Sigma0, [nb_feature nb_feature nb_states nb_gmm]);
    mu0=mu_Cluster';
    Sigma0= reshape(Sigma0, [nb_feature nb_feature nb_states nb_gmm]);

    mixmat0 = mk_stochastic(rand(nb_states, nb_gmm)); %check this matrix

    %mixmat0=reshape(weights, nb_states, nb_gmm);
    %Train
    [LL, prior1, transmat1, mu1, Sigma1, mixmat1] = ...
    mhmm_em(seq_train', prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 30);
 
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
        
        % path Computing the most probable sequence (Viterbi)

        %First you need to evaluate B(t,i) = P(y_t | Q_t=i) for all t,i:
%  B = mixgauss_prob(P2_test', mu1, Sigma1, mixmat1); %(:,:,P1_test')
% % % where data(:,:,ex) is OxT where O is the size of the observation vector. Finally, use
%  [path] = viterbi_path(prior1, transmat1, B);
% % 
% % %if nargin==3 % if cluster centers given, change the transmat
%     idxCL=zeros(size(P2_test,1),1);
%     fprintf('Person %d \n', j);
%         for m=1:size(P2_test,1);  
%            
%             D=zeros(nb_states,1);
%             for n=1:nb_states %equal to number of clusters
%             %find the neares cluster
%             %euclidean distance
%              D(n) = sqrt(sum((C(n,:) -P2_test(m,:)) .^ 2));
%             end
%         [val, pos]=min(D);
%         idxCL(m)=pos; % find the neares cluster
%       
%         end
%  fprintf('Estimated by HMM path\n');  
%  path
%  fprintf('Nearest Clusters\n');
%  idxCL'
% figure
% subplot(3,1,1) 
% plot(P2_test); title('Features');
% subplot(3,1,2) 
% plot(idxCL, 'ro');title('Estimated by KNN path');
% subplot(3,1, 3) 
% plot(path, 'go');  title('Estimated by HMM path');
%  pause();
%   
 
        
        %fprintf('Training data log prob %2.4f\n', ll_P1_test);
        Matrix_results(i,j)=(ll_P1_test);
        Matrix_results_test(i,j)=(ll_P2_test);
      end
 end

 
 % display the results
 [val, ind_r]=max(Matrix_results); %If A is a matrix, then min(A) is a row vector containing the minimum value of each column
 % -> then I have for each person the model which fits the most
 
 [val2, ind2_r]=max(Matrix_results_test);
  fprintf('Person\n');
 1:8
 fprintf('Results on training data\n');
 ind_r
 
 fprintf('Results on testing data\n');
 ind2_r

 
 

% 
% ll_P1_test 		= mhmm_logprob(P1_test', prior1, transmat1, mu1, Sigma1, mixmat1);
% ll_P2_test 		= mhmm_logprob(P2_test', prior2, transmat2, mu2, Sigma2, mixmat2);
% 
% ll_P1_cross_test 	= mhmm_logprob(P2_test', prior2, transmat2, mu2, Sigma2, mixmat2);
% ll_P2_cross_test 	= mhmm_logprob(P1_test', prior1, transmat1, mu1, Sigma1, mixmat1);



% 
% 
% 
% 
% 
% % calculate mu and sigma for each cluster
% mu_p=[];
% sigma=zeros(20:20);
% Sigma0=zeros(20:20:7);
%     for j=1:nb_states
%   
% if (numel(mean(seq_train(find(idx==j), :)))~=1)
%     mu_p=[mu_p;mean(seq_train(find(idx==j), :))];
%         diagS=std(seq_train(find(idx==j), :));
%         for t=1:20
%          sigma(t,t)=diagS(t);
%         end
%     Sigma0(:,:,j)=sigma;
% else
%     mu_p=[mu_p;seq_train(find(idx==j), :) ]; % check this one for the correctness => I am taking mean and variance for the according to the data sequence and here there is 1 frame, so take it directly
%            for t=1:20
%          sigma(t,t)=sigma_Cluster(j,t);  % CHANGE THIS PART FOR SOMETHING correct
%         end
%     Sigma0(:,:,j)=sigma;
% end 
% 
%     end
