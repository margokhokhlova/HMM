%
%
% Unit test with GMM's and human gate data
%
function ex2_simple;

close all;
clear all;

% Load training data

load('data/data_mat.mat');

%Load all the data at the same moment - initial matrix;

load('G:\GitHub\02-21\PCA_margo\data_norm');
cov_mat=1/size(X_norm,1)*X_norm'*X_norm;
[U, S, V]=svd(cov_mat);
% 
 U_red=U(1:5,:);
 
 %save('U_red.mat', 'U_red');
 Z_red=X_norm*U_red';

% from here the HMM

s1 = data{1};
s2 = data{2};
s3 = data{3};
valid_data = [s1; s2];
test_data_norm=s3;
train_data=s3;

%train_data= Z_red; %40x20 => second dimension en deuxieme
% train_data_cyr=[s1];
% train_data_cyr2=[s2];
% val_data 	= [s3];

% Load test data
load('data/data_abnormal.mat');

s1 = data{1};
s2 = data{2};
s3 = data{3};

test_data_abnorm = [s1; s2; s3];
% test_data1 = [s1];
% test_data2=[s2];

nb_states = 7;
% Inital cluster guess using kmeans
% TODO: implement cluster hopping as an inital transition matrix
[idx,C] = kmeans(train_data, nb_states);

% modify data to change dimensions to 1 for test data

%change all data to simple 1D
train_data=idx;
val_norm=findNearestClusters(valid_data, C, nb_states);
test_norm=findNearestClusters(test_data_norm, C, nb_states);
test_abnorm=findNearestClusters(test_data_abnorm,C, nb_states);

% HMM and cluster settings
nb_feature = size(train_data, 2); 

nb_gmm = 1;
cov_type = 'diag';

prior0 		= normalize(rand(nb_states, 1));
transmat0 	= mk_stochastic(rand(nb_states, nb_states));


%[ transmat0 ] = Transission( idx );

% caclulate the variance in the clusters and construct the initial
% transition matrix



[mu0, Sigma0, weights] = mixgauss_init(nb_states*nb_gmm, train_data', cov_type); % change
% normalize weights
mu0 	= reshape(mu0, [nb_feature nb_states nb_gmm]);
Sigma0 	= reshape(Sigma0, [nb_feature nb_feature nb_states nb_gmm]);
mixmat0 = mk_stochastic(rand(nb_states, nb_gmm)); %check this matrix % dimensions weights


%obsmat1 = mk_stochastic(rand(nb_feature, nb_states));

%%
% [mu0, Sigma0] = mixgauss_init(nb_states*nb_gmm, train_data', cov_type);
% mu0 	= reshape(mu0, [nb_feature nb_states nb_gmm]);
% Sigma0 	= reshape(Sigma0, [nb_feature nb_feature nb_states nb_gmm]);
% mixmat0 = mk_stochastic(rand(nb_states, nb_gmm));

%[LL, prior0, transmat0, obsmat0] = dhmm_em(train_data', prior0, transmat0, obsmat1, 'max_iter', 5);
% Improve GMM estimate
 [LL, prior1, transmat1, mu1, Sigma1, mixmat1] = mhmm_em(train_data', prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 20);
% LL- probability to have the model with (one number)
% lp_val 		= mhmm_logprob(val_norm', prior0, transmat0, obsmat0);
% lp_test1 	= mhmm_logprob(test_norm', prior0, transmat0,obsmat0);
% lp_test2 	= mhmm_logprob(test_abnorm', prior0, transmat0, obsmat0);

 lp_val 		= mhmm_logprob(val_norm', prior0, transmat0, mu0, Sigma0, mixmat0); % essayer de changer pour vitterbi -> forward propagation
 lp_test1 	= mhmm_logprob(test_norm', prior0, transmat0, mu0, Sigma0, mixmat0);
 lp_test2 	= mhmm_logprob(test_abnorm', prior0, transmat0, mu0, Sigma0, mixmat0);


fprintf('Validation data log prob %2.4f \n', lp_val);
fprintf('Testing data log prob %2.4f\n', lp_test1);
fprintf('Test abnormal gait data log prob %2.4f and  %2.4f \n', lp_test2);


