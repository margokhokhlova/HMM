%
%
% Unit test with GMM's and human gate data
%
function ex3;

close all;

% Load training data
load('data/data_mat.mat');

s1 = data{1};
s2 = data{2};
s3 = data{3};

% Load test data
load('data/data_abnormal.mat');

s4 = data{1};
s5 = data{2};
s6 = data{3};

cluster_data = [s1; s2; s4; s5];
ind = randperm(size(cluster_data, 1));
cluster_data = cluster_data(ind, :);

% Cluster settings
nb_feature = size(cluster_data, 2);
nb_states = 7;
nb_gmm = 2;
cov_type = 'diag';

prior0 		= normalize(rand(nb_states, 1));
transmat0 	= mk_stochastic(rand(nb_states, nb_states));

% Inital cluster guess using kmeans
% TODO: implement cluster hopping as a inital transition matrix
[mu0, Sigma0, weights] = mixgauss_init(nb_states*nb_gmm, cluster_data', cov_type);

mu0 	= reshape(mu0, [nb_feature nb_states nb_gmm]);
Sigma0 	= reshape(Sigma0, [nb_feature nb_feature nb_states nb_gmm]);
mixmat0 = reshape(weights, nb_states, nb_gmm);

% Now train a HMM per person
P1_train 	= [s1; s2];
P1_test 	= [s3];
P2_train	= [s4; s5];
P2_test		= [s6];


[LL, prior1, transmat1, mu1, Sigma1, mixmat1] = ...
    mhmm_em(P1_train', prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 20);

[LL, prior2, transmat2, mu2, Sigma2, mixmat2] = ...
    mhmm_em(P2_train', prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 20);

ll_P1_test 		= mhmm_logprob(P1_test', prior1, transmat1, mu1, Sigma1, mixmat1);
ll_P2_test 		= mhmm_logprob(P2_test', prior2, transmat2, mu2, Sigma2, mixmat2);

ll_P1_cross_test 	= mhmm_logprob(P2_test', prior2, transmat2, mu2, Sigma2, mixmat2);
ll_P2_cross_test 	= mhmm_logprob(P1_test', prior1, transmat1, mu1, Sigma1, mixmat1);



fprintf('P1 Test on Model 1, log prob %4.4f\n', ll_P1_test);
fprintf('P2 Test on Model 2, log prob %4.4f\n', ll_P2_test);

fprintf('P1 Test on Model 2, log prob %4.4f\n', ll_P1_test);
fprintf('P2 Test on Model 1, log prob %4.4f\n', ll_P2_test);






