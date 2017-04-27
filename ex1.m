%
%
% Unit test with GMM's and human gate data
%
function ex1;

close all;

load('data/data_mat.mat');

s1 = data{1};
s2 = data{2};
s3 = data{3};

train_data = [s1; s2];
test_data = [s3];

data = [s1; s2; s3];


nb_feature = size(data, 2);
nb_states = 12;
nb_gmm = 2;
cov_type = 'diag';

prior0 = normalize(rand(nb_states, 1));

transmat0 = mk_stochastic(rand(nb_states, nb_states));


[mu0, Sigma0] = mixgauss_init(nb_states*nb_gmm, data', cov_type);


mu0 = reshape(mu0, [nb_feature nb_states nb_gmm]);

Sigma0 = reshape(Sigma0, [nb_feature nb_feature nb_states nb_gmm]);
mixmat0 = mk_stochastic(rand(nb_states, nb_gmm));

% Improve GMM estimate

[LL, prior1, transmat1, mu1, Sigma1, mixmat1] = ...
    mhmm_em(train_data', prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 20);


logProb = mhmm_logprob(test_data', prior0, transmat0, mu0, Sigma0, mixmat0)

rand_data = randn(200, 20);

logProb = mhmm_logprob(rand_data', prior0, transmat0, mu0, Sigma0, mixmat0)