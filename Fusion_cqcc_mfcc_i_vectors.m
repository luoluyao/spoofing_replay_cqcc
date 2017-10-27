%% Чистимся

clear; close all; clc;

%% Добавляем директории с тулкитами
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
addpath('MSR Identity Toolkit v1.0/code');
addpath('voicebox');

%% Читаем скоры

feature3 = 'baseline';
feature4 = 'cqcc_i_vec';
feature5 = 'xgb';
feature6 = 'baseline_wcmvn';
feature7 = 'cqcc_i_vec_wcmvn';
feature8 = 'cqcc_i_vec_without_first_wcmvn_xgb';

directory_features_3 = fullfile('D:\Filin\spoofing_replay_cqcc\scores\cqcc_features\scores.mat');
scores3 = load(directory_features_3);
scores3 = scores3.scores';

directory_features_4 = fullfile('D:\Filin\spoofing_replay_cqcc\extracted_features' , 'cqcc_i_vec_without_first', 'scores.mat');
% load(fullfile(directory_features_4, 'scores_genuine_cqcc_i_vector_100_gplda_90.mat'));
% load(fullfile(directory_features_4, 'scores_spoof_cqcc_i_vector_100_gplda_90.mat'));
% scores4 = mean(scores_genuine,1) - mean(scores_spoof,1);
% clear scores_genuine;
% clear scores_spoof;
scores4 = load(directory_features_4);
scores4 = scores4.scores;

directory_features_5 = fullfile('D:\Filin\spoofing_replay_cqcc\extracted_features\cqcc_i_vec_xgb\scores.mat');
scores5 = load(directory_features_5);
scores5 = scores5.prob;

directory_features_6 = fullfile('D:\Filin\spoofing_replay_cqcc\extracted_features\cqcc_features_wcmvn_two_ubm\scores_gmm_wcmvn.mat');
scores6 = load(directory_features_6);
scores6 = scores6.scores';

directory_features_7 = fullfile('D:\Filin\spoofing_replay_cqcc\extracted_features\cqcc_i_vec_wcmvn\scores.mat');
scores7 = load(directory_features_7);
scores7 = scores7.scores;

directory_features_8 = fullfile('D:\Filin\spoofing_replay_cqcc\extracted_features\cqcc_i_vec_wcmvn_xgb\scores.mat');
scores8 = load(directory_features_8);
scores8 = scores8.prob;

%% Готовим метки

evaProtocolFile = fullfile('D:','Data','ASVspoof2017_eval', 'protocol', 'KEY_ASVspoof2017_eval_V2.trl');
fileID = fopen(evaProtocolFile);
protocol = textscan(fileID, '%s%s');
labels = protocol{2};
fclose(fileID);

clear protocol;

%% Считаем EER по моделям

prior = 0.5;
% [actdcf1,mindcf1,prbep1,eer1] = fastEval(scores1(strcmp(labels,'genuine')),scores1(strcmp(labels,'spoof')), prior);
% [~,mindcf1,~,eer1] = fastEval(scores1(strcmp(labels,'genuine')),scores1(strcmp(labels,'spoof')), prior);
% [~,mindcf2,~,eer2] = fastEval(scores2(strcmp(labels,'genuine')),scores2(strcmp(labels,'spoof')), prior);
[~,mindcf3,~,eer3] = fastEval(scores3(strcmp(labels,'genuine')),scores3(strcmp(labels,'spoof')), prior);
[~,mindcf4,~,eer4] = fastEval(scores4(strcmp(labels,'genuine')),scores4(strcmp(labels,'spoof')), prior);
[~,mindcf5,~,eer5] = fastEval(scores5(strcmp(labels,'genuine')),scores5(strcmp(labels,'spoof')), prior);
[~,mindcf6,~,eer6] = fastEval(scores6(strcmp(labels,'genuine')),scores6(strcmp(labels,'spoof')), prior);
[~,mindcf7,~,eer7] = fastEval(scores7(strcmp(labels,'genuine')),scores7(strcmp(labels,'spoof')), prior);
[~,mindcf8,~,eer8] = fastEval(scores8(strcmp(labels,'genuine')),scores8(strcmp(labels,'spoof')), prior);

%% Готовим скоры

scores4 = scores4(:)';
% scores2 = scores2(:)';
scores3 = scores3(:)';

train_labels = zeros(size(labels));

for i = 1:length(labels)
    if strcmp(labels(i), 'genuine')
        train_labels(i) = 1;
    else
        train_labels(i) = -1;
    end
end

%% Fuse all

train_scores_all = [
                scores3;
                scores4;
                scores5;
                scores6;
                scores7;
                scores8
                ];

quiet = true; % don't display output during fusion training
maxiters = 100; % maximum number of training iterations
obj_func = [];  % use the default objective function: cllr objective
fusion_func = train_linear_fuser(train_scores_all,train_labels,prior,true,quiet,maxiters,obj_func);

fused_scores_all = fusion_func(train_scores_all);

[~,mindcf_all,~,eer_all] = fastEval(fused_scores_all(strcmp(labels,'genuine')), fused_scores_all(strcmp(labels,'spoof')), prior);

%% Fuse non-norm
train_scores_best = [
                scores3;
                %scores4;
                scores5;
                %scores6;
                scores7;
                %scores8
                ];

quiet = true; % don't display output during fusion training
maxiters = 100; % maximum number of training iterations
obj_func = [];  % use the default objective function: cllr objective

fusion_func_best = train_linear_fuser(train_scores_best,train_labels,prior,true,quiet,maxiters,obj_func);

fused_scores_best = fusion_func_best(train_scores_best);

[~,mindcf_best,~,eer_best] = fastEval(fused_scores_best(strcmp(labels,'genuine')), fused_scores_best(strcmp(labels,'spoof')), prior);


%% Строим графики

plot_title = 'DET';

plot_type = Det_Plot.make_plot_window_from_string('old');
plot_obj = Det_Plot(plot_type,plot_title);

% plot_obj.set_system(scores2(strcmp(labels,'genuine')), scores2(strcmp(labels,'spoof')),'MFCC, i-vector');
% plot_obj.plot_steppy_det({'c','LineWidth',2},' ');
% plot_obj.plot_mindcf_point(prior,{'k*','MarkerSize',8});
% plot_obj.plot_DR30_fa('k.--','30 false alarms');
% plot_obj.plot_DR30_miss('k.--','30 misses');

plot_obj.set_system(scores3(strcmp(labels,'genuine')), scores3(strcmp(labels,'spoof')), '1. GMM');
plot_obj.plot_steppy_det({'r','LineWidth',2},' ');
% plot_obj.plot_mindcf_point(prior,{'k*','MarkerSize',8});
% plot_obj.plot_DR30_fa('k.--','30 false alarms');
% plot_obj.plot_DR30_miss('k.--','30 misses');

plot_obj.set_system(scores4(strcmp(labels,'genuine')), scores4(strcmp(labels,'spoof')),'2. i-Vector-GPLDA');
plot_obj.plot_steppy_det({'g','LineWidth',2},' ');
% plot_obj.plot_mindcf_point(prior,{'k*','MarkerSize',8});
% plot_obj.plot_DR30_fa('k.--','30 false alarms');
% plot_obj.plot_DR30_miss('k.--','30 misses');

plot_obj.set_system(scores5(strcmp(labels,'genuine')), scores5(strcmp(labels,'spoof')), '3. i-Vector-XGBoost');
plot_obj.plot_steppy_det({'b','LineWidth',2},' ');
% plot_obj.plot_mindcf_point(prior,{'g*','MarkerSize',8},'mindcf');
% plot_obj.plot_DR30_fa('k.--','30 false alarms');
% plot_obj.plot_DR30_miss('k.--','30 misses');

plot_obj.set_system(scores6(strcmp(labels,'genuine')), scores6(strcmp(labels,'spoof')), '4. GMM-STMN');
plot_obj.plot_steppy_det({'r--','LineWidth',2},' ');
% plot_obj.plot_mindcf_point(prior,{'g*','MarkerSize',8},'mindcf');
% plot_obj.plot_DR30_fa('k.--','30 false alarms');
% plot_obj.plot_DR30_miss('k.--','30 misses');

plot_obj.set_system(scores7(strcmp(labels,'genuine')), scores7(strcmp(labels,'spoof')), '5. i-Vector-GPLDA-STMN');
plot_obj.plot_steppy_det({'g--','LineWidth',2},' ');
% plot_obj.plot_mindcf_point(prior,{'g*','MarkerSize',8},'mindcf');
% plot_obj.plot_DR30_fa('k.--','30 false alarms');
% plot_obj.plot_DR30_miss('k.--','30 misses');

plot_obj.set_system(scores8(strcmp(labels,'genuine')), scores8(strcmp(labels,'spoof')), '6. i-Vector-XGBoost-STMN');
plot_obj.plot_steppy_det({'b--','LineWidth',2},' ');
% plot_obj.plot_mindcf_point(prior,{'g*','MarkerSize',8},'mindcf');
% plot_obj.plot_DR30_fa('k.--','30 false alarms');
% plot_obj.plot_DR30_miss('k.--','30 misses');

plot_obj.set_system(fused_scores_best(strcmp(labels,'genuine')), fused_scores_best(strcmp(labels,'spoof')),'Fusion of 1,3 and 5');
plot_obj.plot_steppy_det({'k:','LineWidth',2},' ');
%plot_obj.plot_mindcf_point(prior,{'k*','MarkerSize',8},'mindcf');

plot_obj.set_system(fused_scores_all(strcmp(labels,'genuine')), fused_scores_all(strcmp(labels,'spoof')),'Fusion of all models');
plot_obj.plot_steppy_det({'k','LineWidth',2},' ');
%plot_obj.plot_mindcf_point(prior,{'k*','MarkerSize',8},'mindcf');
% plot_obj.plot_DR30_fa('k.--','30 false alarms');
% plot_obj.plot_DR30_miss('k.--','30 misses');

plot_obj.display_legend();
