clear; close all; clc;

%% добавляем директории с тулкитами
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
addpath('MSR Identity Toolkit v1.0/code');
addpath('voicebox');

%% PATH's для файлов с метками
dir_data = 'D:\Data';
pathToDatabase = fullfile(dir_data,'ASVspoof2017_train_dev','wav');
trainProtocolFile = fullfile(dir_data,'ASVspoof2017_train_dev', 'protocol', 'ASVspoof2017_train.trn');
devProtocolFile = fullfile(dir_data,'ASVspoof2017_train_dev', 'protocol', 'ASVspoof2017_dev.trl');
evaProtocolFile = fullfile(dir_data,'ASVspoof2017_eval', 'protocol', 'KEY_ASVspoof2017_eval_V2.trl');


%% Чтение меток Training и Developmenp

fileID_train = fopen(trainProtocolFile);
protocol_train = textscan(fileID_train, '%s%s%s%s%s%s%s');
filenames_train = protocol_train{1};
labels_train = protocol_train{2};
fclose(fileID_train);

genuineIdx_train = find(strcmp(labels_train,'genuine'));
spoofIdx_train = find(strcmp(labels_train,'spoof'));

fileID_dev = fopen(devProtocolFile);
protocol_dev = textscan(fileID_dev, '%s%s%s%s%s%s%s');
filenames_dev = protocol_dev{1};
labels_dev = protocol_dev{2};
fclose(fileID_dev);

genuineIdx_dev = find(strcmp(labels_dev,'genuine'));
spoofIdx_dev = find(strcmp(labels_dev,'spoof'));

train_labels_genuine = cat(1, labels_train(1:length(genuineIdx_train)),  labels_dev(1:length(genuineIdx_dev)));
train_labels_spoof = cat(1,labels_train(length(genuineIdx_train)+1:length(labels_train)),labels_dev(length(genuineIdx_dev)+1:length(labels_dev)));


%% Читаем метки Evaluation 

fileID = fopen(evaProtocolFile);
protocol = textscan(fileID, '%s%s');
filenames = protocol{1};
eva_labels = protocol{2};
fclose(fileID);
%%
load('pred_proba.mat', 'prob')
%% Вычисляем EER

[Pmiss,Pfa] = rocch(prob(strcmp(eva_labels,'genuine')),prob(strcmp(eva_labels,'spoof')));
[pm,pf] = compute_roc(prob(strcmp(eva_labels,'genuine')),prob(strcmp(eva_labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('EER is %.2f\n', EER);

%% Строим кривые
plot(Pfa,Pmiss,'r-^',pf,pm,'g');
title (strcat('ROCCH and ROC plot for  ', model, ' model'));
axis('square');grid;legend('ROCCH','ROC');
xlabel('false alarm probability (%)') % x-axis label
ylabel('miss probability (%)') % y-axis label
plot_det_curve(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')), model)
