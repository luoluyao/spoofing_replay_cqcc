clear; close all; clc;

dir = fullfile('D:\Filin','spoofing_replay_cqcc','extracted_features', 'addition_features');

fprintf('start')
genuineFeatureCellDev = add_new_features(dir,'genuineFeatureCellDev','snrOriginalDEV', 'excessOriginalDEV', 'sigmaOriginalDEV', 'rangeOriginalDEV' );
save ('../cqcc_and_4_feature/genuineFeatureCellDev.mat', 'genuineFeatureCellDev','-v7.3');

genuineFeatureCellTrain = add_new_features(dir,'genuineFeatureCellTrain','snrOriginalTRAIN', 'excessOriginalTRAIN','sigmaOriginalTRAIN', 'rangeOriginalTRAIN' );
save ('../cqcc_and_4_feature/genuineFeatureCellTrain.mat', 'genuineFeatureCellTrain','-v7.3');

spoofFeatureCellDev = add_new_features(dir,'spoofFeatureCellDev','snrSpoofDEV', 'excessSpoofDEV','sigmaSpoofDEV', 'rangeSpoofDEV' );
save ('../cqcc_and_4_feature/spoofFeatureCellDev.mat', 'spoofFeatureCellDev','-v7.3');

spoofFeatureCellTrain = add_new_features(dir,'spoofFeatureCellTrain','snrSpoofTRAIN', 'excessSpoofTRAIN','sigmaSpoofTRAIN','rangeSpoofTRAIN' );
save ('../cqcc_and_4_feature/spoofFeatureCellTrain.mat', 'spoofFeatureCellTrain','-v7.3');

evaluationFeature = add_new_features(dir,'evaluationFeature','snrEVA', 'excessEVA','sigmaEVA', 'rangeEVA' );
save ('../cqcc_and_4_feature/evaluationFeature.mat', 'evaluationFeature','-v7.3');

