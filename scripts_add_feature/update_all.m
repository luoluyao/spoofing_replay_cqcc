clear; close all; clc;

dir = fullfile('D:\Filin','spoofing_replay_cqcc','extracted_features', 'addition_features');

create_mat_features(dir,'genuineFeatureCellDev','snrOriginalDEV', 'excessOriginalDEV', 'sigmaOriginalDEV', 'rangeOriginalDEV' );
create_mat_features(dir,'genuineFeatureCellTrain','snrOriginalTRAIN', 'excessOriginalTRAIN','sigmaOriginalTRAIN', 'rangeOriginalTRAIN' );
create_mat_features(dir,'spoofFeatureCellDev','snrSpoofDEV', 'excessSpoofDEV','sigmaSpoofDEV', 'rangeSpoofDEV' );
create_mat_features(dir,'spoofFeatureCellTrain','snrSpoofTRAIN', 'excessSpoofTRAIN','sigmaSpoofTRAIN','rangeSpoofTRAIN' );
create_mat_features(dir,'evaluationFeature','snrEVA', 'excessEVA','sigmaEVA', 'rangeEVA' );
