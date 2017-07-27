%% example of using features in the FLGPR image feature package
% see README for dependent packages


%% get example dataset
loadDs = prtDataGenMsrcorid({'signs','chimneys'});

imageSize = size(loadDs.X{1}(:,:,1));
imageDs = prtDataSetClass(nan(loadDs.nObservations,prod(imageSize)),loadDs.targets);

for i = 1:loadDs.nObservations
    imageDs.X(i,:) = reshape(rgb2gray(loadDs.X{i}),1,[]);
end

clear loadDs

imageDs.userData.imageSize = imageSize;
%% image normalization
cFeatExt = feature_keepFullPatchNorm;
featRawDs = cFeatExt.run(imageDs);

%% SIFT feature
cFeatExt = feature_SIFT;
cFeatExt.frameOptionVec  = [320,240,11,0];
featSiftDs = cFeatExt.run(imageDs);

%% LSTAT
cFeatExt = feature_subRegionStats;
featLstatDs = cFeatExt.run(imageDs);

%% 2D FFT
cFeatExt = feature_2DFFT;
feat2dfftDs = cFeatExt.run(imageDs);

%% log-gabor statistical
cFeatExt = feature_logGabor;
featLogGaborDs = cFeatExt.run(imageDs);

%% dense sub-regions
cFeatExt = feature_distinctSubImages;

featSubImageDs = cFeatExt.run(imageDs);

% feature_distinctSubImages adds gridSize and featSize to featDs.userData
% for use with feature learning objects
%% dense SFIT
cFeatExt = feature_SIFT;
cFeatExt.frameOptionVec  = [11,11,1,0];
featDenseSiftDs = cFeatExt.run(imageDs);

% feature_SIFT adds gridSize and featSize to featDs.userData
% for use with feature learning objects

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% using the features with a classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cvKeys = prtUtilEquallySubDivideData(feat2dfftDs.targets,5); % get random 5-fold crossval
%%
classObj = prtPreProcZmuv + prtClassLibSvm;
% classObj = prtClassPlsda;

confDs = classObj.crossValidate(featDenseSiftDs,cvKeys);

%% using dense feature with feature learning
classObj = prtPreProcFisherVector + prtClassPlsda('nComponents',3);
% classObj = prtPreProcBov + prtPreProcZmuv + prtClassPlsda('nComponents',3);

confLearnedDs = classObj.crossValidate(featDenseSiftDs,cvKeys); % run with same cv keys from above for fair comparison

%% plot perfromance (ROC)
figure, prtScoreRoc(confDs)
hold on,
prtScoreRoc(confLearnedDs)
hold off
grid on

legend('Static Features','Learned Features')

