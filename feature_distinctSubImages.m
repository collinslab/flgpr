classdef feature_distinctSubImages
    properties (SetAccess = private)
        name = 'get distinct subimages from patch';
        nameAbbreviation = 'distinctSubImages';
    end
    
    
    properties
        %% Put feature parameters in here        
        gridSize = [24,32];
        subImSize = [11,11];
    end
    
    methods
        function [featureSet] = run(self,featureSet)
            ogSize = featureSet.userData.imageSize;
            
            Xexp = reshape(single(featureSet.X),[featureSet.nObservations,ogSize]);

            regIndsX = floor(linspace(1,ogSize(1) - self.subImSize(1) + 1,self.gridSize(1)));
            regIndsY = floor(linspace(1,ogSize(2) - self.subImSize(2) + 1,self.gridSize(2)));
            
            xC = meshgrid(regIndsX(1:self.gridSize(1)),regIndsY(1:self.gridSize(2)));
            yC = meshgrid(regIndsY(1:self.gridSize(2)),regIndsX(1:self.gridSize(1))).'; % this is correct now and iwll work for non-square patch sizes
            
            d = im2col(squeeze(Xexp(1,:,:,1)),self.subImSize,'sliding');
            imInds = sub2ind([ogSize(1) - self.subImSize(1) + 1,ogSize(2) - self.subImSize(2) + 1],xC(:),yC(:));
            
            imFeatLen = numel(d(:,imInds));
            X = nan(featureSet.nObservations,imFeatLen);
            
            progBar = prtUtilProgressBar(0,'creating distinct wub regions');
            % parfor i = 1:numel(toSubIm) if you want
            for i = 1:featureSet.nObservations
                d = single(im2col(squeeze(Xexp(i,:,:)),self.subImSize,'sliding'));
                X(i,:) = reshape(d(:,imInds),[],1);
                progBar.update(i./featureSet.nObservations);
            end
            
            %%
            featureSet.X = double(X);
            featureSet.userData.featSize = prod(self.subImSize);
            featureSet.userData.gridSize = self.gridSize;
        end
    end

end

