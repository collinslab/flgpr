classdef feature_SIFT
    properties (SetAccess = private)
        name = 'SIFT descriptors';
        nameAbbreviation = 'SIFT';
    end
    
    
    properties
        %% Put feature parameters in here
        frameOptionVec = [11,11,1,0] % [dim 2 stride, dim 1 stride, radius, orientation] (in pixels or radians)
        
        
    end
    
    methods
        
        function [featureSet] = run(self,featureSet)
            
            ogSize = featureSet.userData.imageSize;
            
            Xexp = single(reshape(featureSet.X,[featureSet.nObservations,ogSize])); % get data cube
            
            featureSet.X = [];
            %%
            fGrid = reshape(meshgrid(1:self.frameOptionVec(1):ogSize(2),1:self.frameOptionVec(2):ogSize(1)),1,[]);
            fGrid(2,:) = reshape(meshgrid(1:self.frameOptionVec(2):ogSize(1),1:self.frameOptionVec(1):ogSize(2)).',1,[]);
            fGrid(3,:) = self.frameOptionVec(3);
            fGrid(4,:) = self.frameOptionVec(4);
            
            fGrid(1,:) = fGrid(1,:) + ogSize(1)./(size(fGrid,2) + 1);
            fGrid(2,:) = fGrid(2,:) + ogSize(2)./(size(fGrid,2) + 1);
            
            siftVecs = nan(featureSet.nObservations,128.*size(fGrid,2),'single');
            
            for i = 1:featureSet.nObservations
                [~,tmpDescriptors] = vl_sift(squeeze(Xexp(i,:,:)),'Frames',fGrid);
                siftVecs(i,:) = reshape(tmpDescriptors,1,[]);
            end
            siftVecs(isnan(Xexp(:,1,1)),:) = nan;
            
            featureSet.X = double(cat(2,featureSet.X,siftVecs));
            
            
            
            featureSet.userData.featSize = 128; % length of SIFT descriptor
            featureSet.userData.gridSize = [length(1:self.frameOptionVec(2):ogSize(1)),length(1:self.frameOptionVec(1):ogSize(2))];
        end
    end
    
end