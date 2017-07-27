classdef feature_keepFullPatchNorm
    properties (SetAccess = private)
        name = 'keep full patch norm';
        nameAbbreviation = 'keepFullPatchNorm';

    end
    
    
    properties
        %% Put feature parameters in here
        normToggle = 2;  % switches the normilazation type
        
        imFcnStr = 'abs';
    end
    
    methods
        
        function [featureSet] = run(self,featureSet)
            
            ogSize = featureSet.userData.imageSize;
            
            %%
            imFcn = str2func(self.imFcnStr);
            Xexp = double(imFcn(reshape(featureSet.X,[featureSet.nObservations,prod(ogSize)])));
            
            Xexp = reshape(Xexp,featureSet.nObservations,[]);
            switch self.normToggle
                case 0
                    %%% nothing
                case 1 %%% full patch norm
                    
                    Xexp = (Xexp - repmat(mean(abs(Xexp),2),1,prod(ogSize),1))./repmat(sqrt(var(abs(Xexp),0,2) + eps),1,prod(ogSize),1);
                    
                case 2 %%% bg norming
                    normMask = ones(ogSize);
                    normMask(floor((ogSize(1)-1).*.25):ceil((ogSize(1)-1).*.75),floor((ogSize(2)-1).*.25):ceil((ogSize(2)-1).*.75)) = 0;
                    normMask = repmat(normMask(:)',featureSet.nObservations,1).*(prod(ogSize)/sum(normMask(:)));
                    
                    Xexp = (Xexp - repmat(mean(normMask.*abs(Xexp),2),1,prod(ogSize),1))./repmat(sqrt(var(sqrt(normMask).*abs(Xexp),0,2) + eps),1,prod(ogSize),1);
                case 3 %%% North South norm (0% E/W weighting)
                    normMask = ones(ogSize);
                    normMask(floor((ogSize(1)-1).*.25):ceil((ogSize(1)-1).*.75),:) = 0;
                    normMask = repmat(normMask(:)',featureSet.nObservations,1).*(prod(ogSize)/sum(normMask(:)));
                    
                    Xexp = (Xexp - repmat(mean(normMask.*abs(Xexp),2),1,prod(ogSize),1))./repmat(sqrt(var(sqrt(normMask).*abs(Xexp),0,2) + eps),1,prod(ogSize),1);
                case 4 %%% 25% E/W weighting
                    normMask = ones(ogSize);
                    normMask(floor((ogSize(1)-1).*.25):ceil((ogSize(1)-1).*.75),:) = .25;
                    normMask = repmat(normMask(:)',featureSet.nObservations,1).*(prod(ogSize)/sum(normMask(:)));
                    
                    Xexp = (Xexp - repmat(mean(normMask.*abs(Xexp),2),1,prod(ogSize),1))./repmat(sqrt(var(sqrt(normMask).*abs(Xexp),0,2) + eps),1,prod(ogSize),1);
                case 5 %%% 50% E/W weighting
                    normMask = ones(ogSize);
                    normMask(floor((ogSize(1)-1).*.25):ceil((ogSize(1)-1).*.75),:) = .5;
                    normMask = repmat(normMask(:)',featureSet.nObservations,1).*(prod(ogSize)/sum(normMask(:)));
                    
                    Xexp = (Xexp - repmat(mean(normMask.*abs(Xexp),2),1,prod(ogSize),1))./repmat(sqrt(var(sqrt(normMask).*abs(Xexp),0,2) + eps),1,prod(ogSize),1);
                case 6 %%% 75% E/W weighting
                    normMask = ones(ogSize);
                    normMask(floor((ogSize(1)-1).*.25):ceil((ogSize(1)-1).*.75),:) = .75;
                    normMask = repmat(normMask(:)',featureSet.nObservations,1).*(prod(ogSize)/sum(normMask(:)));
                    
                    Xexp = (Xexp - repmat(mean(normMask.*abs(Xexp),2),1,prod(ogSize),1))./repmat(sqrt(var(sqrt(normMask).*abs(Xexp),0,2) + eps),1,prod(ogSize),1);
            end
            
            featureSet = featureSet.setObservations(reshape(single(Xexp),featureSet.nObservations,[]));
            
        end
    end
    
end

