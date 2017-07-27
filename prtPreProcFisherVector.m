classdef prtPreProcFisherVector < prtPreProc
    
    properties (SetAccess=private)
        name = 'Fisher Vector Image Processing'
        nameAbbreviation = 'fisherVector'
    end
    
    properties
        numClusters = 30;
        poolSize = [nan,nan]; % find pool size based on image dimension (don't overlap)
        poolGrid = [2,2];
        subPreProc = [];
        vlOption = 'Improved';
    end
    
    properties (SetAccess = private)
        means
        covariances
        priors
    end
    
    methods
        function Obj = prtPreProcFisherVector(varargin)
            Obj = prtUtilAssignStringValuePairs(Obj,varargin{:});
            Obj.verboseStorage = false;
        end
    end
    
    methods (Access=protected,Hidden=true)
        
        function self = trainAction(self,ds)
            ds.X = single(ds.X); % I only need single precision
            
            if ~isempty(self.subPreProc) % add processing to the subimages (train, then run on training observations)
                self.subPreProc = prtOutlierRemovalMissingData + self.subPreProc;
                self.subPreProc.verboseStorage = self.verboseStorage;
                self.subPreProc = self.subPreProc.train(ds);
                ds = self.subPreProc.run(ds);
            end
            
            %%% train GMM with ALL observations (this is taking the most
            %%% time right now)
            [self.means, self.covariances, self.priors] = vl_gmm(reshape(ds.data.',ds.userData.featSize,[]),self.numClusters);
            
        end
        
        function ds = runAction(self,ds)
            ds.X = single(ds.X); % I only need single precision
            
            if ~isempty(self.subPreProc) % add processing to the subimages (run)
                ds = self.subPreProc.run(ds);
            end
            
            gridSize(1) = ds.userData.gridSize(1);
            gridSize(2) = ds.userData.gridSize(2);
            
            if isnan(self.poolSize(1)), self.poolSize(1) = ceil(gridSize(1)./self.poolGrid(1)); end
            if isnan(self.poolSize(2)), self.poolSize(2) = ceil(gridSize(2)./self.poolGrid(2)); end
            
            poolIndsX = floor(linspace(1,gridSize(1)- self.poolSize(1) + 1,self.poolGrid(1)));
            poolIndsY = floor(linspace(1,gridSize(2)- self.poolSize(2) + 1,self.poolGrid(2)));
            
            
              %%% do this if we don't really hav ea grid specified and no
            %%% pooling
            if prod(self.poolGrid) == 1
                gridSize(1) = ds.nFeatures/ds.userData.featSize;
                gridSize(2) = 1;
                
                poolIndsX = 1; self.poolSize(1) = gridSize(1);
                poolIndsY = 1; self.poolSize(2) = 1;
            end
            %%
            encoding = zeros(ds.nObservations,2*ds.userData.featSize*self.numClusters.*prod(self.poolGrid)); % each feature has a mean and variance for each cluster, thus the new feature size
            
            %%% get the fisherVecotr based on the trained GMM for every
            %%% observation.
            cFeat = nan(prod(self.poolGrid),2*ds.userData.featSize*self.numClusters);
            if ~isempty(self.vlOption)
                for i = 1:ds.nObservations % find fisher vector in relation to trained GMM
                    cData = reshape(ds.data(i,:),ds.userData.featSize,gridSize(1),gridSize(2));
                    for pI = 1:self.poolGrid(1)
                        for pJ = 1:self.poolGrid(2)
                            cDataVec = reshape(cData(:,poolIndsX(pI):poolIndsX(pI) + self.poolSize(1)- 1,poolIndsY(pJ):poolIndsY(pJ) + self.poolSize(2)- 1),ds.userData.featSize,[]);
                            cFeat((pI-1)*self.poolGrid(2) + pJ,:) = vl_fisher(cDataVec,self.means,self.covariances,self.priors,self.vlOption);
                            encoding(i,:) = cFeat(:);
                        end
                    end
                end
            else
                for i = 1:ds.nObservations % find fisher vector in relation to trained GMM
                    cData = reshape(ds.data(i,:),ds.userData.featSize,gridSize(1),gridSize(2));
                    for pI = 1:self.poolGrid(1)
                        for pJ = 1:self.poolGrid(2)
                            cDataVec = reshape(cData(:,poolIndsX(pI):poolIndsX(pI) + self.poolSize(1)- 1,poolIndsY(pJ):poolIndsY(pJ) + self.poolSize(2)- 1),ds.userData.featSize,[]);
                            cFeat((pI-1)*self.poolGrid(2) + pJ,:) = vl_fisher(cDataVec,self.means,self.covariances,self.priors);
                            encoding(i,:) = cFeat(:);
                        end
                    end
                end
            end
            
            %%
            ds.data = double(encoding);
            
        end
        
    end
end
