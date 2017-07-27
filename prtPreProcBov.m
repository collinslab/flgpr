classdef prtPreProcBov < prtPreProc
    % prtPreProcBov BOV feature learning, inspried by Coates and Ng 2012
    
    properties (SetAccess=private)
        name = 'Bag of visual words'
        nameAbbreviation = 'bov'
    end
    
    properties
        clusterAlgo = prtClusterSphericalKmeans('nClusters',200)
        encodeFcn = @(x) x;
        poolFcn = @(x) sum(x,1);
        poolSize = [2,2];
        poolGrid = [2,2];
        subPreProc = prtPreProcZcaFLGPR
    end
    
    methods
        function Obj = prtPreProcCoates(varargin)
            Obj = prtUtilAssignStringValuePairs(Obj,varargin{:});
            Obj.verboseStorage = false;
        end
    end
    
    methods (Access=protected,Hidden=true)
        
        function self = trainAction(self,ds)
            ds.X = single(ds.X); % I only need single precision
            ds = getDsSepFeats(self,ds); % seperate features into there own observaitons
            
            if ~isempty(self.subPreProc) % add processing to the subimages
                self.subPreProc = prtOutlierRemovalMissingData + self.subPreProc;
                self.subPreProc.verboseStorage = self.verboseStorage;
                self.subPreProc = self.subPreProc.train(ds);
                ds = self.subPreProc.run(ds);
            end
            self.clusterAlgo.verboseStorage = self.verboseStorage;
            self.clusterAlgo = self.clusterAlgo.train(ds);
            
        end
        
        
        function ds = runAction(self,ds)
            ds.X = single(ds.X); % I only need single precision
            ds = getDsSepFeats(self,ds); % seperate features into there own observaitons
                                    
            if ~isempty(self.subPreProc) % add processing to the subimages
                ds = self.subPreProc.run(ds);
            end
            yOutK = self.clusterAlgo.run(ds);
            ds = getCombFeats(self,yOutK);
        end
        
        %%% helper functions
        function dsSepFeats = getDsSepFeats(self,ds) %#ok<INUSL>
            %%% required to have ds.userData.featSize, i.e., the
            %%% dimensionality of the sub-region feature
            nStates = ds.nFeatures/ds.userData.featSize; %number of states
            
            tmpX = reshape(ds.X,ds.nObservations,ds.nFeatures./nStates,nStates);
            tmpX = reshape(shiftdim(tmpX,2),ds.nObservations.*nStates,[]);
            

            targets = reshape(repmat(ds.targets,1,nStates).',ds.nObservations.*nStates,1);
            
            dsSepFeats = prtDataSetClass(tmpX,targets);
            dsSepFeats.userData = ds.userData;
            dsSepFeats.userData.obIds = reshape(repmat((1:ds.nObservations).',1,nStates).',ds.nObservations.*nStates,1);
            dsSepFeats.userData.fSetInds = repmat((1:nStates).',ds.nObservations,1);
            dsSepFeats.userData.oldObservationInfo = ds.observationInfo; % don't get rid of the 
            
        end
        function ds = getCombFeats(self,dsSepFeats)
            X = dsSepFeats.X;
            inds = dsSepFeats.userData.obIds;
            targets = dsSepFeats.targets;
            
            XNew = nan(max(inds),self.clusterAlgo.nClusters.*prod(self.poolGrid));
            targetsNew = nan(max(inds),1);
            
            if self.poolGrid(1) ~= 1, poolIndsX = floor(linspace(1,dsSepFeats.userData.gridSize(1)- self.poolSize(1) + 1,self.poolGrid(1))); else poolIndsX = 1; self.poolSize(1) = dsSepFeats.userData.gridSize(1); end     
            if self.poolGrid(2) ~= 1, poolIndsY = floor(linspace(1,dsSepFeats.userData.gridSize(2)- self.poolSize(2) + 1,self.poolGrid(2))); else poolIndsY = 1; self.poolSize(2) = dsSepFeats.userData.gridSize(2); end
            
            for i = 1:max(dsSepFeats.userData.obIds)
                cInds = (inds == i);
                cX = X(cInds,:);
                if ~any(isnan(cX(:)))
                    cData = reshape(cX,[dsSepFeats.userData.gridSize,self.clusterAlgo.nClusters]);
                    for pI = 1:self.poolGrid(1)
                        for pJ = 1:self.poolGrid(2)
                            cFeat((pI-1)*self.poolGrid(2) + pJ,:) = self.poolFcn(reshape(self.encodeFcn(cData(poolIndsX(pI):poolIndsX(pI) + self.poolSize(1)- 1,poolIndsY(pJ):poolIndsY(pJ) + self.poolSize(2)- 1,:)),[],self.clusterAlgo.nClusters)); %#ok<AGROW>
                        end
                    end
                    XNew(i,:) = cFeat(:);
                end

                cTarg = targets(cInds);
                targetsNew(i) = cTarg(1);
            end
            
            %%
            ds = prtDataSetClass(double(XNew),targetsNew);
            ds.userData = dsSepFeats.userData;
            ds.observationInfo = dsSepFeats.userData.oldObservationInfo;
        end
    end
end
