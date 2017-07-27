classdef feature_subRegionStats
    properties (SetAccess = private)
        name = 'get stats from sub regions in patch';
        nameAbbreviation = 'subRegionStats';
    end
    
    
    properties
        %% Put feature parameters in here
        inputMoment = 12;
        nSubs =[3,3];
    end
    
    methods
        
        function [featureSet] = run(self,featureSet)
            %%
            ogSize = featureSet.userData.imageSize;
            
            Xexp = reshape(single(featureSet.X),[featureSet.nObservations,ogSize]);
            
            regInds(1,:) = [1,floor(ogSize(1).*(1:self.nSubs(1))/self.nSubs(1))];
            regInds(2,:) = [1,floor(ogSize(2).*(1:self.nSubs(2))/self.nSubs(2))];
            
            regionInd = 1;
            for i = 1:self.nSubs(1)
                for j = 1:self.nSubs(2)
                    meanRegCube{regionInd} = Xexp(:,regInds(1,i):regInds(1,i+1),regInds(2,j):regInds(2,j+1),:); %#ok<*AGROW>
                    regionInd = regionInd + 1;
                end
            end
            
            
            switch self.inputMoment
                case 1
                    for k = 1:prod(self.nSubs)
                        cMeanCube = reshape(meanRegCube{k},featureSet.nObservations,[]);
                            momMat(:,k) =   mean(cMeanCube,2);
                    end
                    nMoms = 1;
                case 2
                    for k = 1:prod(self.nSubs)
                        cMeanCube = reshape(meanRegCube{k},featureSet.nObservations,[]);
                            momMat(:,k)=   var(cMeanCube,0,2);
                    end
                    nMoms = 1;
                case 3
                    for k = 1:prod(self.nSubs)
                        cMeanCube = reshape(meanRegCube{k},featureSet.nObservations,[]);
                            momMat(:,k) =   skewness(cMeanCube,0,2);
                    end
                    nMoms = 1;
                case 4
                    for k = 1:prod(self.nSubs)
                        cMeanCube = reshape(meanRegCube{k},featureSet.nObservations,[]);
                            momMat(:,k) =   kurtosis(cMeanCube,0,2);
                    end
                    nMoms = 1;
                case 12
                        for k = 1:prod(self.nSubs)
                            cMeanCube = reshape(meanRegCube{k},featureSet.nObservations,[]);
                            momMat1(:,k) =   mean(cMeanCube(:,:),2);
                            momMat2(:,k)=   var(cMeanCube(:,:),0,2);
                        end
                    momMat = reshape(cat(2,momMat1,momMat2),featureSet.nObservations,[]);
                    nMoms = 2;
                case 12345
                        for k = 1:prod(self.nSubs)
                            cMeanCube = reshape(meanRegCube{k},featureSet.nObservations,[]);
                            momMat1(:,k) =   mean(cMeanCube,2);
                            momMat2(:,k)=   var(cMeanCube,0,2);
                            momMat3(:,k) =   skewness(cMeanCube,0,2);
                            momMat4(:,k) =   kurtosis(cMeanCube,0,2);
                            for q = 1:featureSet.nObservations
                                momMat5(q,k) = norm(cMeanCube(q,:));
                            end
                        end
                    momMat = reshape(cat(2,momMat1,momMat2,momMat3,momMat4,momMat5),featureSet.nObservations,[]);
                    nMoms = 5;
                otherwise
                    for k = 1:prod(self.nSubs)
                        cMeanCube = reshape(meanRegCube{k},featureSet.nObservations,[]);
                            momMat(:,k) =   moment(cMeanCube,self.inputMoment,2);
                    end
                    nMoms = 1;
            end
            
            featureSet.userData.featSize = nMoms.*prod(self.nSubs);
            featureSet.X = double(momMat);
        end
    end
    
end

