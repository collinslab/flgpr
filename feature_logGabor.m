classdef feature_logGabor
    properties (SetAccess = private)
        name = 'gets log gabor features of patch';
        nameAbbreviation = 'logGabor';
    end
    
    
    properties
        nScale = 6;
        nOrient = 6;
        
        scaleFactor = 1;
    end
    
    methods
        
        function [featureSet] = run(self,featureSet)
            %%
            ogSize = featureSet.userData.imageSize;
            
            Xexp = single(reshape(featureSet.X,[featureSet.nObservations,ogSize])); % get data cube
            
            logGaborMat = getLogGaborFilterBank(self,ogSize(1)*2+1,ogSize(2)*2+1); % get filters
            
            nSubs = [3,3];
            regInds(1,:) = [1,floor(ogSize(1).*(1:nSubs(1))/nSubs(1))];
            regInds(2,:) = [1,floor(ogSize(2).*(1:nSubs(2))/nSubs(2))];
            
            X = nan([featureSet.nObservations,5*self.nScale*self.nOrient,nSubs],'single');
            %%
            progBar = prtUtilProgressBar(0,'Creating Log Gabor Feature Set');
            
            for iObs = 1:featureSet.nObservations
                %%
                cFiltData = abs(ifft(ifft(repmat(fftshift(fftshift(fft(fft(squeeze(Xexp(iObs,:,:)),ogSize(1)*2+1,1),ogSize(2)*2+1,2),1),2),1,1,self.nScale,self.nScale).*logGaborMat,[],1),[],2));
                % cFiltData = shiftdim(reshape(shiftdim(cFiltData(1:ogSize(1),1:ogSize(2),:,:),2),self.nScale,self.nOrient,prod(ogSize)),2);
                cFiltData = cFiltData(1:ogSize(1),1:ogSize(2),:,:);
                for r1 = 1:nSubs(1)
                    for r2 = 1:nSubs(2)
                        cRegFiltData = reshape(cFiltData(regInds(1,r1):regInds(1,r1+1),regInds(2,r2):regInds(2,r2+1),:),[],self.nScale*self.nOrient);
                        X(iObs,:,r1,r2) = reshape(cat(2,mean(cRegFiltData,1),var(cRegFiltData,0,1),skewness(cRegFiltData,0,1),kurtosis(cRegFiltData,0,1),sum(abs(cRegFiltData).^2,1).^0.5),[],1);
                    end
                end
                if ~mod(iObs,10), progBar.update(iObs/featureSet.nObservations); end
            end
            progBar.update(1);
            
            
            %% old version
            %                 XexpFFT = fftshift(fftshift(fft(fft(Xexp(:,:,:),ogSize(1)*2+1,2),ogSize(2)*2+1,3),2),3);
            %                 %     XexpFFT = repmat(fftshift(fftshift(fft(fft(cXexp,ogSize(1)*2+1,2),ogSize(2)*2+1,3),2),3),1,1,1,nScale);
            %                 for j = 1:self.nOrient
            %                     for i = 1:self.nScale
            %                         cLogGFilt = repmat(shiftdim(permute(logGaborMat(:,:,i,j),[1,2,3]),-1),[featureSet.nObservations,1,1,1]);
            %                         cFiltIm = abs(ifft(ifft(XexpFFT.*cLogGFilt,[],2),[],3));% ,2),3);
            %                         cFiltIm = cFiltIm(:,1:ogSize(1),1:ogSize(2));
            %                         %cFiltImTmp(j,:,:,:) = cFiltIm(3023,:,:,:);
            %
            %                         for r1 = 1:nSubs(1)
            %                             for r2 = 1:nSubs(2)
            %                                 cRegFiltIm = reshape(shiftdim(cFiltIm(:,regInds(1,r1):regInds(1,r1+1),regInds(2,r2):regInds(2,r2+1),:),3),1,featureSet.nObservations,[]);
            %
            %                                 %%% get norms
            %
            %                                 for q = 1:featureSet.nObservations
            %                                     cNorms(q) = norm(squeeze(cRegFiltIm(1,q,:)));
            %                                 end
            %
            %                                 X(:,:,i,r1,r2,j) =   reshape(shiftdim(cat(3,mean(cRegFiltIm,3),var(cRegFiltIm,0,3),skewness(cRegFiltIm,0,3),kurtosis(cRegFiltIm,0,3),cNorms),1),featureSet.nObservations,[]);
            %                             end
            %                         end
            %                         count = count + 1;
            %                         progBar.update(count/(self.nOrient*self.nScale));
            %                     end
            %                 end
            %%
            featureSet.X = double(reshape(X,featureSet.nObservations,[]));
            
        end
        
        function logGaborMat  = getLogGaborFilterBank(self,m,n)
            u = self.nScale;
            P = self.nOrient;
            scaleFact = self.scaleFactor;
            % LOGGABORFILTERBANK generates a custum Gabor filter bank.
            % It creates a u by v array, whose elements are m by n matries;
            % each matrix being a 2-D Gabor filter.
            %
            %
            % Inputs:
            %       u	:	No. of scales (usually set to 5)
            %       v	:	No. of orientations (usually set to 8)
            %       m	:	No. of rows in a 2-D Gabor filter (an odd integer number usually set to 39)
            %       n	:	No. of columns in a 2-D Gabor filter (an odd integer number usually set to 39)
            %
            % Output:
            %       gaborArray: A m by n by u by P array
            
            %%% Params
            sig_p = .996*sqrt(2/3);
            sig_theta = 0.996*(pi/6)*sqrt(1/3);
            %%%
            %%% getting image sampling ready
            [x,y] = meshgrid([-(m-1)/2:(m-1)/2]/(m/10),...
                [-(n-1)/2:(n-1)/2]/(n/10));
            
            x = x.';
            y = y.';
            
            radius = sqrt(x.^2 + y.^2);
            % Matrix values contain *normalised* radius
            % values ranging from 0 at the centre to
            % 0.5 at the boundary.
            radius(floor((m-1)/2 + 1), floor((n-1)/2+1)) = 1;
            
            theta = atan2(-y,x);              % Matrix values contain polar angle.
            % (note -ve y is used to give +ve
            % anti-clockwise angles)
            sintheta = sin(theta);
            costheta = cos(theta);
            %%%
            %%%initialize matrix
            logGaborMat = nan(m,n,u,P);
            %%%
            %%% loop
            for iKloop = 0:u-1
                iK = iKloop.*scaleFact.*((5)/(u-1)); % make the scaling max out at "6"
                rho_k = log2(6) - (sqrt(1/3)*iK);
                for iP = 0:P-1
                    theta_pk = (pi/P)*iP;
                    ds = sintheta * cos(theta_pk) - costheta * sin(theta_pk);    % Difference in sine.
                    dc = costheta * cos(theta_pk) + sintheta * sin(theta_pk);    % Difference in cosine.
                    dtheta = abs(atan2(ds,dc));
                    
                    
                    logGaborMat(:,:,iKloop+1,iP+1) = exp((-(radius - rho_k).^2) / (2 * (sig_p)^2)).*exp((-dtheta.^2) / (2 * (sig_theta)^2));
                end
            end
        end
    end
    
end