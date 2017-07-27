classdef feature_2DFFT
    properties (SetAccess = private)
        name = 'gets 2D FFT of patch';
        nameAbbreviation = '2DFFT';
    end
    
    methods
        function [featureSet] = run(self,featureSet)
            %%
            ogSize = featureSet.userData.imageSize;
            
            Xexp = single(reshape(featureSet.X,[featureSet.nObservations,ogSize])); % get data cube
            
            %%% create hamming window
            w = repmat(hamming(max(ogSize)),[1,max(ogSize)]);
            w = imresize(w.*w',ogSize);
            
            wBlock = repmat(shiftdim(w,-1),[featureSet.nObservations,1,1]);
            
            %%% take  2Dfft
            XexpFFT = abs(fftshift(fftshift(fft(fft(wBlock.*Xexp,[],2),[],3),2),3));
            X = double(reshape(XexpFFT(:,1:ceil(ogSize(1)/2),1:ceil(ogSize(2)/2),:),featureSet.nObservations,[]));

            featureSet.X = double(X);
        end
    end
    
end