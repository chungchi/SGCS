function [FDist] = ClustDistMembership(Color, FCClusters, FCovs, FWeights, supixels)
% compute weight of GMM for t-link 

NumFClusters = size(FCClusters,2);
numULabels = size(Color,1);
% FDist = zeros(numULabels,1);
% Ftmp = zeros(numULabels, NumFClusters);
number = max(supixels(:));
FDist = zeros(number,1);
Ftmp = zeros(number, NumFClusters);

for i = 1:number
    idx = find(supixels == i);
    mean_color(i,1)=mean(Color(idx,1));
    mean_color(i,2)=mean(Color(idx,2));
    mean_color(i,3)=mean(Color(idx,3));
end

for k = 1:NumFClusters
    M = FCClusters(:,k);
    CovM = FCovs(:,:,k);
    W = FWeights(1,k);
    V = mean_color - repmat(M',number,1);
    Ftmp(:,k) = -log((W / sqrt(det(CovM))) * exp(-( sum( ((V * inv(CovM)) .* V),2) /2))); 
end

FDist = min(Ftmp,[],2);% -log, so we select the minimum value