% This function is to define t-links

function T = Calc_Weights_extend(im_org,im_cosal,par,objGMM,bkgGMM,superpixel)
% im-> input image
% fg-> foreground pixel locations in row matrixes Mx2
% bg-> background pixel locations in row matrixes Nx2
% [row,col] = size(im(:,:,1));
N = max(superpixel(:));
T = zeros(N,2);
lambda = par.alpha2;
lambda_GMM = par.alpha3;
number = max(superpixel(:));
addpath(genpath('EmGm')); % add function path

% define the t-links
if isempty(objGMM) && isempty(bkgGMM)
else
    im_org1 = im_org(:,:,1);
    im_org2 = im_org(:,:,2);
    im_org3 = im_org(:,:,3);
    ClusterNum = size(objGMM.FWeights,2);
    meancolor = zeros(number,3);
    for i = 1:number
        idx = find(superpixel == i);
        meancolor(i,1) = mean(im_org1(idx));
        meancolor(i,2) = mean(im_org2(idx));
        meancolor(i,3) = mean(im_org3(idx));
    end
 
    Fmodel.mu = objGMM.FCClusters;
    Fmodel.Sigma = objGMM.FCovs;
    Fmodel.w = objGMM.FWeights;
    [~,~,fg] = mixGaussPred(meancolor',Fmodel);
    Bmodel.mu = bkgGMM.BCClusters;
    Bmodel.Sigma = bkgGMM.BCovs;
    Bmodel.w = bkgGMM.BWeights;
    [~,~,bg] = mixGaussPred(meancolor',Bmodel);

    fgn = fg./(fg+bg);
    bgn = bg./(fg+bg);
    FDist = fgn;
    BDist = bgn;   
end


avg_sal = zeros(number,1);
for count = 1 : max(superpixel(:))
    idx = find(superpixel==count);
    avg_sal(count) = mean(im_cosal(idx)); % t-link with object,
end

for count = 1 : max(superpixel(:))
    if isempty(objGMM) && isempty(bkgGMM)
        T(count,1) = lambda*(avg_sal(count)); % t-link with object,
        T(count,2) = lambda*(1-avg_sal(count)); % t-link with background,
    else
        T(count,1) = lambda*(avg_sal(count))+lambda_GMM*FDist(count); % t-link with object,
        T(count,2) = lambda*(1-avg_sal(count))+lambda_GMM*BDist(count); % t-link with background,
    end
end
T = sparse(T);
end