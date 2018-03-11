% ICME17: Segmentation Guided Local Proposal Fusion for Co-saliency Detection
% Code Author: Chung-Chi "Charles" Tsai
% Email: chungchi@tamu.edu
% Date: July 2017
% Note: Please download the "CVX" and get the license file from "http://cvxr.com/cvx/download"
% If you think this code is useful, please consider citing:
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% @inproceedings{tsai2017segmentation,
%   title={Segmentation guided local proposal fusion for co-saliency detection},
%   author={Tsai, Chung-Chi and Qian, Xiaoning and Lin, Yen-Yu},
%   booktitle={Multimedia and Expo (ICME), 2017 IEEE International Conference on},
%   pages={523--528},
%   year={2017},
%   organization={IEEE}}
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This source code is for research purpose only.
% CoSegResults folder stores the iterative cosegmentation result
% CoSalResults folder stores the iterative cosaliency fusion result

clc; close all; clear all;
gt.Dir = './cosdata/gt/pair/pair105'; % input gt path
fdir.main ='./cosdata/images/pair/pair105'; % input image path
fdir.sub = dir([fdir.main, '/']); % input image path
fdir.sub = fdir.sub(~ismember({fdir.sub.name},{'.','..','.DS_Store'}));
fileNum = length(fdir.sub);
map.folder = './submaps/pair/pair105'; % local saliency maps path
% Group 1 saliency proposals
%map.names = {'_Achanta.bmp','_Itti.bmp','_Hou.bmp','_color.bmp','_patch.bmp'};
% Group 2 saliency proposals
map.names = {'_CA.png','_SF.png','_RBD.png','_Cor.png','_CoC.png'}; 
map.num = length(map.names);
maxIter = 8;

% Parameter settings for co-segmetnation
addpath(genpath('./coseg_superpixel')); % add function path
NumBClusters = 5; % number of background clusters for bkg GMM
NumFClusters = 5; % number of object clusters for obj GMM

% Parameter settings for co-saliency
addpath(genpath('./External')); % add function path
addpath(genpath('./SLIC')); % add function path
par.neighbors = 0; % number of inter-image responses (pairwise)
par.clambda = 1; % affinity value, standard deviation denominator for color feature (2nd data & pairwise)
par.glambda = 1.5; % affinity value, standard deviation denominator for saliency feature (2nd data & pairwise)
par.ccodebook = 100; % how many bins in color histogram (1st data & pairwise)
par.cclusternum = 150; % how many run times in kmeans (1st data & pairwise)

for fnum = 1:fileNum
    
    fprintf('begin processing image pairs: %s\r', fdir.sub(fnum).name);
    pair_num = fdir.sub(fnum).name;
    
    % Read-in Images
    img.Dir1 = ([fdir.main,'/',(pair_num)]);
    img.Dir2 = [img.Dir1,'/','*.bmp']; % input image path
    img.list = dir(img.Dir2);
    IRGB{1} = (imread(sprintf([img.Dir1,'/%s'],img.list(1).name)));
    IRGB{2} = (imread(sprintf([img.Dir1,'/%s'],img.list(2).name)));
    img.num = length(IRGB);
    
    gt.Dir1 = ([gt.Dir,'/',(pair_num)]);
    gt.Dir2 = [gt.Dir1,'/','*.bmp'];
    gt.list = dir(gt.Dir2);
    MaskList{1} = (imread(sprintf([gt.Dir1,'/%s'],gt.list(1).name)));
    MaskList{2} = (imread(sprintf([gt.Dir1,'/%s'],gt.list(2).name)));
    
    % Generating color and gabor features
    disp('generating features...');
    cvector = []; % color vector
    gvector = []; % gabor vector
    
    clear cvecc gvecc;
    
    for i = 1:img.num
        if strfind(img.list(i).name,'.bmp'),
            img.name{i} = strrep(img.list(i).name,'.bmp','');
        elseif strfind(img.list(i).name,'.png'),
            img.name{i} = strrep(img.list(i).name,'.png','');
        elseif strfind(img.list(i).name,'.jpg'),
            img.name{i} = strrep(img.list(i).name,'.jpg','');
        elseif strfind(img.list(i).name,'.JPG'),
            img.name{i} = strrep(img.list(i).name,'.JPG','');
        end;
        
        % Superpixels generation
        supixels{i} = SLIC_mex(IRGB{i},200,20);
        % Feature generation
        RGB = im2double(IRGB{i});
        R = RGB(:,:,1);
        G = RGB(:,:,2);
        B = RGB(:,:,3);
        rgb = [R(:),G(:),B(:)];
        Ycbr = double(rgb2ycbcr(IRGB{i}));
        Y = Ycbr(:,:,1)/255;
        Cb = Ycbr(:,:,2)/255;
        Cr = Ycbr(:,:,3)/255;
        ybr = [Y(:),Cb(:),Cr(:)];
        [Cl,Ca,Cb] = rgb2lab(IRGB{i}(:,:,1),IRGB{i}(:,:,2),IRGB{i}(:,:,3));
        Lab = [Cl(:)/100,(Ca(:)+110)/220,(Cb(:)+110)/220];
        cvecc{i} = [rgb,Lab,ybr]; % color feature
        cvector = [cvector;cvecc{i}];
        gvecc{i} = create_feature_space(IRGB{i}); % gabor feature
        gvector = [gvector;gvecc{i}];
    end
    
    % K-means clustering for color features
    % Bag-of-words representation
    des = cvector;
    randid = floor(size(des,1)*rand(1,par.ccodebook))'+1;
    randcen = des(randid,:);
    [~,ccenl] = do_kmeans(des',par.ccodebook,par.cclusternum,randcen');
    ccenl = ccenl+1;
    
    % K-means clustering for gabor features
    % Bag-of-words representation
    des = gvector;
    randid = floor(size(des,1)*rand(1,par.ccodebook))'+1;
    randcen = des(randid,:);
    [~,gcenl] = do_kmeans(des',par.ccodebook,par.cclusternum,randcen');
    gcenl = gcenl+1;
    
    % Read-in saliency maps
    submap_dir = sprintf('%s/%s',map.folder,fdir.sub(fnum).name); % local maps path
    Mset = read_maps(img.name,img.num,map.names,map.num,submap_dir);
    
    clear cfeat gfeat
    % Generate the feature histogram for superpixels
    pnum = 0;
    for i = 1:img.num,
        for j = 1:max(supixels{i}(:)),
            idx = find(supixels{i}(:) == j);
            cfeat{i,j} = hist(ccenl(idx+pnum),(1:par.ccodebook))/numel(idx);
            gfeat{i,j} = hist(gcenl(idx+pnum),(1:par.ccodebook))/numel(idx);
        end;
        pnum = pnum + numel(supixels{i});
    end;
    
    % Dervie the penalty for co-saliency unary term
    [w,~] = sacs_accv14_thes_region(map.names,img.name,Mset,img.Dir1,supixels); % Self-adaptive weight fusion
    
    segIm = [];
    for iter = 1:maxIter,
        
        close all; clc;
        par.alpha1 = 5; % weight for cosal data term
        par.alpha2 = 2; % weight for coupling term
        par.alpha3 = 5; % weight for coseg data term
        par.beta1 = 1;  % weight for cosal pairwise term
        par.beta2 = 0.1; % weight for coseg pairwise term
        
        fprintf('%i-th pairs,iter=%i\n',fnum,iter);
        
        % Affinity matrix
        disp('binary term...');
        [par.sigma_c,par.sigma_g] = compute_sigma_new(img.num,supixels,cfeat,gfeat);% Computer the normalization constant
        [M,e] = set_affinity_2ring(supixels,img.num,cfeat,gfeat,par,[]);% compute the affinity matrix
        affinity = M;
        
        % Compute the degree matrix
        D = diag(sum(affinity,2));
        L = D - affinity;
        [~,E,W] = eig(L);
        m = W*E^(1/2);
        
        % Unary term
        disp('unary term...');
        s_avg = avg_sm(supixels,img.num,map.num,Mset);
        [~,~,C,U] = set_U_exp(0,supixels,w,segIm,e,s_avg);
        
        % Optimization
        disp('optimization...');
        Nsum = size(U,1);
        clear x sm;
        cvx_begin quiet
        variable x(Nsum,map.num)
        for k = 1:map.num
            sm(1,k) = quad_form(m'*x(:,k),eye(Nsum));
        end
        minimize(par.alpha1*trace(U*x')+par.alpha2*trace(C*x')+par.beta1*sum(sm)+sum_square(x(:)))
        subject to
        ones(1,map.num)*x' == ones(1,Nsum);
        0 <= x(:) <= 1;
        cvx_end
        
        % Optimiztion weight
        w_new = set_w(img.num,supixels,x);
        % Cosalieny maps
        disp('maps fusion...')
        for i = 1:img.num,
            temp_map = zeros(size(Mset{i,1}));
            for q = 1:max(supixels{i}(:)),
                h = supixels{i} == q;
                for t = 1:map.num,
                    temp_map = temp_map+(w_new{i,q}(t)*h).*double(Mset{i,t});
                end;
            end;
            
            rs1 = temp_map;
            SalMap{i} = im2double(rs1);
            if (iter <= maxIter) || (iter == 1)
                name = 'CoSalResults/pair';
                cosal_dir = sprintf('./%s/%s/%s_%i.png',name,pair_num,img.name{i},iter);
                sp_make_dir(cosal_dir);
                imwrite(normalize(SalMap{i}),cosal_dir,'png');
            end
        end;
        % Setup in the first interation
        if iter == 1,
            ObjColors_AllImage = [];
            BkgGMM = cell(img.num,1);
            for i = 1:img.num,
                im_org = IRGB{i};
                im_cosal = SalMap{i};
                im_org1 = im_org(:,:,1);
                im_org2 = im_org(:,:,2);
                im_org3 = im_org(:,:,3);
                thres = mean(im_cosal(:));
                segIm_init = double(im_cosal < 2*thres);
                
                % Obj pixels
                ObjPixel_ID = find(segIm_init==0);
                ObjColors = zeros(length(ObjPixel_ID),3);
                ObjColors(:,1) = im_org1(ObjPixel_ID);
                ObjColors(:,2) = im_org2(ObjPixel_ID);
                ObjColors(:,3) = im_org3(ObjPixel_ID);
                ObjColors_AllImage = [ObjColors_AllImage;ObjColors];
                
                % Bkg pixels
                BkgPixel_ID = find(segIm_init==1);
                BkgColors = zeros(length(BkgPixel_ID),3);
                BkgColors(:,1) = im_org1(BkgPixel_ID);
                BkgColors(:,2) = im_org2(BkgPixel_ID);
                BkgColors(:,3) = im_org3(BkgPixel_ID);
                
                % Bkg GMM parameters
                [BId, ~] = kmeans(BkgColors,NumBClusters);
                Bdim = size(BkgColors,2);
                BCClusters = zeros(Bdim, NumBClusters);
                BWeights = zeros(1,NumBClusters);
                BCovs = zeros(Bdim, Bdim, NumBClusters);
                
                for k = 1:NumBClusters
                    relColors = BkgColors(BId==k,:); %% Colors belonging to cluster k
                    BCClusters(:,k) = mean(relColors,1)';
                    BCovs(:,:,k) = cov(relColors);
                    BWeights(1,k) = length(find(BId==k)) / length(BId);
                end
                BkgGMM{i}.BCClusters = BCClusters;
                BkgGMM{i}.BCovs = BCovs;
                BkgGMM{i}.BWeights = BWeights;
            end;
            
            % Obj GMM parameters from the obj pixels in all images in the set
            [FId,~] = kmeans(ObjColors_AllImage, NumFClusters);
            Fdim = size(BkgColors,2);
            FCClusters = zeros(Fdim, NumFClusters);
            FWeights = zeros(1,NumFClusters);
            FCovs = zeros(Fdim, Fdim, NumFClusters);
            
            for k = 1:NumFClusters
                relColors = ObjColors_AllImage(FId==k,:); %% Colors belonging to cluster k
                FCClusters(:,k) = mean(relColors,1)';
                FCovs(:,:,k) = cov(relColors);
                FWeights(1,k) = length(find(FId==k)) / length(FId);
            end
            
            ObjGMM.FCClusters = FCClusters;
            ObjGMM.FCovs = FCovs;
            ObjGMM.FWeights = FWeights;
        end
        
        % Minimize the energy function J by iterative Graph Cuts
        fprintf('%s-th Graph Cut\n',num2str(iter))
        ObjColors_AllImage = [];
        clear segIm SegColorIm;
        
        for i = 1:img.num,
            
            im_org = IRGB{i};
            im_cosal = SalMap{i};
            im_org1 = im_org(:,:,1);
            im_org2 = im_org(:,:,2);
            im_org3 = im_org(:,:,3);
            
            [segIm{i},SegColorIm{i}] = Graph_Cuts_extend_2ring_lambda(im_org,im_cosal,ObjGMM,BkgGMM{i},supixels,cfeat,gfeat,par,i,w_new);
            figure,imshow(SegColorIm{i},[]);
            SegColorIm_dir = sprintf('./CoSegResults/pair/%s/%s_%i.png',pair_num,img.name{i},iter);
            if iter <= maxIter
                sp_make_dir(SegColorIm_dir);
                imwrite(SegColorIm{i},SegColorIm_dir,'png');
            end
            
            % Obj pixels
            ObjPixel_ID = find(segIm{i}==0);
            ObjColors = zeros(length(ObjPixel_ID),3);
            ObjColors(:,1) = im_org1(ObjPixel_ID);
            ObjColors(:,2) = im_org2(ObjPixel_ID);
            ObjColors(:,3) = im_org3(ObjPixel_ID);
            ObjColors_AllImage = [ObjColors_AllImage;ObjColors];
            
            % Bkg pixels
            BkgPixel_ID = find(segIm{i}==1);
            BkgColors = zeros(length(BkgPixel_ID),3);
            BkgColors(:,1) = im_org1(BkgPixel_ID);
            BkgColors(:,2) = im_org2(BkgPixel_ID);
            BkgColors(:,3) = im_org3(BkgPixel_ID);
            
            %%%%%% update background GMM %%%%%%
            % bkg GMM parameters
            [BId,~] = kmeans(BkgColors, NumBClusters);
            Bdim = size(BkgColors,2);
            BCClusters = zeros(Bdim, NumBClusters);
            BWeights = zeros(1,NumBClusters);
            BCovs = zeros(Bdim, Bdim, NumBClusters);
            for k = 1:NumBClusters,
                relColors = BkgColors(BId==k,:); %% Colors belonging to cluster k
                BCClusters(:,k) = mean(relColors,1)';
                BCovs(:,:,k) = cov(relColors);
                BWeights(1,k) = length(find(BId==k)) / length(BId);
            end;
            BkgGMM{i}.BCClusters=BCClusters;
            BkgGMM{i}.BCovs=BCovs;
            BkgGMM{i}.BWeights=BWeights;
            
            %%%%%% update object GMM %%%%%%
            if i==img.num
                % Obj GMM parameters from the obj pixels in all images in the set
                [FId , ~] = kmeans(ObjColors_AllImage, NumFClusters);
                Fdim = size(BkgColors,2);
                FCClusters = zeros(Fdim, NumFClusters);
                FWeights = zeros(1,NumFClusters);
                FCovs = zeros(Fdim, Fdim, NumFClusters);
                for k = 1:NumFClusters
                    relColors = ObjColors_AllImage(FId==k,:); %% Colors belonging to cluster k
                    FCClusters(:,k) = mean(relColors,1)';
                    FCovs(:,:,k) = cov(relColors);
                    FWeights(1,k) = length(find(FId==k)) / length(FId);
                end
                ObjGMM.FCClusters = FCClusters;
                ObjGMM.FCovs = FCovs;
                ObjGMM.FWeights = FWeights;
            end
        end
        disp('Coseg code finishes...');
    end % End of iteration
    % mae : mean absolute error
    % ap : average precision
    % auc : area under the ROC curve
    % mF : mean F-measure (be aware there are several different definition)
    % wf : weighted F-measure (How to evaluate foreground maps? CVPR2014)
    [mae(fnum),ap(fnum),auc(fnum),mf(fnum),wf(fnum)] = EvalSalImage(SalMap,MaskList); % Evaluation
end % End of each image pair

