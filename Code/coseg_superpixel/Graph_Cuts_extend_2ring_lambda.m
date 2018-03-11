% Graph cut using max flow algorithm as described
% in Y. Boykov, M. Jolly, "Interactive Graph Cuts for Optimal Boundary and
% Region Segmentation of Objects in N-D Images", ICCV, 2001.

function [SegIm, SegColorIm, flow] = Graph_Cuts_extend_2ring_lambda(im_org,im_cosal,objGMM,bkgGMM,supixels,cfeat,gfeat,par,i,~)

lambda = par.alpha2;% Terminal weight strength
c = par.beta2; % Neighboring weight strength

%mex maxflowmex.cpp maxflow-v3.0/graph.cpp maxflow-v3.0/maxflow.cpp % Mex

disp('building graph');
% Construct graph and Calculate weighted graph
E = supixelsconnected_2ring(supixels{i});
E = double(E);
[numedge,~] = size(E);
Smooth_Dist = [];
for n = 1:numedge
    j = E(n,1);
    temp = E(n,2);        
    d1 = Ka2distance_demo(cfeat{i,j},cfeat{i,temp}); % color distance
    d2 = Ka2distance_demo(gfeat{i,j},gfeat{i,temp}); % gabor distance
    Dist = par.clambda*(d1/par.sigma_c) + par.glambda*(d2/par.sigma_g);
    Smooth_Dist = [Smooth_Dist; Dist];          
end

% define n-links
V = c.*exp((-abs(Smooth_Dist)));
K = max(V)+1;
A = sparse(E(:,1),E(:,2),V);
% define t-links
T = Calc_Weights_extend(im_org,im_cosal,par,objGMM,bkgGMM,supixels{i}); 

% Max flow Algorithm
disp('calculating maximum flow');
[flow,labels] = maxflow(A,T);
[w,h,~] = size(im_org);
SegColorIm = im_org;
SegIm = zeros(w,h);
zero_map = zeros(w,h);
one_map = ones(w,h);

for count = 1:max(supixels{i}(:))
    if(labels(count)==0)
        idx = find(supixels{i} == count);
        [I,J] = ind2sub(size(im_org),idx);
        for nb = 1:numel(idx);
            II = I(nb,1);
            JJ = J(nb,1);
            SegColorIm(II,JJ,1) = im_org(II,JJ,1);
            SegColorIm(II,JJ,2) = im_org(II,JJ,2);
            SegColorIm(II,JJ,3) = im_org(II,JJ,3);
            SegIm(II,JJ) = 0;
        end
    else
        idx = find(supixels{i} == count);
        [I,J] = ind2sub([w,h],idx);
        for nb = 1:numel(idx);
            II = I(nb,1);
            JJ = J(nb,1);
            SegColorIm(II,JJ,1) = 255;
            SegColorIm(II,JJ,2) = 255;
            SegColorIm(II,JJ,3) = 255;
            SegIm(II,JJ) = 1;
        end
    end
end

SegColorIm = uint8(SegColorIm);
SegIm = uint8(SegIm);
