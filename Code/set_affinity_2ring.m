function [M,e,pairSeg,pairEng] = set_affinity_2ring(supixels,img_num,cfeat,gfeat,par,segIm)

for i = 1:img_num
    iii = 0;
    for ii = 1:img_num
        N1 = max(supixels{i}(:));
        N2 = max(supixels{ii}(:));
        W{i,ii} = zeros(N1,N2);
        if (i == ii)
            for j = 1:N1
                pool = [];
                rx = find(supixels{i}==j);
                neigh = getneighbor_demo(supixels{i},rx);
                pool = [pool;neigh.ind];
                for n = 1:neigh.num
                    temp = neigh.ind(n);
                    rx2 = find(supixels{i}==temp);
                    neigh2 = getneighbor_demo(supixels{i},rx2);
                    pool = [pool;neigh2.ind];
                end
                pool = unique(pool);
                setpool = pool(pool~=j);
                num = numel(setpool);
                for n = 1:num
                    temp = setpool(n);
                    d1 = Ka2distance_demo(cfeat{i,j},cfeat{ii,temp}); % color distance
                    d2 = Ka2distance_demo(gfeat{i,j},gfeat{ii,temp}); % gabor distance
                    if isempty(segIm)
                        W{i,ii}(j,temp) = exp(-(par.clambda*(d1/par.sigma_c) + par.glambda*(d2/par.sigma_g)));
                    else
                        ry = find(supixels{i}==temp);
                        f1 =((sum(segIm{i}(rx)==0)-sum(segIm{i}(rx)==1))>0);
                        f2 =((sum(segIm{i}(ry)==0)-sum(segIm{i}(ry)==1))>0);
                        if f1 == f2
                            W{i,ii}(j,temp) = exp(-(par.clambda*(d1/par.sigma_c) + par.glambda*(d2/par.sigma_g)));
                        else
                            W{i,ii}(j,temp) = 0;
                        end
                    end
                end
            end
        else
            iii = iii+1;
            for j = 1:N1
                for k = 1:N2
                    d3 = Ka2distance_demo(cfeat{i,j},cfeat{ii,k}); % color distance
                    d4 = Ka2distance_demo(gfeat{i,j},gfeat{ii,k}); % gabor distance
                    dist(k) = par.clambda*(d3/par.sigma_c) + par.glambda*(d4/par.sigma_g);
                end
                [val,loc] = sort(dist,'ascend');
                for m = 1:min(par.neighbors,N2)
                    W{i,ii}(j,loc(m)) = exp(-val(m));
                end
                e{i}(j,iii) = exp(-val(1));
                clear dist
            end
        end
    end
end

for kk = 1:img_num
    indx = find(W{kk,kk}~=0);
    [I,J] = ind2sub(size(W{kk,kk}),indx);
    pairSeg{kk} = [];
    pairEng{kk} = [];
    for kkk = 1:length(I)
        pairSeg{kk}(kkk,1) = (I(kkk));
        pairSeg{kk}(kkk,2) = (J(kkk));
        pairEng{kk}(kkk,1) = W{kk,kk}(I(kkk),J(kkk));
    end
    pairSeg{kk} = int32(pairSeg{kk});
end

% affinity matrix
M = cell2mat(W);
M = sqrt(M.*M');