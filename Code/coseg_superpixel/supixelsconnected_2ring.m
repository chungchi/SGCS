function E = supixelsconnected_2ring(superpixels)

E = [];
for i = 1:max(unique(superpixels))
    pool = [];
    rx = find(superpixels == i);
    neigh = getneighbor_demo(superpixels,rx);
    pool = [pool;neigh.ind];
    for n = 1:neigh.num
        temp = neigh.ind(n);
        rx2 = find(superpixels == temp);
        neigh2 = getneighbor_demo(superpixels,rx2);
        pool = [pool;neigh2.ind];
    end
    pool = unique(pool);
    setpool = pool(pool~=i);
    num = numel(setpool);
    cand = [double(i)*(ones(num,1)),setpool];
    E = vertcat(E,cand);
end
end