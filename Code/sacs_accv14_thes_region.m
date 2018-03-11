function [w,supixels] = sacs_accv14_thes_region(map_names, inames, Mset, imgDir, supixels)

img_num = length(inames);
map_num = length(map_names);
bins = 256;
knn = 50;
thes = 0.1;

for i = 1:img_num
    imname = dir([imgDir '/' [inames{i},'.bmp']]);
    if isempty(imname)
        imname = dir([imgDir '/' [inames{i},'.jpg']]);
    end
    
    superpixels = supixels{i};
    spnum = max(superpixels(:)); % the actual number of the superpixels
    avg_p = zeros(spnum,2);
    
    for l = 1:spnum
        [xx,yy] = find(superpixels == l);
        avg_p(l,:) = mean([xx,yy]);
    end
    
    [nb,~] = knnsearch(avg_p,avg_p,'k',knn,'distance','euclidean');
    
    for p = 1:map_num
        mest = Mset{i,p};
        mest = im2double(mest);
        avg = zeros(1,spnum);
        for j=1:spnum
            h = superpixels == j;
            % calculate the average saliency score for each superpixel
            avg(j) = mean(mest(h));
        end;
        % compute the color histogram for the p-th map of the i-th image
        for sp = 1:spnum
            window = nb(sp,:);
            sign = avg(window) >= thes * max(avg(window)); % sailency thresholding
            slabels = window(sign); % record the superpixel label over the theshold
            fhis = [];
            for q = 1:numel(slabels)
                h = (superpixels == slabels(q));
                H = (mest(h));
                his = hist(H(:),(0:1:bins-1)/bins);
                fhis = [fhis;his];
            end;
            F{p,sp} = sum(fhis);%feature sp on p-th map
        end;
    end
    
    for sp = 1:spnum
        f_matrix = zeros(map_num,bins);
        for p = 1:map_num
            temp = F{p,sp};
            f_matrix(p,:) = temp;
        end
        f_matrix = f_matrix ./ 10000; % to reduce the scale of each bin, for accelerating the speed.
        %---------------------RPCA---------------------------%
        % lamda is used to control the weight of the saprsity of E
        lamda = 0.05;
        [~ ,E] = exact_alm_rpca(f_matrix',lamda);
        S = double(E');
        w{i,sp} = sqrt(sum(abs(S).^2,2));
        w{i,sp} = w{i,sp} / (max(w{i,sp})-min(w{i,sp})+1e-10); %normalization
        w{i,sp} = exp(-w{i,sp})+1e-10;
        sum_w = sum(w{i,sp},1);
        w{i,sp} = w{i,sp} / (sum_w);
    end
end
end