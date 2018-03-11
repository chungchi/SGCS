function [A,B,C,D] = set_U_exp(bgProbShift,supixels,w,segIm,e,s_avg)
A = []; % penalty from local consensus
B = []; % penalty from global repeatitiveness.
C = []; % penaly from deviation of cosaliency deteciton and cosegmetnation
D = []; % penalty for co-saliency unary term
img_num = length(supixels);
map_num = length(s_avg{1,1});
for id = 1:img_num
    if isempty(segIm)
        e_val = (e{id}-repmat(min(e{id}),size(e{id},1),1))./repmat((max(e{id})-min(e{id})+1E-8),size(e{id},1),1);
        for i = 1:max(supixels{id}(:))
            
            % A term
            A_vec1 = ones(1,map_num)-w{id,i}';
            temp_A = exp(A_vec1)./sum(exp(A_vec1));
            A = vertcat(A,temp_A);
            
            % B term
            mean_val = mean(e_val(i,:));
            std_val = std(e{id}(i,:));
            men = (mean_val)./(1+std_val);
            temp_vec1 = (1-men).*s_avg{id,i}+men.*(1-s_avg{id,i});
            temp_B = exp(temp_vec1)./ sum(exp(temp_vec1));
            B = vertcat(B,temp_B);
            
            temp_D = temp_A + temp_B;
            temp_D = exp(temp_D)./ sum(exp(temp_D));
            D = vertcat(D,temp_D);
            C = zeros(size(D));
            
        end
    else
        e_val = (e{id}-repmat(min(e{id}),size(e{id},1),1))./ repmat(max(e{id})-min(e{id}+1E-8),size(e{id},1),1);
        gt = double(segIm{id}); % read-in segmentation map
        for i = 1:max(supixels{id}(:))
            
            % A term
            A_vec1 = ones(1,map_num)-w{id,i}';
            temp_A = exp(A_vec1)./sum(exp(A_vec1));
            A = vertcat(A,temp_A);
            
            % B term
            mean_val = mean(e_val(i,:));
            std_val = std(e{id}(i,:));
            men = (mean_val)./(1+std_val);
            temp_vec1 = (1-men).*s_avg{id,i}+men.*(1-s_avg{id,i});
            temp_B = exp(temp_vec1)./sum(exp(temp_vec1));
            B = vertcat(B,temp_B);
            
            % C term
            idx = find(supixels{id}(:)==i);
            if sum(gt(idx)==0)>sum(gt(idx)==1)
                temp_C = ones(1,map_num) - s_avg{id,i} - bgProbShift;
            else
                temp_C = s_avg{id,i};
            end
            C = vertcat(C,temp_C);
            
            temp_D = temp_A + temp_B;
            temp_D = exp(temp_D)./sum(exp(temp_D));
            D = vertcat(D,temp_D);
        end
    end
end




