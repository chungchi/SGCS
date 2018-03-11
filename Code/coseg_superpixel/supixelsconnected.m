function E = supixelsconnected(superpixels)

% EDGES8CONNECTED Creates edges where each node
%   is connected to its eight adjacent neighbors on a 
%   height x width grid.
%   E - a vector in which each row i represents an edge
%   E(i,1) --> E(i,2). The edges are listed is in the following 
%   neighbor order: down,up,right,left, where nodes 
%   indices are taken column-major.
%
%   (c) 2008 Michael Rubinstein, WDI R&D and IDC
%    Revised by Min Xian, PR research center@Utah State University
%   $Date 2011/09/17$
%

% N = height*width;
% I = []; 
% J = [];
% % connect vertically (down, then up)
% is = [1:N]'; is([height:height:N])=[];
% js = is+1;
% I = [I;is;js];
% J = [J;js;is];
% % connect horizontally (right, then left)
% is = [1:N-height]';
% js = is+height;
% I = [I;is;js];
% J = [J;js;is];
% % connect 45 (top right)
% is = [1:N-height]'; 
% is([1:height:N-height])=[];
% js = is+height -1;
% I = [I;is;js];
% J = [J;js;is];
% % connect -45 (down right)
% is = [1:N-height]'; 
% is([height:height:N-height])=[];
% js = is+height +1;
% I = [I;is;js];
% J = [J;js;is];
% 
% E = [I,J];

E = [];
for i = 1:max(unique(superpixels))
    rx = find(superpixels == i);
    neigh = getneighbor_demo(superpixels,rx);            
    for n = 1:neigh.num
        temp = neigh.ind(n);
        E = vertcat(E,[i,temp]);
    end
end  


end