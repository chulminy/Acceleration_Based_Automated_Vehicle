function [k, th, a, b, Fx, w] = GentleBoost_parallel(featureVec, featureIdx, Nrounds)

% function [k, th, a, b, Fx, w] = GentleBoost(featureVec, featureIdx, Nrounds)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize output of strong classifiers on training and test set
Fx = 0;

Npt_total = size(featureVec,2);

% Initialize to one all the weights for training samples
w  = ones(1, Npt_total);

for m = 1:Nrounds
    % Weak regression stump: It is defined by four parameters (a,b,k,th)
    % f_m = a * (x_k > th) + b
    [k(m), th(m), a(m), b(m), error] = ...
        selectBestRegressionStump(featureVec, featureIdx, w);
    
    
    % Updating and computing classifier output on training set
    fm = (a(m) * (featureVec(k(m),:)>th(m)) + b(m));
    Fx = Fx + fm;
    
    if error < 0.00001
       featureVec(k(m),:) = rand(1,size(featureVec,2)); % a trick not to select corresponding weak leaner
    end
    % Reweight training samples
    w = w .* exp(-featureIdx.*fm);
end

end
function [featureNdx, th, a , b, error] = selectBestRegressionStump(x, z, w)
% [th, a , b] = fitRegressionStump(x, z);
% z = a * (x>th) + b;
%
% where (a,b,th) are so that it minimizes the weighted error:
% error = sum(w * |z - (a*(x>th) + b)|^2) / sum(w)

[Nfeatures, Nsamples] = size(x); % Nsamples = Number of thresholds that we will consider

th      = zeros(1,Nfeatures);
a       = zeros(1,Nfeatures);
b       = zeros(1,Nfeatures);
error   = zeros(1,Nfeatures);
% 
for n = 1:Nfeatures
    [th(n), a(n) , b(n), error(n)] = fitRegressionStump(x(n,:), z, w);
end

% [th, a , b, error] = fitRegressionStump(x', z, w);

[error, featureNdx] = min(error);
th  = th(featureNdx);
a   = a(featureNdx);
b   = b(featureNdx);

end
function [th, a , b, error] = fitRegressionStump(x, z, w)
% [th, a , b] = fitRegressionStump(x, z);
% The regression has the form:
% z_hat = a * (x>th) + b;
%
% where (a,b,th) are so that it minimizes the weighted error:
% error = sum(w * |z - (a*(x>th) + b)|^2)
%
% x,z and w are vectors of the same length
% x, and z are real values.
% w is a weight of positive values. There is no asumption that it sums to
% one.

% atb, 2003

Nsamples = size(x,2); % Nsamples = Number of thresholds that we will consider

%threshold will be located in between samples.
w = w/sum(w); % just in case... (w should be 1)

[x, j] = sort(x);
% this now becomes the thresholds. I assume that all the values are different.
% If the values are repeated you might need to add some noise.

z = z(j); w = w(j);
Szw = cumsum(z.*w); Ezw = Szw(end);
Sw  = cumsum(w);

% This is 'a' and 'b' for all posible thresholds:
b = Szw ./ Sw;
zz = Sw == 1; % boundary
Sw(zz) = 0;
a = (Ezw - Szw) ./ (1-Sw) - b;
Sw(zz) = 1;

% Now, let's look at the error so that we pick the minimum:
% the error at each threshold is:
% for i=1:Nsamples
%     error(i) = sum(w.*(z - ( a(i)*(x>th(i)) + b(i)) ).^2);
% end
% but with vectorized code it is much faster but also more obscure code:
Error = sum(w.*z.^2) - 2*a.*(Ezw-Szw) - 2*b*Ezw + (a.^2 +2*a.*b) .* (1-Sw) + b.^2;

% Output parameters. Search for best threshold (th):
[error, k] = min(Error);

if k == Nsamples
    th = x(k);
else
    th = (x(k) + x(k+1))/2;
end
a = a(k);
b = b(k);

end