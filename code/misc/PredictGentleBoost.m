function label = PredictGentleBoost(clss,feature)

nRounds = size(clss.k,2); % same with size(clss.th,2) or size(clss.a,2)

k   = clss.k;
th  = clss.th;
a   = clss.a;
b   = clss.b;

nSample = size(feature,1);
label   = zeros(nSample,1);

Fxt     = zeros(nSample,1);
for jj = 1:nRounds
    temp = feature(:,k(jj));
    fm  = (a(jj) * (temp > th(jj)) + b(jj));
    Fxt = Fxt + fm;
end

label(Fxt>=0) = clss.idx(1);
label(Fxt<0)  = clss.idx(2);





