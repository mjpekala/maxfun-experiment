function Z = all_windowed_sums(X, wVals)
% ALL_WINDOWED_SUMS  Uses an integral image to compute
%                    all "windowed" sums of an image.
%
%      Z = all_windowed_sums(X, wVals);
%
%    where:
%      X     := an (m x n) image
%      wVals := a vector of window width values
%
%      Z    := a tensor where Z(:,:,ii) is the result of convolving
%              an averaging filter of width(ii) over the entire
%              image X.  Each Z(:,:,ii) contains on the valid 
%              part of this convolution; the remainder is zero so
%              that all "slices" have the same size.
%

% mjp, april 2016

assert(ndims(X) == 2);

% calculate the integral image
% (used to calculate areas later below)
I = cumsum(cumsum(double(X),1),2);
% zero pad left and top so calculations work out
I = [zeros(size(I,1),1) I];
I = [zeros(1, size(I,2)) ; I];

% TODO: there's probably a clever way to do this calculation even
%       more efficiently... 
Z = zeros(size(X,1), size(X,2), length(wVals));
for ii = 1:length(wVals)
    w = wVals(ii);
    
    % The sum over a rectangle with corners A, B, C, D
    %
    %    A        B
    %    +--------+
    %    |        |
    %    |        |
    %    +--------+
    %    C        D
    %
    %  = I(D) + I(A) - I(B) - I(C).
    % Note this sum excludes the left/top edge (hence the w+1 below)
    % We'll compute all those sums in one fell swoop for a square
    % with side length w.
    filt = zeros(w+1, w+1);
    filt(1,1) = 1;
    filt(1,end) = -1;
    filt(end,1) = -1;
    filt(end,end) = 1;

    % Note: convn does not appear to be substantially faster than repeated
    % calls to conv2 in standalone tests...
    Zi = conv2(I, filt, 'valid') / w / w;     % note: normalizing
                                              % by measure of square
    Z(1:size(Zi,1), 1:size(Zi,2), ii) = Zi;
end

