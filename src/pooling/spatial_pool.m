function [Y, nfo] = spatial_pool(X, poolType, p)
% SPATIAL_POOL  Whole image pooling
%
%      Y = spatial_pool(X, poolType, [p])
%
%    where,
%      X := A tensor whose first two dimensions are spatial
%           dimensions. All other dimensions are treated as 
%           *independent* objects/instances.
%
%           Examples of reasonable semantics for dimensions of X:
%               (height, width, #_objects)
%               (height, width, #_channels, #_objects)
%                etc.
%
%      poolType := one of {'avg', 'max', 'sos', 'pnorm', 'maxfun'}
%
%      p := (optional) a scalar parameter whose meaning depends upon
%           the pool type.  If numel(p) > 1, then this function will
%           compute multiple poolings, one for each parameter in the
%           vector p.  This can be faster than calling spatial_pool()
%           once for each pooling parameter value.
%
%      Y   := the result of pooling over the two spatial dimensions.
%      nfo := (optional) information about the pool
%

% mjp, april 2016

sz = size(X);

% To make the implementation simpler, reshape X into either a two or
% three dimensional object, depending upon whether the underlying
% pooling operator cares about spatial relationships ("commutative" vs
% "non-commutative").  In the former case, X internally will be:
%
%    (size_of_pooling_region,  #_regions_to_pool)
%
% otherwise, X will have shape: 
%
%    (height, width, #_regions_to_pool)
%
% Regardless, we will reshape the result back to the proper dimensions
% before returning.  This is the job of the restore_dimensions function.
%
reshape_commutative = @(M) reshape(M, sz(1)*sz(2), prod(sz(3:end)));
reshape_noncommutative = @(M) reshape(M, sz(1), sz(2), prod(sz(3:end)));

if length(sz) > 3
    % we were asked to pool multiple regions; restore the shape of these
    % pooling regions.
    restore_dimensions = @(z) reshape(z, sz(3:end));
else
    % we were asked to pool a single region; ditch any trivial dimensions
    % that resulted from pooling.
    restore_dimensions = @squeeze;
end


% Apply operation to the pooling region
switch(lower(poolType))
    
  case {'avg', 'average'}
    X = reshape_commutative(X);
    y = mean(X,1);

 
  case {'max', 'maximum'}
    X = reshape_commutative(X);
    y = max(X,[],1);
   

  case {'p-norm', 'pnorm'}
    X = reshape_commutative(X);
    % Note: this isn't really the p-norm (due to scaling)
    y = (sum(X.^p, 1) / size(X,1)).^(1/p);
   
    
  case {'sos'}
    X = reshape_commutative(X);
    Y = sort(X,1,'descend');  % sort each column of X
    
    if numel(p) == 1
        y = sum(Y(1:p,:)) / p;    % average the p largest in each column
    else
        y = cellfun(@(v) sum(Y(1:v,:))/v, num2cell(p), 'UniformOutput', 0);
    end

    
  case {'maxfun', 'fun'}
    % pooling using the (uncentered) maximal function-inspired pooling
    X = reshape_noncommutative(X);  % (h, w, #_regions)
    X = abs(X);                     % def. MAXFUN
    wMin = p(1);
    wMax = min([size(X,1), size(X,2)]);
    wVals = wMin:wMax;
    
    to_col = @(x) x(:);
    
    if numel(p) == 1
        % Only a single parameter; no need to sweep over values
        %
        y = zeros(size(X,3),1);      % the maxfun value
       
        % these next three variables store information about the maxfun support.
        c0 = zeros(size(X,3),1);     % column coordinate (upper left)
        r0 = zeros(size(X,3),1);     % row coordinate (upper left)
        w0 = zeros(size(X,3),1);     % the maxfun window width selected
       
        % process each 2d image separately
        for ii = 1:size(X,3)
            Z = all_windowed_sums(X(:,:,ii), wVals);
            [y(ii), idx] = max(Z(:));

            % only calculate if user requested diagnostics
            if nargout > 1
                [r0(ii), c0(ii), w0(ii)] = ind2sub(size(Z), idx);
                w0(ii) = wVals(w0(ii));
            end
        end
    else
        y = cellfun(@(v) zeros(size(X,3),1), num2cell(1:numel(p)), 'UniformOutput', 0);

        % The calculation below is ordered in such a way that we
        % only need calculate all_windowed_sums once per example.
        % We re-use that same result when sweeping over p.
        for ii = 1:size(X,3)
            Z = all_windowed_sums(X(:,:,ii), wVals);
            for jj = 1:length(p)
                idx = find(p(jj) == wVals);
                y{jj}(ii) = max(to_col(Z(:,:,idx:end)));
            end
        end
    end

    
  otherwise
    error('unsupported pooling type');
end


% restore dimensions
if ~iscell(y)
    Y = restore_dimensions(y);

    % this part only applies to MAXFUN
    if nargout > 1
        nfo.col = restore_dimensions(c0);
        nfo.row = restore_dimensions(r0);
        nfo.w = restore_dimensions(w0);
    end
else
    Y = cellfun(restore_dimensions, y, 'UniformOutput', 0);
end
