function view_dataset(X, y, figdir)
% VIEW_DATASET  Visualize subset of a dataset.

if nargin < 3, figdir = ''; end


%% plot some images
y_all = unique(y);

for yi = y_all
    idx = find(y == yi);
   
    idx = randsample(idx, 1);
    
    figure;
    imagesc(X(:,:,:,idx));
    title(sprintf('object %d, y=%d', idx, yi));
    set(gca, 'XTick', [], 'YTick', []);
  
    if length(figdir)
        fn = sprintf('class_%02d_example.png', yi);
        saveas(gca, fullfile(figdir,fn));
    end
end

drawnow


%% look at class distribution
[n_examp_per_class,~] = hist(double(y), length(unique(y)));
n_examp_per_class
