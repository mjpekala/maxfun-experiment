function classification_report(y, Y_hat, desc)
% CLASSIFICATION_REPORT Analysis of classification performance
%
%  Parameters:
%    y     : nx1 vector of true class labels for n test cases
%
%    Y_hat : an nxm matrix of estimated class labels for n test cases
%            all estimated by m different
%            algorithms/configurations.
%
%    desc : a cell array of length m describing each of the
%    algorithms.

% sept 2016, mjp

acc_mu = sum(bsxfun(@eq, y(:), Y_hat), 1) / numel(y);

figure; 
stem(acc_mu);
set(gca, 'XTick', 1:size(Y_hat,2), 'XTickLabel', desc, ...
         'XTickLabelRotation', 45);
ylabel('accuracy');
title('Classification Accuracy');
xlim([0, size(Y_hat,2)+1]);


for ii = 1:size(Y_hat,2)
    [cm, gorder] = confusionmat(y(:), Y_hat(:,ii));

    labels = cellfun(@(x) sprintf('%d (%d / %d)', gorder(x), cm(x,x), sum(cm(x,:))), ...
                     num2cell(1:numel(gorder)), 'UniformOutput', 0);
    
    figure;
    imagesc(cm);
    title(sprintf('Confusion matrix for %s', desc{ii}));
    xlabel('y hat');
    colorbar;
    set(gca, 'YTick', 1:length(gorder), 'YTickLabel', labels);
    set(gca, 'XTick', 1:length(gorder), 'XTickLabel', num2str(gorder));
end
