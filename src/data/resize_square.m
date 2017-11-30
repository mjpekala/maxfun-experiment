function x_out = resize_square(x, dim)
% RESIZE_SQUARE  "square-ifies" and resizes an image.
    
    [rows, cols, n_channels] = size(x);
    
    if rows ~= cols
        n = max(rows, cols);
        x_sq = zeros([n,n,size(x,3)], class(x));
        
        if rows < cols
            a = floor((n - rows)/2);
            a = max(a,1);
            b = a + size(x,1) - 1;
            x_sq(a:b,:,:) = x;
          
            % fill in background with something reasonable 
            if a > 1
                edge = x_sq(a,:,:);
                %x_sq(1:a-1,:,:) = edge(ones(a-1,1),:,:);
                for c = 1:n_channels
                    x_sq(1:a-1,:,c) = median(edge(:,:,c));
                end
            end
            if b < n
                edge = x_sq(b,:,:);
                %x_sq(b+1:end,:,:) = edge(ones(n-b,1),:,:);
                for c = 1:n_channels
                    x_sq(b+1:end,:,c) = median(edge(:,:,c));
                end
            end
            
        else
            a = floor((n - cols)/2);
            a = max(a,1);
            b = a + size(x,2) - 1;
            x_sq(:,a:b,:) = x;
            
            % fill in background with something reasonable 
            if a > 1
                edge = x_sq(:,a,:);
                %x_sq(:,1:a-1,:) = edge(:,ones(1,a-1),:);
                for c = 1:n_channels
                    x_sq(:,1:a-1,c) = median(edge(:,:,c));
                end
            end
            if b < n
                edge = x_sq(:,b,:);
                %x_sq(:,b+1:end,:) = edge(:,ones(1,n-b),:);
                for c = 1:n_channels
                    x_sq(:,b+1:end,c) = median(edge(:,:,c));
                end
            end
            
        end
    else
        x_sq = x;  % is already square
    end

    % now that x is square, resize to desired dimension.
    x_out = imresize(x_sq, [dim, dim]);
