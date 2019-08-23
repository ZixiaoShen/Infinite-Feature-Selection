function [Rank, Score] = IFS_Zeal(train_X, train_y, alpha, supervision, verbose)

    if (nargin < 3)
        verbose = 0;
    end
    
    %% Standard Deviation over the samples
    if (verbose)
        fprintf('1) Priors/weights estimation \n');
    end
    
    if supervision
        sample_negative = train_X(train_y == -1, :);
        sample_positive = train_X(train_y == 1, :);
        mu_sample_negative = mean(sample_negative);
        mu_sample_positive = mean(sample_positive);
        priors_corr = ([mu_sample_positive - mu_sample_negative].^2);
        
        std_postive = std(sample_postive).^2;
        total_std = std_postive + std(sample_negative).^2;
        total_std(find(total_std == 0)) = 10000;
        
        corr_ij = priors_corr ./ total_std;
        corr_ij = corr_ij' * corr_ij;
        
        corr_ij = corr_ij - min(min(corr_ij));
        corr_ij = corr_ij ./ max(max(corr_ij));
        
    else
        [corr_ij, pval] = corr(train_X, 'type', 'Spearman');
        corr_ij(isnan(corr_ij)) = 0; % remove NaN
        corr_ij(isinf(corr_ij)) = 0; % remove inf
        corr_ij = 1 - abs(corr_ij);
        save(filename, 'corr_ij');
    end
    
    % Standard Deviation Estimation
    STD = std(train_X, [], 1);
    STDMatrix = bsxfun(@max, STD, STD');
    STDMatrix = STDMatrix - min(min(STDMatrix));
    sigma_ij = STDMatrix ./ max(max(STDMatrix));
    sigma_ij(isnan(sigma_ij)) = 0; % remove NaN
    sigma_ij(isinf(sigma_ij)) = 0; % remove inf
    
    %% 2) Building the graph G = <V, E>
    if (verbose)
        fprintf('2) Building the graph G = <V, E> \n');
    end
    A = (alpha * corr_ij + (1 - alpha) * sigma_ij);
    
    factor = 0.99;
    
    %% 3) Letting paths tend to infinite: Inf-FS Core
    if (verbose)
        fprintf('3) Letting paths tend to infinite \n');
    end
    I = eye(size(A, 1));  % Identity Matrix
    
    r = (factor/max(eig(A))); % Set a meaningful value for r
    
    y = I - (r * A);
    
    S = inv(y) - I; % see Gelfand's formula - convergence of the geometric series of matrices
    
    %% 4) Estimating energy scores
    if (verbose)
        fprintf('4) Estimating energy scores \n');
    end
    
    Score = sum(S, 2);  % energy scores s(i)
    
    %% 5) Ranking features according to s
    if (verbose)
        fprintf('5) Features ranking \n');
    end
    [~, Rank] = sort(Score, 'descend');
    
    Rank = Rank';
    Score = Score';
    
end

