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



end