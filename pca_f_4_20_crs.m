function [finalE, finalx, fs, xs, is, end_iter_count] = pca_f_4_20_crs( weightmat, numneur, output, numcycles, threshold, replimit, numtest,plotonot,testname,mattype,data_dir)
%Rand function. Somewhat based on stimrnn7_31. Breaks for 50 repetitions of
%same best solution. within cycle checks best step.
%Cindy Steinhardt
%based on stimrnn7_31
%with output figures of error over time and output values as well as neuron activations
% taking random input from calling script

%% Initialize our functions
trans = @(x) x*weightmat;
err_f = @(x,y) mean((repmat(x, [size(y, 1) 1]) - y).^2, 2);
samples = 100;
cutoff = 2;
potential_comp = cutoff*(2*rand(samples, numneur)-1);
for i = 1:samples,
   testouts(i,:) =  potential_comp(i,:)*weightmat;
end
[A, B,r] = canoncorr(testouts, potential_comp);
 num1= size(A);
[pcomps,score,latent] = pcacov(A');
%num= sum(latent>1);
%Initialize our partitioning ratios
max_dim = round([1:replimit(2)]./replimit(2)*numneur);
pcomps = pcomps(:,1:max(max_dim));% change ideally to num (ones with latent over 1)!

% if numneur > replimit(2),
% pcomps = pcomps(:,1:replimit(2));
% else
%     pcomps = pcomps(:,1:numneur);% change ideally to num (ones with latent over 1)!
% end


%% Simulated Annealing

cutoff = 2;
% Number of accepted solutions
na = 0.0;

% Our target
desired_output = trans(output);

%Initialize our output variables
finalE = NaN*ones(numtest, 1);
finalx = NaN*ones(numtest, numneur);
fs = cell(numtest, 1);
xs = cell(numtest, 1);
is = cell(numtest, 1);
end_iter_count = NaN*ones(numtest, 1);

orig_frac = 0.95;
% Repeat the test multiple times
for cur_test = 1:numtest,
    
    % Initialize x to be random
    clear cur_xs cur_fs cur_is;
    cur_xs(1,:) = 10*(2*rand(1, numneur)-1);
    cur_fs(1,:) = err_f(desired_output, trans(cur_xs(end, :)));
    cur_is(1) = 0;
       
    frac = orig_frac;
    stop = 0; iter_count = 0;
    while ~stop,
        
        %Do simulated annealing locally, but do little runs
        run_xs = repmat(cur_xs(end, :), [replimit(1) 1]);
        run_err = cur_fs(end)*ones(replimit(1), 1);
        for cur_run = 1:replimit(1),
            run_xs(cur_run, :) = cur_xs(end, :);
            run_err(cur_run) = cur_fs(end);
            cur_run_frac = frac*ones(1, numneur);
            %Take this many steps in our run
            for cur_run_step = 1:replimit(2),
                %Start with just using the low dim spaces
                cur_num_dim = cur_run_step;
                cur_dim_range = max_dim(cur_num_dim);
                
                %Sample around our current step
                step_x = cutoff*repmat(cur_run_frac(1:cur_dim_range), [replimit(3) 1]).*(2*rand(replimit(3), cur_dim_range) - 1);
                step_x = step_x*pcomps(1:cur_dim_range,:);            
                temp_x = repmat(run_xs(cur_run, :), [replimit(3) 1]) + step_x;
                temp_x(temp_x < -10) = -10; temp_x(temp_x > 10) = 10; 
    
                temp_err = err_f(desired_output, trans(temp_x));
                [min_err, min_ind] = min(temp_err);
                iter_count = iter_count + replimit(3); %Update our iteration count
                  if iter_count >= numcycles,
                       stop = 1;
                       break;
                 end
                cur_run_frac(1:cur_dim_range) = cur_run_frac(1:cur_dim_range)*frac;
                %Move to the best location
                run_xs(cur_run, :) = temp_x(min_ind, :);
                run_err(cur_run) = min_err;
            end %run steps
            if stop==1,
                break
            end
        end %run tests
        
        [min_err, min_ind] = min(run_err);
        
        %Check to see if this is better than we were doing
        cur_is(end+1) = iter_count;
        if min_err < cur_fs(end),
            cur_fs(end+1) = min_err;
            cur_xs(end+1, :) = run_xs(min_ind, :);
            
        end
        
        %Check to see if we reached our minimum error
%         if cur_fs(end) <= threshold,
%             stop = 1;
%         end
        
        %Check to see if our iteration count is beyond our limit
      
               
        %Slowly slow down the learning?    
        frac = frac.*orig_frac;
        frac(frac <= 0.1) = 0.1;
        
    end %iteration loop
    
    end_iter_count(cur_test) = iter_count;
    finalE(cur_test) = cur_fs(end);
    finalx(cur_test, :) = cur_xs(end, :);
    fs{cur_test} = cur_fs;
    xs{cur_test} = cur_xs;
    is{cur_test} = cur_is;
    
end %test loop

if plotonot,
% figure(1); cla;
% for i = 1:numtest,
%     plot(is{i}, fs{i}, '-'); hold all;
% end
% v = axis;
% for i = 1:numtest,
%     if end_iter_count(i) < numcycles,
%         plot(end_iter_count(i)*[1 1], v(3:4), 'k-');
%     else
%         plot(is{i}(end), fs{i}(end), 'k*', 'MarkerSize', 20);
%     end
% end
% xlabel('Iteration Count'); ylabel('Error');
% title('PCA Subspaces');
%saveas(gcf, sprintf('pca_%d_learning.fig', numneur));

pltfig = figure(1);
[~, best_err] = min(finalE);
%clim = get(gca, 'CLim');
subplot(2,2,1); imagesc(reshape(trans(finalx(best_err, :)), [sqrt(numneur) sqrt(numneur)])); title('PCA - System Output'); %set(gca, 'CLim', [0 20]); %imagesc(reshape(output,[12 12]));
subplot(2,2,2); imagesc(reshape(desired_output, [sqrt(numneur) sqrt(numneur)])); title('Desired Output'); %set(gca, 'CLim', [0 20]); %imagesc(reshape(output,[12 12]));
subplot(2,2,3); imagesc(reshape(desired_output - trans(finalx(best_err, :)), [sqrt(numneur) sqrt(numneur)])); title('PCA - Difference');%imagesc(reshape(output,[12 12]));
%set(gca, 'CLim', clim);
subplot(2,2,4); imagesc(reshape(desired_output - trans(finalx(best_err, :)), [sqrt(numneur) sqrt(numneur)])); colorbar; title('PCA - Difference');%imagesc(reshape(output,[12 12]));
%saveas(gcf, sprintf('dct_%d_perf.fig', numneur));
%saveas(gcf, sprintf('pca_%d_perf.fig', numneur));

drawnow;
end
cd(data_dir);
savefig(pltfig,sprintf(['pca_n%d_' testname mattype '.fig'], numneur));
 close;
cd( data_dir(1:(end-12)));
end
