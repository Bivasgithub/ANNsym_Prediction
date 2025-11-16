data = readtable('DATA ANALYSIS_Nu.csv');

% Extract features (first 6 columns)
X = data{:, 1:6};

% Extract multiple outputs (e.g., columns 7 to 9 — adjust as needed)
Y = data{:, 7:8};  % Modify based on how many outputs you have

% Transpose for neural network input format
X = X';  % [features × samples]
Y = Y';  % [outputs × samples]

% Create a feedforward neural network with 10 hidden neurons
net = feedforwardnet(10, 'trainlm');  % 'trainlm' uses Levenberg-Marquardt

% Configure training parameters
net.trainParam.epochs = 200;
net.trainParam.goal = 1e-6;

% Set train/val/test split ratios
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.15;

% Train the network
net = train(net, X, Y);

% Predict on full dataset
Y_pred = net(X);
%Y_pred_test = net(X_test);
%Y_pred_val = net(X_val);

% Display predictions (transpose back for readability)
disp('Predicted outputs:');
disp(Y_pred');
disp(net.divideParam)
%disp(Y_pred_test');
%disp(Y_test);

%all_test = [X_test', Y_test', Y_pred_test'];
% Transpose back for row-wise format
X_all = X';              % [samples × features]
Y_true = Y';             % [samples × outputs]
Y_predicted = Y_pred';   % [samples × outputs]

% Combine all data into one matrix: [X | Y_true | Y_predicted]
all_data = [X_all, Y_true, Y_predicted];

% Create headers
n_features = size(X_all, 2);
n_outputs = size(Y_true, 2);
headers = [ ...
    arrayfun(@(i) sprintf('Feature_%d', i), 1:n_features, 'UniformOutput', false), ...
    arrayfun(@(i) sprintf('True_Output_%d', i), 1:n_outputs, 'UniformOutput', false), ...
    arrayfun(@(i) sprintf('Pred_Output_%d', i), 1:n_outputs, 'UniformOutput', false)];

% Write to CSV
writetable(array2table(all_data, 'VariableNames', headers), 'all_data_Nu_Nw.csv');














% Calculate R² (coefficient of determination) for each output
Y_true = Y';
Y_predicted = Y_pred';

n_outputs = size(Y_true, 2);

for i = 1:n_outputs
    y_true = Y_true(:, i);
    y_pred = Y_predicted(:, i);

    SS_res = sum((y_true - y_pred).^2);
    SS_tot = sum((y_true - mean(y_true)).^2);

    R_squared = 1 - (SS_res / SS_tot);

    fprintf('R² for Output %d: %.4f\n', i, R_squared);
end



% Assume residuals is a column vector of error values
% Example: residuals = Y_true(i,:)' - Y_predicted(i,:)';

residuals = Y_true(i,:)' - Y_predicted(i,:)';  % i = output index
maxLag = 20;

% Manually compute autocorrelation with negative lags
[acf, lags] = xcorr(residuals - mean(residuals), maxLag, 'coeff');

% Confidence limits (95%)
confLimit = 1.96 / sqrt(length(residuals));

% Plot
figure;
bar(lags, acf, 'b'); hold on;
yline(0, 'k');  % Zero correlation line
yline(confLimit, 'r--');  % Upper confidence limit
yline(-confLimit, 'r--'); % Lower confidence limit
xlabel('Lag');
ylabel('Correlation');
title(sprintf('Autocorrelation of Error %d', i));
legend('Correlations', 'Zero Correlation', 'Confidence Limit');
grid on;
