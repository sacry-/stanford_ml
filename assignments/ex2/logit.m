function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
    m = length(y);
    J_history = zeros(num_iters, 1);
    
    for iter = 1:num_iters
        h0s = zeros(m,1);
        for i = 1:m
            h0s(i) = h0(X(i,:),theta);
        end

        thetas_updated = zeros(1,length(theta));
        for j = 1:length(theta)
            cost = 0;
            for i = 1:m
                cost = cost + (h0s(i) - y(i))*X(i,j);
            end
            thetas_updated(j) = theta(j) - alpha * (1/m) * cost;
        end
        
        theta = thetas_updated;
        
        J_history(iter) = computeCost(X, y, theta);
    end
    
    figure;
    plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
    xlabel('Number of iterations');
    ylabel('Cost J');
end

function J = computeCost(X, y, theta)
    m = length(y);
    a = 0;
    for i = 1:m
       a = a + Cost(h0(X(i), theta), y(i)); 
    end
    J = -(1/m)*a;
end

function c = Cost(h, yi)
    c = (yi * log(h)) + ((1 - yi) * log(1 - h));
end

function h = h0(xi, theta)
    h = 1/(1 + exp(1)^(-sum(theta*xi')));
end
