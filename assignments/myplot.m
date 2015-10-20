function [] = myplot(f, D, axis_dim, col)
    % handles = {@sin, @cos, tan};for i = 1:length(handles);myplot(handles{i});end
    % for i = 1:length(handles);myplot(handles{i});end
    %
    % D = linspace(-10,10,100);
    % a = [-10 10 -2 2];
    % myplot(@cos, D, a, '-r')
    % myplot(@tan, D, a, '-b')
    % myplot(@tan, D, '-m')
    if ~exist('f','var'), f = @(x) x.^3 + 3.*x; end
    if ~exist('D','var'), D = linspace(-10,10,100); end
    if ~exist('axis_dim','var'), axis_dim = [-5 5 -10 10]; end
    if ~exist('col','var'), col = '-b'; end
    
    Y = f(D);
    hold on;
    plot(D, Y, 'rx', 'MarkerSize', 4)
    plot(D, Y, col, 'LineWidth', 1)
    plot([0 0], axis_dim(3:4), '-k', 'LineWidth', 0.6)
    plot(axis_dim(1:2), [0 0], '-k', 'LineWidth', 0.6)
    plot([axis_dim(1) axis_dim(1)], axis_dim(3:4), '-k', 'LineWidth', 0.6)
    plot([axis_dim(2) axis_dim(2)], axis_dim(3:4), '-k', 'LineWidth', 0.6)
    axis(axis_dim)
    xlabel('D Data')
    ylabel('y = f(D)')
    hold off;
end

