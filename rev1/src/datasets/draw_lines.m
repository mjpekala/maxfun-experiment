function X = draw_line(X, p0, p1, value)
%  DRAW_LINE  Draws a line on matrix data.
%
%   A more serious implementation might use
%   Bresenham's algorithm:
%     https://en.wikipedia.org/wiki/Bresenham's_line_algorithm
%
%   X     : (m x n) matrix to draw lines on (treated as quadrant 4)
%   p0    : (p x 2) matrix of start points (x,y) 
%   p1    : (p x 2) matrix of end points (x,y)
%   value : the value to write in the matrix X

[m,n] = size(X);
assert(all(1 <= p0(:,1)) && all(p0(:,1) <= n));
assert(all(1 <= p1(:,1)) && all(p1(:,1) <= n));
assert(all(1 <= p0(:,2)) && all(p0(:,2) <= m));
assert(all(1 <= p1(:,2)) && all(p1(:,2) <= m));

x0 = p0(:,1);  y0 = p0(:,2);
x1 = p1(:,1);  y1 = p1(:,2);

slope = (y1 - y0) ./ (x1 - x0);

for ii = 1:length(x0)
    dx = (x1(ii) - x0(ii))/1000;
    x = x0(ii):dx:x1(ii);
    y = y0(ii) + slope(ii)*(x-x0(ii));
    ind = sub2ind(size(X), round(y), round(x));
    X(ind) = value;
end



