function X = draw_line(X, x1, y1, x2, y2, value)
%  DRAW_LINE  Draws a line on matrix data.
%
%   A more serious implementation might use
%   Bresenham's algorithm:
%     https://en.wikipedia.org/wiki/Bresenham's_line_algorithm
%
%   X     : an (m x n) matrix.
%   x1,y1 : the starting point (in matrix coordinates)
%   x2,y2 : the ending point 
%   value : the value to write in the matrix X

[m,n] = size(X);
assert(1 <= x1 && 1 <= x2 && 1 <= y1 && 1 <= y2);
assert(x1 <= n && x2 <= n && y1 <= m && y2 <= m);

slope = (y2-y1) / (x2 - x1);

dx = (x2-x1)/1000;

x = x1:dx:x2;
y = y1+slope*(x-x1);
ind = sub2ind(size(X), round(y), round(x));

X(ind) = value;
