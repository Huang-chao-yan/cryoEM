function [IoU, area] = compute_IoU(region_a, region_b)
%COMPUTE_IOU Is compute the two region overlap area.
%    
%   ************************
%   *                      *
%   *      (x_a,y_a)******************
%   *          *           *         *
%   *          *           *         *
%   *          *           *         *
%   *******************(x_b,y_b)     *
%              *                     *
%              *                     *
%              ***********************

x_a = max(region_a(1), region_b(1));
y_a = max(region_a(2), region_b(2));
x_b = min(region_a(3), region_b(3));
y_b = min(region_a(4), region_b(4));

area_a = (region_a(3) - region_a(1) + 1) * (region_a(4) - region_a(2) + 1);
area_b = (region_b(3) - region_b(1) + 1) * (region_b(4) - region_b(2) + 1);

area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1);
IoU = area / (area_a + area_b - area);

end


