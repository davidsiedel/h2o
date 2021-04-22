//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, 0.044, 0, 1.0};
//+
Point(3) = {0.048, 0.06, 0, 1.0};
//+
Point(4) = {0.048, 0.044, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Physical Curve("TOP", 5) = {2};
//+
Physical Curve("BOTTOM", 6) = {4};
//+
Physical Curve("LEFT", 7) = {1};
//+
Physical Curve("RIGHT", 8) = {3};
//+
Curve Loop(1) = {2, 3, 4, 1};
//+
Plane Surface(1) = {1};
//+
Transfinite Curve {2, 1, 4, 3} = 2 Using Progression 1;
//+
Transfinite Surface {1};
//+
Recombine Surface {1};
