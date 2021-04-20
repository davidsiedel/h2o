H = DefineNumber[ 0.01, Name "Parameters/H" ];
//+
L = DefineNumber[ 0.01, Name "Parameters/L" ];
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {L, 0, 0, 1.0};
//+
Point(3) = {L, L, 0, 1.0};
//+
Point(4) = {0, L, 0, 1.0};
//+
Point(5) = {L/2, 0, 0, 1.0};
//+
Point(6) = {L/2, L/2, 0, 1.0};
//+
Point(7) = {0, L/2, 0, 1.0};
//+
Point(8) = {0, 0, H, 1.0};
//+
Point(9) = {L, 0, H, 1.0};
//+
Point(10) = {L, L, H, 1.0};
//+
Point(11) = {0, L, H, 1.0};
//+
Point(12) = {L/2, 0, H, 1.0};
//+
Point(13) = {L/2, L/2, H, 1.0};
//+
Point(14) = {0, L/2, H, 1.0};
//+



Line(1) = {5, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 7};
//+
Line(5) = {1, 5};
//+
Line(6) = {5, 6};
//+
Line(7) = {6, 7};
//+
Line(8) = {7, 1};
//+



//+
Line(9) = {1, 8};
//+
Line(10) = {5, 12};
//+
Line(11) = {7, 14};
//+
Line(12) = {6, 13};
//+
Line(13) = {13, 14};
//+
Line(14) = {13, 12};
//+
Line(15) = {12, 8};
//+
Line(16) = {8, 14};
//+
Line(17) = {12, 9};
//+
Line(18) = {2, 9};
//+
Line(19) = {10, 9};
//+
Line(20) = {10, 3};
//+
Line(21) = {10, 11};
//+
Line(22) = {11, 4};
//+
Line(23) = {4, 11};
//+
Line(24) = {14, 11};
//+
Curve Loop(1) = {9, 16, -11, 8};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {8, 5, 6, 7};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {7, 11, -13, -12};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {9, -15, -10, -5};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {14, 15, 16, -13};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {24, -21, 19, -17, -14, 13};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {11, 24, 22, 4};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {7, -4, -3, -2, -1, 6};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {22, -3, -20, 21};
//+
Plane Surface(9) = {9};
//+
Curve Loop(10) = {20, -2, 18, -19};
//+
Plane Surface(10) = {10};
//+
Curve Loop(11) = {18, -17, -10, 1};
//+
Plane Surface(11) = {11};
//+
Surface Loop(1) = {8, 7, 6, 9, 10, 11, 4, 1, 5, 2};
//+
Volume(1) = {1};
//+
Physical Surface("indent", 25) = {2};
//+
Physical Surface("bottom", 26) = {5, 6};
