// Gmsh project created on Fri May  7 13:35:06 2021
SetFactory("OpenCASCADE");
//+
H = DefineNumber[ 0.18, Name "Parameters/H" ];
//+
L = DefineNumber[ 0.1, Name "Parameters/L" ];
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {L/2, 0, 0, 1.0};
//+
Point(3) = {L, 0, 0, 1.0};
//+
Point(4) = {L, H/2, 0, 1.0};
//+
Point(5) = {L, H, 0, 1.0};
//+
Point(6) = {L/2, H, 0, 1.0};
//+
Point(7) = {0, H, 0, 1.0};
//+
Point(8) = {0, H/2, 0, 1.0};
//+
Point(9) = {L/2, H/2, 0, 1.0};
//+
Point(10) = {0.05, 0.03, 0, 1.0};
//+
//+
Point(11) = {0, 0.03, 0, 1.0};
//+
Point(12) = {0.1, 0.03, 0, 1.0};
//+




Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 12};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {8, 11};
//+
Line(9) = {8, 9};
//+
Line(10) = {9, 4};
//+
Line(11) = {6, 9};
//+
Line(12) = {9, 10};
//+
Line(13) = {10, 2};
//+
Line(14) = {1, 11};
//+
Line(15) = {12, 4};
//+
//+
Physical Curve("top", 16) = {6, 5};
//+
Physical Curve("bottom", 17) = {2};
//+
Physical Curve("left", 18) = {7, 8, 14};

Curve Loop(1) = {6, 7, 9, -11};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {5, 11, 10, 4};
//+
Plane Surface(2) = {2};




//+
Curve Loop(3) = {8, -14, 1, -13, -12, -9};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {10, -15, -3, -2, -13, -12};
//+
Plane Surface(4) = {4};
//+
Transfinite Surface {1} = {7, 6, 9, 8};
//+
Transfinite Surface {2} = {6, 5, 4, 9};
//+
Transfinite Surface {3} = {8, 9, 2, 1};
//+
Transfinite Surface {4} = {9, 4, 3, 2};

//+
Transfinite Curve {15, 12, 8} = 8 Using Progression 1;
//+
Transfinite Curve {4, 11, 7} = 8 Using Progression 1;
//+
Transfinite Curve {14, 13, 3} = 16 Using Progression 1;
//+
Transfinite Curve {2, 10, 5} = 8 Using Progression 1;
//+
Transfinite Curve {6, 9, 1} = 8 Using Progression 1;
//+
Recombine Surface {1, 2, 4, 3};
