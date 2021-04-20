// Gmsh project created on Fri Apr 16 15:03:32 2021
SetFactory("OpenCASCADE");
//+
H = DefineNumber[ 0.18, Name "Parameters/H" ];
//+
L = DefineNumber[ 0.1, Name "Parameters/L" ];
//+
R = DefineNumber[ 0.05, Name "Parameters/R" ];
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {R, 0, 0, 1.0};
//+
Point(3) = {L, 0, 0, 1.0};
//+
Point(4) = {L, H, 0, 1.0};
//+
Point(5) = {0, H, 0, 1.0};
//+
Point(6) = {0, R, 0, 1.0};
//+
Line(1) = {2, 3};
//+
Line(2) = {3, 4};
//+
Line(3) = {4, 5};
//+
Line(4) = {5, 6};
//+
Circle(5) = {6, 1, 2};
//+
Curve Loop(1) = {4, 5, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Physical Curve("top", 6) = {3};
//+
Physical Curve("left", 7) = {4};
//+
Physical Curve("right", 8) = {2};
//+
Physical Curve("bottom", 9) = {1};
//+
Physical Curve("arc", 10) = {5};
