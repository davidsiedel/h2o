// Gmsh project created on Tue Apr 13 14:47:50 2021
SetFactory("OpenCASCADE");
//+
H = DefineNumber[ 10, Name "Parameters/H" ];
//+
L = DefineNumber[ 10, Name "Parameters/L" ];
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
Curve Loop(1) = {7, 8, 5, 6};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {4, -7, -6, 1, 2, 3};
//+
Plane Surface(2) = {2};
//+
Physical Surface("indent", 9) = {1};
//+
Transfinite Surface {1};
//+
Recombine Surface {1};
//+
Recombine Surface {2};
//+


//+
Extrude {0, 0, H} {
  Curve{5}; Curve{6}; Curve{7}; Curve{4}; Curve{8}; Curve{3}; Curve{2}; Curve{1}; Surface{1}; Layers {20}; Recombine;
}
