//+
H = DefineNumber[ 0.053334, Name "Parameters/H" ];
//+
Lh = DefineNumber[ 0.012826, Name "Parameters/Lh" ];
//+
Lb = DefineNumber[ 0.012595, Name "Parameters/Lb" ];
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {Lb/2, 0, 0, 1.0};
//+
Point(3) = {Lh/2, H/2, 0, 1.0};
//+
Point(4) = {0, H/2, 0, 1.0};
//+
Point(5) = {0.0025,0 , 0, 1.0};
//+
Point(6) = {0.0025,H/2 , 0, 1.0};
//+
Point(7) = {0,0.0055 , 0, 1.0};
//+
Point(8) = {Lb/2,0.0055 , 0, 1.0};
//+


Line(1) = {1, 5};
//+
Line(2) = {2, 8};
//+
Line(3) = {3, 6};
//+
Line(4) = {4, 7};
//+
Line(5) = {5, 2};
//+
Line(6) = {6, 4};
//+
Line(7) = {7, 1};
//+
Line(8) = {8, 3};
//+
Curve Loop(1) = {3, 4, 1, 2, 5,6,7,8};
//+
Plane Surface(1) = {1};
//+
Physical Curve("top", 5) = {3};
//+
Physical Curve("bottom", 6) = {1};
//+
Physical Curve("right", 7) = {2};
//+
Physical Curve("left", 8) = {4};

//+
Transfinite Surface {1} = {4, 3, 2, 1};
//+
Transfinite Curve {6, 1} = 8 Using Progression 1;
//+
Transfinite Curve {3, 5} = 8 Using Progression 1;
//+
Transfinite Curve {4, 8} = 10 Using Progression 1;
//+
Transfinite Curve {7, 2} = 15 Using Progression 1;
//+
Recombine Surface {1};