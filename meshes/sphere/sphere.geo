//+
Point(1) = {0.8, 0, 0, 1.0};
//+
Point(2) = {1, 0, 0, 1.0};
//+
Point(3) = {0, 0, 0, 1.0};
//+
Point(4) = {0, 0.8, 0, 1.0};
//+
Point(5) = {0, 1, 0, 1.0};
//+
//Extrude {{0, 1, 0}, {0, 0, 0}, Pi/4} {
//  Surface{1}; Layers{5}; Recombine;
//}
//+
//+
Circle(1) = {1, 3, 4};
//+
Circle(2) = {2, 3, 5};
//+
Line(3) = {5, 4};
//+
Line(4) = {1, 2};
//+
Curve Loop(1) = {2, 3, -1, 4};
//+
Plane Surface(1) = {1};
//+
//Rotate {{0, 1, 0}, {0, 0, 0}, Pi/2} {
// Surface{1}; 
//}
//+
Extrude {{0, 1, 0}, {0, 0, 0}, Pi/2} {
  //Curve{1}; Curve{2}; Curve{3}; Curve{4}; Layers{5}; Recombine;
  Surface{1}; 
}
//+
Physical Surface("BOTTOM", 22) = {20};
//+
Physical Surface("RIGHT", 23) = {1};
//+
Physical Surface("LEFT", 24) = {21};
//+
Physical Surface("INTERIOR", 25) = {16};
//+
Physical Surface("EXTERIOR", 26) = {12};
//+
Physical Volume("SPHERE", 27) = {1};
