// Define global settings for smoother curves
$fn = 64;

// --- Dimension Definitions ---

// Overall dimensions
total_height = 75;
base_width = 60;
base_depth = 50;
top_width = 25;

// Feature dimensions
hole_diameter = 25;
hole_depth = 65; // Through hole
hole_x_offset = 30.81;
hole_y_offset = 24.71;

fillet_radius = 35;
fillet_height = 45;

// Derived dimensions
base_height = 35;
top_height = total_height - fillet_height;

// --- Module Definitions ---

// Module for the base block
module base() {
    cube([base_width, base_depth, base_height]);
}

// Module for the top block
module top() {
    translate([0, base_depth - top_width, base_height])
    cube([base_width, top_width, top_height]);
}

// Module for the fillet
module fillet() {
    translate([0, base_depth - top_width, base_height])
    rotate_extrude(angle = 90)
    translate([fillet_radius, 0, 0])
    circle(r = fillet_radius);
}

// Module for the hole
module hole() {
    translate([hole_x_offset, hole_y_offset, base_height/2])
    cylinder(h = hole_depth, d = hole_diameter, center = true);
}

// --- Main Assembly Module ---

module main() {
    union() {
        base();
        top();
        fillet();
        difference(){
            union(){}
            hole();
        }
    }
}

main();
