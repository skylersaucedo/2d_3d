// Core dimensions
width = 60;  // Width
depth = 50;  // Depth
height = 75;  // Total height
top_width = 25;  // Width of top portion
top_height = 30;  // Height of top portion
corner_radius = 35;  // Corner radius
hole_diameter = 25;  // Hole diameter
hole_center_y = 24.71;  // Hole center y position from bottom edge

module main() {
    difference() {
        union() {
            // Main body
            cube([width, depth, height-top_height]);
            
            // Top portion
            translate([0, 0, height-top_height])
                cube([top_width, depth, top_height]);
        }
        
        // Rounded corner cutout
        translate([top_width, 0, height-top_height])
            difference() {
                cube([corner_radius, corner_radius, top_height]);
                translate([0, corner_radius, 0])
                    cylinder(r=corner_radius, h=top_height);
            }
        
        // Through hole
        translate([30.81, hole_center_y, -1])
            cylinder(d=hole_diameter, h=height-top_height+2);
    }
}

main();
