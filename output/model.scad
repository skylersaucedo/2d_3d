// Main part dimensions
width = 60;
depth = 50;
height = 75;
corner_radius = 35;
hole_diameter = 25;

difference() {
    // Base block
    union() {
        // Main body
        translate([0, 0, 0])
            cube([width, depth, height]);
        
        // Cut out the corner with radius
        difference() {
            translate([width, 0, height-corner_radius])
                rotate([0, 0, 180])
                difference() {
                    cube([corner_radius, corner_radius, corner_radius]);
                    translate([0, 0, 0])
                        cylinder(r=corner_radius, h=corner_radius);
                }
        }
    }
    
    // Circular hole
    translate([width/2, depth/2, height/2])
        rotate([90, 0, 0])
        cylinder(d=hole_diameter, h=depth*2, center=true);
}
