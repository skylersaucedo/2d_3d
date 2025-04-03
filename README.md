# 2D to 3D Model Generator

This project uses Claude AI to analyze three orthographic views of an object and generate a 3D model in STL format using OpenSCAD.

## Test Example

### Input Images

The project includes three test images in `sample_images/` showing different views of a mechanical part:

1. Front/Side View (`side1.jpg`, 13KB):
   - Height: 75 units
   - Width: 50 units
   - Features a rounded corner (R35) on the top right
   - Contains horizontal section markers

2. Top/Bottom View (`side2.jpg`, 11KB):
   - Width: 50 units
   - Height: 35 units (partial)
   - Shows vertical dividing line
   - Contains corresponding section markers

3. Side View with Hole (`side3.jpg`, 17KB):
   - Height: 75 units
   - Width: 60 units
   - Features a circular hole (Ø25 units)
   - Shows section markers

### Generated 3D Model

The system generates three output files in the `output/` directory:

1. OpenSCAD Source (`model.scad`, 645B):
   - Contains the parametric 3D model definition
   - Uses CSG operations for the main shape
   - Implements the rounded corner and cylindrical hole

2. STL Model (`output.stl`, 21KB):
   - Final 3D model ready for viewing or 3D printing
   - Represents a block with dimensions: 60x50x75 units
   - Features:
     - Rounded corner (R35) on one edge
     - Cylindrical hole (Ø25) through part of the body

3. Dimensions File (`output.brep`):
   - Contains the extracted dimensions in text format
   - Records width: 60, height: 75, depth: 50

## Setup and Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install OpenSCAD from [https://openscad.org/downloads.html](https://openscad.org/downloads.html)

3. Set up your environment variables:
```bash
ANTHROPIC_API_KEY=your_api_key_here
```

4. Run the test script:
```bash
python test_generation.py
```

The script will:
1. Load the three sample images
2. Send them to Claude for analysis
3. Generate OpenSCAD code based on the analysis
4. Create an STL file using OpenSCAD

## Project Structure

```
.
├── sample_images/          # Input test images
│   ├── side1.jpg          # Front/Side view
│   ├── side2.jpg          # Top/Bottom view
│   └── side3.jpg          # Side view with hole
├── output/                 # Generated files
│   ├── model.scad         # OpenSCAD source
│   ├── output.stl         # 3D model
│   └── output.brep        # Dimensions
├── claude_mcp_builder.py  # Main implementation
├── test_generation.py     # Test script
└── requirements.txt       # Python dependencies
```

## Dependencies

- FastAPI
- Uvicorn
- Python-multipart
- Python-dotenv
- Anthropic
- OpenSCAD (external software)

## Viewing the 3D Model

You can view the generated STL file using:
1. OpenSCAD itself
2. Online viewers like [ViewSTL](https://www.viewstl.com/)
3. 3D printing slicers like Cura or PrusaSlicer
4. CAD software like FreeCAD or Fusion 360 