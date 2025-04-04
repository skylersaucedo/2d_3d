import os
import time
import asyncio
import base64
import google.generativeai as genai
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta
from collections import deque

load_dotenv()

class TokenBucket:
    def __init__(self, tokens_per_minute, window_size=60):
        self.capacity = tokens_per_minute
        self.window_size = window_size  # in seconds
        self.tokens = deque()  # Store timestamps and token counts
        
    async def consume(self, tokens):
        now = datetime.now()
        
        # Remove old tokens outside the window
        while self.tokens and (now - self.tokens[0][0]).total_seconds() > self.window_size:
            self.tokens.popleft()
        
        # Calculate current token usage in window
        current_usage = sum(count for _, count in self.tokens)
        
        if current_usage + tokens > self.capacity:
            # Calculate wait time needed
            oldest_timestamp = self.tokens[0][0] if self.tokens else now
            wait_seconds = self.window_size - (now - oldest_timestamp).total_seconds()
            if wait_seconds > 0:
                print(f"Rate limit reached. Waiting {wait_seconds:.2f} seconds...")
                await asyncio.sleep(wait_seconds)
        
        # Add new token usage
        self.tokens.append((now, tokens))

class GeminiBuilder:
    def __init__(self):
        # Configure the Gemini API
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Initialize Gemini gemini-2.0-flash
        generation_config = {
            "temperature": 0.1,  # Lower temperature for more precise outputs
            "top_p": 0.5,       # More focused sampling
            "top_k": 32,        # Limit vocabulary diversity
            "max_output_tokens": 2048,  # Allow for longer responses
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Initialize token buckets for rate limiting
        self.request_bucket = TokenBucket(60)  # 60 requests/minute
        
        # Create output directories if they don't exist
        os.makedirs("output", exist_ok=True)
        
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and prepare image for Gemini"""
        try:
            img = Image.open(image_path)
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize if too large (Gemini has a 4MB limit per image)
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            return img
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {str(e)}")
        
    async def generate_model(self, side1_path: str, side2_path: str, side3_path: str, side4_path: str = None) -> tuple[str, str]:
        """
        Generate a 3D model from three or four side images using Gemini Pro Vision
        """
        try:
            # Load images
            side1_img = self._load_image(side1_path)
            side2_img = self._load_image(side2_path)
            side3_img = self._load_image(side3_path)
            side4_img = self._load_image(side4_path) if side4_path else None

            # Create prompt for Gemini
            prompt = self._create_prompt()
            
            # Wait if needed based on request rate limit
            await self.request_bucket.consume(1)

            # Prepare content list
            contents = [prompt, side1_img, side2_img, side3_img]
            if side4_img:
                contents.append(side4_img)

            # Get Gemini's response
            response = self.model.generate_content(
                contents=contents,
                stream=False
            )
            
            # Check if response was blocked
            if response.prompt_feedback.block_reason:
                raise ValueError(f"Response blocked: {response.prompt_feedback.block_reason}")
            
            # Debug: Print response
            print(f"Response type: {type(response.text)}")
            print(f"Response content: {response.text}")
            
            # Extract the dictionary containing OpenSCAD code and dimensions
            result = self._extract_dict_from_response(response.text)
            
            # Generate 3D model using OpenSCAD
            stl_path, brep_path = self._generate_3d_files(result)

            return stl_path, brep_path
            
        except Exception as e:
            print(f"Error in generate_model: {str(e)}")
            raise e

    def _create_prompt(self) -> str:
        """Create prompt for Gemini based on the images"""
        return """
        You are a CAD expert. Analyze these technical drawings and create a precise 3D model based on the provided views. Generate complete, executable OpenSCAD code.

        1. DIMENSIONAL ANALYSIS:
           - Carefully measure and identify ALL critical dimensions from the images
           - Document each dimension with a clear, descriptive name
           - Include both primary and secondary feature dimensions
           - Note any relationships between dimensions
           - Consider both explicit and derived measurements
           - Pay special attention to:
             * Radii of all curved surfaces and fillets
             * Center points of circular features
             * Angular measurements for non-orthogonal features
             * Depth of hidden features

        2. FEATURE ANALYSIS:
           - Identify the primary geometric forms (cylinders, cubes, etc.)
           - Document all secondary features (holes, cuts, threads, etc.)
           - Note symmetry, patterns, and relationships
           - Analyze hidden geometry:
             * Use all available views to infer hidden features
             * Cross-reference dimensions across views
             * Consider internal structures suggested by visible features
             * Look for hints of internal voids or channels
           - Curve Analysis:
             * Parameterize ALL curved surfaces with exact equations
             * Define control points for Bézier curves
             * Specify sweep paths for complex curves
             * Document curve tangency and continuity
           - Pay special attention to:
             * Feature positions and alignments
             * Thread characteristics (pitch, depth, angle)
             * Connection points and transitions
             * Surface details and finishes
             * Intersections between curved surfaces

        3. OPENSCAD IMPLEMENTATION:
           - Start with clear variable definitions for ALL identified dimensions
           - Create a logical module hierarchy:
             * Base/core geometry module
             * Feature-specific modules for each major component
             * Curve generation modules using exact mathematical definitions
             * Assembly module for final composition
           - For curved features:
             * Use mathematical functions to define curves (sin, cos, etc.)
             * Implement proper sweep and extrude operations
             * Maintain tangency between curved surfaces
             * Use hull() or minkowski() for complex blends
           - For threaded features:
             * Use linear_extrude with FIXED twist value (not calculated)
             * OR use a limited for loop (max 360 steps)
             * Keep thread profile simple (2-3 points maximum)
             * Use small thread depth (0.1-0.2 times diameter)
           - Performance optimization:
             * Limit recursive operations
             * Use simple boolean operations
             * Avoid complex calculations in loops
             * Keep polygon counts reasonable
           - End with a single main() module that:
             * Takes no parameters
             * Assembles all components
             * Is called on the last line

        After analysis, respond with ONLY a dictionary in this EXACT format:

        {
            "openscad_code": "// Your complete OpenSCAD code here\\n",
            "dimensions": {
                // ALL dimensions you identified from the images
                // Each with a descriptive name and numeric value
                // Include curve parameters and control points
                // Example: "total_height": 50.0,
                // Example: "fillet_radius": 3.0,
                // Example: "curve_control_point_1": [10.0, 5.0, 0.0]
            }
        }

        CRITICAL REQUIREMENTS:
        1. Use DOUBLE QUOTES for all strings
        2. Escape newlines with \\n in OpenSCAD code
        3. Include ONLY the dictionary in your response
        4. Set $fn=64 for better performance
        5. Use descriptive variable names
        6. Include comprehensive comments
        7. End with 'main();' on its own line
        8. Define ALL dimensions as variables
        9. Use proper modular design
        10. Document ALL curve parameters
        11. Cross-reference features across ALL views
        12. Verify geometric continuity"""

    def _extract_dict_from_response(self, response_text: str) -> dict:
        """Extract the dictionary from Gemini's response"""
        try:
            # Find the dictionary start
            dict_start = response_text.find('{')
            if dict_start == -1:
                raise ValueError("No dictionary found in response")

            # Find the dictionary end
            dict_end = response_text.rfind('}')
            if dict_end == -1:
                raise ValueError("No closing brace found")

            # Extract the dictionary string
            dict_str = response_text[dict_start:dict_end + 1]
            dict_str = dict_str.strip()
            dict_str = dict_str.replace("'", '"')

            # Clean up newlines in OpenSCAD code
            import json
            try:
                result = json.loads(dict_str)
            except json.JSONDecodeError:
                # If failed, try to clean up the OpenSCAD code
                import re
                code_match = re.search(r'"openscad_code":\s*"([^"]*)"', dict_str)
                if code_match:
                    code = code_match.group(1)
                    code = code.replace('\n', '\\n')
                    dict_str = re.sub(r'"openscad_code":\s*"[^"]*"', f'"openscad_code": "{code}"', dict_str)
                    result = json.loads(dict_str)
                else:
                    raise ValueError("Could not find OpenSCAD code in response")

            # Validate the structure
            if not isinstance(result, dict):
                raise ValueError("Extracted content is not a dictionary")

            if 'openscad_code' not in result or 'dimensions' not in result:
                raise ValueError("Missing required fields in response")

            # Validate dimensions
            dims = result['dimensions']
            if not isinstance(dims, dict) or not dims:
                raise ValueError("Dimensions must be a non-empty dictionary")

            # Ensure all dimension values are numeric
            for key, value in dims.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Dimension '{key}' must have a numeric value")

            return result

        except Exception as e:
            print(f"Error extracting dictionary: {str(e)}")
            if 'dims' in locals():
                print(f"Found dimensions: {list(dims.keys())}")
            raise ValueError(f"Failed to extract dictionary from response: {str(e)}")

    def _find_openscad_path(self) -> str:
        """Find the OpenSCAD executable path"""
        # Common installation paths for OpenSCAD on Windows
        possible_paths = [
            r"C:\Program Files\OpenSCAD\openscad.exe",
            r"C:\Program Files (x86)\OpenSCAD\openscad.exe",
            os.path.expanduser("~\\AppData\\Local\\Programs\\OpenSCAD\\openscad.exe")
        ]
        
        # Check common installation paths first
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        # Check if OpenSCAD is in PATH
        import subprocess
        try:
            result = subprocess.run(['where', 'openscad'], capture_output=True, text=True)
            if result.returncode == 0:
                # Take the first path found
                return result.stdout.strip().split('\n')[0]
        except Exception:
            pass
            
        raise ValueError(
            "OpenSCAD not found. Please install it from https://openscad.org/downloads.html "
            "and make sure it's in your system PATH or installed in the default location."
        )

    def _generate_3d_files(self, model_dict: dict) -> tuple[str, str]:
        """Generate STL and BREP files using OpenSCAD"""
        try:
            # Get the OpenSCAD code and ensure it ends with a newline
            openscad_code = model_dict['openscad_code'].replace('\\n', '\n').strip() + '\n'
            
            # Create output directory if it doesn't exist
            output_dir = os.path.abspath(os.path.join("output"))
            os.makedirs(output_dir, exist_ok=True)
            
            # Create paths using proper Windows path handling
            scad_path = os.path.join(output_dir, "model.scad")
            stl_path = os.path.join(output_dir, "output.stl")
            brep_path = os.path.join(output_dir, "output.brep")
            
            # Write OpenSCAD file
            with open(scad_path, 'w') as f:
                f.write(openscad_code)
            
            print(f"Generated OpenSCAD file at: {scad_path}")
            print(f"OpenSCAD code:\n{openscad_code}")
            
            # Find OpenSCAD executable
            openscad_exe = self._find_openscad_path()
            
            # Run OpenSCAD using subprocess for better path handling
            import subprocess
            try:
                result = subprocess.run([
                    openscad_exe,
                    "-o",
                    stl_path,
                    scad_path
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print("OpenSCAD Error Output:")
                    print(result.stderr)
                    raise ValueError(f"OpenSCAD command failed with exit code {result.returncode}")
                
            except Exception as e:
                print(f"Error running OpenSCAD: {str(e)}")
                print(f"Command attempted: {openscad_exe} -o {stl_path} {scad_path}")
                raise ValueError("Failed to run OpenSCAD command. Please ensure OpenSCAD is properly installed.")
            
            # Save dimensions as BREP
            with open(brep_path, 'w') as f:
                f.write(str(model_dict['dimensions']))
            
            return stl_path, brep_path
            
        except Exception as e:
            print(f"Error generating 3D files: {str(e)}")
            raise e 