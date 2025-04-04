import os
import time
import asyncio
import base64
from anthropic import Anthropic
import open3d as o3d
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

class ClaudeMCPBuilder:
    def __init__(self):
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Initialize token buckets for rate limiting
        self.input_token_bucket = TokenBucket(20000)  # 20k input tokens/minute
        self.output_token_bucket = TokenBucket(8000)  # 8k output tokens/minute
        self.request_bucket = TokenBucket(50)  # 50 requests/minute
        
        # Approximate token counts for different content types
        self.BASE_PROMPT_TOKENS = 100  # Base prompt template tokens
        self.IMAGE_TOKENS = 1500  # Approximate tokens per image
        
        # Create output directories if they don't exist
        os.makedirs("output", exist_ok=True)
        
    def _encode_image(self, image_path: str) -> str:
        """Encode image file to base64 string"""
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            return base64.b64encode(image_bytes).decode('utf-8')
        
    async def generate_model(self, side1_path: str, side2_path: str, side3_path: str, side4_path: str = None) -> tuple[str, str]:
        """
        Generate a 3D model from three or four side images using Claude 3.7 MCP
        """
        # Read and encode images
        side1_b64 = self._encode_image(side1_path)
        side2_b64 = self._encode_image(side2_path)
        side3_b64 = self._encode_image(side3_path)
        side4_b64 = self._encode_image(side4_path) if side4_path else None

        # Create prompt for Claude
        prompt = self._create_prompt()
        
        # Calculate approximate input tokens
        total_input_tokens = self.BASE_PROMPT_TOKENS + (self.IMAGE_TOKENS * (4 if side4_path else 3))
        
        # Wait if needed based on input token rate limit
        await self.input_token_bucket.consume(total_input_tokens)
        
        # Wait if needed based on request rate limit
        await self.request_bucket.consume(1)

        PROMPT = "Here are multiple side view images of an object. Please analyze them carefully to understand all features and dimensions."
        
        # Prepare message content
        message_content = [
            {
                "type": "text",
                "text": PROMPT
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": side1_b64
                }
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": side2_b64
                }
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": side3_b64
                }
            }
        ]

        # Add fourth image if provided
        if side4_b64:
            message_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": side4_b64
                }
            })

        # Add the prompt
        message_content.append({
            "type": "text",
            "text": prompt
        })

        # Get Claude's response
        response = self.anthropic.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": message_content
            }]
        )
        
        # Debug: Print response type and content
        print(f"Response type: {type(response.content)}")
        print(f"Response content: {response.content}")
        
        # Extract the text content from response
        if isinstance(response.content, list):
            response_text = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
        else:
            response_text = response.content.text if hasattr(response.content, 'text') else str(response.content)
        
        # Debug: Print processed response
        print(f"Processed response: {response_text}")
        
        # Approximate output tokens
        output_tokens = len(response_text) * 2
        await self.output_token_bucket.consume(int(output_tokens))

        # Generate 3D model using OpenSCAD
        stl_path, brep_path = self._generate_3d_files(response_text)

        return stl_path, brep_path

    def _create_prompt(self) -> str:
        """Create prompt for Claude based on the images"""
        return """
        You are a CAD expert. Analyze these technical drawings and create a precise 3D model based on the provided orthographic views. Your task is to generate complete, executable OpenSCAD code.

        1. DIMENSIONAL ANALYSIS:
           - Extract all explicit dimensions from the drawings
           - Calculate any implicit dimensions based on scale and relationships
           - Define clear variable names for all dimensions
           - Document units (assume mm if not specified)

        2. FEATURE ANALYSIS:
           - Identify the base shape and primary features
           - Note all secondary features (holes, cuts, threads, etc.)
           - Document feature relationships and positions
           - Identify any patterns or symmetry

        3. OPENSCAD IMPLEMENTATION:
           - Start with clear variable definitions for ALL dimensions
           - Create separate modules for complex features
           - Use proper boolean operations (union, difference, intersection)
           - ALWAYS include a final module call to render the object
           - Use high resolution ($fn) for curved surfaces
           - Add clear comments explaining each major step

        After analysis, respond with ONLY a dictionary in this EXACT format:

        {
            "openscad_code": "// Your OpenSCAD code with proper newline escaping (\\n)",
            "dimensions": {
                "head_diameter": 10,
                "head_height": 5,
                "shaft_length": 20,
                "shaft_diameter": 8,
                "thread_pitch": 1.25,
                "head_width": 12
            }
        }

        CRITICAL REQUIREMENTS:
        1. Use DOUBLE QUOTES for all strings
        2. Escape newlines with \\n in OpenSCAD code
        3. Include ONLY the dictionary in your response
        4. Set $fn=64 or higher for curved surfaces
        5. Use clear variable names prefixed with their category
        6. Include comprehensive comments in the code
        7. ALWAYS end the code with a module call to render the object
        8. Ensure all dimensions are defined as variables at the start
        9. Use proper scoping and modular design
        10. All dimensions must be numeric values"""

    def _extract_dict_from_response(self, response_text: str) -> dict:
        """Extract the dictionary from Claude's response"""
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

            if 'openscad_code' not in result:
                raise ValueError("Missing OpenSCAD code in response")

            # Handle specialized dimensions
            if 'dimensions' in result:
                dims = result['dimensions']
                # Map specialized dimensions to required format
                if all(key in dims for key in ['head_diameter', 'shaft_length', 'head_height']):
                    # Use the largest dimension for width/depth
                    max_diameter = max(float(dims.get('head_diameter', 0)), float(dims.get('shaft_diameter', 0)))
                    result['dimensions'] = {
                        'width': int(max_diameter),
                        'depth': int(max_diameter),
                        'height': int(float(dims.get('shaft_length', 0)) + float(dims.get('head_height', 0)))
                    }
                else:
                    print(f"Found dimensions: {list(dims.keys())}")
                    raise ValueError("Missing required dimensions")

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

    def _generate_3d_files(self, claude_response: str) -> tuple[str, str]:
        """Generate STL and BREP files using OpenSCAD"""
        try:
            # Extract the dictionary containing OpenSCAD code and dimensions
            model_dict = self._extract_dict_from_response(claude_response)
            
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