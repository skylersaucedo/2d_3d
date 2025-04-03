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
        
    async def generate_model(self, side1_path: str, side2_path: str, side3_path: str) -> tuple[str, str]:
        """
        Generate a 3D model from three side images using Claude 3.7 MCP
        """
        # Read and encode images
        side1_b64 = self._encode_image(side1_path)
        side2_b64 = self._encode_image(side2_path)
        side3_b64 = self._encode_image(side3_path)

        # Create prompt for Claude
        prompt = self._create_prompt()
        
        # Calculate approximate input tokens
        total_input_tokens = self.BASE_PROMPT_TOKENS + (self.IMAGE_TOKENS * 3)
        
        # Wait if needed based on input token rate limit
        await self.input_token_bucket.consume(total_input_tokens)
        
        # Wait if needed based on request rate limit
        await self.request_bucket.consume(1)
        
        # Get Claude's response
        response = self.anthropic.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here are three side view images of an object. Please analyze them."
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
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
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
        Based on these three side view images, please:

        1. First, describe the key features you observe from each view:
        - Exact measurements and proportions
        - Surface characteristics
        - Any special features or details

        2. Then, generate OpenSCAD code that will recreate this object precisely. The code should:
        - Use exact measurements based on your analysis
        - Properly implement all geometric features you observed
        - Use OpenSCAD's CSG operations (union, difference, intersection) as needed
        - Include detailed comments explaining each operation

        3. Finally, provide the code in this format (and ONLY this format, no other text):
        {
            'openscad_code': 'your complete OpenSCAD code here',
            'dimensions': {
                'width': measured_width,
                'height': measured_height,
                'depth': measured_depth
            }
        }

        Be precise and thorough in your analysis. The goal is to create an exact replica of the object shown in the images.
        """

    def _extract_dict_from_response(self, response_text: str) -> dict:
        """Extract the dictionary from Claude's response"""
        try:
            # Find the last occurrence of openscad_code and dimensions
            # This helps avoid matching any earlier mentions in the analysis
            code_start = response_text.rfind("'openscad_code':")
            if code_start == -1:
                raise ValueError("No OpenSCAD code found in response")
            
            dims_start = response_text.rfind("'dimensions':")
            if dims_start == -1:
                raise ValueError("No dimensions found in response")
            
            # Extract OpenSCAD code
            code_text = response_text[code_start:dims_start].strip()
            code_text = code_text.replace("'openscad_code':", "").strip()
            
            # Clean up the code text
            if code_text.startswith("'"):
                code_text = code_text[1:]
            if code_text.endswith(","):
                code_text = code_text[:-1]
            if code_text.endswith("'"):
                code_text = code_text[:-1]
            
            # Clean up the OpenSCAD code
            code_text = code_text.replace('\\n', '\n')
            code_text = code_text.replace('\\\\', '\\')
            
            # Extract dimensions section
            dims_text = response_text[dims_start:]
            # Find the closing brace of the entire dictionary
            brace_count = 0
            dims_end = -1
            
            for i, char in enumerate(dims_text):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        dims_end = i + 1
                        break
            
            if dims_end == -1:
                raise ValueError("Could not find end of dimensions section")
                
            dims_text = dims_text[:dims_end]
            
            # Extract just the dimensions dictionary part
            dims_dict_start = dims_text.find('{', dims_text.find("'dimensions':"))
            if dims_dict_start == -1:
                raise ValueError("Could not find dimensions dictionary")
            
            dims_text = dims_text[dims_dict_start:].strip()
            
            # Remove outer braces
            dims_text = dims_text.strip('{}')
            
            # Parse dimensions
            dimensions = {}
            for pair in dims_text.split(','):
                if ':' not in pair:
                    continue
                key, value = pair.split(':')
                key = key.strip().strip("'").strip('"')
                # Clean up the value and convert to int
                value = value.strip().strip("'").strip('"')
                try:
                    dimensions[key] = int(value)
                except ValueError:
                    print(f"Warning: Could not parse dimension value: {value}")
                    continue
            
            # Create the final dictionary
            result = {
                'openscad_code': code_text,
                'dimensions': dimensions
            }
            
            # Print debug info
            print("\nExtracted OpenSCAD code:")
            print(code_text)
            print("\nExtracted dimensions:")
            print(dimensions)
            
            # Validate required keys
            required_keys = ['openscad_code', 'dimensions']
            if not all(key in result for key in required_keys):
                raise ValueError(f"Missing required keys. Found: {list(result.keys())}")
            
            # Validate dimensions
            required_dims = ['width', 'height', 'depth']
            if not all(key in result['dimensions'] for key in required_dims):
                raise ValueError(f"Missing required dimensions. Found: {list(result['dimensions'].keys())}")
            
            return result
            
        except Exception as e:
            print(f"Error extracting dictionary: {str(e)}")
            print(f"Response text: {response_text}")
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
            openscad_code = model_dict['openscad_code'].strip() + '\n'
            
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
            
            # Find OpenSCAD executable
            openscad_exe = self._find_openscad_path()
            
            # Use subprocess.run instead of os.system for better path handling
            import subprocess
            try:
                # Run OpenSCAD with properly quoted paths
                result = subprocess.run([
                    openscad_exe,
                    "-o", stl_path,
                    scad_path
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print("OpenSCAD Error Output:")
                    print(result.stderr)
                    raise ValueError(f"OpenSCAD command failed with exit code {result.returncode}")
                
            except Exception as e:
                print(f"Error running OpenSCAD: {str(e)}")
                raise ValueError("Failed to run OpenSCAD command. Please ensure OpenSCAD is properly installed.")
            
            # Save dimensions as BREP
            with open(brep_path, 'w') as f:
                f.write(str(model_dict['dimensions']))
            
            return stl_path, brep_path
            
        except Exception as e:
            print(f"Error generating 3D files: {str(e)}")
            raise e 