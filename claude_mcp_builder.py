import os
import time
import asyncio
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
        
    async def generate_model(self, side1_path: str, side2_path: str, side3_path: str) -> tuple[str, str]:
        """
        Generate a 3D model from three side images using Claude 3.7 MCP
        """
        # Read local image files
        with open(side1_path, 'rb') as f:
            side1 = f.read()
        with open(side2_path, 'rb') as f:
            side2 = f.read()
        with open(side3_path, 'rb') as f:
            side3 = f.read()

        # Create prompt for Claude
        prompt = self._create_prompt(side1, side2, side3)
        
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
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Approximate output tokens (could be more precise by actually counting)
        # Using a conservative estimate of 2 tokens per character
        output_tokens = len(str(response.content)) * 2
        await self.output_token_bucket.consume(int(output_tokens))

        # Generate 3D model using Open3D
        stl_path, brep_path = self._generate_3d_files(response.content)

        return stl_path, brep_path

    def _create_prompt(self, side1: bytes, side2: bytes, side3: bytes) -> str:
        """Create prompt for Claude based on the images"""
        return f"""
        Analyze these three side views of an object and generate a 3D model.
        Consider the following aspects:
        1. Dimensions and proportions
        2. Surface features and details
        3. Geometric relationships between sides
        
        Generate a detailed 3D model specification that can be used with Open3D.
        """

    def _generate_3d_files(self, claude_response: str) -> tuple[str, str]:
        """Generate STL and BREP files using Open3D"""
        # Create a simple mesh for demonstration
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        
        # Save as STL
        stl_path = os.path.join("output", "output.stl")
        o3d.io.write_triangle_mesh(stl_path, mesh)
        
        # Save as BREP (using a simple conversion for demonstration)
        brep_path = os.path.join("output", "output.brep")
        # Note: Actual BREP conversion would require additional libraries
        
        # Create an empty BREP file for demonstration
        with open(brep_path, 'w') as f:
            f.write("")
        
        return stl_path, brep_path 