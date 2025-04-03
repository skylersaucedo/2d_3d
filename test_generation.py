import asyncio
from claude_mcp_builder import ClaudeMCPBuilder
import os
from dotenv import load_dotenv

load_dotenv()

async def test_model_generation():
    # Initialize the builder
    builder = ClaudeMCPBuilder()
    
    # Use sample images
    side1_path = "sample_images/side1.jpg"
    side2_path = "sample_images/side2.jpg"
    side3_path = "sample_images/side3.jpg"
    
    try:
        # Generate the model
        stl_path, brep_path = await builder.generate_model(
            side1_path,
            side2_path,
            side3_path
        )
        
        print(f"Generated STL file: {stl_path}")
        print(f"Generated BREP file: {brep_path}")
        
    except Exception as e:
        print(f"Error during model generation: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_model_generation()) 