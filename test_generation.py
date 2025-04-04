import asyncio
from gemini_builder import GeminiBuilder
import os
from dotenv import load_dotenv

load_dotenv()

async def test_model_generation():
    try:
        # Initialize the builder
        builder = GeminiBuilder()
        
        # Use v2 schematic images
        side1_path = "sample_images/side1v2.jpg"
        side2_path = "sample_images/side2v2.jpg"
        side3_path = "sample_images/side3v2.jpg"
        side4_path = "sample_images/side4v2.jpg"
        
        print("Starting model generation with gemini-2.0-flash.")
        print(f"Processing images: {side1_path}, {side2_path}, {side3_path}, {side4_path}")
        
        # Generate the model
        stl_path, brep_path = await builder.generate_model(
            side1_path,
            side2_path,
            side3_path,
            side4_path
        )
        
        print(f"Successfully generated STL file: {stl_path}")
        print(f"Successfully generated BREP file: {brep_path}")
        
    except Exception as e:
        print(f"Error during model generation: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Caused by: {str(e.__cause__)}")

if __name__ == "__main__":
    asyncio.run(test_model_generation()) 