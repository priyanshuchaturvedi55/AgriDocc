#!/usr/bin/env python3
"""
Test script to verify Gemini AI is working correctly
"""
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import analyze_with_gemini
from PIL import Image

# Test with a sample image
test_image = '../test_images/Apple_scab.JPG'

if os.path.exists(test_image):
    print(f"Testing Gemini AI with image: {test_image}")
    print("=" * 60)
    
    result = analyze_with_gemini(test_image)
    
    if result:
        print("\n✓ Gemini AI Analysis Successful!")
        print("=" * 60)
        print(f"Disease Name: {result['disease_name']}")
        print(f"Plant Type: {result['plant_type']}")
        print(f"Severity: {result['severity']}")
        print(f"Confidence: {result['confidence']}")
        print(f"\nDescription:\n{result['description'][:300]}...")
        print(f"\nTreatment:\n{result['treatment'][:300]}...")
        print("=" * 60)
    else:
        print("\n✗ Gemini AI Analysis Failed!")
        print("Check the error messages above.")
else:
    print(f"Test image not found: {test_image}")

