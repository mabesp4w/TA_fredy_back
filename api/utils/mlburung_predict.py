"""
MLBurung prediction integration for DjangoProject
This module provides a way to use the original MLBurung model in DjangoProject
"""

import os
import sys
import subprocess
import tempfile
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

def predict_with_mlburung_model(audio_file):
    """
    Use MLBurung model to predict bird species
    This function calls the MLBurung predict.py script directly
    """
    try:
        # Save uploaded file to temporary location
        temp_path = None
        if hasattr(audio_file, 'read'):
            # Django uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                for chunk in audio_file.chunks():
                    tmp_file.write(chunk)
                temp_path = tmp_file.name
        else:
            # File path
            temp_path = audio_file
        
        # MLBurung project path (relative to DjangoProject)
        mlburung_path = os.path.join(settings.BASE_DIR, "..", "MLBurung")
        
        # Command to run MLBurung prediction
        cmd = [
            "python", "predict.py", 
            temp_path, 
            "--cpu", 
            "--model", "model_20251007_063007/bird_sound_classifier.h5"
        ]
        
        logger.info(f"Running MLBurung prediction for: {temp_path}")
        
        # Run prediction
        result = subprocess.run(
            cmd, 
            cwd=mlburung_path,
            capture_output=True, 
            text=True, 
            timeout=60
        )
        
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path) and hasattr(audio_file, 'read'):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")
        
        if result.returncode == 0:
            # Parse the output to extract prediction results
            output_lines = result.stdout.strip().split('\n')
            
            # Find prediction results section
            predictions = []
            in_results_section = False
            
            for line in output_lines:
                if "Prediction Results:" in line:
                    in_results_section = True
                    continue
                
                if in_results_section and line.strip():
                    if line.startswith("-"):
                        continue
                    
                    # Parse prediction line: "1. Eclectus roratus: 0.699"
                    if ". " in line and ":" in line:
                        try:
                            parts = line.split(". ", 1)
                            if len(parts) == 2:
                                class_part = parts[1].split(": ")
                                if len(class_part) == 2:
                                    class_name = class_part[0].strip()
                                    confidence = float(class_part[1].strip())
                                    predictions.append({
                                        'class': class_name,
                                        'confidence': confidence
                                    })
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Failed to parse prediction line: {line}, error: {e}")
                            continue
            
            if predictions:
                # Return the top prediction
                top_prediction = predictions[0]
                logger.info(f"MLBurung prediction: {top_prediction['class']} (confidence: {top_prediction['confidence']:.3f})")
                
                return {
                    'scientific_nm': top_prediction['class'],
                    'confidence': top_prediction['confidence'],
                    'all_predictions': predictions,
                    'method': 'mlburung_original'
                }
            else:
                logger.error("No predictions found in MLBurung output")
                return {"error": "No predictions found in MLBurung output"}
        
        else:
            logger.error(f"MLBurung prediction failed: {result.stderr}")
            return {"error": f"MLBurung prediction failed: {result.stderr}"}
            
    except subprocess.TimeoutExpired:
        logger.error("MLBurung prediction timed out")
        return {"error": "MLBurung prediction timed out"}
    except Exception as e:
        logger.error(f"Error in MLBurung prediction: {e}")
        return {"error": f"Error in MLBurung prediction: {str(e)}"}

def predict_with_mlburung_model_from_path(file_path):
    """
    Use MLBurung model to predict bird species from file path
    """
    try:
        # MLBurung project path (relative to DjangoProject)
        mlburung_path = os.path.join(settings.BASE_DIR, "..", "MLBurung")
        
        # Command to run MLBurung prediction
        cmd = [
            "python", "predict.py", 
            file_path, 
            "--cpu", 
            "--model", "model_20251007_063007/bird_sound_classifier.h5"
        ]
        
        logger.info(f"Running MLBurung prediction for: {file_path}")
        
        # Run prediction
        result = subprocess.run(
            cmd, 
            cwd=mlburung_path,
            capture_output=True, 
            text=True, 
            timeout=60
        )
        
        if result.returncode == 0:
            # Parse the output to extract prediction results
            output_lines = result.stdout.strip().split('\n')
            
            # Find prediction results section
            predictions = []
            in_results_section = False
            
            for line in output_lines:
                if "Prediction Results:" in line:
                    in_results_section = True
                    continue
                
                if in_results_section and line.strip():
                    if line.startswith("-"):
                        continue
                    
                    # Parse prediction line: "1. Eclectus roratus: 0.699"
                    if ". " in line and ":" in line:
                        try:
                            parts = line.split(". ", 1)
                            if len(parts) == 2:
                                class_part = parts[1].split(": ")
                                if len(class_part) == 2:
                                    class_name = class_part[0].strip()
                                    confidence = float(class_part[1].strip())
                                    predictions.append({
                                        'class': class_name,
                                        'confidence': confidence
                                    })
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Failed to parse prediction line: {line}, error: {e}")
                            continue
            
            if predictions:
                logger.info(f"MLBurung prediction: {predictions[0]['class']} (confidence: {predictions[0]['confidence']:.3f})")
                return predictions
            else:
                logger.error("No predictions found in MLBurung output")
                return None
        
        else:
            logger.error(f"MLBurung prediction failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("MLBurung prediction timed out")
        return None
    except Exception as e:
        logger.error(f"Error in MLBurung prediction: {e}")
        return None
