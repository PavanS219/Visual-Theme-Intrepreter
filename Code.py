import json
import base64
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import KMeans
import cv2
from collections import Counter
from typing import Dict, List, Tuple, Any
import colorsys
import logging
from pathlib import Path

# LangChain imports
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema import BaseMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for structured output
class ColorAnalysis(BaseModel):
    """Model for color analysis output"""
    dominant_emotion: str = Field(description="Primary emotion evoked by colors")
    brand_personality: List[str] = Field(description="Brand personality traits suggested by colors")
    user_behavior_impact: str = Field(description="How colors might affect user behavior")
    color_harmony: str = Field(description="Type of color harmony present")

class DesignPatterns(BaseModel):
    """Model for design pattern analysis"""
    typography_style: str = Field(description="Typography style category")
    layout_pattern: str = Field(description="Primary layout pattern")
    visual_hierarchy: str = Field(description="Visual hierarchy approach")
    interaction_style: str = Field(description="Suggested interaction style")

class MoodAtmosphere(BaseModel):
    """Model for mood and atmosphere analysis"""
    energy_level: str = Field(description="Energy level of the design")
    formality: str = Field(description="Formality level")
    time_period: str = Field(description="Associated time period or era")
    target_audience: str = Field(description="Primary target audience")

class ThemeAnalysis(BaseModel):
    """Complete theme analysis model"""
    color_analysis: ColorAnalysis
    design_patterns: DesignPatterns
    mood_atmosphere: MoodAtmosphere
    key_recommendations: List[str] = Field(description="Top 5 actionable recommendations")
    confidence_score: float = Field(description="Analysis confidence score (0-1)")

class VisualThemeInterpreter:
    """
    A comprehensive system for analyzing images and generating theme configurations using LangChain and Mistral
    """
    
    def __init__(self, mistral_api_key: str = None, model_name: str = "pixtral-12b-2409"):
        self.mistral_api_key = mistral_api_key
        self.model_name = model_name
        
        # Initialize LangChain components
        if self.mistral_api_key:
            self.llm = ChatMistralAI(
                model=model_name,
                mistral_api_key=mistral_api_key,
                temperature=0.3,
                max_tokens=1500
            )
            self._setup_chains()
        else:
            logger.warning("Mistral API key not provided, AI analysis will be unavailable")
            self.llm = None
    
    def _setup_chains(self):
        """Setup LangChain chains for different analysis types"""
        
        # Color Psychology Chain
        color_system_template = SystemMessagePromptTemplate.from_template(
            """You are an expert in color psychology and brand design. Analyze the dominant colors 
            in the provided image and provide insights about their psychological impact.
            
            Focus on:
            - Emotional responses these colors typically evoke
            - Brand personality traits they suggest
            - How they might influence user behavior in digital interfaces
            - The type of color harmony present
            
            Provide your analysis in a structured JSON format."""
        )
        
        color_human_template = HumanMessagePromptTemplate.from_template(
            """Analyze the colors in this image for the theme: "{theme_prompt}"
            
            Image data: {image_data}"""
        )
        
        self.color_prompt = ChatPromptTemplate.from_messages([
            color_system_template,
            color_human_template
        ])
        
        # Design Patterns Chain
        design_system_template = SystemMessagePromptTemplate.from_template(
            """You are a UX/UI design expert. Analyze the visual design patterns and elements 
            in the provided image.
            
            Focus on:
            - Typography style and character
            - Layout patterns and grid systems
            - Visual hierarchy principles
            - Suggested interaction patterns and affordances
            
            Provide specific, actionable insights in JSON format."""
        )
        
        design_human_template = HumanMessagePromptTemplate.from_template(
            """Analyze the design patterns in this image for the theme: "{theme_prompt}"
            
            Image data: {image_data}"""
        )
        
        self.design_prompt = ChatPromptTemplate.from_messages([
            design_system_template,
            design_human_template
        ])
        
        # Mood and Atmosphere Chain
        mood_system_template = SystemMessagePromptTemplate.from_template(
            """You are a brand strategist specializing in visual communication. Analyze the overall 
            mood, atmosphere, and brand implications of the provided image.
            
            Focus on:
            - Energy level and emotional tone
            - Formality and professionalism level
            - Time period associations or style era
            - Target audience implications
            
            Provide insights in structured JSON format."""
        )
        
        mood_human_template = HumanMessagePromptTemplate.from_template(
            """Analyze the mood and atmosphere of this image for the theme: "{theme_prompt}"
            
            Image data: {image_data}"""
        )
        
        self.mood_prompt = ChatPromptTemplate.from_messages([
            mood_system_template,
            mood_human_template
        ])
        
        # Comprehensive Analysis Chain
        comprehensive_system_template = SystemMessagePromptTemplate.from_template(
            """You are a senior visual design consultant. Provide a comprehensive analysis of this image 
            that combines color psychology, design patterns, and mood assessment.
            
            Based on your analysis, provide:
            1. Complete structured analysis covering all aspects
            2. Top 5 actionable recommendations for implementing this theme
            3. A confidence score for your analysis
            
            Consider the specific theme context: "{theme_prompt}"
            
            Output your analysis in the specified JSON structure."""
        )
        
        comprehensive_human_template = HumanMessagePromptTemplate.from_template(
            """Provide a comprehensive theme analysis for: "{theme_prompt}"
            
            Image data: {image_data}"""
        )
        
        self.comprehensive_prompt = ChatPromptTemplate.from_messages([
            comprehensive_system_template,
            comprehensive_human_template
        ])
        
        # Setup output parsers
        self.json_parser = JsonOutputParser(pydantic_object=ThemeAnalysis)
        self.str_parser = StrOutputParser()
        
        # Create chains
        self.color_chain = self.color_prompt | self.llm | self.str_parser
        self.design_chain = self.design_prompt | self.llm | self.str_parser
        self.mood_chain = self.mood_prompt | self.llm | self.str_parser
        self.comprehensive_chain = self.comprehensive_prompt | self.llm | self.str_parser
    
    def extract_color_palette(self, image_path: str, n_colors: int = 8) -> List[Dict]:
        """
        Extract dominant colors using K-means clustering
        """
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Reshape image to be a list of pixels
            pixels = image.reshape((-1, 3))
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get colors and their frequencies
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            # Calculate color frequencies
            color_counts = Counter(labels)
            total_pixels = len(labels)
            
            palette = []
            for i, color in enumerate(colors):
                frequency = color_counts[i] / total_pixels
                hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
                
                # Calculate color properties
                hsl = colorsys.rgb_to_hls(color[0]/255, color[1]/255, color[2]/255)
                
                palette.append({
                    "hex": hex_color,
                    "rgb": color.tolist(),
                    "hsl": [round(hsl[0]*360), round(hsl[2]*100), round(hsl[1]*100)],
                    "frequency": round(frequency, 3),
                    "role": self._determine_color_role(color, frequency)
                })
            
            # Sort by frequency
            palette.sort(key=lambda x: x["frequency"], reverse=True)
            return palette
            
        except Exception as e:
            logger.error(f"Error extracting color palette: {e}")
            return []
    
    def _determine_color_role(self, color: np.ndarray, frequency: float) -> str:
        """Determine the likely role of a color in the design"""
        r, g, b = color
        brightness = (r + g + b) / 3
        
        if frequency > 0.3:
            return "primary" if brightness < 128 else "background"
        elif frequency > 0.15:
            return "secondary"
        elif brightness > 200:
            return "highlight"
        elif brightness < 50:
            return "text"
        else:
            return "accent"
    
    def analyze_composition(self, image_path: str) -> Dict:
        """Analyze image composition and layout patterns"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            height, width = image.shape
            
            # Edge detection for structure analysis
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Detect lines for grid analysis
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            # Calculate composition metrics
            composition = {
                "aspect_ratio": round(width / height, 2),
                "complexity": self._calculate_complexity(edges),
                "symmetry": self._calculate_symmetry(image),
                "grid_detected": lines is not None and len(lines) > 10,
                "dominant_direction": self._get_dominant_direction(edges),
                "visual_weight_distribution": self._analyze_visual_weight(image)
            }
            
            return composition
            
        except Exception as e:
            logger.error(f"Error analyzing composition: {e}")
            return {}
    
    def _calculate_complexity(self, edges: np.ndarray) -> str:
        """Calculate visual complexity based on edge density"""
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density < 0.05:
            return "minimal"
        elif edge_density < 0.15:
            return "moderate"
        else:
            return "complex"
    
    def _calculate_symmetry(self, image: np.ndarray) -> Dict:
        """Analyze symmetry in the image"""
        height, width = image.shape
        
        # Vertical symmetry
        left_half = image[:, :width//2]
        right_half = np.fliplr(image[:, width//2:])
        min_width = min(left_half.shape[1], right_half.shape[1])
        
        vertical_diff = np.mean(np.abs(left_half[:, :min_width] - right_half[:, :min_width]))
        
        # Horizontal symmetry
        top_half = image[:height//2, :]
        bottom_half = np.flipud(image[height//2:, :])
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        
        horizontal_diff = np.mean(np.abs(top_half[:min_height, :] - bottom_half[:min_height, :]))
        
        return {
            "vertical": "high" if vertical_diff < 30 else "low",
            "horizontal": "high" if horizontal_diff < 30 else "low"
        }
    
    def _get_dominant_direction(self, edges: np.ndarray) -> str:
        """Determine dominant directional flow"""
        # Sobel operators for gradient direction
        grad_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
        
        horizontal_strength = np.sum(np.abs(grad_x))
        vertical_strength = np.sum(np.abs(grad_y))
        
        if horizontal_strength > vertical_strength * 1.2:
            return "horizontal"
        elif vertical_strength > horizontal_strength * 1.2:
            return "vertical"
        else:
            return "balanced"
    
    def _analyze_visual_weight(self, image: np.ndarray) -> Dict:
        """Analyze distribution of visual weight"""
        height, width = image.shape
        
        # Divide image into quadrants
        top_left = np.mean(image[:height//2, :width//2])
        top_right = np.mean(image[:height//2, width//2:])
        bottom_left = np.mean(image[height//2:, :width//2])
        bottom_right = np.mean(image[height//2:, width//2:])
        
        weights = [top_left, top_right, bottom_left, bottom_right]
        max_weight_idx = np.argmax(weights)
        
        quadrants = ["top_left", "top_right", "bottom_left", "bottom_right"]
        
        return {
            "primary_focus": quadrants[max_weight_idx],
            "balance": "centered" if max(weights) - min(weights) < 50 else "asymmetrical"
        }
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API consumption"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return ""
    
    def analyze_with_langchain(self, image_path: str, theme_prompt: str, analysis_type: str = "comprehensive") -> Dict:
        """
        Use LangChain with Mistral to analyze the image
        """
        if not self.llm:
            logger.warning("LangChain/Mistral not available, using fallback analysis")
            return self._rule_based_interpretation(theme_prompt)
        
        try:
            # Encode image
            encoded_image = self._encode_image(image_path)
            if not encoded_image:
                return self._rule_based_interpretation(theme_prompt)
            
            # Create image data string for the model
            image_data = f"data:image/jpeg;base64,{encoded_image}"
            
            # Choose analysis chain based on type
            if analysis_type == "color":
                result = self.color_chain.invoke({
                    "theme_prompt": theme_prompt,
                    "image_data": image_data
                })
            elif analysis_type == "design":
                result = self.design_chain.invoke({
                    "theme_prompt": theme_prompt,
                    "image_data": image_data
                })
            elif analysis_type == "mood":
                result = self.mood_chain.invoke({
                    "theme_prompt": theme_prompt,
                    "image_data": image_data
                })
            else:  # comprehensive
                result = self.comprehensive_chain.invoke({
                    "theme_prompt": theme_prompt,
                    "image_data": image_data
                })
            
            # Try to parse as JSON, fallback to string analysis
            try:
                parsed_result = json.loads(result)
                return {
                    "analysis_type": analysis_type,
                    "result": parsed_result,
                    "confidence": parsed_result.get("confidence_score", 0.8),
                    "raw_response": result
                }
            except json.JSONDecodeError:
                return {
                    "analysis_type": analysis_type,
                    "result": {"analysis": result},
                    "confidence": 0.7,
                    "raw_response": result
                }
                
        except Exception as e:
            logger.error(f"Error with LangChain analysis: {e}")
            return self._rule_based_interpretation(theme_prompt)
    
    def run_multi_perspective_analysis(self, image_path: str, theme_prompt: str) -> Dict:
        """
        Run multiple analysis perspectives and combine results
        """
        if not self.llm:
            return self._rule_based_interpretation(theme_prompt)
        
        analyses = {}
        
        # Run different types of analysis
        for analysis_type in ["color", "design", "mood"]:
            try:
                result = self.analyze_with_langchain(image_path, theme_prompt, analysis_type)
                analyses[analysis_type] = result
            except Exception as e:
                logger.error(f"Error in {analysis_type} analysis: {e}")
                analyses[analysis_type] = {"error": str(e)}
        
        # Run comprehensive analysis
        try:
            comprehensive = self.analyze_with_langchain(image_path, theme_prompt, "comprehensive")
            analyses["comprehensive"] = comprehensive
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            analyses["comprehensive"] = {"error": str(e)}
        
        return analyses
    
    def _rule_based_interpretation(self, theme_prompt: str) -> Dict:
        """Fallback rule-based interpretation when AI is not available"""
        theme_lower = theme_prompt.lower()
        
        interpretations = {
            "style": "modern" if any(word in theme_lower for word in ["modern", "contemporary", "sleek"]) else "classic",
            "energy": "high" if any(word in theme_lower for word in ["energetic", "vibrant", "dynamic"]) else "calm",
            "target_audience": "tech-savvy" if any(word in theme_lower for word in ["tech", "digital", "futuristic"]) else "general",
            "formality": "professional" if any(word in theme_lower for word in ["corporate", "business", "professional"]) else "casual",
            "analysis_type": "rule_based",
            "confidence": 0.5
        }
        
        return interpretations
    
    def generate_theme_config(self, image_path: str, theme_prompt: str, use_multi_perspective: bool = True) -> Dict:
        """
        Main method to generate complete theme configuration using LangChain
        """
        logger.info(f"Analyzing theme for: {theme_prompt}")
        
        # Extract visual data
        color_palette = self.extract_color_palette(image_path)
        composition = self.analyze_composition(image_path)
        
        # Get AI analysis
        if use_multi_perspective and self.llm:
            ai_analyses = self.run_multi_perspective_analysis(image_path, theme_prompt)
            primary_analysis = ai_analyses.get("comprehensive", {}).get("result", {})
        else:
            ai_analysis = self.analyze_with_langchain(image_path, theme_prompt, "comprehensive")
            primary_analysis = ai_analysis.get("result", {})
            ai_analyses = {"comprehensive": ai_analysis}
        
        # Generate comprehensive config
        theme_config = {
            "metadata": {
                "theme_name": theme_prompt,
                "generated_at": "2025-06-21",
                "version": "2.0",
                "analysis_method": "langchain_mistral",
                "analysis_confidence": primary_analysis.get("confidence_score", 0.7),
                "multi_perspective": use_multi_perspective
            },
            
            "ai_analyses": ai_analyses if use_multi_perspective else {"comprehensive": primary_analysis},
            
            "colors": {
                "palette": color_palette,
                "primary": color_palette[0]["hex"] if color_palette else "#000000",
                "secondary": color_palette[1]["hex"] if len(color_palette) > 1 else "#ffffff",
                "accent": next((c["hex"] for c in color_palette if c["role"] == "accent"), 
                             color_palette[2]["hex"] if len(color_palette) > 2 else "#808080"),
                "background": next((c["hex"] for c in color_palette if c["role"] == "background"), "#ffffff"),
                "text": next((c["hex"] for c in color_palette if c["role"] == "text"), "#000000")
            },
            
            "typography": self._generate_typography_config(primary_analysis, theme_prompt),
            
            "layout": {
                "grid_system": "12-column" if composition.get("grid_detected") else "flexbox",
                "spacing_scale": self._generate_spacing_scale(composition),
                "border_radius": self._determine_border_radius(theme_prompt, primary_analysis),
                "aspect_ratio": composition.get("aspect_ratio", 1.0)
            },
            
            "components": self._generate_component_styles(color_palette, primary_analysis),
            
            "interactions": self._generate_interaction_config(theme_prompt, primary_analysis),
            
            "brand_personality": {
                "primary_traits": self._extract_brand_traits(primary_analysis, theme_prompt),
                "mood": primary_analysis.get("mood_atmosphere", {}).get("energy_level", "balanced"),
                "target_audience": primary_analysis.get("mood_atmosphere", {}).get("target_audience", "general"),
                "style_direction": primary_analysis.get("design_patterns", {}).get("typography_style", "modern")
            },
            
            "recommendations": primary_analysis.get("key_recommendations", []),
            
            "technical_specs": {
                "responsive_breakpoints": {
                    "mobile": "320px",
                    "tablet": "768px", 
                    "desktop": "1024px",
                    "wide": "1440px"
                },
                "animation_preferences": self._determine_animation_style(theme_prompt, primary_analysis),
                "accessibility": {
                    "contrast_ratio": self._calculate_contrast_ratio(color_palette),
                    "recommended_improvements": []
                }
            }
        }
        
        return theme_config
    
    def _generate_typography_config(self, ai_analysis: Dict, theme_prompt: str) -> Dict:
        """Generate typography configuration based on AI analysis"""
        typography_style = ai_analysis.get("design_patterns", {}).get("typography_style", "clean")
        theme_lower = theme_prompt.lower()
        
        # Map AI insights to font choices
        if "serif" in typography_style.lower() or "elegant" in typography_style.lower():
            font_stack = ["Playfair Display", "Georgia", "serif"]
            style = "serif"
        elif "geometric" in typography_style.lower() or "tech" in theme_lower:
            font_stack = ["Roboto", "Arial", "sans-serif"]
            style = "geometric"
        else:
            font_stack = ["Inter", "Helvetica", "sans-serif"]
            style = "clean"
        
        return {
            "primary_font": font_stack,
            "heading_font": font_stack,
            "body_font": font_stack,
            "style_category": style,
            "ai_recommendation": typography_style,
            "scale": {
                "h1": "2.5rem",
                "h2": "2rem", 
                "h3": "1.5rem",
                "body": "1rem",
                "small": "0.875rem"
            },
            "weights": ["400", "600", "700"]
        }
    
    def _generate_spacing_scale(self, composition: Dict) -> Dict:
        """Generate spacing scale based on composition analysis"""
        complexity = composition.get("complexity", "moderate")
        
        if complexity == "minimal":
            base_unit = 8
        elif complexity == "complex":
            base_unit = 4
        else:
            base_unit = 6
        
        return {
            "base_unit": f"{base_unit}px",
            "scale": {
                "xs": f"{base_unit}px",
                "sm": f"{base_unit * 2}px",
                "md": f"{base_unit * 3}px", 
                "lg": f"{base_unit * 4}px",
                "xl": f"{base_unit * 6}px",
                "xxl": f"{base_unit * 8}px"
            }
        }
    
    def _determine_border_radius(self, theme_prompt: str, ai_analysis: Dict) -> str:
        """Determine appropriate border radius based on theme and AI analysis"""
        layout_pattern = ai_analysis.get("design_patterns", {}).get("layout_pattern", "")
        theme_lower = theme_prompt.lower()
        
        if "sharp" in theme_lower or "geometric" in layout_pattern.lower():
            return "0px"
        elif "soft" in theme_lower or "organic" in layout_pattern.lower():
            return "12px"
        else:
            return "6px"
    
    def _generate_component_styles(self, color_palette: List[Dict], ai_analysis: Dict) -> Dict:
        """Generate component-specific styling based on AI insights"""
        primary_color = color_palette[0]["hex"] if color_palette else "#007bff"
        interaction_style = ai_analysis.get("design_patterns", {}).get("interaction_style", "standard")
        
        # Adjust button styles based on AI recommendations
        if "playful" in interaction_style.lower():
            button_radius = "12px"
            button_padding = "14px 28px"
        elif "minimal" in interaction_style.lower():
            button_radius = "2px"
            button_padding = "10px 20px"
        else:
            button_radius = "6px"
            button_padding = "12px 24px"
        
        return {
            "buttons": {
                "primary": {
                    "background": primary_color,
                    "color": "#ffffff",
                    "border": "none",
                    "padding": button_padding,
                    "border_radius": button_radius
                },
                "secondary": {
                    "background": "transparent",
                    "color": primary_color,
                    "border": f"2px solid {primary_color}",
                    "padding": button_padding,
                    "border_radius": button_radius
                }
            },
            "cards": {
                "background": "#ffffff",
                "border": "1px solid #e0e0e0",
                "border_radius": "8px",
                "shadow": "0 2px 8px rgba(0,0,0,0.1)"
            },
            "forms": {
                "input_background": "#ffffff",
                "input_border": "#d0d0d0", 
                "input_focus_border": primary_color,
                "label_color": "#333333"
            },
            "ai_insights": {
                "interaction_style": interaction_style,
                "recommended_adjustments": ai_analysis.get("key_recommendations", [])
            }
        }
    
    def _generate_interaction_config(self, theme_prompt: str, ai_analysis: Dict) -> Dict:
        """Generate interaction and animation preferences"""
        energy_level = ai_analysis.get("mood_atmosphere", {}).get("energy_level", "medium")
        theme_lower = theme_prompt.lower()
        
        if energy_level == "high" or "energetic" in theme_lower:
            animation_style = "dynamic"
            duration = "0.2s"
        elif energy_level == "low" or "professional" in theme_lower:
            animation_style = "subtle"
            duration = "0.4s"
        else:
            animation_style = "smooth"
            duration = "0.3s"
        
        return {
            "hover_effects": True,
            "transition_duration": duration,
            "animation_style": animation_style,
            "micro_interactions": energy_level == "high",
            "ai_insights": {
                "energy_level": energy_level,
                "recommended_style": animation_style
            }
        }
    
    def _extract_brand_traits(self, ai_analysis: Dict, theme_prompt: str) -> List[str]:
        """Extract brand personality traits from AI analysis"""
        # Try to get traits from AI analysis first
        color_analysis = ai_analysis.get("color_analysis", {})
        brand_traits = color_analysis.get("brand_personality", [])
        
        if brand_traits and isinstance(brand_traits, list):
            return brand_traits[:3]
        
        # Fallback to rule-based extraction
        traits = []
        theme_lower = theme_prompt.lower()
        
        trait_keywords = {
            "innovative": ["futuristic", "tech", "modern", "cutting-edge"],
            "trustworthy": ["professional", "reliable", "corporate"],
            "creative": ["artistic", "creative", "unique", "expressive"],
            "friendly": ["warm", "approachable", "casual", "welcoming"],
            "sophisticated": ["elegant", "luxury", "premium", "refined"]
        }
        
        for trait, keywords in trait_keywords.items():
            if any(keyword in theme_lower for keyword in keywords):
                traits.append(trait)
        
        return traits[:3]
    
    def _determine_animation_style(self, theme_prompt: str, ai_analysis: Dict) -> Dict:
        """Determine animation preferences from AI analysis"""
        energy_level = ai_analysis.get("mood_atmosphere", {}).get("energy_level", "medium")
        formality = ai_analysis.get("mood_atmosphere", {}).get("formality", "casual")
        
        if energy_level == "high":
            preference = "dynamic"
            duration = "fast"
        elif formality == "professional":
            preference = "subtle"
            duration = "slow"
        else:
            preference = "smooth"
            duration = "medium"
        
        return {
            "preference": preference,
            "duration": duration,
            "easing": "ease-in-out",
            "ai_insights": {
                "energy_level": energy_level,
                "formality": formality
            }
        }
    
    def _calculate_contrast_ratio(self, color_palette: List[Dict]) -> str:
        """Calculate and assess contrast ratio"""
        if len(color_palette) < 2:
            return "unknown"
        
        # Simplified contrast calculation
        primary = color_palette[0]["rgb"]
        secondary = color_palette[1]["rgb"]
        
        # Calculate relative luminance (simplified)
        l1 = (primary[0] * 0.299 + primary[1] * 0.587 + primary[2] * 0.114) / 255
        l2 = (secondary[0] * 0.299 + secondary[1] * 0.587 + secondary[2] * 0.114) / 255
        
        contrast = (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)
        
        if contrast >= 7:
            return "AAA"
        elif contrast >= 4.5:
            return "AA"
        else:
            return "poor"
    
    def save_config(self, config: Dict, output_path: str):
        """Save the generated configuration to a JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def export_config_formats(self, config: Dict, base_filename: str):
        """Export configuration in multiple formats"""
        try:
            # JSON format
            with open(f"{base_filename}.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            # CSS Variables format
            self._export_css_variables(config, f"{base_filename}.css")
            
            # SCSS Variables format
            self._export_scss_variables(config, f"{base_filename}.scss")
            
            # Tailwind Config format
            self._export_tailwind_config(config, f"{base_filename}_tailwind.js")
            
            logger.info(f"Configuration exported in multiple formats with base name: {base_filename}")
            
        except Exception as e:
            logger.error(f"Error exporting configurations: {e}")
    
    def _export_css_variables(self, config: Dict, output_path: str):
        """Export theme as CSS custom properties"""
        colors = config.get("colors", {})
        typography = config.get("typography", {})
        layout = config.get("layout", {})
        
        css_content = ":root {\n"
        
        # Color variables
        css_content += "  /* Colors */\n"
        for key, value in colors.items():
            if key != "palette":
                css_content += f"  --color-{key.replace('_', '-')}: {value};\n"
        
        # Typography variables
        css_content += "\n  /* Typography */\n"
        if "scale" in typography:
            for size, value in typography["scale"].items():
                css_content += f"  --font-size-{size}: {value};\n"
        
        # Spacing variables
        css_content += "\n  /* Spacing */\n"
        if "spacing_scale" in layout and "scale" in layout["spacing_scale"]:
            for size, value in layout["spacing_scale"]["scale"].items():
                css_content += f"  --spacing-{size}: {value};\n"
        
        # Border radius
        if "border_radius" in layout:
            css_content += f"  --border-radius: {layout['border_radius']};\n"
        
        css_content += "}\n"
        
        with open(output_path, 'w') as f:
            f.write(css_content)
    
    def _export_scss_variables(self, config: Dict, output_path: str):
        """Export theme as SCSS variables"""
        colors = config.get("colors", {})
        typography = config.get("typography", {})
        layout = config.get("layout", {})
        
        scss_content = "// Theme Variables\n\n"
        
        # Color variables
        scss_content += "// Colors\n"
        for key, value in colors.items():
            if key != "palette":
                scss_content += f"${key.replace('_', '-')}: {value};\n"
        
        # Typography variables
        scss_content += "\n// Typography\n"
        if "scale" in typography:
            for size, value in typography["scale"].items():
                scss_content += f"$font-size-{size}: {value};\n"
        
        # Spacing variables
        scss_content += "\n// Spacing\n"
        if "spacing_scale" in layout and "scale" in layout["spacing_scale"]:
            for size, value in layout["spacing_scale"]["scale"].items():
                scss_content += f"$spacing-{size}: {value};\n"
        
        # Border radius
        if "border_radius" in layout:
            scss_content += f"$border-radius: {layout['border_radius']};\n"
        
        with open(output_path, 'w') as f:
            f.write(scss_content)
    
    def _export_tailwind_config(self, config: Dict, output_path: str):
        """Export theme as Tailwind CSS configuration"""
        colors = config.get("colors", {})
        typography = config.get("typography", {})
        layout = config.get("layout", {})
        
        tailwind_config = {
            "theme": {
                "extend": {
                    "colors": {},
                    "fontSize": {},
                    "spacing": {},
                    "borderRadius": {}
                }
            }
        }
        
        # Add colors
        for key, value in colors.items():
            if key != "palette":
                tailwind_config["theme"]["extend"]["colors"][key.replace('_', '-')] = value
        
        # Add typography
        if "scale" in typography:
            for size, value in typography["scale"].items():
                tailwind_config["theme"]["extend"]["fontSize"][size] = value
        
        # Add spacing
        if "spacing_scale" in layout and "scale" in layout["spacing_scale"]:
            for size, value in layout["spacing_scale"]["scale"].items():
                tailwind_config["theme"]["extend"]["spacing"][size] = value
        
        # Add border radius
        if "border_radius" in layout:
            tailwind_config["theme"]["extend"]["borderRadius"]["default"] = layout["border_radius"]
        
        js_content = f"module.exports = {json.dumps(tailwind_config, indent=2)}"
        
        with open(output_path, 'w') as f:
            f.write(js_content)
    
    def batch_analyze_images(self, image_directory: str, theme_prompts: List[str]) -> Dict:
        """
        Analyze multiple images with different theme prompts
        """
        results = {}
        image_dir = Path(image_directory)
        
        if not image_dir.exists():
            logger.error(f"Directory {image_directory} does not exist")
            return results
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            logger.warning(f"No image files found in {image_directory}")
            return results
        
        # Analyze each image with each theme prompt
        for image_file in image_files:
            image_name = image_file.stem
            results[image_name] = {}
            
            for theme_prompt in theme_prompts:
                logger.info(f"Analyzing {image_name} with theme: {theme_prompt}")
                
                try:
                    config = self.generate_theme_config(str(image_file), theme_prompt)
                    results[image_name][theme_prompt] = config
                    
                    # Save individual config
                    output_filename = f"{image_name}_{theme_prompt.replace(' ', '_').lower()}"
                    self.save_config(config, f"{output_filename}.json")
                    
                except Exception as e:
                    logger.error(f"Error analyzing {image_name} with theme '{theme_prompt}': {e}")
                    results[image_name][theme_prompt] = {"error": str(e)}
        
        # Save batch results
        batch_output_path = f"batch_analysis_results_{len(image_files)}images_{len(theme_prompts)}themes.json"
        self.save_config(results, batch_output_path)
        
        return results
    
    def compare_theme_variations(self, image_path: str, base_theme: str, variations: List[str]) -> Dict:
        """
        Compare different variations of a theme on the same image
        """
        comparison_results = {
            "image_path": image_path,
            "base_theme": base_theme,
            "variations": {}
        }
        
        # Analyze base theme
        logger.info(f"Analyzing base theme: {base_theme}")
        base_config = self.generate_theme_config(image_path, base_theme)
        comparison_results["base_analysis"] = base_config
        
        # Analyze variations
        for variation in variations:
            full_theme = f"{base_theme} - {variation}"
            logger.info(f"Analyzing variation: {full_theme}")
            
            try:
                variation_config = self.generate_theme_config(image_path, full_theme)
                comparison_results["variations"][variation] = variation_config
                
                # Add comparison metrics
                comparison_results["variations"][variation]["comparison_with_base"] = {
                    "color_similarity": self._compare_color_palettes(
                        base_config.get("colors", {}), 
                        variation_config.get("colors", {})
                    ),
                    "confidence_difference": abs(
                        base_config.get("metadata", {}).get("analysis_confidence", 0) - 
                        variation_config.get("metadata", {}).get("analysis_confidence", 0)
                    )
                }
                
            except Exception as e:
                logger.error(f"Error analyzing variation '{variation}': {e}")
                comparison_results["variations"][variation] = {"error": str(e)}
        
        # Save comparison results
        comparison_filename = f"theme_comparison_{base_theme.replace(' ', '_').lower()}.json"
        self.save_config(comparison_results, comparison_filename)
        
        return comparison_results
    
    def _compare_color_palettes(self, palette1: Dict, palette2: Dict) -> float:
        """
        Compare two color palettes and return similarity score (0-1)
        """
        try:
            colors1 = [palette1.get("primary"), palette1.get("secondary"), palette1.get("accent")]
            colors2 = [palette2.get("primary"), palette2.get("secondary"), palette2.get("accent")]
            
            # Remove None values
            colors1 = [c for c in colors1 if c]
            colors2 = [c for c in colors2 if c]
            
            if not colors1 or not colors2:
                return 0.0
            
            # Simple color similarity based on hex values
            matches = sum(1 for c1 in colors1 if c1 in colors2)
            return matches / max(len(colors1), len(colors2))
            
        except Exception:
            return 0.0
    
    def generate_style_guide(self, config: Dict, output_path: str):
        """
        Generate an HTML style guide from the theme configuration
        """
        colors = config.get("colors", {})
        typography = config.get("typography", {})
        components = config.get("components", {})
        brand_personality = config.get("brand_personality", {})
        recommendations = config.get("recommendations", [])
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Style Guide - {config.get('metadata', {}).get('theme_name', 'Theme')}</title>
    <style>
        body {{
            font-family: {', '.join(typography.get('primary_font', ['Arial', 'sans-serif']))};
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: {colors.get('background', '#ffffff')};
            color: {colors.get('text', '#000000')};
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .section {{
            margin-bottom: 40px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .color-palette {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .color-swatch {{
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            color: white;
        }}
        
        .color-swatch.light {{
            color: black;
        }}
        
        .typography-sample {{
            margin: 20px 0;
        }}
        
        .component-preview {{
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }}
        
        .btn-primary {{
            background: {colors.get('primary', '#007bff')};
            color: white;
            border: none;
            padding: {components.get('buttons', {}).get('primary', {}).get('padding', '12px 24px')};
            border-radius: {components.get('buttons', {}).get('primary', {}).get('border_radius', '6px')};
            cursor: pointer;
            margin-right: 10px;
        }}
        
        .btn-secondary {{
            background: transparent;
            color: {colors.get('primary', '#007bff')};
            border: 2px solid {colors.get('primary', '#007bff')};
            padding: {components.get('buttons', {}).get('secondary', {}).get('padding', '12px 24px')};
            border-radius: {components.get('buttons', {}).get('secondary', {}).get('border_radius', '6px')};
            cursor: pointer;
        }}
        
        .card {{
            background: {components.get('cards', {}).get('background', '#ffffff')};
            border: {components.get('cards', {}).get('border', '1px solid #e0e0e0')};
            border-radius: {components.get('cards', {}).get('border_radius', '8px')};
            box-shadow: {components.get('cards', {}).get('shadow', '0 2px 8px rgba(0,0,0,0.1)')};
            padding: 20px;
            margin: 20px 0;
        }}
        
        .recommendations {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid {colors.get('primary', '#007bff')};
        }}
        
        .recommendations li {{
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Style Guide: {config.get('metadata', {}).get('theme_name', 'Theme')}</h1>
            <p>Generated on {config.get('metadata', {}).get('generated_at', '')} using {config.get('metadata', {}).get('analysis_method', 'AI analysis')}</p>
            <p>Confidence Score: {config.get('metadata', {}).get('analysis_confidence', 'N/A')}</p>
        </header>
        
        <section class="section">
            <h2>Brand Personality</h2>
            <p><strong>Primary Traits:</strong> {', '.join(brand_personality.get('primary_traits', []))}</p>
            <p><strong>Mood:</strong> {brand_personality.get('mood', 'N/A')}</p>
            <p><strong>Target Audience:</strong> {brand_personality.get('target_audience', 'N/A')}</p>
            <p><strong>Style Direction:</strong> {brand_personality.get('style_direction', 'N/A')}</p>
        </section>
        
        <section class="section">
            <h2>Color Palette</h2>
            <div class="color-palette">
                <div class="color-swatch" style="background-color: {colors.get('primary', '#007bff')}">
                    <h3>Primary</h3>
                    <p>{colors.get('primary', '#007bff')}</p>
                </div>
                <div class="color-swatch" style="background-color: {colors.get('secondary', '#6c757d')}">
                    <h3>Secondary</h3>
                    <p>{colors.get('secondary', '#6c757d')}</p>
                </div>
                <div class="color-swatch" style="background-color: {colors.get('accent', '#28a745')}">
                    <h3>Accent</h3>
                    <p>{colors.get('accent', '#28a745')}</p>
                </div>
                <div class="color-swatch light" style="background-color: {colors.get('background', '#ffffff')}; border: 1px solid #ddd;">
                    <h3>Background</h3>
                    <p>{colors.get('background', '#ffffff')}</p>
                </div>
            </div>
        </section>
        
        <section class="section">
            <h2>Typography</h2>
            <div class="typography-sample">
                <h1 style="font-size: {typography.get('scale', {}).get('h1', '2.5rem')}">Heading 1 - {typography.get('scale', {}).get('h1', '2.5rem')}</h1>
                <h2 style="font-size: {typography.get('scale', {}).get('h2', '2rem')}">Heading 2 - {typography.get('scale', {}).get('h2', '2rem')}</h2>
                <h3 style="font-size: {typography.get('scale', {}).get('h3', '1.5rem')}">Heading 3 - {typography.get('scale', {}).get('h3', '1.5rem')}</h3>
                <p style="font-size: {typography.get('scale', {}).get('body', '1rem')}">Body text - {typography.get('scale', {}).get('body', '1rem')}</p>
                <small style="font-size: {typography.get('scale', {}).get('small', '0.875rem')}">Small text - {typography.get('scale', {}).get('small', '0.875rem')}</small>
            </div>
            <p><strong>Font Stack:</strong> {', '.join(typography.get('primary_font', ['Arial', 'sans-serif']))}</p>
            <p><strong>Style Category:</strong> {typography.get('style_category', 'N/A')}</p>
        </section>
        
        <section class="section">
            <h2>Components</h2>
            
            <div class="component-preview">
                <h3>Buttons</h3>
                <button class="btn-primary">Primary Button</button>
                <button class="btn-secondary">Secondary Button</button>
            </div>
            
            <div class="component-preview">
                <h3>Cards</h3>
                <div class="card">
                    <h4>Card Title</h4>
                    <p>This is a sample card component using the generated theme styles.</p>
                </div>
            </div>
        </section>
        
        {f'''
        <section class="section">
            <h2>AI Recommendations</h2>
            <div class="recommendations">
                <ul>
                    {"".join(f"<li>{rec}</li>" for rec in recommendations)}
                </ul>
            </div>
        </section>
        ''' if recommendations else ''}
    </div>
</body>
</html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Style guide generated: {output_path}")


# Example usage with enhanced LangChain integration
if __name__ == "__main__":
    # Initialize the interpreter with LangChain and Mistral
    interpreter = VisualThemeInterpreter(
        mistral_api_key="",
        model_name="pixtral-12b-2409"
    )
    
    # Example theme analysis
    image_path = "C:/Users/Pavan/Desktop/batman-laptop-ciluigv1xjrqmfgk.jpg"
    theme_prompt = "Dark, gothic superhero theme with high-tech elements. Embodies justice, fear, and urban vigilance in a dystopian city."
    
    try:
        # Generate theme configuration with multi-perspective analysis
        config = interpreter.generate_theme_config(
            image_path, 
            theme_prompt, 
            use_multi_perspective=True
        )
        
        # Save in multiple formats
        base_filename = "batman_theme_config"
        interpreter.export_config_formats(config, base_filename)
        
        # Generate HTML style guide
        interpreter.generate_style_guide(config, f"{base_filename}_style_guide.html")
        
        # Print summary
        print("Theme Analysis Complete with LangChain + Mistral!")
        print(f"Primary Color: {config['colors']['primary']}")
        print(f"Style Direction: {config['brand_personality']['style_direction']}")
        print(f"AI Confidence: {config['metadata']['analysis_confidence']}")
        
        # Print AI recommendations if available
        if config.get('recommendations'):
            print("\nAI Recommendations:")
            for i, rec in enumerate(config['recommendations'], 1):
                print(f"{i}. {rec}")
        
        # Example: Compare theme variations
        variations = [
            "with neon accents", 
            "more minimalist approach", 
            "retro 80s influence"
        ]
        
        comparison = interpreter.compare_theme_variations(
            image_path, 
            theme_prompt, 
            variations
        )
        
        print(f"\nGenerated {len(variations)} theme variations for comparison")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print("Please ensure you have a valid image file and Mistral API key configured.")
        print("Required packages: pip install langchain-mistralai langchain-core")
