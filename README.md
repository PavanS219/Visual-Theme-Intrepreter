# Visual Theme Interpreter 🎨🤖

A comprehensive AI-powered system for analyzing images and generating complete theme configurations using LangChain and Mistral AI. This tool combines computer vision, color psychology, and advanced AI analysis to create production-ready design themes from any image.

## ✨ Features

### 🧠 AI-Powered Analysis
- **Multi-perspective Analysis**: Color psychology, design patterns, and mood assessment
- **LangChain Integration**: Structured AI workflows with Mistral's Pixtral vision model
- **Confidence Scoring**: AI-generated confidence metrics for reliability assessment
- **Natural Language Processing**: Theme prompts processed for nuanced understanding

### 🎨 Visual Processing
- **Color Palette Extraction**: K-means clustering for dominant color identification
- **Composition Analysis**: Edge detection, symmetry, and visual weight distribution
- **Design Pattern Recognition**: Typography styles, layouts, and interaction patterns
- **Brand Personality Mapping**: Psychological trait extraction from visual elements

### 📦 Multi-Format Export
- **JSON Configuration**: Complete structured theme data
- **CSS Variables**: Ready-to-use custom properties
- **SCSS Variables**: Sass-compatible variable definitions
- **Tailwind Config**: Direct integration with Tailwind CSS
- **HTML Style Guide**: Interactive visual documentation

### 🔄 Batch Processing
- **Multi-image Analysis**: Process entire directories
- **Theme Variations**: Compare different approaches to the same image
- **Batch Export**: Automated processing with organized output

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
pip package manager
Mistral AI API key
```

### Install Dependencies
```bash
# Core dependencies
pip install langchain-mistralai langchain-core
pip install pillow numpy scikit-learn opencv-python
pip install colorsys pathlib

# Additional utilities
pip install pydantic typing-extensions
```

### Package Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/visual-theme-interpreter.git
cd visual-theme-interpreter

# Install requirements
pip install -r requirements.txt
```

## 🔧 Configuration

### API Key Setup
```python
# Method 1: Direct initialization
interpreter = VisualThemeInterpreter(
    mistral_api_key="your-mistral-api-key-here"
)

# Method 2: Environment variable
import os
os.environ['MISTRAL_API_KEY'] = 'your-mistral-api-key-here'
interpreter = VisualThemeInterpreter()
```

### Model Configuration
```python
interpreter = VisualThemeInterpreter(
    mistral_api_key="your-key",
    model_name="pixtral-12b-2409"  # Vision-capable model
)
```

## 📖 Usage Examples

### Basic Theme Generation
```python
from visual_theme_interpreter import VisualThemeInterpreter

# Initialize
interpreter = VisualThemeInterpreter(mistral_api_key="your-key")

# Generate theme
config = interpreter.generate_theme_config(
    image_path="path/to/your/image.jpg",
    theme_prompt="Modern minimalist design with corporate professionalism"
)

# Save configuration
interpreter.save_config(config, "my_theme.json")
```

### Multi-Format Export
```python
# Export in all formats
interpreter.export_config_formats(config, "my_theme")

# Generates:
# - my_theme.json (structured data)
# - my_theme.css (CSS variables)
# - my_theme.scss (SCSS variables)
# - my_theme_tailwind.js (Tailwind config)
```

### Generate Style Guide
```python
# Create interactive HTML style guide
interpreter.generate_style_guide(
    config, 
    "my_theme_style_guide.html"
)
```

### Batch Processing
```python
# Process multiple images
themes = ["modern corporate", "creative artistic", "tech startup"]
results = interpreter.batch_analyze_images(
    image_directory="./images/",
    theme_prompts=themes
)
```

### Theme Variations
```python
# Compare different approaches
variations = [
    "with dark mode",
    "mobile-first approach",
    "accessibility focused"
]

comparison = interpreter.compare_theme_variations(
    image_path="design.jpg",
    base_theme="e-commerce platform",
    variations=variations
)
```

## 🏗️ Architecture

### Core Components

#### 1. VisualThemeInterpreter Class
The main orchestrator that coordinates all analysis components.

#### 2. LangChain Integration
```python
# Specialized chains for different analysis types
- color_chain: Color psychology analysis
- design_chain: Design pattern recognition
- mood_chain: Atmosphere and brand assessment
- comprehensive_chain: Complete integrated analysis
```

#### 3. Computer Vision Pipeline
```python
- Color extraction via K-means clustering
- Composition analysis with OpenCV
- Visual weight distribution calculation
- Symmetry and balance assessment
```

#### 4. AI Analysis Models
```python
class ColorAnalysis(BaseModel):
    dominant_emotion: str
    brand_personality: List[str]
    user_behavior_impact: str
    color_harmony: str

class DesignPatterns(BaseModel):
    typography_style: str
    layout_pattern: str
    visual_hierarchy: str
    interaction_style: str

class MoodAtmosphere(BaseModel):
    energy_level: str
    formality: str
    time_period: str
    target_audience: str
```

## 📊 Output Structure

### Complete Theme Configuration
```json
{
  "metadata": {
    "theme_name": "Your Theme Name",
    "generated_at": "2025-06-21",
    "analysis_confidence": 0.87,
    "analysis_method": "langchain_mistral"
  },
  "colors": {
    "palette": [...],
    "primary": "#007bff",
    "secondary": "#6c757d",
    "accent": "#28a745"
  },
  "typography": {
    "primary_font": ["Inter", "Helvetica", "sans-serif"],
    "style_category": "clean",
    "scale": { "h1": "2.5rem", "body": "1rem" }
  },
  "layout": {
    "grid_system": "12-column",
    "spacing_scale": {...},
    "border_radius": "6px"
  },
  "components": {
    "buttons": {...},
    "cards": {...},
    "forms": {...}
  },
  "brand_personality": {
    "primary_traits": ["innovative", "trustworthy"],
    "mood": "professional",
    "target_audience": "tech-savvy professionals"
  },
  "recommendations": [
    "Use consistent spacing for visual harmony",
    "Implement micro-interactions for engagement"
  ]
}
```

## 🎯 Use Cases

### Web Development
- **Design System Generation**: Create comprehensive design systems from brand images
- **Theme Customization**: Generate multiple theme variations for A/B testing
- **Brand Consistency**: Ensure visual coherence across digital products

### Brand Strategy
- **Visual Identity Analysis**: Extract brand personality from existing materials
- **Competitor Analysis**: Analyze competitor visual strategies
- **Brand Evolution**: Generate modern interpretations of classic designs

### UX/UI Design
- **Rapid Prototyping**: Quick theme generation for design exploration
- **Accessibility Assessment**: Color contrast and usability analysis
- **Design Documentation**: Automated style guide generation

### Marketing
- **Campaign Theming**: Generate themes aligned with campaign imagery
- **Social Media Branding**: Consistent visual identity across platforms
- **Event Branding**: Theme generation from venue or concept images

## 🔬 Advanced Features

### Multi-Perspective Analysis
```python
# Get detailed insights from multiple AI perspectives
config = interpreter.generate_theme_config(
    image_path="complex_design.jpg",
    theme_prompt="luxury hospitality brand",
    use_multi_perspective=True
)

# Access individual analyses
color_insights = config["ai_analyses"]["color"]["result"]
design_patterns = config["ai_analyses"]["design"]["result"] 
mood_assessment = config["ai_analyses"]["mood"]["result"]
```

### Custom Analysis Chains
```python
# Extend functionality with custom chains
def create_accessibility_chain():
    # Custom accessibility analysis
    pass

# Add to interpreter
interpreter.accessibility_chain = create_accessibility_chain()
```

### Confidence-Based Filtering
```python
# Only process high-confidence results
if config["metadata"]["analysis_confidence"] > 0.8:
    # Proceed with high-confidence theme
    apply_theme(config)
else:
    # Request manual review or alternative analysis
    manual_review_required(config)
```

## 📈 Performance Optimization

### Image Preprocessing
```python
# Optimize images before analysis
def preprocess_image(image_path, max_size=1024):
    # Resize large images for faster processing
    # Maintain aspect ratio and quality
    pass
```

### Caching Strategies
```python
# Cache color palettes and composition analysis
# Reduce redundant computer vision operations
cache_key = f"{image_hash}_{analysis_type}"
```

### Batch Optimization
```python
# Process multiple images efficiently
# Parallel processing for independent analyses
# Memory management for large datasets
```

## 🛠️ Customization

### Theme Prompt Engineering
```python
# Effective prompt patterns
prompts = {
    "brand_focused": "Professional {industry} brand emphasizing {values}",
    "user_focused": "Design for {demographic} users who value {priorities}",
    "context_focused": "{platform} interface optimized for {use_case}"
}
```

### Color Role Customization
```python
def custom_color_role_detector(color, frequency, context):
    # Custom logic for determining color roles
    # Based on industry, brand type, or use case
    pass
```

### Component Style Generation
```python
def generate_custom_components(config, component_types):
    # Generate styles for specific component libraries
    # Material-UI, Ant Design, Chakra UI, etc.
    pass
```

## 🔍 Troubleshooting

### Common Issues

#### API Connection Problems
```python
# Check API key validity
if not interpreter.llm:
    print("Mistral API not configured properly")
    print("Verify API key and model availability")
```

#### Image Processing Errors
```python
# Validate image files
supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
if not any(path.lower().endswith(fmt) for fmt in supported_formats):
    raise ValueError("Unsupported image format")
```

#### Low Confidence Scores
```python
# Strategies for improving analysis confidence:
# 1. Use higher resolution images
# 2. Provide more specific theme prompts
# 3. Ensure good image composition and lighting
# 4. Use images with clear design elements
```

### Performance Issues
```python
# Optimize for large images
def optimize_image_size(image_path, target_size=1024):
    # Resize while maintaining quality
    # Balance processing speed and accuracy
    pass
```

## 📚 API Reference

### Main Methods

#### `generate_theme_config(image_path, theme_prompt, use_multi_perspective=True)`
Generates complete theme configuration from image analysis.

**Parameters:**
- `image_path` (str): Path to image file
- `theme_prompt` (str): Natural language theme description
- `use_multi_perspective` (bool): Enable multi-angle AI analysis

**Returns:** Dictionary containing complete theme configuration

#### `extract_color_palette(image_path, n_colors=8)`
Extracts dominant colors using K-means clustering.

**Parameters:**
- `image_path` (str): Path to image file
- `n_colors` (int): Number of colors to extract

**Returns:** List of color dictionaries with hex, RGB, HSL, and role information

#### `analyze_composition(image_path)`
Analyzes image composition and layout patterns.

**Parameters:**
- `image_path` (str): Path to image file

**Returns:** Dictionary with composition metrics and analysis

#### `export_config_formats(config, base_filename)`
Exports theme in multiple formats (JSON, CSS, SCSS, Tailwind).

**Parameters:**
- `config` (dict): Theme configuration dictionary
- `base_filename` (str): Base name for output files

### Utility Methods

#### `save_config(config, output_path)`
Saves configuration to JSON file.

#### `generate_style_guide(config, output_path)`
Generates interactive HTML style guide.

#### `batch_analyze_images(image_directory, theme_prompts)`
Processes multiple images with various theme prompts.

#### `compare_theme_variations(image_path, base_theme, variations)`
Compares different theme interpretations of the same image.

## 🤝 Contributing

### Development Setup
```bash
# Clone for development
git clone https://github.com/yourusername/visual-theme-interpreter.git
cd visual-theme-interpreter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install in development mode
pip install -e .
```

### Code Style
```bash
# Format code
black visual_theme_interpreter/
flake8 visual_theme_interpreter/

# Type checking
mypy visual_theme_interpreter/
```

### Testing
```bash
# Run tests
pytest tests/

# Coverage report
pytest --cov=visual_theme_interpreter tests/
```

### Pull Request Process
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **Mistral AI** for providing advanced vision-language models
- **LangChain** for the flexible AI application framework
- **OpenCV** and **scikit-learn** for computer vision capabilities
- **PIL/Pillow** for image processing utilities

## 📞 Support

### Documentation
- [Full API Documentation](docs/api.md)
- [Advanced Usage Examples](docs/examples.md)
- [Troubleshooting Guide](docs/troubleshooting.md)

### Community
- [GitHub Issues](https://github.com/yourusername/visual-theme-interpreter/issues)
- [Discussion Forum](https://github.com/yourusername/visual-theme-interpreter/discussions)
- [Discord Community](https://discord.gg/your-invite)

### Professional Support
For enterprise support, custom development, or consulting services:
- Email: support@yourcompany.com
- Website: https://yourcompany.com/enterprise

---

**Made with ❤️ by Pavan **

*Transforming visual inspiration into production-ready design systems through AI-powered analysis.*
