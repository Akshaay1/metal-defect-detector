import requests
import json
import os

# You would typically use your own API key for a proper implementation
# For demonstration, we'll simulate LLM responses

# Dictionary of pre-defined responses for common defect types
DEFECT_INFO = {
    'Crazing': {
        'description': 'Crazing defects are characterized by a network of fine cracks or lines that appear on the metal surface.',
        'causes': 'Crazing is typically caused by thermal cycling, improper cooling rates, or surface stresses during manufacturing.',
        'prevention': 'To prevent crazing, implement controlled cooling rates, proper alloy selection, and stress relief procedures.'
    },
    'Scratches': {
        'description': 'Scratches are linear marks or grooves on the metal surface caused by mechanical damage.',
        'causes': 'Scratches typically result from improper handling, tools contacting the surface, or debris in manufacturing equipment.',
        'prevention': 'To prevent scratches, improve material handling procedures, use proper tooling, and maintain clean manufacturing environments.'
    },
    'inclusion': {
        'description': 'Inclusions are foreign particles or impurities embedded in the metal matrix.',
        'causes': 'Inclusions are caused by contamination during metal production, inadequate refining, or improper melting procedures.',
        'prevention': 'To prevent inclusions, improve melt cleanliness, use proper filtration, and control raw material quality.'
    },
    'oil spot': {
        'description': 'Oil spots appear as discolored areas on the metal surface where oil or lubricant has contaminated the material.',
        'causes': 'Oil spots are typically caused by lubricant leaks in machinery, improper cleaning before processing, or contaminated handling equipment.',
        'prevention': 'To prevent oil spots, maintain machinery to prevent leaks, implement thorough cleaning procedures, and use proper handling techniques.'
    },
    'water spot': {
        'description': 'Water spots are circular marks or stains on the metal surface caused by water droplets.',
        'causes': 'Water spots typically result from water droplets drying on the surface, improper drying after cleaning, or condensation during processing.',
        'prevention': 'To prevent water spots, ensure complete drying after wet processes, control humidity in processing areas, and implement proper storage conditions.'
    }
}

def get_defect_info(defect_class):
    """
    Get information about a defect type using an LLM or predefined responses
    
    Args:
        defect_class: The defect class to get information about
        
    Returns:
        defect_info: Dictionary containing information about the defect
    """
    # Check if we have predefined information for this defect
    if defect_class in DEFECT_INFO:
        return DEFECT_INFO[defect_class]
    
    # If not, we would typically query an LLM API
    # For demonstration, we'll generate a generic response
    
    # In a real implementation, you would use something like:
    # response = requests.post(
    #     "https://api.openai.com/v1/chat/completions",
    #     headers={
    #         "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    #         "Content-Type": "application/json"
    #     },
    #     json={
    #         "model": "gpt-3.5-turbo",
    #         "messages": [
    #             {"role": "system", "content": "You are a metallurgy expert. Provide concise information about metal defects."},
    #             {"role": "user", "content": f"Describe the {defect_class} metal defect, its causes, and prevention methods."}
    #         ]
    #     }
    # )
    # response_data = response.json()
    # llm_response = response_data["choices"][0]["message"]["content"]
    
    # For demonstration, return a generic response
    generic_info = {
        'description': f'{defect_class} is a type of metal surface defect that affects material quality and performance.',
        'causes': f'Common causes of {defect_class} include manufacturing process variations, material inconsistencies, and handling issues.',
        'prevention': f'To prevent {defect_class}, implement proper quality control, maintain equipment, and follow material-specific processing guidelines.'
    }
    
    return generic_info

def format_defect_info(defect_info):
    """
    Format defect information for display
    
    Args:
        defect_info: Dictionary containing information about the defect
        
    Returns:
        formatted_info: Formatted information as HTML
    """
    html = f"""
    <div class="defect-info">
        <h3>Defect Information</h3>
        <div class="info-section">
            <h4>Description</h4>
            <p>{defect_info['description']}</p>
        </div>
        <div class="info-section">
            <h4>Common Causes</h4>
            <p>{defect_info['causes']}</p>
        </div>
        <div class="info-section">
            <h4>Prevention Methods</h4>
            <p>{defect_info['prevention']}</p>
        </div>
    </div>
    """
    
    return html 