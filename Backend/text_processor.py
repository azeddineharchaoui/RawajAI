import re

def clean_response_for_tts(text, language="en"):
    """Clean and format a response to make it suitable for text-to-speech"""
    # Basic cleanup
    response = text.strip()
    if not response:
        return response
    
    # Remove markdown code blocks completely
    response = re.sub(r'```[\s\S]*?```', '', response)
    
    # Remove inline code formatting
    response = re.sub(r'`([^`]+)`', r'\1', response)
    
    # Remove markdown links but keep the text
    response = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', response)
    
    # Replace bullet points with proper verbal markers and add pauses
    response = re.sub(r'^\s*[-*]\s+(.+)$', r'• \1. ', response, flags=re.MULTILINE)
    
    # Format numbers for better TTS pronunciation
    def format_number(match):
        num = match.group(0)
        if '.' in num:  # It's a decimal
            # Improve pronunciation of decimals with proper spacing
            return num.replace('.', ' point ')
        # For large numbers add spaces for better pronunciation
        if len(num) > 6:  # For numbers > 999,999
            return ' '.join(num)
        return num  # Return as is for whole numbers
        
    response = re.sub(r'\b\d+\.\d+\b', format_number, response)
    
    # Convert percentages for better pronunciation
    response = re.sub(r'(\d+)%', r'\1 percent', response)
    
    # Replace common abbreviations with full forms for better TTS
    abbr_dict = {
        # Supply chain specific abbreviations
        r'\bEOQ\b': 'Economic Order Quantity',
        r'\bJIT\b': 'Just in Time',
        r'\bSCM\b': 'Supply Chain Management',
        r'\bKPI\b': 'Key Performance Indicator',
        r'\bROI\b': 'Return on Investment',
        r'\bCRM\b': 'Customer Relationship Management',
        r'\bERP\b': 'Enterprise Resource Planning',
        r'\bRFID\b': 'Radio Frequency Identification',
        r'\bIoT\b': 'Internet of Things',
        r'\bAI\b': 'Artificial Intelligence',
        r'\bML\b': 'Machine Learning',
        r'\bNLP\b': 'Natural Language Processing',
        
        # Common abbreviations
        r'\be\.g\.\b': 'for example',
        r'\bi\.e\.\b': 'that is',
        r'\betc\.\b': 'etcetera',
        r'\bvs\.\b': 'versus',
        r'\bapprox\.\b': 'approximately',
        
        # Business terms
        r'\bQ(\d+)\b': r'Quarter \1',  # Q1 -> Quarter 1
        r'\bFY(\d{2,4})\b': r'Fiscal Year \1',  # FY22 -> Fiscal Year 22
        r'\bYOY\b': 'Year over Year',
        r'\bMOM\b': 'Month over Month',
        r'\bCOGS\b': 'Cost of Goods Sold',
        r'\bB2B\b': 'Business to Business',
        r'\bB2C\b': 'Business to Consumer'
    }
    
    for abbr, full in abbr_dict.items():
        response = re.sub(abbr, full, response)
    
    # Add appropriate pauses for better audio readability and natural speech rhythm
    response = response.replace('. ', '. <break time="0.5s"/> ')
    response = response.replace('! ', '! <break time="0.5s"/> ')
    response = response.replace('? ', '? <break time="0.6s"/> ')
    response = response.replace(': ', ': <break time="0.3s"/> ')
    response = response.replace('; ', '; <break time="0.4s"/> ')
    
    # Add pauses for paragraph breaks for more natural speech
    response = re.sub(r'\n\s*\n', '\n<break time="1s"/>\n', response)
    
    # Remove any HTML/XML-like tags that aren't SSML
    response = re.sub(r'<(?!(break|prosody|phoneme|say-as|sub|p|s|voice|emphasis)\s)[^>]*>', '', response)
    
    # Remove any JSON fragments that might have slipped through
    response = re.sub(r'{.*?}', '', response)
    response = re.sub(r'\[.*?\]', '', response)
    
    # Remove URLs as they don't read well in speech
    response = re.sub(r'https?://\S+', 'website link', response)
    
    # Handle tables by converting them to spoken form
    if '|' in response:
        # Simple table detection
        table_rows = re.findall(r'^.*\|.*\|.*$', response, re.MULTILINE)
        if table_rows and len(table_rows) > 1:
            # Replace table with a spoken description
            table_text = "Here's a summary of key information: <break time=\"0.5s\"/> "
            for row in table_rows[1:]:  # Skip header row
                cells = [cell.strip() for cell in row.split('|') if cell.strip()]
                if cells:
                    table_text += ", ".join(cells) + ". <break time=\"0.5s\"/> "
            
            # Replace table with spoken text
            for row in table_rows:
                response = response.replace(row, '')
            response += table_text
    
    # Add special handling for languages
    if language == "fr":
        # Format French numbers
        response = re.sub(r'\b(\d+),(\d+)\b', r'\1 virgule \2', response)
        
        # French-specific abbreviations
        fr_abbr = {
            r'\bex\.\b': 'exemple',
            r'\bcf\.\b': 'confère',
            r'\bc-à-d\b': 'c\'est-à-dire',
            r'\bN\.B\.\b': 'nota bene'
        }
        for abbr, full in fr_abbr.items():
            response = re.sub(abbr, full, response)
            
    elif language == "ar":
        # Arabic-specific formatting
        # Convert Western digits to Arabic if needed
        pass
    
    # Clean up potential hallucinations and standard model endings
    common_endings = [
        "I hope this helps", 
        "Hope this helps", 
        "Let me know if you have any questions",
        "Feel free to ask", 
        "Is there anything else", 
        "Do you need more information",
        "I'm happy to assist further",
        "Don't hesitate to ask",
        "Please let me know if you need anything else",
        "Thanks for asking",
        "I hope that answers your question"
    ]
    
    for ending in common_endings:
        if ending.lower() in response[-100:].lower():
            response = response[:response.lower().rfind(ending.lower())].strip()
    
    # Remove multiple consecutive spaces
    response = re.sub(r' {2,}', ' ', response)
    
    # Check for and add concluding period if missing
    if response and response[-1] not in '.!?':
        response += '.'
    
    return response.strip()
