import re

def remove_negation(text):
    if not isinstance(text, str):
        return text, False  # Return non-string values as is

    negation_patterns = ['never had', 'unremarkable', 'normal', 'no']  # Add more patterns as needed
    exceptions = ['without particularity', 'without significance', 'was normal', 'not found']  # Add patterns that should not be removed at the beginning

    negation_found = False

    # Iterate over negation patterns and apply removal based on context
    for pattern in negation_patterns:
        # Create a regular expression pattern with word boundaries around negation words
        pattern_regex = r'\b' + re.escape(pattern) + r'\b'

        # Use re.sub to remove the negation part from the text
        text, count = re.subn(pattern_regex, '', text, flags=re.IGNORECASE)

        # Check if negation pattern was found
        if count > 0:
            negation_found = True

    return text.strip(), negation_found  # Remove leading/trailing whitespaces and return negation flag