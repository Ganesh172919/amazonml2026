"""
Preprocessor module for Business Entity Resolution.
Provides text normalization, token cleaning, and entity parsing for multilingual records (US, India, France).
"""

import re
import unicodedata

# Common legal suffixes across US, India, France
LEGAL_SUFFIXES = {
    # English / US / International
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'llp', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'plc', 'lp', 'gmbh', 'sa', 'ag', 'holdings', 'group', 'services', 'solutions',
    'enterprises', 'international', 'industries', 'partners', 'associates',
    # French
    'sarl', 'sasu', 'eurl', 'sas', 'sci', 'snc', 'gie', 'association', 'societe', 'et', 'cie',
    # Common stopwords & particles
    'and', '&', 'the', 'of', 'in', 'at', 'by', 'for', 'with', 'de', 'du', 'la', 'le', 'des', 'les', 'd', 'l'
}

# Common address abbreviations and noise tokens
ADDR_ABBREVIATIONS = {
    'st': 'street', 'rd': 'road', 'ave': 'avenue', 'av': 'avenue', 'dr': 'drive',
    'ln': 'lane', 'blvd': 'boulevard', 'hwy': 'highway', 'ct': 'court', 'pl': 'place',
    'fl': 'floor', 'flr': 'floor', 'apt': 'apartment', 'ste': 'suite', 'unit': 'unit',
    'bldg': 'building', 'opp': 'opposite', 'nr': 'near', 'hno': 'house', 'no': 'number'
}

ADDR_STOPWORDS = {
    'road', 'street', 'avenue', 'drive', 'lane', 'boulevard', 'court', 'place', 'way',
    'near', 'opp', 'opposite', 'behind', 'beside', 'next', 'floor', 'unit', 'apartment',
    'suite', 'building', 'block', 'tower', 'house', 'flat', 'shop', 'plot', 'sector',
    'phase', 'nagar', 'colony', 'city', 'state', 'district', 'town', 'village', 'post',
    'po', 'pin', 'rue', 'boulevard', 'avenue', 'chemin', 'impasse', 'place', 'route'
}


def normalize_text(text: str) -> str:
    """Normalize unicode, convert to lowercase, and replace punctuation with spaces."""
    if not text or not isinstance(text, str):
        return ""
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    # Lowercase
    text = text.lower()
    # Replace non-alphanumeric (keep unicode word characters) with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse multiple whitespaces
    return ' '.join(text.split())


def clean_name_tokens(name: str) -> list[str]:
    """Extract significant name tokens excluding legal suffixes and stopwords."""
    norm = normalize_text(name)
    tokens = []
    for w in norm.split():
        # Expand abbreviation if known
        w = ADDR_ABBREVIATIONS.get(w, w)
        if w not in LEGAL_SUFFIXES and len(w) > 1 and not w.isdigit():
            tokens.append(w)
    return tokens


def clean_addr_tokens(addr: str) -> list[str]:
    """Extract significant address tokens excluding address stopwords."""
    norm = normalize_text(addr)
    tokens = []
    for w in norm.split():
        w = ADDR_ABBREVIATIONS.get(w, w)
        if w not in ADDR_STOPWORDS and len(w) > 2 and not w.isdigit():
            tokens.append(w)
    return tokens


def extract_numbers(text: str) -> list[str]:
    """Extract all sequences of digits (building numbers, postal codes, etc.)."""
    if not text or not isinstance(text, str):
        return []
    # Match digit sequences of length 1 to 8
    nums = re.findall(r'\b\d{1,8}\b', text)
    # Normalize numbers (strip leading zeros for building numbers, e.g. 0684 -> 684)
    norm_nums = []
    for n in nums:
        norm_nums.append(n)
        stripped = n.lstrip('0')
        if stripped and stripped != n:
            norm_nums.append(stripped)
    return norm_nums


def get_compressed_name(name: str) -> str:
    """Get alphanumeric-only compressed string for typo and spacing tolerance."""
    if not name or not isinstance(name, str):
        return ""
    norm = normalize_text(name)
    # Keep only alphanumeric characters without spaces
    return re.sub(r'[^a-z0-9]', '', norm)
