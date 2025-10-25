from nemo.utils import logging
import re
from nemo.collections.tts.g2p.models.espeakphonemizer.phonemize import create_espeak_phonemizer, phonemize

_DEF_PUNCS=',.!?-:;/"()[]{}।॥|~`\'\'"``;:,.!?¡¿—…"«»“”।@'
phonemizers = {}

# Regular expressions to detect scripts, numbers, SSML tags, and time formats
DEVANAGARI_REGEX = re.compile(r'[\u0900-\u097F]+')  # Hindi, Marathi, Nepali (Devanagari script)
BENGALI_REGEX = re.compile(r'[\u0980-\u09FF]+')  # Bengali script
GUJARATI_REGEX = re.compile(r'[\u0A80-\u0AFF]+')  # Gujarati script
GURMUKHI_REGEX = re.compile(r'[\u0A00-\u0A7F]+')  # Punjabi (Gurmukhi script)
ORIYA_REGEX = re.compile(r'[\u0B00-\u0B7F]+')  # Odia (Oriya script)
TAMIL_REGEX = re.compile(r'[\u0B80-\u0BFF]+')  # Tamil script
TELUGU_REGEX = re.compile(r'[\u0C00-\u0C7F]+')  # Telugu script
KANNADA_REGEX = re.compile(r'[\u0C80-\u0CFF]+')  # Kannada script
MALAYALAM_REGEX = re.compile(r'[\u0D00-\u0D7F]+')  # Malayalam script
LATIN_REGEX = re.compile(r'[A-Za-z]+')  # Latin script

class IndicG2P():
    def __init__(self):
        self.phonemizers = {}


    def initialize_phonemizers(self, languages=None):
        if languages is None:   
            languages = ['hi', 'en-gb', 'bn', 'gu', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'ta', 'te']
        for lang in languages:
            try:
                self.phonemizers[lang] = create_espeak_phonemizer(language=lang, punctuation_marks=_DEF_PUNCS)
                logging.info(f"Successfully initialized phonemizer for {lang}")
            except Exception as e:
                logging.warning(f"Failed to initialize phonemizer for {lang}: {e}")
                self.phonemizers[lang] = None

    
    def detect_language(self,word):
        """
        Detect the language of a word based on its Unicode script.
        
        Args:
            word (str): The word to analyze
            
        Returns:
            str: Language code ('hi', 'bn', 'gu', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'ta', 'te', 'en', 'unknown')
            
        Note: Devanagari script defaults to Hindi but could be Marathi or Nepali
        """
        if DEVANAGARI_REGEX.search(word):
            # Devanagari script is used by Hindi, Marathi, and Nepali
            # For now, default to Hindi, but this could be enhanced with more sophisticated detection
            return 'hi'  # Could be 'mr' or 'ne' as well
        elif BENGALI_REGEX.search(word):
            return 'bn'
        elif GUJARATI_REGEX.search(word):
            return 'gu'
        elif GURMUKHI_REGEX.search(word):
            return 'pa'
        elif ORIYA_REGEX.search(word):
            return 'or'
        elif TAMIL_REGEX.search(word):
            return 'ta'
        elif TELUGU_REGEX.search(word):
            return 'te'
        elif KANNADA_REGEX.search(word):
            return 'kn'
        elif MALAYALAM_REGEX.search(word):
            return 'ml'
        elif LATIN_REGEX.search(word):
            return 'en'
        return 'unknown'
    
    def phonemize_text(self, text: str):
        words = text.strip().split()
        phonemized_words = [self.get_phonemization(word) for word in words]
        return phonemized_words

    def get_phonemization(self,word):
        """
        Get phonemization for a word based on its detected language
        """
        language = self.detect_language(word)
        
        # Map language codes to phonemizer keys
        phonemizer_key_map = {
            'hi': 'hi',
            'en': 'en-gb',  # Use 'en-gb' phonemizer for English
            'bn': 'bn',
            'gu': 'gu',
            'kn': 'kn',
            'ml': 'ml',
            'mr': 'mr',
            'ne': 'ne',
            'or': 'or',
            'pa': 'pa',
            'ta': 'ta',
            'te': 'te'
        }
        
        phonemizer_key = phonemizer_key_map.get(language)
        phonemizer = phonemizers.get(phonemizer_key) if phonemizer_key else None
        
        if phonemizer:
            try:
                phonemized_word = phonemize(phonemizer, word)
                # Remove language tags like (en), (hi), (bn), etc. from phonemizer output
                phonemized_word = re.sub(r'\([a-z]{2,3}(-[a-z]+)?\)', '', phonemized_word)
                # Clean up any extra whitespace
                phonemized_word = ' '.join(phonemized_word.split())
                return phonemized_word
            except Exception as e:
                logging.warning(f"Phonemization failed for word '{word}' in language '{language}': {e}")
                return word
        else:
            # If no phonemizer available, return the word as is
            logging.debug(f"No phonemizer available for language '{language}', returning word as-is: '{word}'")
            return word
