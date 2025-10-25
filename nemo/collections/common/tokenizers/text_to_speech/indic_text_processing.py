
from abc import ABC, abstractmethod
from typing import List, Optional
import re
import time
import inflect
import logging as logger

_DEF_PUNCS=',.!?-:;/"()[]{}।॥|~`\'\'"``;:,.!?¡¿—…"«»“”।@'
PUNCTUATION_REGEX = re.compile(r'([{}])'.format(re.escape(_DEF_PUNCS)))

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

# Number detection patterns for different scripts
NUMBER_REGEX = re.compile(r'\d+')  # Latin numerals (0-9)
DEVANAGARI_NUMBER_REGEX = re.compile(r'[\u0966-\u096F]+')  # Hindi/Marathi/Nepali numerals (०-९)
BENGALI_NUMBER_REGEX = re.compile(r'[\u09E6-\u09EF]+')  # Bengali numerals (০-৯)
GUJARATI_NUMBER_REGEX = re.compile(r'[\u0AE6-\u0AEF]+')  # Gujarati numerals (૦-૯)
GURMUKHI_NUMBER_REGEX = re.compile(r'[\u0A66-\u0A6F]+')  # Punjabi numerals (੦-੯)
ORIYA_NUMBER_REGEX = re.compile(r'[\u0B66-\u0B6F]+')  # Odia numerals (୦-୯)
TAMIL_NUMBER_REGEX = re.compile(r'[\u0BE6-\u0BEF]+')  # Tamil numerals (௦-௯)
TELUGU_NUMBER_REGEX = re.compile(r'[\u0C66-\u0C6F]+')  # Telugu numerals (౦-౯)
KANNADA_NUMBER_REGEX = re.compile(r'[\u0CE6-\u0CEF]+')  # Kannada numerals (೦-೯)
MALAYALAM_NUMBER_REGEX = re.compile(r'[\u0D66-\u0D6F]+')  # Malayalam numerals (൦-൯)

# Combined pattern for all local number systems
ALL_NUMBERS_REGEX = re.compile(r'[\d\u0966-\u096F\u09E6-\u09EF\u0AE6-\u0AEF\u0A66-\u0A6F\u0B66-\u0B6F\u0BE6-\u0BEF\u0C66-\u0C6F\u0CE6-\u0CEF\u0D66-\u0D6F]+')

# Mathematical equation patterns
SIMPLE_MATH_REGEX = re.compile(r'(\d+(?:\.\d+)?)\s*([+\-×*÷/])\s*(\d+(?:\.\d+)?)')  # Basic math operations without equals
EQUATION_WITH_RESULT_REGEX = re.compile(r'([\d+\-×*÷/\s.]+)\s*=\s*(\d+(?:\.\d+)?)')  # Any expression = result
MATH_EXPRESSION_REGEX = re.compile(r'(\d+(?:\.\d+)?)(\s*[+\-×*÷/]\s*\d+(?:\.\d+)?)+(\s*=\s*\d+(?:\.\d+)?)?')  # Full math expressions

TIME_REGEX = re.compile(r'\b(\d{1,2}):(\d{2})\b')  # Detects time in HH:MM format with Latin numerals

# Time regex patterns for local numerals
LOCAL_TIME_REGEX = re.compile(r'\b([\d\u0966-\u096F\u09E6-\u09EF\u0AE6-\u0AEF\u0A66-\u0A6F\u0B66-\u0B6F\u0BE6-\u0BEF\u0C66-\u0C6F\u0CE6-\u0CEF\u0D66-\u0D6F]{1,2}):([\d\u0966-\u096F\u09E6-\u09EF\u0AE6-\u0AEF\u0A66-\u0A6F\u0B66-\u0B6F\u0BE6-\u0BEF\u0C66-\u0C6F\u0CE6-\u0CEF\u0D66-\u0D6F]{2})\b')  # Time with any numeral system (Latin or local scripts)
URL_REGEX = re.compile(
    
    r'((https?|ftp)://)?'  # Protocol (optional)
    r'([-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b)'  # Domain name
    r'([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'  # Path/query (optional)
)
SENTENCE_END_REGEX = re.compile(r'[.?!।]')  # Detect sentence-ending punctuation
SSML_SAY_AS_CARDINAL = re.compile(r'<say-as interpret-as="cardinal">(.*?)</say-as>')
SSML_SAY_AS_DIGITS = re.compile(r'<say-as interpret-as="digits">(.*?)</say-as>')
SSML_SAY_AS_TIME = re.compile(r'<say-as interpret-as="time">(.*?)</say-as>')
SSML_LANGUAGE_TAG = re.compile(r'<lang xml:lang="(.*?)">(.*?)</lang>')
SSML_COMBINED_TAG = re.compile(r'<lang xml:lang="(.*?)"><say-as interpret-as="(cardinal|digits|time)">(.*?)</say-as></lang>|<say-as interpret-as="(cardinal|digits|time)"><lang xml:lang="(.*?)">(.*?)</lang></say-as>')

# English inflector for number-to-word conversion
eng_inflector = inflect.engine()

# Mathematical operator mappings for different languages
math_operators = {
    'en': {
        '+': 'plus',
        '-': 'minus',
        '×': 'times',
        '*': 'times',
        '÷': 'divided by',
        '/': 'divided by',
        '=': 'equals'
    },
    'hi': {
        '+': 'जमा',
        '-': 'घटा',
        '×': 'गुणा',
        '*': 'गुणा',
        '÷': 'भाग',
        '/': 'भाग',
        '=': 'बराबर'
    },
    'bn': {
        '+': 'যোগ',
        '-': 'বিয়োগ',
        '×': 'গুণ',
        '*': 'গুণ',
        '÷': 'ভাগ',
        '/': 'ভাগ',
        '=': 'সমান'
    },
    'gu': {
        '+': 'વત્તા',
        '-': 'ઓછું',
        '×': 'ગુણાકાર',
        '*': 'ગુણાકાર',
        '÷': 'ભાગાકાર',
        '/': 'ભાગાકાર',
        '=': 'સમાન'
    },
    'kn': {
        '+': 'ಕೂಡಿಸಿ',
        '-': 'ಕಳೆಯಿರಿ',
        '×': 'ಗುಣಿಸಿ',
        '*': 'ಗುಣಿಸಿ',
        '÷': 'ಭಾಗಿಸಿ',
        '/': 'ಭಾಗಿಸಿ',
        '=': 'ಸಮಾನ'
    },
    'ml': {
        '+': 'കൂട്ടുക',
        '-': 'കുറയ്ക്കുക',
        '×': 'ഗുണിക്കുക',
        '*': 'ഗുണിക്കുക',
        '÷': 'ഭാഗിക്കുക',
        '/': 'ഭാഗിക്കുക',
        '=': 'തുല്യം'
    },
    'mr': {
        '+': 'जमा',
        '-': 'वजा',
        '×': 'गुणाकार',
        '*': 'गुणाकार',
        '÷': 'भागाकार',
        '/': 'भागाकार',
        '=': 'समान'
    },
    'ne': {
        '+': 'जमा',
        '-': 'घटाउ',
        '×': 'गुणन',
        '*': 'गुणन',
        '÷': 'भाग',
        '/': 'भाग',
        '=': 'बराबर'
    },
    'or': {
        '+': 'ଯୋଗ',
        '-': 'ବିୟୋଗ',
        '×': 'ଗୁଣ',
        '*': 'ଗୁଣ',
        '÷': 'ଭାଗ',
        '/': 'ଭାଗ',
        '=': 'ସମାନ'
    },
    'pa': {
        '+': 'ਜਮਾਂ',
        '-': 'ਘਟਾਓ',
        '×': 'ਗੁਣਾ',
        '*': 'ਗੁਣਾ',
        '÷': 'ਭਾਗ',
        '/': 'ਭਾਗ',
        '=': 'ਬਰਾਬਰ'
    },
    'ta': {
        '+': 'கூட்டல்',
        '-': 'கழித்தல்',
        '×': 'பெருக்கல்',
        '*': 'பெருக்கல்',
        '÷': 'வகுத்தல்',
        '/': 'வகுத்தல்',
        '=': 'சமம்'
    },
    'te': {
        '+': 'కూడిక',
        '-': 'తీసివేత',
        '×': 'గుణకారం',
        '*': 'గుణకారం',
        '÷': 'భాగహారం',
        '/': 'భాగహారం',
        '=': 'సమానం'
    }
}

# Number-to-words dictionaries for all supported languages

# Hindi number words for units, tens, hundreds, thousands, and up to one lakh
hindi_numbers = {
    '0': 'शून्य', '1': 'एक', '2': 'दो', '3': 'तीन', '4': 'चार',
    '5': 'पांच', '6': 'छह', '7': 'सात', '8': 'आठ', '9': 'नौ',
    '10': 'दस', '11': 'ग्यारह', '12': 'बारह', '13': 'तेरह', '14': 'चौदह',
    '15': 'पंद्रह', '16': 'सोलह', '17': 'सत्रह', '18': 'अठारह', '19': 'उन्नीस',
    '20': 'बीस', '21': 'इक्कीस', '22': 'बाईस', '23': 'तेईस', '24': 'चौबीस',
    '25': 'पच्चीस', '26': 'छब्बीस', '27': 'सत्ताईस', '28': 'अट्ठाईस', '29': 'उनतीस',
    '30': 'तीस', '31': 'इकतीस', '32': 'बत्तीस', '33': 'तैंतीस', '34': 'चौंतीस',
    '35': 'पैंतीस', '36': 'छत्तीस', '37': 'सैंतीस', '38': 'अड़तीस', '39': 'उनतालीस',
    '40': 'चालीस', '41': 'इकतालीस', '42': 'बयालीस', '43': 'तैंतालीस', '44': 'चवालीस',
    '45': 'पैंतालीस', '46': 'छियालीस', '47': 'सैंतालीस', '48': 'अड़तालीस', '49': 'उनचास',
    '50': 'पचास', '51': 'इक्यावन', '52': 'बावन', '53': 'तिरेपन', '54': 'चौवन',
    '55': 'पचपन', '56': 'छप्पन', '57': 'सत्तावन', '58': 'अट्ठावन', '59': 'उनसाठ',
    '60': 'साठ', '61': 'इकसठ', '62': 'बासठ', '63': 'तिरसठ', '64': 'चौंसठ',
    '65': 'पैंसठ', '66': 'छियासठ', '67': 'सड़सठ', '68': 'अड़सठ', '69': 'उनहत्तर',
    '70': 'सत्तर', '71': 'इकहत्तर', '72': 'बहत्तर', '73': 'तिहत्तर', '74': 'चौहत्तर',
    '75': 'पचहत्तर', '76': 'छिहत्तर', '77': 'सतहत्तर', '78': 'अठहत्तर', '79': 'उन्यासी',
    '80': 'अस्सी', '81': 'इक्यासी', '82': 'बयासी', '83': 'तिरासी', '84': 'चौरासी',
    '85': 'पचासी', '86': 'छियासी', '87': 'सतासी', '88': 'अठासी', '89': 'नवासी',
    '90': 'नब्बे', '91': 'इक्यानवे', '92': 'बानवे', '93': 'तिरेनवे', '94': 'चौरानवे',
    '95': 'पचानवे', '96': 'छियानवे', '97': 'सत्तानवे', '98': 'अट्ठानवे', '99': 'निन्यानवे',
    '100': 'सौ',
    '200': 'दो सौ', '300': 'तीन सौ', '400': 'चार सौ', '500': 'पांच सौ', 
    '600': 'छह सौ', '700': 'सात सौ', '800': 'आठ सौ', '900': 'नौ सौ',
    '1000': 'हज़ार', '2000': 'दो हज़ार', '3000': 'तीन हज़ार', '4000': 'चार हज़ार', 
    '5000': 'पांच हज़ार', '6000': 'छह हज़ार', '7000': 'सात हज़ार', '8000': 'आठ हज़ार', 
    '9000': 'नौ हज़ार', '10000': 'दस हज़ार', '20000': 'बीस हज़ार', 
    '30000': 'तीस हज़ार', '40000': 'चालीस हज़ार', '50000': 'पचास हज़ार', 
    '60000': 'साठ हज़ार', '70000': 'सत्तर हज़ार', '80000': 'अस्सी हज़ार', 
    '90000': 'नब्बे हज़ार', '100000': 'एक लाख'
}

# Bengali number words (complete set 0-99)
bengali_numbers = {
    '0': 'শূন্য', '1': 'এক', '2': 'দুই', '3': 'তিন', '4': 'চার',
    '5': 'পাঁচ', '6': 'ছয়', '7': 'সাত', '8': 'আট', '9': 'নয়',
    '10': 'দশ', '11': 'এগারো', '12': 'বারো', '13': 'তেরো', '14': 'চৌদ্দ',
    '15': 'পনেরো', '16': 'ষোলো', '17': 'সতেরো', '18': 'আঠারো', '19': 'উনিশ',
    '20': 'বিশ', '21': 'একুশ', '22': 'বাইশ', '23': 'তেইশ', '24': 'চব্বিশ',
    '25': 'পঁচিশ', '26': 'ছাব্বিশ', '27': 'সাতাশ', '28': 'আটাশ', '29': 'উনত্রিশ',
    '30': 'ত্রিশ', '31': 'একত্রিশ', '32': 'বত্রিশ', '33': 'তেত্রিশ', '34': 'চৌত্রিশ',
    '35': 'পঁয়ত্রিশ', '36': 'ছত্রিশ', '37': 'সাইত্রিশ', '38': 'আটত্রিশ', '39': 'উনচল্লিশ',
    '40': 'চল্লিশ', '41': 'একচল্লিশ', '42': 'বিয়াল্লিশ', '43': 'তেতাল্লিশ', '44': 'চুয়াল্লিশ',
    '45': 'পঁয়তাল্লিশ', '46': 'ছেচল্লিশ', '47': 'সাতচল্লিশ', '48': 'আটচল্লিশ', '49': 'উনপঞ্চাশ',
    '50': 'পঞ্চাশ', '51': 'একান্ন', '52': 'বায়ান্ন', '53': 'তিপ্পান্ন', '54': 'চুয়ান্ন',
    '55': 'পঞ্চান্ন', '56': 'ছাপ্পান্ন', '57': 'সাতান্ন', '58': 'আটান্ন', '59': 'উনষাট',
    '60': 'ষাট', '61': 'একষট্টি', '62': 'বাষট্টি', '63': 'তেষট্টি', '64': 'চৌষট্টি',
    '65': 'পঁয়ষট্টি', '66': 'ছেষট্টি', '67': 'সাতষট্টি', '68': 'আটষট্টি', '69': 'উনসত্তর',
    '70': 'সত্তর', '71': 'একাত্তর', '72': 'বাহাত্তর', '73': 'তিয়াত্তর', '74': 'চুয়াত্তর',
    '75': 'পঁচাত্তর', '76': 'ছিয়াত্তর', '77': 'সাতাত্তর', '78': 'আটাত্তর', '79': 'উনআশি',
    '80': 'আশি', '81': 'একাশি', '82': 'বিরাশি', '83': 'তিরাশি', '84': 'চুরাশি',
    '85': 'পঁচাশি', '86': 'ছিয়াশি', '87': 'সাতাশি', '88': 'আটাশি', '89': 'উননব্বই',
    '90': 'নব্বই', '91': 'একানব্বই', '92': 'বিরানব্বই', '93': 'তিরানব্বই', '94': 'চুরানব্বই',
    '95': 'পঁচানব্বই', '96': 'ছিয়ানব্বই', '97': 'সাতানব্বই', '98': 'আটানব্বই', '99': 'নিরানব্বই',
    '100': 'একশ',
    '200': 'দুইশ', '300': 'তিনশ', '400': 'চারশ', '500': 'পাঁচশ',
    '600': 'ছয়শ', '700': 'সাতশ', '800': 'আটশ', '900': 'নয়শ',
    '1000': 'হাজার'
}

# Gujarati number words (complete set 0-99)
gujarati_numbers = {
    '0': 'શૂન્ય', '1': 'એક', '2': 'બે', '3': 'ત્રણ', '4': 'ચાર',
    '5': 'પાંચ', '6': 'છ', '7': 'સાત', '8': 'આઠ', '9': 'નવ',
    '10': 'દસ', '11': 'અગિયાર', '12': 'બાર', '13': 'તેર', '14': 'ચૌદ',
    '15': 'પંદર', '16': 'સોળ', '17': 'સત્તર', '18': 'અઢાર', '19': 'ઓગણીસ',
    '20': 'વીસ', '21': 'એકવીસ', '22': 'બાવીસ', '23': 'ત્રેવીસ', '24': 'ચોવીસ',
    '25': 'પચીસ', '26': 'છવીસ', '27': 'સત્તાવીસ', '28': 'અઠ્ઠાવીસ', '29': 'ઓગણત્રીસ',
    '30': 'ત્રીસ', '31': 'એકત્રીસ', '32': 'બત્રીસ', '33': 'તેત્રીસ', '34': 'ચોત્રીસ',
    '35': 'પાંત્રીસ', '36': 'છત્રીસ', '37': 'સાડત્રીસ', '38': 'અડત્રીસ', '39': 'ઓગણચાલીસ',
    '40': 'ચાળીસ', '41': 'એકતાળીસ', '42': 'બેતાળીસ', '43': 'ત્રેતાળીસ', '44': 'ચુંતાળીસ',
    '45': 'પિંતાળીસ', '46': 'છિયાળીસ', '47': 'સુડતાળીસ', '48': 'અઢતાળીસ', '49': 'ઓગણપચાસ',
    '50': 'પચાસ', '51': 'એકાવન', '52': 'બાવન', '53': 'ત્રેપન', '54': 'ચોવન',
    '55': 'પચપન', '56': 'છપન', '57': 'સત્તાવન', '58': 'અઠાવન', '59': 'ઓગણસાઠ',
    '60': 'સાઠ', '61': 'એકસઠ', '62': 'બાસઠ', '63': 'ત્રેસઠ', '64': 'ચોસઠ',
    '65': 'પાંસઠ', '66': 'છિયાસઠ', '67': 'સડસઠ', '68': 'અડસઠ', '69': 'ઓગણસિત્તેર',
    '70': 'સિત્તેર', '71': 'એકોત્તેર', '72': 'બોત્તેર', '73': 'ત્રોત્તેર', '74': 'ચોમત્તેર',
    '75': 'પંચોત્તેર', '76': 'છોત્તેર', '77': 'સિત્યોત્તેર', '78': 'અઠોત્તેર', '79': 'ઓગણએંસી',
    '80': 'એંસી', '81': 'એક્યાસી', '82': 'બ્યાસી', '83': 'ત્ર્યાસી', '84': 'ચોર્યાસી',
    '85': 'પંચાસી', '86': 'છ્યાસી', '87': 'સત્યાસી', '88': 'અઠ્યાસી', '89': 'ઓગણનેવું',
    '90': 'નેવું', '91': 'એકાણું', '92': 'બાણું', '93': 'ત્રાણું', '94': 'ચોરાણું',
    '95': 'પંચાણું', '96': 'છાણું', '97': 'સત્તાણું', '98': 'અઠાણું', '99': 'નવ્વાણું',
    '100': 'સો',
    '200': 'બસો', '300': 'ત્રણસો', '400': 'ચારસો', '500': 'પાંચસો',
    '600': 'છસો', '700': 'સાતસો', '800': 'આઠસો', '900': 'નવસો',
    '1000': 'હજાર'
}

# Kannada number words (complete set with compound numbers)
kannada_numbers = {
    '0': 'ಸೊನ್ನೆ', '1': 'ಒಂದು', '2': 'ಎರಡು', '3': 'ಮೂರು', '4': 'ನಾಲ್ಕು',
    '5': 'ಐದು', '6': 'ಆರು', '7': 'ಏಳು', '8': 'ಎಂಟು', '9': 'ಒಂಬತ್ತು',
    '10': 'ಹತ್ತು', '11': 'ಹನ್ನೊಂದು', '12': 'ಹನ್ನೆರಡು', '13': 'ಹದಿಮೂರು', '14': 'ಹದಿನಾಲ್ಕು',
    '15': 'ಹದಿನೈದು', '16': 'ಹದಿನಾರು', '17': 'ಹದಿನೇಳು', '18': 'ಹದಿನೆಂಟು', '19': 'ಹತ್ತೊಂಬತ್ತು',
    '20': 'ಇಪ್ಪತ್ತು', '21': 'ಇಪ್ಪತ್ತೊಂದು', '22': 'ಇಪ್ಪತ್ತೆರಡು', '23': 'ಇಪ್ಪತ್ತ್ಮೂರು', '24': 'ಇಪ್ಪತ್ತ್ನಾಲ್ಕು',
    '25': 'ಇಪ್ಪತ್ತೈದು', '26': 'ಇಪ್ಪತ್ತಾರು', '27': 'ಇಪ್ಪತ್ತೇಳು', '28': 'ಇಪ್ಪತ್ತೆಂಟು', '29': 'ಇಪ್ಪತ್ತೊಂಬತ್ತು',
    '30': 'ಮೂವತ್ತು', '31': 'ಮೂವತ್ತೊಂದು', '32': 'ಮೂವತ್ತೆರಡು', '33': 'ಮೂವತ್ತ್ಮೂರು', '34': 'ಮೂವತ್ತ್ನಾಲ್ಕು',
    '35': 'ಮೂವತ್ತೈದು', '36': 'ಮೂವತ್ತಾರು', '37': 'ಮೂವತ್ತೇಳು', '38': 'ಮೂವತ್ತೆಂಟು', '39': 'ಮೂವತ್ತೊಂಬತ್ತು',
    '40': 'ನಲವತ್ತು', '41': 'ನಲವತ್ತೊಂದು', '42': 'ನಲವತ್ತೆರಡು', '43': 'ನಲವತ್ತ್ಮೂರು', '44': 'ನಲವತ್ತ್ನಾಲ್ಕು',
    '45': 'ನಲವತ್ತೈದು', '46': 'ನಲವತ್ತಾರು', '47': 'ನಲವತ್ತೇಳು', '48': 'ನಲವತ್ತೆಂಟು', '49': 'ನಲವತ್ತೊಂಬತ್ತು',
    '50': 'ಐವತ್ತು', '51': 'ಐವತ್ತೊಂದು', '52': 'ಐವತ್ತೆರಡು', '53': 'ಐವತ್ತ್ಮೂರು', '54': 'ಐವತ್ತ್ನಾಲ್ಕು',
    '55': 'ಐವತ್ತೈದು', '56': 'ಐವತ್ತಾರು', '57': 'ಐವತ್ತೇಳು', '58': 'ಐವತ್ತೆಂಟು', '59': 'ಐವತ್ತೊಂಬತ್ತು',
    '60': 'ಅರವತ್ತು', '61': 'ಅರವತ್ತೊಂದು', '62': 'ಅರವತ್ತೆರಡು', '63': 'ಅರವತ್ತ್ಮೂರು', '64': 'ಅರವತ್ತ್ನಾಲ್ಕು',
    '65': 'ಅರವತ್ತೈದು', '66': 'ಅರವತ್ತಾರು', '67': 'ಅರವತ್ತೇಳು', '68': 'ಅರವತ್ತೆಂಟು', '69': 'ಅರವತ್ತೊಂಬತ್ತು',
    '70': 'ಎಪ್ಪತ್ತು', '71': 'ಎಪ್ಪತ್ತೊಂದು', '72': 'ಎಪ್ಪತ್ತೆರಡು', '73': 'ಎಪ್ಪತ್ತ್ಮೂರು', '74': 'ಎಪ್ಪತ್ತ್ನಾಲ್ಕು',
    '75': 'ಎಪ್ಪತ್ತೈದು', '76': 'ಎಪ್ಪತ್ತಾರು', '77': 'ಎಪ್ಪತ್ತೇಳು', '78': 'ಎಪ್ಪತ್ತೆಂಟು', '79': 'ಎಪ್ಪತ್ತೊಂಬತ್ತು',
    '80': 'ಎಂಬತ್ತು', '81': 'ಎಂಬತ್ತೊಂದು', '82': 'ಎಂಬತ್ತೆರಡು', '83': 'ಎಂಬತ್ತ್ಮೂರು', '84': 'ಎಂಬತ್ತ್ನಾಲ್ಕು',
    '85': 'ಎಂಬತ್ತೈದು', '86': 'ಎಂಬತ್ತಾರು', '87': 'ಎಂಬತ್ತೇಳು', '88': 'ಎಂಬತ್ತೆಂಟು', '89': 'ಎಂಬತ್ತೊಂಬತ್ತು',
    '90': 'ತೊಂಬತ್ತು', '91': 'ತೊಂಬತ್ತೊಂದು', '92': 'ತೊಂಬತ್ತೆರಡು', '93': 'ತೊಂಬತ್ತ್ಮೂರು', '94': 'ತೊಂಬತ್ತ್ನಾಲ್ಕು',
    '95': 'ತೊಂಬತ್ತೈದು', '96': 'ತೊಂಬತ್ತಾರು', '97': 'ತೊಂಬತ್ತೇಳು', '98': 'ತೊಂಬತ್ತೆಂಟು', '99': 'ತೊಂಬತ್ತೊಂಬತ್ತು',
    '100': 'ನೂರು',
    '200': 'ಇನ್ನೂರು', '300': 'ಮೂನೂರು', '400': 'ನಾನೂರು', '500': 'ಐನೂರು',
    '600': 'ಆರುನೂರು', '700': 'ಏಳುನೂರು', '800': 'ಎಂಟುನೂರು', '900': 'ಒಂಬತ್ತುನೂರು',
    '1000': 'ಸಾವಿರ'
}

# Malayalam number words (complete set 0-99)
malayalam_numbers = {
    '0': 'പൂജ്യം', '1': 'ഒന്ന്', '2': 'രണ്ട്', '3': 'മൂന്ന്', '4': 'നാല്',
    '5': 'അഞ്ച്', '6': 'ആറ്', '7': 'ഏഴ്', '8': 'എട്ട്', '9': 'ഒമ്പത്',
    '10': 'പത്ത്', '11': 'പതിനൊന്ന്', '12': 'പന്ത്രണ്ട്', '13': 'പതിമൂന്ന്', '14': 'പതിനാല്',
    '15': 'പതിനഞ്ച്', '16': 'പതിനാറ്', '17': 'പതിനേഴ്', '18': 'പതിനെട്ട്', '19': 'പത്തൊമ്പത്',
    '20': 'ഇരുപത്', '21': 'ഇരുപത്തിയൊന്ന്', '22': 'ഇരുപത്തിരണ്ട്', '23': 'ഇരുപത്തിമൂന്ന്', '24': 'ഇരുപത്തിനാല്',
    '25': 'ഇരുപത്തിയഞ്ച്', '26': 'ഇരുപത്തിയാറ്', '27': 'ഇരുപത്തിയേഴ്', '28': 'ഇരുപത്തിയെട്ട്', '29': 'ഇരുപത്തൊമ്പത്',
    '30': 'മുപ്പത്', '31': 'മുപ്പത്തിയൊന്ന്', '32': 'മുപ്പത്തിരണ്ട്', '33': 'മുപ്പത്തിമൂന്ന്', '34': 'മുപ്പത്തിനാല്',
    '35': 'മുപ്പത്തിയഞ്ച്', '36': 'മുപ്പത്തിയാറ്', '37': 'മുപ്പത്തിയേഴ്', '38': 'മുപ്പത്തിയെട്ട്', '39': 'മുപ്പത്തൊമ്പത്',
    '40': 'നാല്പത്', '41': 'നാല്പത്തിയൊന്ന്', '42': 'നാല്പത്തിരണ്ട്', '43': 'നാല്പത്തിമൂന്ന്', '44': 'നാല്പത്തിനാല്',
    '45': 'നാല്പത്തിയഞ്ച്', '46': 'നാല്പത്തിയാറ്', '47': 'നാല്പത്തിയേഴ്', '48': 'നാല്പത്തിയെട്ട്', '49': 'നാല്പത്തൊമ്പത്',
    '50': 'അമ്പത്', '51': 'അമ്പത്തിയൊന്ന്', '52': 'അമ്പത്തിരണ്ട്', '53': 'അമ്പത്തിമൂന്ന്', '54': 'അമ്പത്തിനാല്',
    '55': 'അമ്പത്തിയഞ്ച്', '56': 'അമ്പത്തിയാറ്', '57': 'അമ്പത്തിയേഴ്', '58': 'അമ്പത്തിയെട്ട്', '59': 'അമ്പത്തൊമ്പത്',
    '60': 'അറുപത്', '61': 'അറുപത്തിയൊന്ന്', '62': 'അറുപത്തിരണ്ട്', '63': 'അറുപത്തിമൂന്ന്', '64': 'അറുപത്തിനാല്',
    '65': 'അറുപത്തിയഞ്ച്', '66': 'അറുപത്തിയാറ്', '67': 'അറുപത്തിയേഴ്', '68': 'അറുപത്തിയെട്ട്', '69': 'അറുപത്തൊമ്പത്',
    '70': 'എഴുപത്', '71': 'എഴുപത്തിയൊന്ന്', '72': 'എഴുപത്തിരണ്ട്', '73': 'എഴുപത്തിമൂന്ന്', '74': 'എഴുപത്തിനാല്',
    '75': 'എഴുപത്തിയഞ്ച്', '76': 'എഴുപത്തിയാറ്', '77': 'എഴുപത്തിയേഴ്', '78': 'എഴുപത്തിയെട്ട്', '79': 'എഴുപത്തൊമ്പത്',
    '80': 'എണ്‍പത്', '81': 'എണ്‍പത്തിയൊന്ന്', '82': 'എണ്‍പത്തിരണ്ട്', '83': 'എണ്‍പത്തിമൂന്ന്', '84': 'എണ്‍പത്തിനാല്',
    '85': 'എണ്‍പത്തിയഞ്ച്', '86': 'എണ്‍പത്തിയാറ്', '87': 'എണ്‍പത്തിയേഴ്', '88': 'എണ്‍പത്തിയെട്ട്', '89': 'എണ്‍പത്തൊമ്പത്',
    '90': 'തൊണ്ണൂറ്', '91': 'തൊണ്ണൂറ്റിയൊന്ന്', '92': 'തൊണ്ണൂറ്റിരണ്ട്', '93': 'തൊണ്ണൂറ്റിമൂന്ന്', '94': 'തൊണ്ണൂറ്റിനാല്',
    '95': 'തൊണ്ണൂറ്റിയഞ്ച്', '96': 'തൊണ്ണൂറ്റിയാറ്', '97': 'തൊണ്ണൂറ്റിയേഴ്', '98': 'തൊണ്ണൂറ്റിയെട്ട്', '99': 'തൊണ്ണൂറ്റൊമ്പത്',
    '100': 'നൂറ്',
    '200': 'ഇരുനൂറ്', '300': 'മുന്നൂറ്', '400': 'നാനൂറ്', '500': 'അഞ്ഞൂറ്',
    '600': 'അറുനൂറ്', '700': 'എഴുനൂറ്', '800': 'എണ്ണൂറ്', '900': 'തൊള്ളായിരം',
    '1000': 'ആയിരം'
}

# Marathi number words (similar to Hindi due to Devanagari script)
marathi_numbers = hindi_numbers.copy()  # Marathi uses similar number words as Hindi

# Nepali number words (similar to Hindi due to Devanagari script)
nepali_numbers = hindi_numbers.copy()  # Nepali uses similar number words as Hindi

# Odia number words (complete set 0-99)
odia_numbers = {
    '0': 'ଶୂନ୍ୟ', '1': 'ଏକ', '2': 'ଦୁଇ', '3': 'ତିନି', '4': 'ଚାରି',
    '5': 'ପାଞ୍ଚ', '6': 'ଛଅ', '7': 'ସାତ', '8': 'ଆଠ', '9': 'ନଅ',
    '10': 'ଦଶ', '11': 'ଏଗାର', '12': 'ବାର', '13': 'ତେର', '14': 'ଚଉଦ',
    '15': 'ପନ୍ଦର', '16': 'ଷୋଳ', '17': 'ସତର', '18': 'ଅଠର', '19': 'ଉଣେଇଶ',
    '20': 'କୋଡିଏ', '21': 'ଏକୋଇଶ', '22': 'ବାଇଶ', '23': 'ତେଇଶ', '24': 'ଚବିଶ',
    '25': 'ପଚିଶ', '26': 'ଛବିଶ', '27': 'ସତାଇଶ', '28': 'ଅଠାଇଶ', '29': 'ଅଣତିରିଶ',
    '30': 'ତିରିଶ', '31': 'ଏକତିରିଶ', '32': 'ବତିରିଶ', '33': 'ତେତିରିଶ', '34': 'ଚଉତିରିଶ',
    '35': 'ପଞ୍ଚତିରିଶ', '36': 'ଛତିରିଶ', '37': 'ସଇତିରିଶ', '38': 'ଅଠତିରିଶ', '39': 'ଅଣଚାଳିଶ',
    '40': 'ଚାଳିଶ', '41': 'ଏକଚାଳିଶ', '42': 'ବିଚାଳିଶ', '43': 'ତେଚାଳିଶ', '44': 'ଚଉଚାଳିଶ',
    '45': 'ପଞ୍ଚଚାଳିଶ', '46': 'ଛଚାଳିଶ', '47': 'ସଇଚାଳିଶ', '48': 'ଅଠଚାଳିଶ', '49': 'ଅଣପଚାଶ',
    '50': 'ପଚାଶ', '51': 'ଏକାବନ', '52': 'ବାବନ', '53': 'ତେପନ', '54': 'ଚଉବନ',
    '55': 'ପଞ୍ଚପନ', '56': 'ଛପନ', '57': 'ସତାବନ', '58': 'ଅଠାବନ', '59': 'ଅଣଷାଠିଏ',
    '60': 'ଷାଠିଏ', '61': 'ଏକଷଠି', '62': 'ବାଷଠି', '63': 'ତେଷଠି', '64': 'ଚଉଷଠି',
    '65': 'ପଞ୍ଚଷଠି', '66': 'ଛଷଠି', '67': 'ସଇଷଠି', '68': 'ଅଠଷଠି', '69': 'ଅଣସତୁରି',
    '70': 'ସତୁରି', '71': 'ଏକସତୁରି', '72': 'ବାସତୁରି', '73': 'ତେସତୁରି', '74': 'ଚଉସତୁରି',
    '75': 'ପଞ୍ଚସତୁରି', '76': 'ଛସତୁରି', '77': 'ସଇସତୁରି', '78': 'ଅଠସତୁରି', '79': 'ଅଣଅଶୀ',
    '80': 'ଅଶୀ', '81': 'ଏକାଶୀ', '82': 'ବିରାଶୀ', '83': 'ତିରାଶୀ', '84': 'ଚଉରାଶୀ',
    '85': 'ପଞ୍ଚାଶୀ', '86': 'ଛିଆଶୀ', '87': 'ସଇଆଶୀ', '88': 'ଅଠଆଶୀ', '89': 'ଅଣନବେ',
    '90': 'ନବେ', '91': 'ଏକାନବେ', '92': 'ବିରାନବେ', '93': 'ତିରାନବେ', '94': 'ଚଉରାନବେ',
    '95': 'ପଞ୍ଚାନବେ', '96': 'ଛିଆନବେ', '97': 'ସଇଆନବେ', '98': 'ଅଠଆନବେ', '99': 'ନିରାନବେ',
    '100': 'ଏକଶ',
    '200': 'ଦୁଇଶ', '300': 'ତିନିଶ', '400': 'ଚାରିଶ', '500': 'ପାଞ୍ଚଶ',
    '600': 'ଛଅଶ', '700': 'ସାତଶ', '800': 'ଆଠଶ', '900': 'ନଅଶ',
    '1000': 'ହଜାର'
}

# Punjabi number words (complete set 0-99)
punjabi_numbers = {
    '0': 'ਸਿਫ਼ਰ', '1': 'ਇੱਕ', '2': 'ਦੋ', '3': 'ਤਿੰਨ', '4': 'ਚਾਰ',
    '5': 'ਪੰਜ', '6': 'ਛੇ', '7': 'ਸੱਤ', '8': 'ਅੱਠ', '9': 'ਨੌਂ',
    '10': 'ਦਸ', '11': 'ਗਿਆਰਾਂ', '12': 'ਬਾਰਾਂ', '13': 'ਤੇਰਾਂ', '14': 'ਚੌਦਾਂ',
    '15': 'ਪੰਦਰਾਂ', '16': 'ਸੋਲਾਂ', '17': 'ਸਤਾਰਾਂ', '18': 'ਅਠਾਰਾਂ', '19': 'ਉਨੀਂ',
    '20': 'ਵੀਹ', '21': 'ਇੱਕੀ', '22': 'ਬਾਈ', '23': 'ਤੇਈ', '24': 'ਚੌਵੀ',
    '25': 'ਪੱਚੀ', '26': 'ਛਬੀ', '27': 'ਸਤਾਈ', '28': 'ਅਠਾਈ', '29': 'ਉਣੱਤੀ',
    '30': 'ਤੀਹ', '31': 'ਇਕੱਤੀ', '32': 'ਬੱਤੀ', '33': 'ਤੇਤੀ', '34': 'ਚੌਂਤੀ',
    '35': 'ਪੈਂਤੀ', '36': 'ਛੱਤੀ', '37': 'ਸੈਂਤੀ', '38': 'ਅੜਤੀ', '39': 'ਉਣਤਾਲੀ',
    '40': 'ਚਾਲੀ', '41': 'ਇਕਤਾਲੀ', '42': 'ਬਿਆਲੀ', '43': 'ਤਿਰਤਾਲੀ', '44': 'ਚਵਾਲੀ',
    '45': 'ਪੰਤਾਲੀ', '46': 'ਛਿਆਲੀ', '47': 'ਸੈਂਤਾਲੀ', '48': 'ਅੜਤਾਲੀ', '49': 'ਉਣਜਾਹ',
    '50': 'ਪੰਜਾਹ', '51': 'ਇਕਵੰਜਾ', '52': 'ਬਵੰਜਾ', '53': 'ਤਿਰਵੰਜਾ', '54': 'ਚਉਵੰਜਾ',
    '55': 'ਪਚਵੰਜਾ', '56': 'ਛਪੰਜਾ', '57': 'ਸਤਵੰਜਾ', '58': 'ਅਠਵੰਜਾ', '59': 'ਉਣਸੱਠ',
    '60': 'ਸੱਠ', '61': 'ਇਕਸਠ', '62': 'ਬਾਸਠ', '63': 'ਤਿਰਸਠ', '64': 'ਚੌਂਸਠ',
    '65': 'ਪੈਂਸਠ', '66': 'ਛਿਆਸਠ', '67': 'ਸਤਸਠ', '68': 'ਅੜਸਠ', '69': 'ਉਣਸੱਤਰ',
    '70': 'ਸੱਤਰ', '71': 'ਇਕਹੱਤਰ', '72': 'ਬਹੱਤਰ', '73': 'ਤਿਹੱਤਰ', '74': 'ਚੌਂਹੱਤਰ',
    '75': 'ਪੰਜਹੱਤਰ', '76': 'ਛਿਆਹੱਤਰ', '77': 'ਸਤਹੱਤਰ', '78': 'ਅਠਹੱਤਰ', '79': 'ਉਣਅੱਸੀ',
    '80': 'ਅੱਸੀ', '81': 'ਇਕਿਆਸੀ', '82': 'ਬਿਆਸੀ', '83': 'ਤਿਰਿਆਸੀ', '84': 'ਚੌਰਾਸੀ',
    '85': 'ਪਚਾਸੀ', '86': 'ਛਿਆਸੀ', '87': 'ਸਤਾਸੀ', '88': 'ਅਠਾਸੀ', '89': 'ਉਣਨੱਬੇ',
    '90': 'ਨੱਬੇ', '91': 'ਇਕਾਨੱਬੇ', '92': 'ਬਾਨੱਬੇ', '93': 'ਤਿਰਾਨੱਬੇ', '94': 'ਚੌਰਾਨੱਬੇ',
    '95': 'ਪਚਾਨੱਬੇ', '96': 'ਛਿਆਨੱਬੇ', '97': 'ਸਤਾਨੱਬੇ', '98': 'ਅਠਾਨੱਬੇ', '99': 'ਨਿਨਾਨੱਬੇ',
    '100': 'ਸੌ',
    '200': 'ਦੋ ਸੌ', '300': 'ਤਿੰਨ ਸੌ', '400': 'ਚਾਰ ਸੌ', '500': 'ਪੰਜ ਸੌ',
    '600': 'ਛੇ ਸੌ', '700': 'ਸੱਤ ਸੌ', '800': 'ਅੱਠ ਸੌ', '900': 'ਨੌਂ ਸੌ',
    '1000': 'ਹਜ਼ਾਰ'
}

# Tamil number words (complete set 0-99)
tamil_numbers = {
    '0': 'பூஜ்யம்', '1': 'ஒன்று', '2': 'இரண்டு', '3': 'மூன்று', '4': 'நான்கு',
    '5': 'ஐந்து', '6': 'ஆறு', '7': 'ஏழு', '8': 'எட்டு', '9': 'ஒன்பது',
    '10': 'பத்து', '11': 'பதினொன்று', '12': 'பன்னிரண்டு', '13': 'பதிமூன்று', '14': 'பதினான்கு',
    '15': 'பதினைந்து', '16': 'பதினாறு', '17': 'பதினேழு', '18': 'பதினெட்டு', '19': 'பத்தொன்பது',
    '20': 'இருபது', '21': 'இருபத்தியொன்று', '22': 'இருபத்தியிரண்டு', '23': 'இருபத்திமூன்று', '24': 'இருபத்தினான்கு',
    '25': 'இருபத்தியைந்து', '26': 'இருபத்தியாறு', '27': 'இருபத்தியேழு', '28': 'இருபத்தியெட்டு', '29': 'இருபத்தொன்பது',
    '30': 'முப்பது', '31': 'முப்பத்தியொன்று', '32': 'முப்பத்தியிரண்டு', '33': 'முப்பத்திமூன்று', '34': 'முப்பத்தினான்கு',
    '35': 'முப்பத்தியைந்து', '36': 'முப்பத்தியாறு', '37': 'முப்பத்தியேழு', '38': 'முப்பத்தியெட்டு', '39': 'முப்பத்தொன்பது',
    '40': 'நாற்பது', '41': 'நாற்பத்தியொன்று', '42': 'நாற்பத்தியிரண்டு', '43': 'நாற்பத்திமூன்று', '44': 'நாற்பத்தினான்கு',
    '45': 'நாற்பத்தியைந்து', '46': 'நாற்பத்தியாறு', '47': 'நாற்பத்தியேழு', '48': 'நாற்பத்தியெட்டு', '49': 'நாற்பத்தொன்பது',
    '50': 'ஐம்பது', '51': 'ஐம்பத்தியொன்று', '52': 'ஐம்பத்தியிரண்டு', '53': 'ஐம்பத்திமூன்று', '54': 'ஐம்பத்தினான்கு',
    '55': 'ஐம்பத்தியைந்து', '56': 'ஐம்பத்தியாறு', '57': 'ஐம்பத்தியேழு', '58': 'ஐம்பத்தியெட்டு', '59': 'ஐம்பத்தொன்பது',
    '60': 'அறுபது', '61': 'அறுபத்தியொன்று', '62': 'அறுபத்தியிரண்டு', '63': 'அறுபத்திமூன்று', '64': 'அறுபத்தினான்கு',
    '65': 'அறுபத்தியைந்து', '66': 'அறுபத்தியாறு', '67': 'அறுபத்தியேழு', '68': 'அறுபத்தியெட்டு', '69': 'அறுபத்தொன்பது',
    '70': 'எழுபது', '71': 'எழுபத்தியொன்று', '72': 'எழுபத்தியிரண்டு', '73': 'எழுபத்திமூன்று', '74': 'எழுபத்தினான்கு',
    '75': 'எழுபத்தியைந்து', '76': 'எழுபத்தியாறு', '77': 'எழுபத்தியேழு', '78': 'எழுபத்தியெட்டு', '79': 'எழுபத்தொன்பது',
    '80': 'எண்பது', '81': 'எண்பத்தியொன்று', '82': 'எண்பத்தியிரண்டு', '83': 'எண்பத்திமூன்று', '84': 'எண்பத்தினான்கு',
    '85': 'எண்பத்தியைந்து', '86': 'எண்பத்தியாறு', '87': 'எண்பத்தியேழு', '88': 'எண்பத்தியெட்டு', '89': 'எண்பத்தொன்பது',
    '90': 'தொண்ணூறு', '91': 'தொண்ணூற்றியொன்று', '92': 'தொண்ணூற்றியிரண்டு', '93': 'தொண்ணூற்றிமூன்று', '94': 'தொண்ணூற்றினான்கு',
    '95': 'தொண்ணூற்றியைந்து', '96': 'தொண்ணூற்றியாறு', '97': 'தொண்ணூற்றியேழு', '98': 'தொண்ணூற்றியெட்டு', '99': 'தொண்ணூற்றொன்பது',
    '100': 'நூறு',
    '200': 'இருநூறு', '300': 'முன்னூறு', '400': 'நானூறு', '500': 'ஐநூறு',
    '600': 'அறுநூறு', '700': 'எழுநூறு', '800': 'எண்ணூறு', '900': 'தொள்ளாயிரம்',
    '1000': 'ஆயிரம்'
}

# Telugu number words (complete set 0-99)
telugu_numbers = {
    '0': 'శూన్యం', '1': 'ఒకటి', '2': 'రెండు', '3': 'మూడు', '4': 'నాలుగు',
    '5': 'ఐదు', '6': 'ఆరు', '7': 'ఏడు', '8': 'ఎనిమిది', '9': 'తొమ్మిది',
    '10': 'పది', '11': 'పదకొండు', '12': 'పన్నెండు', '13': 'పదమూడు', '14': 'పద్నాలుగు',
    '15': 'పదిహేను', '16': 'పదహారు', '17': 'పదిహేడు', '18': 'పద్దెనిమిది', '19': 'పందొమ్మిది',
    '20': 'ఇరవై', '21': 'ఇరవైయొకటి', '22': 'ఇరవైరెండు', '23': 'ఇరవైమూడు', '24': 'ఇరవైనాలుగు',
    '25': 'ఇరవైయైదు', '26': 'ఇరవైయారు', '27': 'ఇరవైయేడు', '28': 'ఇరవైయెనిమిది', '29': 'ఇరవైతొమ్మిది',
    '30': 'ముప్పై', '31': 'ముప్పైయొకటి', '32': 'ముప్పైరెండు', '33': 'ముప్పైమూడు', '34': 'ముప్పైనాలుగు',
    '35': 'ముప్పైయైదు', '36': 'ముప్పైయారు', '37': 'ముప్పైయేడు', '38': 'ముప్పైయెనిమిది', '39': 'ముప్పైతొమ్మిది',
    '40': 'నలభై', '41': 'నలభైయొకటి', '42': 'నలభైరెండు', '43': 'నలభైమూడు', '44': 'నలభైనాలుగు',
    '45': 'నలభైయైదు', '46': 'నలభైయారు', '47': 'నలభైయేడు', '48': 'నలభైయెనిమిది', '49': 'నలభైతొమ్మిది',
    '50': 'యాభై', '51': 'యాభైయొకటి', '52': 'యాభైరెండు', '53': 'యాభైమూడు', '54': 'యాభైనాలుగు',
    '55': 'యాభైయైదు', '56': 'యాభైయారు', '57': 'యాభైయేడు', '58': 'యాభైయెనిమిది', '59': 'యాభైతొమ్మిది',
    '60': 'అరవై', '61': 'అరవైయొకటి', '62': 'అరవైరెండు', '63': 'అరవైమూడు', '64': 'అరవైనాలుగు',
    '65': 'అరవైయైదు', '66': 'అరవైయారు', '67': 'అరవైయేడు', '68': 'అరవైయెనిమిది', '69': 'అరవైతొమ్మిది',
    '70': 'డెబ్బై', '71': 'డెబ్బైయొకటి', '72': 'డెబ్బైరెండు', '73': 'డెబ్బైమూడు', '74': 'డెబ్బైనాలుగు',
    '75': 'డెబ్బైయైదు', '76': 'డెబ్బైయారు', '77': 'డెబ్బైయేడు', '78': 'డెబ్బైయెనిమిది', '79': 'డెబ్బైతొమ్మిది',
    '80': 'ఎనభై', '81': 'ఎనభైయొకటి', '82': 'ఎనభైరెండు', '83': 'ఎనభైమూడు', '84': 'ఎనభైనాలుగు',
    '85': 'ఎనభైయైదు', '86': 'ఎనభైయారు', '87': 'ఎనభైయేడు', '88': 'ఎనభైయెనిమిది', '89': 'ఎనభైతొమ్మిది',
    '90': 'తొంభై', '91': 'తొంభైయొకటి', '92': 'తొంభైరెండు', '93': 'తొంభైమూడు', '94': 'తొంభైనాలుగు',
    '95': 'తొంభైయైదు', '96': 'తొంభైయారు', '97': 'తొంభైయేడు', '98': 'తొంభైయెనిమిది', '99': 'తొంభైతొమ్మిది',
    '100': 'వంద',
    '200': 'రెండువందలు', '300': 'మూడువందలు', '400': 'నాలుగువందలు', '500': 'ఐదువందలు',
    '600': 'ఆరువందలు', '700': 'ఏడువందలు', '800': 'ఎనిమిదివందలు', '900': 'తొమ్మిదివందలు',
    '1000': 'వేయి'
}

# Dictionary mapping language codes to their number dictionaries
language_numbers = {
    'hi': hindi_numbers,
    'bn': bengali_numbers,
    'gu': gujarati_numbers,
    'kn': kannada_numbers,
    'ml': malayalam_numbers,
    'mr': marathi_numbers,
    'ne': nepali_numbers,
    'or': odia_numbers,
    'pa': punjabi_numbers,
    'ta': tamil_numbers,
    'te': telugu_numbers,
    'en': {}  # English uses inflect library
}

# Local numeral to Latin numeral conversion mappings
local_numeral_mappings = {
    # Devanagari numerals (Hindi/Marathi/Nepali) ०१२३४५६७८९
    'devanagari': {
        '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
        '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
    },
    # Bengali numerals ০১২৩৪৫৬৭৮৯
    'bengali': {
        '০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4',
        '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9'
    },
    # Gujarati numerals ૦૧૨૩૪૫૬૭૮૯
    'gujarati': {
        '૦': '0', '૧': '1', '૨': '2', '૩': '3', '૪': '4',
        '૫': '5', '૬': '6', '૭': '7', '૮': '8', '૯': '9'
    },
    # Gurmukhi numerals (Punjabi) ੦੧੨੩੪੫੬੭੮੯
    'gurmukhi': {
        '੦': '0', '੧': '1', '੨': '2', '੩': '3', '੪': '4',
        '੫': '5', '੬': '6', '੭': '7', '੮': '8', '੯': '9'
    },
    # Oriya numerals (Odia) ୦୧୨୩୪୫୬୭୮୯
    'oriya': {
        '୦': '0', '୧': '1', '୨': '2', '୩': '3', '୪': '4',
        '୫': '5', '୬': '6', '୭': '7', '୮': '8', '୯': '9'
    },
    # Tamil numerals ௦௧௨௩௪௫௬௭௮௯
    'tamil': {
        '௦': '0', '௧': '1', '௨': '2', '௩': '3', '௪': '4',
        '௫': '5', '௬': '6', '௭': '7', '௮': '8', '௯': '9'
    },
    # Telugu numerals ౦౧౨౩౪౫౬౭౮౯
    'telugu': {
        '౦': '0', '౧': '1', '౨': '2', '౩': '3', '౪': '4',
        '౫': '5', '౬': '6', '౭': '7', '౮': '8', '౯': '9'
    },
    # Kannada numerals ೦೧೨೩೪೫೬೭೮೯
    'kannada': {
        '೦': '0', '೧': '1', '೨': '2', '೩': '3', '೪': '4',
        '೫': '5', '೬': '6', '೭': '7', '೮': '8', '೯': '9'
    },
    # Malayalam numerals ൦൧൨൩൪൫൬൭൮൯
    'malayalam': {
        '൦': '0', '൧': '1', '൨': '2', '൩': '3', '൪': '4',
        '൫': '5', '൬': '6', '൭': '7', '൮': '8', '൯': '9'
    }
}

# Create phonemizers for all supported languages
# Initialize phonemizers with error handling
phonemizers = {}

def initialize_phonemizers():
    """
    Initialize phonemizers for all supported languages.
    
    Creates espeak-based phonemizers for 12 languages with error handling.
    If espeak is not installed, phonemizers will be None and words will be 
    returned as-is during phonemization.
    
    Supported languages: Bengali, English, Gujarati, Hindi, Kannada, Malayalam,
    Marathi, Nepali, Odia, Punjabi, Tamil, Telugu
    """
    languages = ['hi', 'en-gb', 'bn', 'gu', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'ta', 'te']
    
    for lang in languages:
        try:
            phonemizers[lang] = create_phonemizer(language=lang, punctuation_marks=_DEF_PUNCS)
            logger.info(f"Successfully initialized phonemizer for {lang}")
        except Exception as e:
            logger.warning(f"Failed to initialize phonemizer for {lang}: {e}")
            phonemizers[lang] = None

# Initialize phonemizers
initialize_phonemizers()

# Phonemizers are now stored in the 'phonemizers' dictionary

def process_sentence_with_spoken_currency(sentence):
    """
    Processes a sentence to format currency strings by placing the spoken name of the currency
    after the numeric value, while leaving other text unchanged.
    Supports multilingual currency names based on context detection.

    Args:
        sentence (str): A sentence containing a mix of currency strings and other text.

    Returns:
        str: The sentence with currency strings formatted, appending spoken names after numbers.
    """
    # Handle percentage first
    sentence = sentence.replace("%", " percent ")
    
    # Multilingual currency name mappings
    rupee_names = {
        'en': 'rupees',
        'hi': 'रुपए', 
        'bn': 'টাকা',
        'gu': 'રૂપિયા',
        'kn': 'ರೂಪಾಯಿಗಳು',
        'ml': 'രൂപ',
        'mr': 'रुपये',
        'ne': 'रुपैया',
        'or': 'ଟଙ୍କା',
        'pa': 'ਰੁਪਏ',
        'ta': 'ரூபாய்',
        'te': 'రూపాయలు'
    }
    
    currency_names = {
        "₹": rupee_names,
        "Rs": rupee_names,
        "Rs.": rupee_names,
        "INR": rupee_names,
        "$": {
            'en': 'dollars',
            'hi': 'डॉलर',
            'bn': 'ডলার',
            'gu': 'ડોલર',
            'kn': 'ಡಾಲರ್',
            'ml': 'ഡോളർ',
            'mr': 'डॉलर',
            'ne': 'डलर',
            'or': 'ଡଲାର',
            'pa': 'ਡਾਲਰ',
            'ta': 'டாலர்',
            'te': 'డాలర్'
        },
        "€": {
            'en': 'euros',
            'hi': 'यूरो',
            'bn': 'ইউরো',
            'gu': 'યુરો',
            'kn': 'ಯುರೋ',
            'ml': 'യൂറോ',
            'mr': 'युरो',
            'ne': 'युरो',
            'or': 'ୟୁରୋ',
            'pa': 'ਯੂਰੋ',
            'ta': 'யூரோ',
            'te': 'యూరో'
        },
        "£": {
            'en': 'pounds',
            'hi': 'पाउंड',
            'bn': 'পাউন্ড',
            'gu': 'પાઉન્ડ',
            'kn': 'ಪೌಂಡ್',
            'ml': 'പൗണ്ട്',
            'mr': 'पाउंड',
            'ne': 'पाउन्ड',
            'or': 'ପାଉଣ୍ଡ',
            'pa': 'ਪਾਉਂਡ',
            'ta': 'பவுண்ட்',
            'te': 'పౌండ్'
        },
        "¥": {
            'en': 'yen',
            'hi': 'येन',
            'bn': 'ইয়েন',
            'gu': 'યેન',
            'kn': 'ಯೆನ್',
            'ml': 'യെൻ',
            'mr': 'येन',
            'ne': 'येन',
            'or': 'ୟେନ',
            'pa': 'ਯੇਨ',
            'ta': 'யென்',
            'te': 'యెన్'
        },
        "元": {
            'en': 'yuan',
            'hi': 'युआन',
            'bn': 'ইউয়ান',
            'gu': 'યુઆન',
            'kn': 'ಯುವಾನ್',
            'ml': 'യുവാൻ',
            'mr': 'युआन',
            'ne': 'युआन',
            'or': 'ୟୁଆନ',
            'pa': 'ਯੁਆਨ',
            'ta': 'யுவான்',
            'te': 'యువాన్'
        }
    }

    # Enhanced regex to handle currency symbols before or after numbers, with optional spaces
    # Also includes common rupee abbreviations like "Rs", "Rs.", "INR"
    currency_pattern = re.compile(r"([₹$€£¥元]|Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)")
    
    # Also handle numbers followed by currency symbols (like "100₹", "100 Rs", "100 INR")
    # More precise pattern to avoid capturing trailing periods
    currency_after_pattern = re.compile(r"([\d,]+(?:\.\d+)?)\s*([₹$€£¥元]|Rs\.?|INR)")

    def detect_currency_context_language(text, currency_pos):
        """Detect language context around currency for appropriate translation"""
        # Look at words before and after the currency
        words = text.split()
        
        # Find the position of currency in the word list
        for i, word in enumerate(words):
            if any(symbol in word for symbol in ["₹", "$", "€", "£", "¥", "元", "Rs", "Rs.", "INR"]):
                # Check surrounding words for language detection
                context_words = []
                if i > 0:
                    context_words.extend(words[max(0, i-2):i])
                if i < len(words) - 1:
                    context_words.extend(words[i+1:min(len(words), i+3)])
                
                for context_word in context_words:
                    lang = detect_language(context_word)
                    if lang != 'unknown' and lang != 'en':
                        return lang
                break
        
        return 'en'  # Default to English

    def format_currency(match):
        """Format currency with appropriate language context"""
        # Handle both patterns: currency before number and currency after number
        if match.group(1):  # Currency symbol before number
            symbol = match.group(1)
            numeric_value = re.sub(r"[^\d\.]", "", match.group(2))
        elif match.group(3):  # Alternative pattern
            symbol = match.group(3)
            numeric_value = re.sub(r"[^\d\.]", "", match.group(4))
        else:
            return match.group(0)  # Return original if no match
        
        # Clean up symbol (remove trailing periods for display)
        clean_symbol = symbol.rstrip('.')
        
        # Detect language context
        language = detect_currency_context_language(sentence, match.start())
        
        # Get appropriate currency name - use cleaned symbol for lookup
        currency_translations = currency_names.get(clean_symbol, currency_names.get(symbol, {}))
        currency_name = currency_translations.get(language, currency_translations.get('en', symbol))
        
        return f"{numeric_value} {currency_name}"

    def format_currency_after(match):
        """Format currency when symbol comes after number"""
        numeric_value = re.sub(r"[^\d\.]", "", match.group(1))
        symbol = match.group(2)
        
        # Clean up symbol (remove trailing periods for display)
        clean_symbol = symbol.rstrip('.')
        
        # Detect language context
        language = detect_currency_context_language(sentence, match.start())
        
        # Get appropriate currency name - use cleaned symbol for lookup
        currency_translations = currency_names.get(clean_symbol, currency_names.get(symbol, {}))
        currency_name = currency_translations.get(language, currency_translations.get('en', symbol))
        
        return f"{numeric_value} {currency_name}"

    # Process currency symbols before numbers
    formatted_sentence = currency_pattern.sub(format_currency, sentence)
    
    # Process currency symbols after numbers
    formatted_sentence = currency_after_pattern.sub(format_currency_after, formatted_sentence)

    return formatted_sentence


def convert_number_to_words_generic(number, language):
    """
    Convert number to words for any supported language.
    Uses Indian numbering system (lakhs and crores) for all languages including English.
    """
    # Get the appropriate number dictionary for the language
    numbers_dict = language_numbers.get(language, hindi_numbers)
    
    # For English, use inflect only for numbers below 100,000 (before lakhs start)
    # Above that, we'll use the Indian system
    if language == 'en' and number < 100000:
        return eng_inflector.number_to_words(number).replace("-", " ")
    
    # Always check for direct dictionary lookup first (handles compound numbers properly)
    if str(number) in numbers_dict:
        return numbers_dict[str(number)]

    if number < 100:
        # If not found directly, construct from tens and units
        tens = (number // 10) * 10
        units = number % 10
        if str(tens) in numbers_dict and str(units) in numbers_dict:
            return f"{numbers_dict[str(tens)]} {numbers_dict[str(units)]}" if units != 0 else numbers_dict[str(tens)]
    
    elif number < 1000:
        hundreds_digit = number // 100
        remainder = number % 100
        
        # Handle hundreds properly
        if hundreds_digit > 0:
            if hundreds_digit == 1:
                # For 100-199, use "एक सौ", "one hundred", etc.
                if language == 'en':
                    hundreds_word = "one hundred"
                else:
                    one_word = numbers_dict.get('1', 'one')
                    hundred_word = numbers_dict.get('100', 'hundred')
                    hundreds_word = f"{one_word} {hundred_word}"
            else:
                # For 200+, use "दो सौ", "three hundred", etc.
                if language == 'en':
                    hundreds_word = f"{eng_inflector.number_to_words(hundreds_digit)} hundred"
                else:
                    hundreds_digit_word = numbers_dict.get(str(hundreds_digit), str(hundreds_digit))
                    hundred_word = numbers_dict.get('100', 'hundred')
                    hundreds_word = f"{hundreds_digit_word} {hundred_word}"
            
            if remainder != 0:
                remainder_words = convert_number_to_words_generic(remainder, language)
                return f"{hundreds_word} {remainder_words}"
            else:
                return hundreds_word
    
    elif number < 100000:
        thousands = number // 1000
        thousands_part = convert_number_to_words_generic(thousands, language)
        remainder = number % 1000
        # Use appropriate thousand word based on language
        thousand_word = numbers_dict.get('1000', 'thousand')
        thousands_word = f"{thousands_part} {thousand_word}"
        if remainder != 0:
            remainder_words = convert_number_to_words_generic(remainder, language)
            return f"{thousands_word} {remainder_words}"
        else:
            return thousands_word
    
    elif number < 10000000:  # Less than 1 crore (Indian numbering system)
        # Handle lakhs (1 lakh = 100,000)
        lakhs = number // 100000
        lakhs_part = convert_number_to_words_generic(lakhs, language)
        remainder = number % 100000
        
        # Lakh word for different languages
        lakh_words = {
            'en': 'lakh' if lakhs == 1 else 'lakhs',  # Proper pluralization for English
            'hi': 'लाख',
            'bn': 'লক্ষ',
            'gu': 'લાખ',
            'kn': 'ಲಕ್ಷ',
            'ml': 'ലക്ഷം',
            'mr': 'लाख',
            'ne': 'लाख',
            'or': 'ଲକ୍ଷ',
            'pa': 'ਲੱਖ',
            'ta': 'லட்சம்',
            'te': 'లక్ష'
        }
        lakh_word = lakh_words.get(language, 'lakh')
        lakhs_word = f"{lakhs_part} {lakh_word}"
        
        if remainder != 0:
            remainder_words = convert_number_to_words_generic(remainder, language)
            return f"{lakhs_word} {remainder_words}"
        else:
            return lakhs_word
    
    elif number < 1000000000:  # Less than 100 crore
        # Handle crores (1 crore = 10,000,000)
        crores = number // 10000000
        crores_part = convert_number_to_words_generic(crores, language)
        remainder = number % 10000000
        
        # Crore word for different languages
        crore_words = {
            'en': 'crore' if crores == 1 else 'crores',  # Proper pluralization for English
            'hi': 'करोड़',
            'bn': 'কোটি',
            'gu': 'કરોડ',
            'kn': 'ಕೋಟಿ',
            'ml': 'കോടി',
            'mr': 'कोटी',
            'ne': 'करोड',
            'or': 'କୋଟି',
            'pa': 'ਕਰੋੜ',
            'ta': 'கோடி',
            'te': 'కోటి'
        }
        crore_word = crore_words.get(language, 'crore')
        crores_word = f"{crores_part} {crore_word}"
        
        if remainder != 0:
            remainder_words = convert_number_to_words_generic(remainder, language)
            return f"{crores_word} {remainder_words}"
        else:
            return crores_word

    return str(number)

# Backward compatibility function
def convert_number_to_hindi_words(number):
    """Legacy function - use convert_number_to_words_generic instead"""
    return convert_number_to_words_generic(number, 'hi')

def detect_numeral_script(numeral_string):
    """
    Detect which script the numerals belong to
    """
    if DEVANAGARI_NUMBER_REGEX.fullmatch(numeral_string):
        return 'devanagari'
    elif BENGALI_NUMBER_REGEX.fullmatch(numeral_string):
        return 'bengali'
    elif GUJARATI_NUMBER_REGEX.fullmatch(numeral_string):
        return 'gujarati'
    elif GURMUKHI_NUMBER_REGEX.fullmatch(numeral_string):
        return 'gurmukhi'
    elif ORIYA_NUMBER_REGEX.fullmatch(numeral_string):
        return 'oriya'
    elif TAMIL_NUMBER_REGEX.fullmatch(numeral_string):
        return 'tamil'
    elif TELUGU_NUMBER_REGEX.fullmatch(numeral_string):
        return 'telugu'
    elif KANNADA_NUMBER_REGEX.fullmatch(numeral_string):
        return 'kannada'
    elif MALAYALAM_NUMBER_REGEX.fullmatch(numeral_string):
        return 'malayalam'
    elif NUMBER_REGEX.fullmatch(numeral_string):
        return 'latin'
    return 'unknown'

def convert_local_numerals_to_latin(numeral_string):
    """
    Convert local language numerals to Latin numerals
    """
    script = detect_numeral_script(numeral_string)
    
    if script == 'latin':
        return numeral_string  # Already Latin numerals
    
    if script == 'unknown':
        return numeral_string  # Can't convert, return as is
    
    mapping = local_numeral_mappings.get(script, {})
    if not mapping:
        return numeral_string
    
    # Convert each character
    converted = ''
    for char in numeral_string:
        converted += mapping.get(char, char)
    
    return converted

def get_language_from_numeral_script(script):
    """
    Map numeral script to language code
    """
    script_to_language = {
        'devanagari': 'hi',  # Default to Hindi for Devanagari
        'bengali': 'bn',
        'gujarati': 'gu',
        'gurmukhi': 'pa',
        'oriya': 'or',
        'tamil': 'ta',
        'telugu': 'te',
        'kannada': 'kn',
        'malayalam': 'ml',
        'latin': 'en'  # Default to English for Latin numerals
    }
    return script_to_language.get(script, 'en')

def spell_out_url(match):
    """
    Spells out a URL, replacing '.' with 'dot' and '/' with 'slash', among others.
    """
    url = match.group(0)
    spelled_out = url.replace('.', ' dot ').replace('/', ' slash ').replace(':', ' colon ').replace('-', ' dash ')
    return ' '.join(spelled_out.split())  # Normalize spaces

def process_websites(text):
    """
    Detects and spells out website URLs in the text.
    """
    return URL_REGEX.sub(spell_out_url, text)

def process_mathematical_equations(text, default_language='en'):
    """
    Processes mathematical equations in the text and converts them to spoken form.
    
    Args:
        text (str): Input text containing mathematical equations
        default_language (str): Default language for mathematical operators
        
    Returns:
        str: Text with mathematical equations converted to spoken form
        
    Examples:
        "2+2" -> "two plus two"
        "5×3=15" -> "five times three equals fifteen"
        "10÷2" -> "ten divided by two"
        "5×2-3=7" -> "five times two minus three equals seven"
    """
    def convert_math_expression_to_words(expression, language):
        """Convert a mathematical expression to words"""
        operators_dict = math_operators.get(language, math_operators['en'])
        
        # Split the expression into tokens (numbers and operators)
        tokens = re.findall(r'\d+(?:\.\d+)?|[+\-×*÷/=]', expression)
        result_words = []
        
        for token in tokens:
            if token.replace('.', '').isdigit():
                # It's a number
                try:
                    num_words = convert_number_to_words_generic(int(float(token)), language)
                    result_words.append(num_words)
                except (ValueError, TypeError):
                    result_words.append(token)
            else:
                # It's an operator
                operator_word = operators_dict.get(token, token)
                result_words.append(operator_word)
        
        return ' '.join(result_words)
    
    def replace_math_expression(match):
        """Replace a mathematical expression with its spoken form"""
        expression = match.group(0)
        language = default_language
        return convert_math_expression_to_words(expression, language)
    
    # Process full mathematical expressions (including complex ones like 5×2-3=7)
    text = MATH_EXPRESSION_REGEX.sub(replace_math_expression, text)
    
    return text

def detect_equation_context_language(text, equation_start_pos):
    """
    Detect the language context around a mathematical equation.
    
    Args:
        text (str): Full text containing the equation
        equation_start_pos (int): Starting position of the equation
        
    Returns:
        str: Detected language code
    """
    # Look at words before and after the equation for language detection
    before_text = text[:equation_start_pos].split()
    after_text = text[equation_start_pos:].split()
    
    # Check the last few words before the equation
    for word in reversed(before_text[-3:]):
        lang = detect_language(word)
        if lang != 'unknown' and lang != 'en':
            return lang
    
    # Check the first few words after the equation
    for word in after_text[1:4]:  # Skip the equation itself
        lang = detect_language(word)
        if lang != 'unknown' and lang != 'en':
            return lang
    
    return 'en'  # Default to English

def spell_out_uppercase_words(text):
    """
    Checks for words with all uppercase letters and splits them into individual characters with spaces.
    """
    def spell_out(word):
        # If the word is all uppercase, split into characters
        if word.isupper():
            return ' '.join(word)
        return word

    # Split the text into words, process each, and join them back
    words = text.split()
    processed_words = [spell_out(word) for word in words]
    return ' '.join(processed_words)

def number_to_words(word, language, interpret_as="cardinal"):
    """
    Convert numbers to words in the specified language
    """
    if interpret_as == "digits":
        # Return each digit individually, considering language
        digits = []
        for d in word:
            if language == 'en':
                digits.append(eng_inflector.number_to_words(d))
            else:
                numbers_dict = language_numbers.get(language, hindi_numbers)
                digits.append(numbers_dict.get(d, d))
        return " ".join(digits)
    
    if interpret_as == "time":
        # Handle both Latin and local numeral time formats
        match = TIME_REGEX.fullmatch(word) or LOCAL_TIME_REGEX.fullmatch(word)
        if match:
            hours_str, minutes_str = match.groups()
            
            # Convert local numerals to Latin if needed
            hours_latin = convert_local_numerals_to_latin(hours_str)
            minutes_latin = convert_local_numerals_to_latin(minutes_str)
            
            # If time is in local numerals, determine language from script
            if not TIME_REGEX.fullmatch(word):
                # This is a local numeral time format, determine language from script
                script = detect_numeral_script(hours_str + minutes_str)
                if script != 'unknown' and script != 'latin':
                    language = get_language_from_numeral_script(script)
            
            if language == "en":
                return f"{eng_inflector.number_to_words(hours_latin)} hours {eng_inflector.number_to_words(minutes_latin)} minutes"
            else:
                # Get time words for different languages
                time_words = {
                    'hi': ('घंटे', 'मिनट'),
                    'bn': ('ঘন্টা', 'মিনিট'),
                    'gu': ('કલાક', 'મિનિટ'),
                    'kn': ('ಗಂಟೆ', 'ನಿಮಿಷ'),
                    'ml': ('മണിക്കൂർ', 'മിനിറ്റ്'),
                    'mr': ('तास', 'मिनिट'),
                    'ne': ('घण्टा', 'मिनेट'),
                    'or': ('ଘଣ୍ଟା', 'ମିନିଟ୍'),
                    'pa': ('ਘੰਟਾ', 'ਮਿੰਟ'),
                    'ta': ('மணி', 'நிமிடம்'),
                    'te': ('గంట', 'నిମిషం')
                }
                hour_word, minute_word = time_words.get(language, ('hours', 'minutes'))
                hours_text = convert_number_to_words_generic(int(hours_latin), language)
                minutes_text = convert_number_to_words_generic(int(minutes_latin), language)
                return f"{hours_text} {hour_word} {minutes_text} {minute_word}"
    
    # Handle cardinal numbers
    try:
        number = int(word)
        return convert_number_to_words_generic(number, language)
    except ValueError:
        # If it's not a number, return as is with punctuation handling
        word = add_spaces_around_punctuation(word)
        word = word.replace(",", " , ")
        return word

def detect_language(word):
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

def add_spaces_around_punctuation(text):
    return PUNCTUATION_REGEX.sub(r' \1 ', text)

def determine_context_language(words, index):
    """
    Determine the language context for a word based on surrounding words
    """
    # Check if we're at the beginning of a sentence
    if index == 0 or (index > 0 and SENTENCE_END_REGEX.match(words[index - 1])):
        # Look at the next word to determine context
        if len(words) > index + 1:
            next_lang = detect_language(words[index + 1])
            if next_lang != 'unknown':
                return next_lang
        return 'en'  # Default to English
    
    # Check previous word
    if index > 0:
        prev_lang = detect_language(words[index - 1])
        if prev_lang != 'unknown':
            return prev_lang
    
    # Check next word
    if index < len(words) - 1:
        next_lang = detect_language(words[index + 1])
        if next_lang != 'unknown':
            return next_lang
    
    return 'en'  # Default to English

def handle_ssml_tags(text):
    """
    Handle SSML tags for all supported languages
    """
    # Language code mapping from SSML to internal codes
    ssml_lang_map = {
        'hi-IN': 'hi',
        'en-US': 'en',
        'en-GB': 'en',
        'bn-IN': 'bn',
        'gu-IN': 'gu',
        'kn-IN': 'kn',
        'ml-IN': 'ml',
        'mr-IN': 'mr',
        'ne-NP': 'ne',
        'or-IN': 'or',
        'pa-IN': 'pa',
        'ta-IN': 'ta',
        'te-IN': 'te',
        # Also support without country codes
        'hi': 'hi',
        'en': 'en',
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
    
    # Process combined <lang><say-as interpret-as="cardinal|digits|time"> or <say-as><lang> tags
    def combined_tag_replacer(match):
        language = match.group(1) or match.group(5)
        interpret_as = match.group(2) or match.group(4)
        content = match.group(3) or match.group(6)
        lang_code = ssml_lang_map.get(language, 'en')
        return number_to_words(content, lang_code, interpret_as=interpret_as)

    text = SSML_COMBINED_TAG.sub(combined_tag_replacer, text)

    # Process <say-as interpret-as="cardinal"> by converting the content to cardinal numbers
    text = SSML_SAY_AS_CARDINAL.sub(lambda match: number_to_words(match.group(1), 'en', interpret_as="cardinal"), text)

    # Process <say-as interpret-as="digits"> by converting the content to individual digits
    text = SSML_SAY_AS_DIGITS.sub(lambda match: number_to_words(match.group(1), 'en', interpret_as="digits"), text)

    # Process <say-as interpret-as="time"> by interpreting the content as time
    text = SSML_SAY_AS_TIME.sub(lambda match: number_to_words(match.group(1), 'en', interpret_as="time"), text)
    
    # Process <lang xml:lang="..."> by converting content to specified language as cardinal
    def language_tag_replacer(match):
        language = match.group(1)
        content = match.group(2)
        lang_code = ssml_lang_map.get(language, 'en')
        return number_to_words(content, lang_code, interpret_as="cardinal")
    
    text = SSML_LANGUAGE_TAG.sub(language_tag_replacer, text)
    
    # Process time formats before adding spaces around punctuation
    def process_time_formats(text):
        """Process time formats in the text before punctuation spacing"""
        words = text.split()
        processed_words = []
        
        for i, word in enumerate(words):
            if TIME_REGEX.fullmatch(word) or LOCAL_TIME_REGEX.fullmatch(word):
                # Handle both Latin and local numeral time formats
                if LOCAL_TIME_REGEX.fullmatch(word):
                    # For local time format, determine language from numeral script
                    match = LOCAL_TIME_REGEX.fullmatch(word)
                    if match:
                        hours_str, minutes_str = match.groups()
                        script = detect_numeral_script(hours_str + minutes_str)
                        if script != 'unknown' and script != 'latin':
                            language = get_language_from_numeral_script(script)
                        else:
                            language = 'en'  # Default fallback
                    else:
                        language = 'en'
                else:
                    # For Latin time format, default to English or detect from context
                    language = 'en'
                
                word_in_words = number_to_words(word, language, interpret_as="time")
                processed_words.append(word_in_words)
            else:
                processed_words.append(word)
        
        return " ".join(processed_words)
    
    text = process_time_formats(text)
    text = add_spaces_around_punctuation(text)

    return text

def preprocess_numbers(text):
    text = handle_ssml_tags(text) 
    words = text.split()
    processed_words = []

    for i, word in enumerate(words):
        if TIME_REGEX.fullmatch(word) or LOCAL_TIME_REGEX.fullmatch(word):
            # Handle both Latin and local numeral time formats
            if LOCAL_TIME_REGEX.fullmatch(word):
                # For local time format, determine language from numeral script
                match = LOCAL_TIME_REGEX.fullmatch(word)
                if match:
                    hours_str, minutes_str = match.groups()
                    script = detect_numeral_script(hours_str + minutes_str)
                    if script != 'unknown' and script != 'latin':
                        language = get_language_from_numeral_script(script)
                    else:
                        language = determine_context_language(words, i)
                else:
                    language = determine_context_language(words, i)
            else:
                # For Latin time format, use context
                language = determine_context_language(words, i)
            
            word_in_words = number_to_words(word, language, interpret_as="time")
            processed_words.append(word_in_words)
        elif ALL_NUMBERS_REGEX.fullmatch(word):
            # Handle both Latin and local numerals
            script = detect_numeral_script(word)
            
            if script != 'unknown':
                # Convert local numerals to Latin numerals for processing
                latin_numeral = convert_local_numerals_to_latin(word)
                
                # Determine language based on numeral script or context
                if script != 'latin':
                    # Use language based on numeral script
                    language = get_language_from_numeral_script(script)
                else:
                    # For Latin numerals, use context
                    language = determine_context_language(words, i)
                
                interpret_as = "digits" if len(latin_numeral) > 8 else "cardinal"
                word_in_words = number_to_words(latin_numeral, language, interpret_as=interpret_as)
                processed_words.append(word_in_words)
            else:
                processed_words.append(word)
        else:
            processed_words.append(word)

    return " ".join(processed_words)

def remove_commas_in_numbers(text):
    """
    Converts numbers with commas into plain numbers without commas.
    Handles both Western (1,234,567) and Indian (1,23,45,678) numbering formats.
    Also handles number ranges, but avoids interfering with mathematical equations.
    """
    # Remove ALL commas from numbers by repeatedly applying the regex until no more matches
    # This handles both Indian format (1,00,000) and Western format (1,000,000)
    while re.search(r'(\d+),(\d+)', text):
        text = re.sub(r'(\d+),(\d+)', r'\1\2', text)
    
    # Handle number ranges (e.g., "5-10" -> "5 to 10") 
    # But only if it's not part of a mathematical equation
    # Look for patterns that are clearly ranges, not math operations
    text = re.sub(r'\b(\d+)\s*-\s*(\d+)\b(?!\s*[=])', r'\1 to \2', text)
    
    return text

def IndicTextProcessing(text):
    """
    Core text processing pipeline for multilingual TTS preprocessing.
    
    Processing order:
    1. Remove commas from numbers (1,234 -> 1234)
    2. Spell out uppercase words (API -> A P I)  
    3. Process currency symbols (₹123 -> 123 Indian Rupees)
    4. Process mathematical equations (2+2 -> two plus two)
    5. Spell out websites (www.test.com -> www dot test dot com)
    6. Process numbers and time formats (123 -> one hundred twenty three)
    7. Remove any remaining SSML tags
    8. Add spaces around punctuation for TTS
    9. Phonemize each word with appropriate language model
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Phonemized text with words separated by " | "
    """
    text = remove_commas_in_numbers(text)
    text = spell_out_uppercase_words(text)
    text = process_sentence_with_spoken_currency(text)
    text = process_mathematical_equations(text)
    text = process_websites(text) 
    text = preprocess_numbers(text)
    text = re.sub(r'<.*?>', '', text)  # Remove any remaining SSML tags
    text = add_spaces_around_punctuation(text)
    text = text.replace(",", " , ")
    return text

