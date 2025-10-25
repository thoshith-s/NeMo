import os
import sys
from logging import Logger
from typing import Optional, Union, List, Pattern
from nemo.collections.tts.g2p.models.espeakphonemizer.backend import BACKENDS
from nemo.collections.tts.g2p.models.espeakphonemizer.backend.base import BaseBackend
from nemo.collections.tts.g2p.models.espeakphonemizer.backend.espeak.language_switch import LanguageSwitch
from nemo.collections.tts.g2p.models.espeakphonemizer.backend.espeak.words_mismatch import WordMismatch
from nemo.collections.tts.g2p.models.espeakphonemizer.logger import get_logger
from nemo.collections.tts.g2p.models.espeakphonemizer.punctuation import Punctuation
from nemo.collections.tts.g2p.models.espeakphonemizer.separator import default_separator, Separator
from nemo.collections.tts.g2p.models.espeakphonemizer.utils import list2str, str2list



def create_phonemizer(  # pylint: disable=too-many-arguments
        language: str = 'en-us',
        separator: Optional[Separator] = Separator(phone=' ', word=' | '),
        strip: bool = True,
        preserve_punctuation: bool = True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
        with_stress: bool = True,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = 'keep-flags',
        words_mismatch: WordMismatch = 'ignore',
        logger: Logger = get_logger()):

    phonemizer = BACKENDS["espeak"](
        language,
        punctuation_marks=punctuation_marks,
        preserve_punctuation=preserve_punctuation,
        with_stress=with_stress,
        tie=tie,
        language_switch=language_switch,
        words_mismatch=words_mismatch,
        logger=logger)
    
    return phonemizer

def phonemize(phonemizer, text):
    text_type = type(text)
    separator = Separator(phone=' ', word=' | ')
    strip = True
    # force the text as a list
    text = [line.strip(os.linesep) for line in str2list(text)]

    text = [line for line in text if line.strip()]

    if (text):
        phonemized = phonemizer.phonemize(text, separator=separator, strip=strip, njobs=1)
    else:
        phonemized = []

    if text_type == str:
        return list2str(phonemized)
    
    return phonemized
