import pytest
import sys
from revcaptcha.text import normalize_transcript 

def test_normal_string():
    assert normalize_transcript("up, left!") == ["UP", "LEFT"]
    assert normalize_transcript("DOWN! OFF.") == ["DOWN", "OFF"]
    assert normalize_transcript("on off") == ["ON", "OFF"]

def test_punctuation_removal():
    assert normalize_transcript("up,,, left!!!!") == ["UP", "LEFT"]
    assert normalize_transcript("....") == []
    assert normalize_transcript(",,,...on! ...,,,left--!-") == ["ON", "LEFT"]

def test_whitespace_removal():
    assert normalize_transcript("up     left") == ["UP", "LEFT"]
    assert normalize_transcript("    ") == []
    assert normalize_transcript("on,  .left.") == ["ON", "LEFT"]

def test_empty_string():
    assert normalize_transcript("") == []
    assert normalize_transcript(" .... ") == []
    assert normalize_transcript("         ") == []

def test_whitespace_replacemnet_on_punctuation():
    assert normalize_transcript("up,,, left!!!!") == ["UP", "LEFT"]
    assert normalize_transcript("seven. .six,,,") == ["SEVEN", "SIX"]
    assert normalize_transcript("on- left") == ["ON", "LEFT"]

    
