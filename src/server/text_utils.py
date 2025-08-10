#!/usr/bin/env python3
"""
Text processing utilities for streaming TTS generation.
"""
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .sentence_splitter import split_text_into_sentences


@dataclass
class TextValidationResult:
    """Result of text validation with details"""
    is_valid: bool
    normalized_text: str
    issues: List[str]
    sentence_count: int
    total_length: int
    estimated_duration: float  # Rough estimate based on reading speed


class TextProcessor:
    """Text processing utilities for streaming TTS"""
    
    # Rough estimates for duration calculation
    AVERAGE_WORDS_PER_MINUTE = 150  # Average reading speed
    CHARS_PER_WORD = 5  # Average characters per word
    
    def __init__(self, max_text_length: int = 1000, min_sentence_length: int = 3):
        self.max_text_length = max_text_length
        self.min_sentence_length = min_sentence_length
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile commonly used regex patterns"""
        
        # Pattern for excessive whitespace
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Pattern for problematic characters that might cause TTS issues
        self.problematic_chars_pattern = re.compile(r'[^\w\s\.,!?;:\'"()\-–—\[\]{}/@#$%&*+=<>|\\`~]')
        
        # Pattern for repeated punctuation
        self.repeated_punct_pattern = re.compile(r'([.!?]){2,}')
        
        # Pattern for excessive commas
        self.excessive_commas_pattern = re.compile(r',{2,}')
        
        # Pattern for numbers that might need special handling
        self.number_pattern = re.compile(r'\b\d+(?:\.\d+)?\b')
        
        # Pattern for URLs/emails that might disrupt flow
        self.url_email_pattern = re.compile(
            r'(?:https?://[^\s]+|www\.[^\s]+|[\w.-]+@[\w.-]+\.[a-z]{2,})',
            re.IGNORECASE
        )
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for better TTS processing.
        
        Args:
            text: Raw input text
            
        Returns:
            Normalized text ready for TTS processing
        """
        if not text:
            return ""
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize whitespace (collapse multiple spaces/newlines)
        text = self.whitespace_pattern.sub(' ', text)
        
        # Fix repeated punctuation (... becomes , for better flow)
        text = self.repeated_punct_pattern.sub(lambda m: ',' if m.group(1) == '.' else m.group(1), text)
        
        # Fix excessive commas
        text = self.excessive_commas_pattern.sub(',', text)
        
        # Ensure sentences end with proper punctuation
        text = self._ensure_sentence_endings(text)
        
        # Clean up problematic characters (replace with space)
        text = self.problematic_chars_pattern.sub(' ', text)
        
        # Final whitespace cleanup
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    def _ensure_sentence_endings(self, text: str) -> str:
        """Ensure text ends with appropriate punctuation"""
        if not text:
            return text
        
        text = text.rstrip()
        
        # If doesn't end with sentence punctuation, add period
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def validate_text(self, text: str) -> TextValidationResult:
        """
        Validate text for streaming TTS processing.
        
        Args:
            text: Input text to validate
            
        Returns:
            TextValidationResult with validation details
        """
        issues = []
        
        # Check if text is empty or too short
        if not text or not text.strip():
            return TextValidationResult(
                is_valid=False,
                normalized_text="",
                issues=["Text is empty"],
                sentence_count=0,
                total_length=0,
                estimated_duration=0.0
            )
        
        # Normalize the text
        normalized = self.normalize_text(text)
        
        # Check length constraints
        if len(normalized) > self.max_text_length:
            issues.append(f"Text too long ({len(normalized)} > {self.max_text_length} chars)")
        
        if len(normalized) < 3:
            issues.append(f"Text too short ({len(normalized)} < 3 chars)")
        
        # Split into sentences
        sentences = split_text_into_sentences(normalized, self.min_sentence_length)
        
        # Validate sentences
        if not sentences:
            issues.append("No valid sentences found after processing")
        else:
            # Check for sentences that might be problematic
            for i, sentence in enumerate(sentences):
                if len(sentence) < self.min_sentence_length:
                    issues.append(f"Sentence {i+1} too short: '{sentence[:50]}...'")
                
                if len(sentence) > 500:
                    issues.append(f"Sentence {i+1} very long ({len(sentence)} chars)")
        
        # Check for potentially problematic content
        if self.url_email_pattern.search(normalized):
            issues.append("Contains URLs/emails that might affect pronunciation")
        
        # Estimate duration (rough calculation)
        estimated_duration = self._estimate_duration(normalized)
        
        is_valid = len(issues) == 0 or all('very long' in issue or 'URLs/emails' in issue for issue in issues)
        
        return TextValidationResult(
            is_valid=is_valid,
            normalized_text=normalized,
            issues=issues,
            sentence_count=len(sentences),
            total_length=len(normalized),
            estimated_duration=estimated_duration
        )
    
    def _estimate_duration(self, text: str) -> float:
        """
        Estimate audio duration based on text length.
        
        Args:
            text: Input text
            
        Returns:
            Estimated duration in seconds
        """
        if not text:
            return 0.0
        
        # Rough calculation based on average reading speed
        char_count = len(text)
        word_count = char_count / self.CHARS_PER_WORD
        minutes = word_count / self.AVERAGE_WORDS_PER_MINUTE
        return minutes * 60.0  # Convert to seconds
    
    def prepare_sentences_for_streaming(self, text: str) -> List[Dict[str, any]]:
        """
        Prepare text sentences for streaming generation.
        
        Args:
            text: Input text
            
        Returns:
            List of sentence dictionaries with metadata
        """
        validation = self.validate_text(text)
        
        if not validation.is_valid:
            raise ValueError(f"Text validation failed: {', '.join(validation.issues)}")
        
        sentences = split_text_into_sentences(validation.normalized_text, self.min_sentence_length)
        
        sentence_data = []
        total_chars = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            estimated_duration = self._estimate_duration(sentence)
            
            sentence_info = {
                'index': i,
                'text': sentence,
                'length': sentence_length,
                'estimated_duration': estimated_duration,
                'start_char': total_chars,
                'end_char': total_chars + sentence_length
            }
            
            sentence_data.append(sentence_info)
            total_chars += sentence_length
        
        return sentence_data
    
    def get_text_complexity_score(self, text: str) -> float:
        """
        Calculate a complexity score for text (0.0 to 1.0).
        Higher scores indicate more complex text that might take longer to process.
        
        Args:
            text: Input text
            
        Returns:
            Complexity score between 0.0 and 1.0
        """
        if not text:
            return 0.0
        
        factors = []
        
        # Length factor (longer text is more complex)
        length_score = min(len(text) / 500.0, 1.0)
        factors.append(length_score * 0.3)
        
        # Sentence count factor
        sentences = split_text_into_sentences(text)
        sentence_score = min(len(sentences) / 10.0, 1.0)
        factors.append(sentence_score * 0.2)
        
        # Punctuation density
        punct_count = len(re.findall(r'[.,!?;:]', text))
        punct_score = min(punct_count / len(text) * 20, 1.0) if text else 0.0
        factors.append(punct_score * 0.2)
        
        # Special characters (URLs, emails, numbers)
        special_count = len(self.url_email_pattern.findall(text)) + len(self.number_pattern.findall(text))
        special_score = min(special_count / 5.0, 1.0)
        factors.append(special_score * 0.3)
        
        return sum(factors)


def normalize_text_for_tts(text: str, max_length: int = 1000) -> str:
    """
    Convenience function for normalizing text for TTS.
    
    Args:
        text: Raw input text
        max_length: Maximum allowed text length
        
    Returns:
        Normalized text
        
    Raises:
        ValueError: If text is invalid or too long
    """
    processor = TextProcessor(max_text_length=max_length)
    validation = processor.validate_text(text)
    
    if not validation.is_valid:
        raise ValueError(f"Text validation failed: {', '.join(validation.issues)}")
    
    return validation.normalized_text


def prepare_text_for_streaming(text: str, max_length: int = 1000) -> List[Dict[str, any]]:
    """
    Convenience function for preparing text for streaming TTS.
    
    Args:
        text: Raw input text
        max_length: Maximum allowed text length
        
    Returns:
        List of sentence dictionaries ready for streaming
        
    Raises:
        ValueError: If text is invalid
    """
    processor = TextProcessor(max_text_length=max_length)
    return processor.prepare_sentences_for_streaming(text)


if __name__ == "__main__":
    # Test the text processor
    processor = TextProcessor()
    
    test_texts = [
        "Hello world. How are you today?",
        "This is a test...   with   extra    spaces!!! And repeated punctuation???",
        "",  # Empty text
        "A",  # Too short
        "Mr. Smith went to visit Dr. Johnson at https://example.com. Email him at test@example.com for more details.",
        "This is a single sentence without proper ending",
    ]
    
    print("=== Text Processing Tests ===\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        try:
            validation = processor.validate_text(text)
            print(f"  Valid: {validation.is_valid}")
            print(f"  Normalized: '{validation.normalized_text[:50]}{'...' if len(validation.normalized_text) > 50 else ''}'")
            print(f"  Sentences: {validation.sentence_count}")
            print(f"  Length: {validation.total_length}")
            print(f"  Est. Duration: {validation.estimated_duration:.2f}s")
            if validation.issues:
                print(f"  Issues: {', '.join(validation.issues)}")
            
            if validation.is_valid:
                sentences = processor.prepare_sentences_for_streaming(text)
                print(f"  Prepared sentences: {len(sentences)}")
                for j, sent in enumerate(sentences):
                    print(f"    {j+1}: '{sent['text'][:40]}{'...' if len(sent['text']) > 40 else ''}' ({sent['length']} chars)")
            
            complexity = processor.get_text_complexity_score(text)
            print(f"  Complexity: {complexity:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")
        
        print()