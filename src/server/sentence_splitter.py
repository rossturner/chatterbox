#!/usr/bin/env python3
import re
from typing import List, Tuple


class SentenceSplitter:
    """
    Robust sentence splitter that handles common edge cases in text-to-speech contexts.
    
    Handles:
    - Abbreviations (Mr., Dr., Jr., etc.)
    - Decimal numbers (3.14, $29.99)
    - Dialogue and quotations
    - Multiple sentence endings (... !!! ???)
    - URLs and email addresses
    - Time formats (2:30 PM)
    """
    
    def __init__(self):
        # Common abbreviations that shouldn't trigger sentence breaks
        self.abbreviations = {
            'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'etc', 'inc', 'corp',
            'ltd', 'co', 'st', 'ave', 'blvd', 'rd', 'ln', 'ct', 'pl', 'sq',
            'ft', 'in', 'cm', 'mm', 'km', 'mi', 'lb', 'kg', 'oz', 'hr', 'min', 'sec',
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
            'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
            'am', 'pm', 'est', 'pst', 'gmt', 'utc',
            'usa', 'uk', 'eu', 'nato', 'un', 'fbi', 'cia', 'nasa', 'dna', 'cpu', 'gpu',
            'e.g', 'i.e', 'et al', 'cf', 'viz', 'vs'
        }
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for sentence splitting"""
        
        # Pattern to match potential sentence endings
        self.sentence_ending_pattern = re.compile(
            r'([.!?]+)'  # Capture sentence ending punctuation
            r'(\s*)'     # Optional whitespace
            r'(?=["\'\s]*[A-Z]|["\'\s]*$)'  # Lookahead for capital letter or end of string
        )
        
        # Pattern to match decimal numbers (to avoid false splits)
        self.decimal_pattern = re.compile(r'\d+\.\d+')
        
        # Pattern to match URLs and emails (to avoid false splits)
        self.url_email_pattern = re.compile(
            r'(?:https?://|www\.|[\w.-]+@[\w.-]+\.[a-z]{2,})'
        )
        
        # Pattern to match time formats
        self.time_pattern = re.compile(r'\d{1,2}:\d{2}(?:\s*[ap]m)?', re.IGNORECASE)
        
        # Pattern to match abbreviations
        abbr_list = '|'.join(re.escape(abbr) for abbr in self.abbreviations)
        self.abbreviation_pattern = re.compile(
            rf'\b(?:{abbr_list})\.(?:\s+[a-z])', 
            re.IGNORECASE
        )
    
    def split_sentences(self, text: str, min_sentence_length: int = 3) -> List[str]:
        """
        Split text into sentences, handling common edge cases.
        
        Args:
            text: Input text to split
            min_sentence_length: Minimum characters per sentence (to avoid tiny fragments)
            
        Returns:
            List of sentences
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        # Step 1: Protect URLs, emails, and special patterns from splitting
        protected_segments = []
        text = self._protect_patterns(text, protected_segments)
        
        # Step 2: Split on sentence boundaries
        sentences = self._split_on_boundaries(text)
        
        # Step 3: Restore protected segments
        sentences = self._restore_protected_patterns(sentences, protected_segments)
        
        # Step 4: Clean up and filter sentences
        sentences = self._cleanup_sentences(sentences, min_sentence_length)
        
        # Step 5: Handle edge cases for very short/long sentences
        sentences = self._balance_sentences(sentences, min_sentence_length)
        
        return sentences
    
    def _protect_patterns(self, text: str, protected_segments: List[str]) -> str:
        """Replace URLs, emails, and other special patterns with placeholders"""
        
        # Protect URLs and emails
        def replace_url_email(match):
            placeholder = f"__PROTECTED_{len(protected_segments)}__"
            protected_segments.append(match.group(0))
            return placeholder
        
        text = self.url_email_pattern.sub(replace_url_email, text)
        
        # Protect decimal numbers
        def replace_decimal(match):
            placeholder = f"__DECIMAL_{len(protected_segments)}__"
            protected_segments.append(match.group(0))
            return placeholder
        
        text = self.decimal_pattern.sub(replace_decimal, text)
        
        # Protect time formats
        def replace_time(match):
            placeholder = f"__TIME_{len(protected_segments)}__"
            protected_segments.append(match.group(0))
            return placeholder
        
        text = self.time_pattern.sub(replace_time, text)
        
        return text
    
    def _split_on_boundaries(self, text: str) -> List[str]:
        """Split text on sentence boundaries while respecting abbreviations"""
        
        sentences = []
        current_pos = 0
        
        for match in self.sentence_ending_pattern.finditer(text):
            punct_start, punct_end = match.span(1)  # Get punctuation group
            full_end = match.end()  # End of full match including whitespace
            
            # Check if this is a false positive (abbreviation)
            if self._is_abbreviation_boundary(text, punct_start):
                continue
            
            # Extract sentence up to this boundary (including punctuation)
            sentence = text[current_pos:full_end].strip()
            if sentence:
                sentences.append(sentence)
            
            current_pos = full_end
        
        # Add remaining text as final sentence
        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if remaining:
                sentences.append(remaining)
        
        return sentences
    
    def _is_abbreviation_boundary(self, text: str, pos: int) -> bool:
        """Check if a period at position is part of an abbreviation"""
        
        # Look back to find the start of the potential abbreviation
        start = pos
        while start > 0 and (text[start - 1].isalnum() or text[start - 1] == '.'):
            start -= 1
        
        if start < pos:
            potential_abbr = text[start:pos].lower().rstrip('.')
            if potential_abbr in self.abbreviations:
                # Additional check: make sure next character after period is not uppercase
                # Unless it's end of string, which is fine
                if pos + 1 < len(text) and text[pos + 1].isspace():
                    next_char_pos = pos + 1
                    while next_char_pos < len(text) and text[next_char_pos].isspace():
                        next_char_pos += 1
                    if next_char_pos < len(text) and text[next_char_pos].isupper():
                        return False  # This looks like a sentence boundary
                return True
        
        return False
    
    def _restore_protected_patterns(self, sentences: List[str], protected_segments: List[str]) -> List[str]:
        """Restore protected patterns back to sentences"""
        
        restored_sentences = []
        
        for sentence in sentences:
            # Restore protected segments
            for i, segment in enumerate(protected_segments):
                sentence = sentence.replace(f"__PROTECTED_{i}__", segment)
                sentence = sentence.replace(f"__DECIMAL_{i}__", segment)
                sentence = sentence.replace(f"__TIME_{i}__", segment)
            
            restored_sentences.append(sentence)
        
        return restored_sentences
    
    def _cleanup_sentences(self, sentences: List[str], min_length: int) -> List[str]:
        """Clean up sentences and filter by minimum length"""
        
        cleaned = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip empty sentences
            if not sentence:
                continue
            
            # Normalize whitespace
            sentence = ' '.join(sentence.split())
            
            # Add ending punctuation if missing
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            
            # Skip too-short sentences after normalization
            if len(sentence) < min_length:
                continue
            
            cleaned.append(sentence)
        
        return cleaned
    
    def _balance_sentences(self, sentences: List[str], min_length: int) -> List[str]:
        """Balance very short and very long sentences for better TTS streaming"""
        
        if not sentences:
            return sentences
        
        balanced = []
        i = 0
        
        while i < len(sentences):
            current = sentences[i]
            
            # If current sentence is very short and there's a next sentence,
            # consider combining them
            if (len(current) < min_length * 2 and 
                i + 1 < len(sentences) and 
                len(current) + len(sentences[i + 1]) < 200):  # Don't create super long sentences
                
                combined = f"{current} {sentences[i + 1]}"
                balanced.append(combined)
                i += 2  # Skip the next sentence since we combined it
            else:
                # Handle very long sentences by splitting on commas if needed
                if len(current) > 200:  # Lower threshold for better streaming
                    sub_sentences = self._split_long_sentence(current)
                    balanced.extend(sub_sentences)
                else:
                    balanced.append(current)
                i += 1
        
        return balanced
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split overly long sentences on commas or other natural break points"""
        
        # Try splitting on commas first
        parts = sentence.split(',')
        
        if len(parts) > 1:
            # Combine parts to create reasonably sized chunks
            chunks = []
            current_chunk = ""
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                # If adding this part would make chunk too long, start new chunk
                if current_chunk and len(current_chunk) + len(part) > 200:
                    chunks.append(current_chunk)
                    current_chunk = part
                else:
                    if current_chunk:
                        current_chunk += ", " + part
                    else:
                        current_chunk = part
            
            # Add final chunk
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks
        
        # If no commas, just return the original sentence
        return [sentence]
    
    def get_sentence_stats(self, text: str) -> dict:
        """Get statistics about sentence structure in the text"""
        
        sentences = self.split_sentences(text)
        
        if not sentences:
            return {
                'total_sentences': 0,
                'avg_length': 0,
                'min_length': 0,
                'max_length': 0,
                'total_chars': 0
            }
        
        lengths = [len(s) for s in sentences]
        
        return {
            'total_sentences': len(sentences),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'total_chars': sum(lengths),
            'sentences': sentences
        }


def split_text_into_sentences(text: str, min_sentence_length: int = 3) -> List[str]:
    """
    Convenience function for splitting text into sentences.
    
    Args:
        text: Input text to split
        min_sentence_length: Minimum characters per sentence
        
    Returns:
        List of sentences suitable for sequential TTS processing
    """
    splitter = SentenceSplitter()
    return splitter.split_sentences(text, min_sentence_length)


if __name__ == "__main__":
    # Test the sentence splitter
    test_texts = [
        "Hello world. How are you today?",
        "Mr. Smith went to Dr. Johnson's office at 3:30 PM. The temperature was 98.6 degrees.",
        "She said, 'Hello there!' Then she left.",
        "Visit https://example.com for more info. Email us at test@example.com.",
        "This is a very short sentence. A. B. This is longer.",
        "This is an incredibly long sentence that goes on and on and on, with many clauses, subclauses, and additional information that might make it challenging for text-to-speech systems to process in a single chunk, so we should probably split it into smaller, more manageable pieces.",
    ]
    
    splitter = SentenceSplitter()
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n=== Test {i} ===")
        print(f"Input: {text}")
        sentences = splitter.split_sentences(text)
        print(f"Sentences ({len(sentences)}):")
        for j, sentence in enumerate(sentences, 1):
            print(f"  {j}: {sentence}")
        
        stats = splitter.get_sentence_stats(text)
        print(f"Stats: {stats}")