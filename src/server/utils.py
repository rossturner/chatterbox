import logging
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text for TTS synthesis.
    Based on punc_norm from chatterbox.tts
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."
    
    # Remove multiple space chars
    text = " ".join(text.split())
    
    # Capitalise first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    # Replace uncommon/llm punctuation
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (""", "\""),
        (""", "\""),
        ("'", "'"),
        ("'", "'"),
    ]
    
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)
    
    # Add full stop if no ending punctuation
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."
    
    return text


def get_memory_usage() -> Optional[float]:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        logger.debug("psutil not available for memory monitoring")
        return None
    except Exception as e:
        logger.error(f"Failed to get memory usage: {e}")
        return None


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"