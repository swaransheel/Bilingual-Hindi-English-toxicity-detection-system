import argparse
import os
import re
from typing import List, Dict, Tuple, Set, Union, Optional
from pydub import AudioSegment
from pydub.generators import Sine
from faster_whisper import WhisperModel
from transformers import pipeline
import langid
from nltk.corpus import stopwords

import torch

def generate_beep(duration_ms: int) -> AudioSegment:
    """1kHz beep of at least 1ms, −6dB."""
    duration_ms = max(1, duration_ms)
    return Sine(1000).to_audio_segment(duration=duration_ms).apply_gain(-6)

def create_fade_beep(duration_ms: int) -> AudioSegment:
    """Fade-in/out on the beep for smooth masking."""
    duration_ms = max(1, duration_ms)
    fade = min(50, max(1, duration_ms // 5))
    beep = generate_beep(duration_ms)
    return beep.fade_in(fade).fade_out(fade)

def transcribe_words(audio_path: str, model_size: str, device: str):
    """Run Faster-Whisper and return list of {'word','start','end'}."""
    model = WhisperModel(model_size, device=device)
    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    
    # Get the detected language from Whisper
    detected_language = info.language
    
    words = []
    for seg in segments:
        for w in seg.words:
            if w.end > w.start:
                words.append({
                    'word': w.word,
                    'start': int(w.start * 1000),
                    'end':   int(w.end   * 1000)
                })
    
    # Free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return words, detected_language

class ToxicDetector:
    """Centralized toxic content detector that dynamically loads models based on need."""
    
    def __init__(self, device: int, tox_model_path: str):  # Fix: Changed _init to __init_
        """Initialize the ToxicDetector.
        
        Args:
            device: Device ID (-1 for CPU, >=0 for GPU)
            tox_model_path: Path to toxicity model
        """
        self.device = device
        self.tox_model_path = tox_model_path
        self.pipes = {}  # Lazy-loaded models
        
        # Define language-specific whitelists and known toxic words
        self.common_words_whitelist = {
            # English whitelist
            'the', 'a', 'an', 'and', 'or', 'but', 'for', 'nor', 'on', 'at', 'to', 'you',
            'from', 'by', 'about', 'like', 'through', 'after', 'over', 'between', 
            'out', 'against', 'during', 'without', 'before', 'under', 'around', 
            'among', 'yes', 'no', 'maybe', 'sure', 'ok', 'okay', 'i', 'you', 'he', 
            'she', 'we', 'they', 'it', 'this', 'that', 'these', 'those', 'my', 
            'your', 'his', 'her', 'our', 'their', 'its', 'of', 'with', 'in', 'into',
            "don't", "can't", "won't", "it's", "that's", "there's", "here's", "what's",
            "who's", "how's", "where's", "when's", "why's", "ain't", "aren't", "isn't",
            "wasn't", "weren't", "haven't", "hasn't", "hadn't", "doesn't", "didn't",
            "couldn't", "shouldn't", "wouldn't", "mightn't", "mustn't", "wanna", "gonna",
            "gotta", "lemme", "you're", "your", "yeah", "oh", "pain", "summer", "nobody", 
            "since", "part", "have", "same", "phone", "sin", "decided", "something", 
            "somewhere", "one", "two", "three", "four", "five", "six", "seven", "eight", 
            "nine", "ten", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
            
            # Hindi/Hinglish whitelist
            'है', 'हूँ', 'हो', 'मैं', 'तुम', 'आप', 'और', 'या', 'में', 'पर', 'से', 'के', 'का', 'की',
            'एक', 'दो', 'तीन', 'हाँ', 'नहीं', 'क्या', 'कब', 'कौन', 'कहाँ', 'क्यों', 'कैसे',
            'तो', 'अभी', 'यहाँ', 'वहाँ', 'अगर', 'मगर', 'लेकिन', 'फिर', 'बस', 'कुछ', 'सब', 
            'मेरा', 'तेरा', 'उसका', 'हमारा', 'आना', 'जाना', 'करना', 'होना', 'रहना', 'कहना',
            'बात', 'लोग', 'आदमी', 'औरत', 'बच्चा', 'दिन', 'रात', 'समय', 'साल', 'महीना',
            'काम', 'पैसा', 'घर', 'गाड़ी', 'स्कूल', 'ऑफिस', 'आफिस', 'प्रिन्सिपल',
            'सोचो', 'अपने', 'बुलाया', 'आया', 'ही', 'पे', 'गान', 'मारने', 'होगा',
            'होता', 'कि', 'उसका', 'टाइगराम', 'प्राक्टिकल', 'आओ', 'कैबिन', 'में',
            'main', 'mein', 'hai', 'hoon', 'ho', 'aap', 'tum', 'aur', 'ya', 'par', 'se', 'ke', 'ka', 'ki',
            'ek', 'do', 'teen', 'haan', 'nahi', 'kya', 'kab', 'kaun', 'kahan', 'kyun', 'kaise',
        }
        
        self.common_words_whitelist.update({
            # Additional Hindi words from logs
            'है', 'मैं', 'तेरी', 'कि', 'में', 'ले', 'भाहू', 'तिरको', 'वेट्टे', 'को',
            'बुला', 'उट', 'खडर', 'रहे', 'हुथा', 'अरहे', 'पट', 'हू', 'बिट',
            'लवडे', 'खडरे', 'पे', 'मारना',
            
            # Common variations
            'थे', 'था', 'थी', 'हैं', 'होगी', 'होगा', 'होगें', 'करो', 'करें',
            'जाओ', 'जाएं', 'आओ', 'आएं', 'लो', 'दो', 'कहो', 'सुनो'
        })
        
        self.known_toxic_words = {
            # English
            "fuck", "shit", "piss", "nigga", "niggas", "bitch", "cunt", "ass", "asshole",
            "faggot", "dyke", "whore", "slut",
            
            # Hindi/Hinglish - expanded list
            'bhenchod', 'madarchod', 'chutiya', 'gaandu', 'randi', 'harami',
            'behen ke lode', 'bhosdike', 'लवडे', 'भोसडी', 'मादरचोद', 'बहनचोद', 'चुतिया',
            'गांडू', 'रंडी',
            # Additional Hindi variants
            'gand', 'gaand', 'gandu' ,'गांद', 'गंद', 'गन्द','गांड',
            'gaand', 'गाँड', 'गांड', 
            'chut', 'चुत', 'चूत',
            'lund', 'लंड', 'लुंड',
            'lauda', 'लौड़ा',
            'bhosda', 'भोसड़ा',
        }
        
        # Store model configurations for lazy loading
        self.model_configs = {
            'toxicity': {'model': self.tox_model_path, 'task': 'text-classification'},
            'toxicity2': {'model': 'unitary/toxic-bert', 'task': 'text-classification'},
            'hate': {'model': 'facebook/roberta-hate-speech-dynabench-r4-target', 'task': 'text-classification'},
            'abusive': {'model': 'DunnBC22/ibert-roberta-base-Abusive_Or_Threatening_Speech', 'task': 'text-classification'},
            'sentiment_en': {'model': 'distilbert-base-uncased-finetuned-sst-2-english', 'task': 'sentiment-analysis'},
            'political': {'model': 'Newtral/xlm-r-finetuned-toxic-political-tweets-es', 'task': 'text-classification'},
            'hindi': {'model': 'Hate-speech-CNERG/hindi-codemixed-abusive-MuRIL', 'task': 'text-classification'},
            'sentiment_hi': {'model': 'pascalrai/hinglish-twitter-roberta-base-sentiment', 'task': 'sentiment-analysis'},
        }
        
        # Define required models by language
        self.required_models = {
            'en': ['toxicity', 'toxicity2', 'hate', 'abusive', 'sentiment_en', 'political'],
            'hi': ['toxicity', 'toxicity2', 'hate', 'hindi', 'sentiment_hi', 'political']
        }
        
        # Add NLTK stopwords initialization
        self.stop_words = {}
        for lang in ['english', 'hindi']:
            try:
                self.stop_words[lang[:2]] = set(stopwords.words(lang))
            except:
                self.stop_words[lang[:2]] = set()
                
        # Add obfuscation patterns
        self.obfuscation_patterns = {
            'a': ['@', '4', 'а', 'α'],
            'b': ['8', '6', 'ƅ', 'β'],
            'c': ['(', '{', '[', '<', 'с'],
            'e': ['3', 'є', 'е', 'э'],
            'i': ['1', '!', '|', 'і', 'ї'],
            'l': ['1', '|', 'і', 'ӏ'],
            'o': ['0', '()', 'ο', 'о', 'σ'],
            's': ['5', '$', 'ѕ'],
            't': ['7', '+', 'т'],
            'u': ['μ', 'υ', 'ц'],
            'v': ['\\/', 'ν', 'v'],
            'w': ['vv', 'ш', 'щ'],
            'x': ['×', 'х', '}{'],
            'y': ['ч', 'у', 'γ'],
            'z': ['2', 'ӡ', 'ʒ']
        }

        # Add methods for handling obfuscated text
        
    def normalize_obfuscated_text(self, text: str) -> str:
        """Handle obfuscated text (e.g., l33tspeak) by normalizing it."""
        normalized = text.lower()
        
        # Replace common leet speak substitutions
        for char, replacements in self.obfuscation_patterns.items():
            for replacement in replacements:
                normalized = normalized.replace(replacement, char)
                
        # Remove spaces that might be inserted to avoid detection
        normalized = re.sub(r'\s+', '', normalized)
        
        return normalized
    
    def check_for_obfuscated_toxic_words(self, word: str) -> bool:
        """Check if a word might be an obfuscated version of a toxic word."""
        if len(word) <= 2:
            return False
            
        normalized = self.normalize_obfuscated_text(word)
        
        for toxic in self.known_toxic_words:
            if len(toxic) <= 3:
                continue
                
            norm_toxic = self.normalize_obfuscated_text(toxic)
            
            # Direct match after normalization
            if norm_toxic in normalized:
                return True
                
            # Similarity check for longer words
            if len(norm_toxic) >= 4 and len(normalized) >= 4:
                common_chars = sum(1 for c in norm_toxic if c in normalized)
                similarity = common_chars / max(len(norm_toxic), len(normalized))
                
                if similarity > 0.7:  # 70% character overlap threshold
                    return True
        
        return False
        
    def get_pipe(self, name: str):
        """Lazy-load models only when needed."""
        if name not in self.pipes:
            config = self.model_configs.get(name)
            if not config:
                raise ValueError(f"Unknown model: {name}")
                
            self.pipes[name] = pipeline(config['task'], model=config['model'], device=self.device)
        return self.pipes[name]
        
    def load_models_for_language(self, language: str):
        """Pre-load all models needed for a specific language."""
        models = self.required_models.get(language, self.required_models['en'])
        for model_name in models:
            self.get_pipe(model_name)
            
    def unload_models(self):
        """Unload all models to free memory."""
        self.pipes.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def detect_toxic_content(self, 
                             words: List[Dict], 
                             language: str, 
                             threshold: float, 
                             min_votes: int, 
                             strict_mode: bool = True,
                             custom_blocklist: Set[str] = None) -> List[Tuple]:
        """Detect toxic content using language-specific approach."""
        # Process blocklist first
        spans = []
        if custom_blocklist:
            for w in words:
                if w['word'].lower().strip() in custom_blocklist:
                    spans.append((w['start'], w['end'], w['word'], ['blocklist:manual:1.0']))
        
        # Prepare batch processing
        texts = [w['word'] for w in words]
        
        # Determine which models to use
        models_to_use = self.required_models.get(language, self.required_models['en'])
        
        # Adjust parameters based on language
        if language == 'hi':
            threshold += 0.2  # Stricter threshold for Hindi
            min_votes = max(3, min_votes + 1)  # Require more votes
            
        # Run batch inference for each required model
        results = {}
        for model_name in models_to_use:
            pipe = self.get_pipe(model_name)
            results[model_name] = pipe(texts)
        
        # Process results
        for i, w in enumerate(words):
            word = w['word'].strip().lower()
            
            # Skip processing if Hindi-specific model indicates non-toxic
            if language == 'hi' and 'hindi' in results:
                hi = results['hindi'][i]
                # Trust Hindi model more - if it says non-toxic, skip unless known toxic
                if hi['score'] < threshold + 0.1 or hi['label'].lower() in ('normal', 'not_offensive'):
                    if word not in self.known_toxic_words:
                        continue
            
            # Improved whitelist check
            is_whitelisted = (word in self.common_words_whitelist 
                            and word not in self.known_toxic_words
                            and not any(c.isdigit() for c in word))  # Exclude numbers
                            
            if (len(word) <= 1 or is_whitelisted) and strict_mode:
                continue
                
            votes = 0
            detection_types = []
            toxicity_score = 0
            
            # Process results from each model based on language
            if language == 'en':
                # Process English-specific models
                
                # toxicity - primary model
                if 'toxicity' in results:
                    t = results['toxicity'][i]
                    if 'toxic' in t['label'].lower() and t['score'] >= threshold:
                        votes += 1
                        toxicity_score += t['score'] * 1.2
                        detection_types.append(f"toxicity:{t['label']}:{t['score']:.2f}")
                
                # toxicity2
                if 'toxicity2' in results:
                    t2 = results['toxicity2'][i]
                    if 'toxic' in t2['label'].lower() and t2['score'] >= threshold:
                        votes += 1
                        toxicity_score += t2['score'] * 1.2
                        detection_types.append(f"toxicity2:{t2['label']}:{t2['score']:.2f}")
                
                # abusive
                if 'abusive' in results:
                    ab = results['abusive'][i]
                    if any(tok in ab['label'].lower() for tok in ('abusive','threatening')) and ab['score'] >= threshold:
                        votes += 1
                        toxicity_score += ab['score'] * 1.3
                        detection_types.append(f"abusive:{ab['label']}:{ab['score']:.2f}")
                
                # hate
                if 'hate' in results:
                    h = results['hate'][i]
                    if any(tok in h['label'].lower() for tok in ('hate','offensive','abusive')) and h['score'] >= threshold:
                        votes += 1
                        toxicity_score += h['score'] * 1.1
                        detection_types.append(f"hate:{h['label']}:{h['score']:.2f}")
                
                # sentiment
                if 'sentiment_en' in results:
                    s = results['sentiment_en'][i]
                    if s['label'].lower().startswith('neg') and s['score'] >= threshold + 0.15:
                        votes += 0.5
                        toxicity_score += s['score'] * 0.6
                        detection_types.append(f"sentiment:{s['label']}:{s['score']:.2f}")
                        
                # political
                if 'political' in results:
                    p = results['political'][i]
                    if any(tok in p['label'].lower() for tok in ('toxic','biased','negative')) and p['score'] >= threshold + 0.1:
                        votes += 0.5
                        toxicity_score += p['score'] * 0.7
                        detection_types.append(f"political:{p['label']}:{p['score']:.2f}")
                
            else:  # Hindi/other languages
                # Process Hindi-specific models
                
                # hindi - primary model for Hindi
                if 'hindi' in results:
                    hi = results['hindi'][i]
                    if hi['score'] >= threshold + 0.1 and hi['label'].lower() not in ('normal','not_offensive'):
                        votes += 2.0  # Increased weight for Hindi model
                        toxicity_score += hi['score'] * 1.5
                        detection_types.append(f"hindi:{hi['label']}:{hi['score']:.2f}")
                
                # sentiment for Hindi
                if 'sentiment_hi' in results:
                    s = results['sentiment_hi'][i]
                    if s['label'].lower().startswith('neg') and s['score'] >= threshold:
                        votes += 1
                        toxicity_score += s['score'] * 1.2
                        detection_types.append(f"sentiment:{s['label']}:{s['score']:.2f}")
                
                # Common models with adjusted weights for Hindi
                if 'toxicity' in results:
                    t = results['toxicity'][i]
                    if 'toxic' in t['label'].lower() and t['score'] >= threshold:
                        votes += 0.8
                        toxicity_score += t['score']
                        detection_types.append(f"toxicity:{t['label']}:{t['score']:.2f}")
                
                if 'toxicity2' in results:
                    t2 = results['toxicity2'][i]
                    if 'toxic' in t2['label'].lower() and t2['score'] >= threshold:
                        votes += 0.8
                        toxicity_score += t2['score']
                        detection_types.append(f"toxicity2:{t2['label']}:{t2['score']:.2f}")
                
                if 'hate' in results:
                    h = results['hate'][i]
                    if any(tok in h['label'].lower() for tok in ('hate','offensive','abusive')) and h['score'] >= threshold:
                        votes += 0.7
                        toxicity_score += h['score'] * 0.8
                        detection_types.append(f"hate:{h['label']}:{h['score']:.2f}")
                        
                if 'political' in results:
                    p = results['political'][i]
                    if any(tok in p['label'].lower() for tok in ('toxic','biased','negative')) and p['score'] >= threshold + 0.1:
                        votes += 0.5
                        toxicity_score += p['score'] * 0.7
                        detection_types.append(f"political:{p['label']}:{p['score']:.2f}")
            
            # Check if the word is known toxic
            is_known_toxic = word in self.known_toxic_words
            
            # Calculate average toxicity and determine if word should be flagged
            avg_toxicity = toxicity_score / max(1, len(detection_types)) if detection_types else 0
            
            # Adjust for language-specific thresholds
            adjusted_threshold = threshold - 0.05 if language == 'hi' else threshold
            
            if (votes >= min_votes and avg_toxicity >= adjusted_threshold) or is_known_toxic:
                # For known toxic words with no detections
                if is_known_toxic and not detection_types:
                    detection_types = [f"blocklist:known:{1.0}"]
                    
                spans.append((w['start'], w['end'], w['word'], detection_types))
        
        return spans


def merge_spans_with_words(spans):
    """
    Merge overlapping or back-to-back spans while preserving word and type info.
    """
    if not spans:
        return []
        
    # Sort by start time
    spans = sorted(spans, key=lambda x: x[0])
    
    merged = []
    current_span = list(spans[0])
    current_span[2] = [current_span[2]]  # Convert word to list
    current_span[3] = [current_span[3]]  # Convert detection_types to list
    
    for s, e, word, types in spans[1:]:
        ps, pe, pwords, ptypes = current_span
        
        # If spans are close enough to merge (within 100ms)
        if s <= pe + 100:
            # Expand the end time if needed
            current_span[1] = max(pe, e)
            # Add word and types
            current_span[2].append(word)
            current_span[3].append(types)
        else:
            # Finalize the current span and start a new one
            merged.append(tuple(current_span))
            current_span = [s, e, [word], [types]]
    
    # Add the final span
    merged.append(tuple(current_span))
    return merged


def format_span_info(span):
    """Format span information for output display."""
    start, end, words, types_list = span
    words_str = " ".join(words)
    flat_types = []
    for types in types_list:
        flat_types.extend(types)
    
    # Group by detection type
    type_groups = {}
    for t in flat_types:
        type_name = t.split(':')[0]
        if type_name not in type_groups:
            type_groups[type_name] = []
        type_groups[type_name].append(t)
    
    # Format the type information
    types_str = "; ".join([f"{k}: {len(v)}" for k, v in type_groups.items()])
    
    return f"({start}-{end}ms) '{words_str}' - {types_str}"


def censor_audio(audio: AudioSegment, spans):
    """
    Replace toxic spans with beeps instead of overlaying for cleaner censoring.
    
    Args:
        audio: Input AudioSegment
        spans: List of tuples (start_ms, end_ms, word, types)
        
    Returns:
        AudioSegment with toxic spans replaced by beeps
    """
    if not spans:
        return audio

    # Work with a copy
    out = audio[:]

    # Process spans in reverse to avoid shifting issues
    for start, end, _, _ in sorted(spans, key=lambda x: -x[0]):
        # Generate beep for the exact duration
        duration = max(1, end - start)
        beep = create_fade_beep(duration)
        
        # Split and replace segment
        before = out[:start]
        after = out[end:]
        out = before + beep + after

    return out


def load_custom_blocklist(blocklist_path=None):
    """Load custom blocklist of words to always censor."""
    if not blocklist_path or not os.path.exists(blocklist_path):
        return set()
    
    with open(blocklist_path, 'r') as f:
        return {line.strip().lower() for line in f if line.strip()}


def save_detection_report(spans, report_path, language):
    """Save a detailed detection report to a file."""
    with open(report_path, 'w') as f:
        f.write("TOXIC WORD DETECTION REPORT\n")
        f.write("=========================\n\n")
        f.write(f"Detected language: {language}\n\n")
        
        for i, span in enumerate(spans, 1):
            start, end, words, types_list = span
            words_str = " ".join(words)
            duration = end - start
            
            f.write(f"Span #{i}: {start}-{end}ms ({duration}ms)\n")
            f.write(f"Words: '{words_str}'\n")
            f.write("Detection types:\n")
            
            for j, word_types in enumerate(types_list):
                if j < len(words):  # Ensure we don't go out of bounds
                    f.write(f"  Word '{words[j]}':\n")
                    for t in word_types:
                        model, label, score = t.split(':')
                        f.write(f"    - {model}: {label} ({score})\n")
            
            f.write("\n")
        
        f.write(f"Total: {len(spans)} toxic spans detected\n")


def determine_language(whisper_language, words):
    """Determine processing language from detection results."""
    # List of languages to process using Hindi pipeline
    indian_languages = ['hi', 'pa', 'mr', 'ur', 'bn', 'gu']
    
    if whisper_language in indian_languages:
        return 'hi'  # Use Hindi pipeline for Indian languages
    elif whisper_language == 'en':
        return 'en'  # English detected
    else:
        # Fallback to langid for confirmation
        detected = langid.classify(" ".join([w['word'] for w in words]))[0]
        return 'hi' if detected in indian_languages else 'en'
    


def censor_audio_file(input_path: str, output_path: str):
    class Args:
        input = input_path
        output = output_path
        model = 'base'
        device = 'cpu'
        threshold = 0.4
        min_votes = 2
        tox_model = 'unitary/toxic-bert'
        language = 'auto'
        blocklist = None
        report = None
        permissive = False
        verbose = False

    args = Args()

    # Everything inside your main() but using args directly, not parser

    # Ensure output directory exists
    odir = os.path.dirname(args.output) or '.'
    os.makedirs(odir, exist_ok=True)

    if args.verbose:
        print(f"[+] Transcribing {args.input} with Whisper-{args.model} on {args.device}")

    # Transcribe and detect
    words, whisper_language = transcribe_words(args.input, args.model, args.device)
    language = determine_language(whisper_language, words) if args.language == 'auto' else args.language

    custom_blocklist = load_custom_blocklist(args.blocklist)

    device_id = 0 if args.device == 'cuda' else -1
    detector = ToxicDetector(device_id, args.tox_model)

    try:
        detector.load_models_for_language(language)

        spans = detector.detect_toxic_content(
            words,
            language,
            args.threshold,
            args.min_votes,
            strict_mode=not args.permissive,
            custom_blocklist=custom_blocklist
        )

        merged_spans = merge_spans_with_words(spans)

        audio = AudioSegment.from_file(args.input)
        censored = censor_audio(audio, merged_spans)

        ext = os.path.splitext(args.output)[1].lstrip('.') or 'mp3'
        censored.export(args.output, format=ext)

        print(f"[✓] Saved censored audio to {args.output}")
    
    finally:
        detector.unload_models()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



def main():
    parser = argparse.ArgumentParser(
        description="Language-aware ensemble-based toxic-word censor"
    )
    parser.add_argument('-i', '--input', required=True, help='input audio file (mp3/wav/etc)')
    parser.add_argument('-o', '--output', required=True, help='output (censored) audio file')
    parser.add_argument('-m', '--model', default='base', help='whisper model: tiny/base/small/medium/large')
    parser.add_argument('-d', '--device', default='cuda', choices=['cuda','cpu'], help='inference device')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='score threshold (0–1)')
    parser.add_argument('--min_votes', type=int, default=2, help='minimum number of models that must agree')
    parser.add_argument('--tox_model', default='unitary/toxic-bert', help='toxic classifier model or directory')
    parser.add_argument('--language', choices=['auto', 'en', 'hi'], default='auto', 
                         help='force language (auto=detect from audio)')
    parser.add_argument('--blocklist', help='path to custom blocklist file (one word per line)')
    parser.add_argument('--report', help='path to save detailed detection report')
    parser.add_argument('--permissive', action='store_true', help='less strict filtering (more matches)')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose logging')
    args = parser.parse_args()

    # Ensure output directory exists
    odir = os.path.dirname(args.output) or '.'
    os.makedirs(odir, exist_ok=True)

    if args.verbose:
        print(f"[+] Transcribing {args.input} with Whisper-{args.model} on {args.device}")
    
    # Get words and detected language from Whisper
    words, whisper_language = transcribe_words(args.input, args.model, args.device)
    
    if args.verbose:
        print(f"    → {len(words)} words extracted")
        print(f"    → Whisper detected language: {whisper_language}")
    
    # Determine language mode - from args or auto-detect
    if args.language == 'auto':
        language = determine_language(whisper_language, words)
    else:
        language = args.language

    if args.verbose:
        print(f"[+] Using language mode: {language} (Whisper detected: {whisper_language})")

    # Load custom blocklist if provided
    custom_blocklist = load_custom_blocklist(args.blocklist)
    if args.verbose and custom_blocklist:
        print(f"    → Loaded {len(custom_blocklist)} words in custom blocklist")

    # Convert device string to int for the transformers pipeline
    device_id = 0 if args.device == 'cuda' else -1
    
    # Initialize the toxicity detector
    detector = ToxicDetector(device_id, args.tox_model)
    
    try:
        # Pre-load models for the detected language
        detector.load_models_for_language(language)
        
        if args.verbose:
            print(f"[+] Running ensemble detection (threshold {args.threshold}, min votes {args.min_votes})…")
        
        # Detect toxic spans with language-specific algorithm
        spans = detector.detect_toxic_content(
            words,
            language,
            args.threshold,
            args.min_votes,
            strict_mode=not args.permissive,
            custom_blocklist=custom_blocklist
        )
        
        # Merge spans that are close to each other
        merged_spans = merge_spans_with_words(spans)
        
        if args.verbose:
            print(f"    → {len(merged_spans)} toxic spans to censor")
            for span in merged_spans:
                print(f"      {format_span_info(span)}")

        # Save detailed report if requested
        if args.report:
            save_detection_report(merged_spans, args.report, language)
            if args.verbose:
                print(f"    → Saved detailed detection report to {args.report}")

        # Load and censor audio
        audio = AudioSegment.from_file(args.input)
        censored = censor_audio(audio, merged_spans)
        
        # Determine output format from file extension
        ext = os.path.splitext(args.output)[1].lstrip('.') or 'mp3'
        censored.export(args.output, format=ext)

        print(f"[✓] Saved censored audio to {args.output}")
        print(f"    Language detected: {language}")
        if merged_spans:
            print(f"    {len(merged_spans)} toxic spans were censored:")
            for span in merged_spans:
                print(f"      {format_span_info(span)}")
        else:
            print("    No toxic spans detected")
    
    finally:
        # Always clean up models to free memory
        detector.unload_models()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()