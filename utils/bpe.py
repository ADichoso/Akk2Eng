import re
import collections
import json

# NOTE: AI was used to help debug and annotate this python file. This source was used to help develop the class file (https://www.geeksforgeeks.org/nlp/byte-pair-encoding-bpe-in-nlp/)
class BytePairEncoder:
    def __init__(self):
        # Merge rules learned during training: list of (pattern, replacement) tuples
        self.merges = None
        # The base character set seen in training data (before any merges)
        self.characters = None
        # Frequency counter for all tokens (characters and merged subwords)
        self.tokens = None
        # The final vocabulary: word -> frequency mapping after all merges are applied
        self.vocab = None

    def save(self, filepath):
        # Saves the model state to a JSON file so it can be reloaded later
        state = {
            "merges": self.merges,
            "characters": list(self.characters),   # cast to list for JSON serialization
            "tokens": dict(self.tokens),            # cast Counter to plain dict
            "vocab": self.vocab
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=4)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        # Loads a previously saved model state from a JSON file
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.merges = state["merges"]
        self.characters = set(state["characters"])          # restore as set for O(1) lookup
        self.tokens = collections.Counter(state["tokens"]) # restore as Counter
        self.vocab = state["vocab"]
        print(f"Model loaded from {filepath}")

    def format_word(self, text, space_token='_'):
        # Splits a word into space-separated characters and appends an end-of-word marker '_'
        # The marker lets the model distinguish word-final tokens from prefixes (e.g. "the_" vs "the")
        # Special tokens like <god> or <person> are returned as-is without splitting
        if text.startswith('<') and text.endswith('>'):
            return text  # treat special tokens as atomic units, don't split
        # e.g. "cat" -> "c a t _"
        return ' '.join(list(text)) + ' ' + space_token

    def initialize_vocab(self, text):
        # Builds the initial character-level vocabulary and token frequencies from raw text
        # Returns vocab (word -> frequency) and tokens (character -> frequency)

        # Isolate hyphens as separate tokens so they don't merge into adjacent characters
        text = text.replace('-', ' - ')
        # Normalize all whitespace to single spaces
        text = re.sub('\s+', ' ', text)

        all_words = text.split()

        # Count how often each character-split word form appears in the corpus
        vocab = {}
        for word in all_words:
            word = self.format_word(word)  # e.g. "cat" -> "c a t _"
            vocab[word] = vocab.get(word, 0) + 1
        
        # Count individual character frequencies, weighted by word frequency
        tokens = collections.Counter()
        for word, freq in vocab.items():
            for symbol in word.split():
                tokens[symbol] += freq

        return vocab, tokens

    def get_bigram_counts(self, vocab):
        # Counts every adjacent symbol pair (bigram) across the vocabulary
        # Each pair's count is weighted by the frequency of the word it appears in
        pairs = {}
        for word, count in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pair = (symbols[i], symbols[i+1])
                pairs[pair] = pairs.get(pair, 0) + count
        return pairs

    def merge_vocab(self, pair, vocab_in):
        # Applies one merge operation across the whole vocabulary
        # Concatenates every occurrence of the given pair into a single new token
        # Word-boundary assertions prevent merging inside already-merged tokens
        vocab_out = {}
        bigram = re.escape(' '.join(pair))          # e.g. ('a', 'b') -> r'a\ b'
        # Only match the pair when surrounded by whitespace or string boundaries
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        bytepair = ''.join(pair)                    # the merged token, e.g. 'ab'
        for word in vocab_in:
            w_out = p.sub(bytepair, word)           # replace pair with merged token
            vocab_out[w_out] = vocab_out.get(w_out, 0) + vocab_in[word]
        return vocab_out, (bigram, bytepair)

    def find_merges(self, vocab, tokens, num_merges):
        # Runs the BPE training loop for num_merges iterations
        # Each step: count bigrams -> pick most frequent -> merge -> record rule
        merges = []
        for i in range(num_merges):
            pairs = self.get_bigram_counts(vocab)
            best_pair = max(pairs, key=pairs.get)   # greedily pick the most frequent bigram
            best_count = pairs[best_pair]
            vocab, (bigram, bytepair) = self.merge_vocab(best_pair, vocab)
            # Store the merge as a regex rule for fast application during encoding
            merges.append((r'(?<!\S)' + bigram + r'(?!\S)', bytepair))
            tokens[bytepair] = best_count           # register the new merged token
        return vocab, tokens, merges

    def fit(self, text, num_merges):
        # Trains the BPE model on text by running num_merges merge operations
        # Populates merges, characters, tokens, and vocab
        vocab, tokens = self.initialize_vocab(text)
        self.characters = set(tokens.keys())        # snapshot of base character inventory
        self.vocab, self.tokens, self.merges = self.find_merges(vocab, tokens, num_merges)

    def encode_word(self, word):
        # Encodes a single word into BPE subword tokens
        # Replays all merge rules in training order — earlier merges take priority

        # Split into characters with end-of-word marker, e.g. "cat" -> "c a t _"
        word = self.format_word(word)

        # Apply each merge rule sequentially
        for pattern, repl in self.merges:
            word = re.sub(pattern, repl, word)

        return word.split()

    def encode(self, text):
        # Encodes a full string into a flat list of BPE tokens
        # Splits on whitespace and encodes each word individually
        words = text.split()
        tokens = []

        for word in words:
            tokens.extend(self.encode_word(word))

        return tokens