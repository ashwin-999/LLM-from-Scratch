import pytest
from tokenizer import Tokenizer

@pytest.fixture
def tokenizer():
    vocab = ['hello', 'world', 'how', 'are', 'you', '<UNK>']
    return Tokenizer(vocab)

def test_single_sentence_encoding_decoding(tokenizer):
    text = "hello world how are you today"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    encode_decoded = tokenizer.encode(decoded)
    assert encoded == encode_decoded, (
        "Encoding and decoding are not consistent for a single sentence.\n"
        f"Original encoded: {encoded}\n"
        f"Re-encoded after decoding: {encode_decoded}"
    )

def test_multiple_sentences_encoding_decoding(tokenizer):
    text1 = "hello world how are you today"
    text2 = "this is a test"
    encoded = tokenizer.encode(text1 + " <ENDOFTEXT> " + text2)
    decoded = tokenizer.decode(encoded)
    encode_decoded = tokenizer.encode(decoded)
    assert encoded == encode_decoded, (
        "Encoding and decoding are not consistent for multiple sentences."
    )