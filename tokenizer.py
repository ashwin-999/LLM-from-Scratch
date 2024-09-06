import pytest

class Tokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary.copy()
        if '<UNK>' not in self.vocabulary:
            self.vocabulary.append('<UNK>')
        if '<ENDOFTEXT>' not in self.vocabulary:
            self.vocabulary.append('<ENDOFTEXT>')           
        self.token_to_id = {token: id for id, token in enumerate(self.vocabulary)}
        self.id_to_token = {id: token for id, token in enumerate(self.vocabulary)}

    def encode(self, text):
        tokens = []
        for word in text.split():
            tokens.append(self.token_to_id.get(word, self.token_to_id['<UNK>']))
        # tokens.append(self.token_to_id['<ENDOFTEXT>'])
        return tokens

    def decode(self, token_ids):
        return ' '.join([self.id_to_token[id] for id in token_ids])

if __name__ == "__main__":
    vocab = ['hello', 'world', 'how', 'are', 'you', '<UNK>'] 
    tokenizer = Tokenizer(vocab)

    text = "hello world how are you today"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    encode_decoded = tokenizer.encode(decoded)
    assert encoded == encode_decoded

    text1 = "hello world how are you today"
    text2 = "this is a test"
    encoded = tokenizer.encode(text1 + " <ENDOFTEXT> " + text2)
    decoded = tokenizer.decode(encoded)
    encode_decoded = tokenizer.encode(decoded)
    assert encoded == encode_decoded



