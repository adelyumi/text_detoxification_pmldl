import sys
import torch
import transformers

def load_toxic_words():
    with open('./data/external/en.txt') as file:
        toxic_words = [line.rstrip() for line in file]
    return toxic_words

def load_model():
    model_name = 'bert-base-cased'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForMaskedLM.from_pretrained(model_name)

    return tokenizer, model

def replace_toxic_words(text, vocab, tokenizer, model):
    tokenized_text = tokenizer.tokenize(text)
    
    toxic_word_indices = []
    
    masked_text = [token if token.lower() not in vocab else '[MASK]' for token in tokenized_text]
    masked_text = " ".join(masked_text)
    
    input_ids = tokenizer.encode(masked_text, add_special_tokens=True)
    input_ids_tensor = torch.tensor(input_ids).unsqueeze(0)
    
    with torch.no_grad():
        predictions = model(input_ids_tensor)[0]
        predicted_tokens = []
        
        for i, token in enumerate(tokenized_text):
            if token.lower() in vocab:
                predicted_word = tokenizer.convert_ids_to_tokens(torch.argmax(predictions[0, i + 1]).item())
                predicted_tokens.append(predicted_word)
            else:
                predicted_tokens.append(token)

    replaced_text = tokenizer.convert_tokens_to_string(predicted_tokens)
    return replaced_text

if __name__ == "__main__":
    text = input("Enter toxic text: ") if len(sys.argv) == 1 else " ".join(sys.argv[1:])

    transformers.logging.set_verbosity_error()

    vocab = load_toxic_words()
    tokenizer, model = load_model()
    
    detox_text = replace_toxic_words(text, vocab, tokenizer, model)

    print("Detoxified text:", detox_text)
