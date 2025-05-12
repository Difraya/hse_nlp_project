# from transformers import GPTNeoForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

checkpoint_path = "./author_style_gpt2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка модели и токенизатора
model = GPT2LMHeadModel.from_pretrained(checkpoint_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)

def generate_text(author, prompt, max_length=150, temperature=0.7):
    # Формируем вход с указанием автора
    input_text = f"[AUTHOR_{author}] {prompt}"

    # Токенизация и перенос на GPU
    inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Генерация текста с улучшенными параметрами
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.2,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )

    # Декодирование и очистка от служебных токенов
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Удаляем метку автора из результата
    clean_text = generated_text.replace(f'[AUTHOR_{author}]', '').strip()

    return clean_text


if __name__ == "__main__":
    author_name = 'Stephen_King'
    prompt_text = 'The most intresting story about vampire is'

    result = generate_text(author_name, prompt_text)
    print("Generated text:")
    print(result)
