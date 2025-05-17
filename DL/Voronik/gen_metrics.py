import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import bert_score
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')

# Пути к модели и токенизатору
GPT_MODEL_PATH = "C:/_1/2/author_style_gpt2_copy_2"

# Загрузка модели и токенизатора
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpt_tokenizer = GPT2Tokenizer.from_pretrained(GPT_MODEL_PATH)
gpt_model = GPT2LMHeadModel.from_pretrained(GPT_MODEL_PATH).to(device)

# Убедитесь, что специальные токены добавлены
authors = pd.read_parquet('books2.pq')['author'].unique().tolist()

# Генерация текста
def generate_text(author, prompt, max_new_tokens=100):
    """Генерация текста в стиле автора"""
    input_text = f"[AUTHOR_{author}] {prompt}"
    inputs = gpt_tokenizer(input_text, return_tensors='pt').to(device)
    
    outputs = gpt_model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.2,
        pad_token_id=gpt_tokenizer.pad_token_id or gpt_tokenizer.eos_token_id,
        num_return_sequences=1,
        do_sample=True
    )
    generated_text = gpt_tokenizer.decode(outputs[0], skip_special_tokens=False)
    return generated_text.replace(f'[AUTHOR_{author}]', '').strip()

# Оценка качества генерации
def evaluate_generation(model, tokenizer, reference_texts, generated_texts):

    min_len = min(len(generated_texts), len(reference_texts))
    generated_texts = generated_texts[:min_len]
    reference_texts = reference_texts[:min_len]

    # Вычисление BERTScore
    P, R, F1 = bert_score.score(generated_texts, reference_texts, lang="en")
    
    # Вычисление Perplexity
    perplexities = []
    for text in generated_texts:
        encodings = tokenizer(text, return_tensors='pt').to(device)
        with torch.no_grad():
            loss = model(**encodings, labels=encodings['input_ids']).loss
        perplexities.append(torch.exp(loss).item())
    
    return {
        "BERTScore-F1": round(F1.mean().item() * 100, 2),
        "Perplexity": round(np.mean(perplexities), 2)
    }


# Пример использования
if __name__ == "__main__":
    sample_author = "Fyodor_Dostoyevsky"
    sample_prompt = "The meaning of life is"
    
    # Генерация текста
    generated = generate_text(sample_author, sample_prompt)
    print(f"Сгенерированный текст: {generated[:100]}...")
    
    # Загрузка эталонных текстов
    df = pd.read_parquet('books2.pq')
    reference_texts = df[df['author'] == sample_author]['text'].tolist()
    
    # Вычисление метрик
    metrics = evaluate_generation(gpt_model, gpt_tokenizer, reference_texts, [generated])
    print(f"Метрики: {metrics}")
