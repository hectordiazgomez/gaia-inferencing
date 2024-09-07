import os
import json
import boto3
import typing as tp
from functools import lru_cache
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

models = "models/"

@lru_cache(maxsize=1)
def load_from_s3(path, bucket_name='gaiacloud'):
    s3 = boto3.client('s3')
    print(f"Path: {path}")
    os.makedirs(models, exist_ok=True)

    model_path = f"{models}/{path}/pytorch_model.bin"
    tokenizer_path = f"{models}/{path}/tokenizer_config.json"
    config = f"{models}/{path}/config.json"
    generation = f"{models}/{path}/generation_config.json"
    sentence = f"{models}/{path}/sentencepiece.bpe.model"
    special_tokens = f"{models}/{path}/special_tokens_map.json"

    if not os.path.exists(model_path):
        print("Downloading model from S3...")
        os.makedirs(f"{models}/{path}", exist_ok=True)
        s3.download_file(bucket_name, f"models/{path}/pytorch_model.bin", model_path)
        s3.download_file(bucket_name, f"models/{path}/tokenizer_config.json", tokenizer_path)
        s3.download_file(bucket_name, f"models/{path}/config.json", config)
        s3.download_file(bucket_name, f"models/{path}/generation_config.json", generation)
        s3.download_file(bucket_name, f"models/{path}/sentencepiece.bpe.model", sentence)
        s3.download_file(bucket_name, f"models/{path}/special_tokens_map.json", special_tokens)
    else:
        print("Model already cached, reusing...")

    model = AutoModelForSeq2SeqLM.from_pretrained(f"{models}/{path}")
    tokenizer = NllbTokenizer.from_pretrained(f"{models}/{path}")

    return model, tokenizer
 

def tokenizer_for_translation(tokenizer, new_lang):
    print("Starting tokenizer_for_translation function")
    old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang] = old_len - 1
    tokenizer.id_to_lang_code[old_len - 1] = new_lang
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset
    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    if new_lang not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append(new_lang)
    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {}
    print("Finished tokenizer_for_translation function")

def translate2(text, model, tokenizer, src_lang='spa_Latn', tgt_lang='eng_Latn', max_input_length=1024, a=32, b=3, num_beams=4, **kwargs):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    print("Going to inputs now")
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length)
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams,
        **kwargs
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)

@csrf_exempt
def translation_endpoint(request):
    print("Translation endpoint called")
    try:
        data = json.loads(request.body)
        text = data.get('text')
        src_lang = data.get('src_lang')
        tgt_lang = data.get('tgt_lang')
        model_path = data.get('path')
        max_length = data.get('max_length', 128)

        print(f"Received request: text={text}, src_lang={src_lang}, tgt_lang={tgt_lang}, max_length={max_length}")

        if not all([text, src_lang, tgt_lang, model_path]):
            print("Missing required parameters")
            return JsonResponse({'error': 'Missing required parameters'}, status=400)

        try:
            print("Loading model and tokenizer")
            model, tokenizer = load_from_s3(model_path, bucket_name='gaiacloud')
            
            print("Preparing tokenizer for translation")
            tokenizer_for_translation(tokenizer, src_lang)
            
            print("Starting translation")
            translated_text = translate2(
                text,
                model,
                tokenizer,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                max_input_length=max_length
            )

            print("Translation completed")
            return JsonResponse({'translated_text': translated_text})

        except Exception as e:
            print(f"An error occurred during model loading or translation: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

    except json.JSONDecodeError:
        print("Invalid JSON received")
        return JsonResponse({'error': 'Invalid JSON'}, status=400)