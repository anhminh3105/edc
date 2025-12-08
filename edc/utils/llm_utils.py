import os
import openai
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import ast
from sentence_transformers import SentenceTransformer
from typing import List
import gc
import torch
import logging

logger = logging.getLogger(__name__)

# Global client for single-provider usage (optional)
# These are read at import time but won't fail if not set.
# For multi-provider parallel crawling, use create_openai_client() instead.
_global_api_key = os.environ.get("OPENAI_KEY")
_global_base_url = os.environ.get("OPENAI_API_BASE")
_global_model = os.environ.get("OPENAI_MODEL")

# Only create global client if environment variables are set
client = None
if _global_api_key and _global_model:
    client = openai.OpenAI(api_key=_global_api_key, base_url=_global_base_url)


def create_openai_client(config):
    """Create an OpenAI client from a provider config dict.
    
    Args:
        config: Dict containing OPENAI_KEY, OPENAI_API_BASE (optional), OPENAI_MODEL
        
    Returns:
        Tuple of (OpenAI client, model name)
    """
    return openai.OpenAI(
        api_key=config["OPENAI_KEY"],
        base_url=config.get("OPENAI_API_BASE")
    ), config.get("OPENAI_MODEL")


def openai_chat_completion_with_client(client, model, system_prompt, history, temperature=0.1, max_tokens=512, max_retries=3):
    """Make a chat completion call using a specific client and model.
    
    Args:
        client: OpenAI client instance
        model: Model name to use
        system_prompt: System prompt (or None)
        history: List of message dicts
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        max_retries: Number of retries on failure
        
    Returns:
        The content of the assistant's response
    """
    response = None
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}] + history
    else:
        messages = history

    retries = 0
    while response is None:
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as e:
            retries += 1
            logger.warning(f"API call failed (attempt {retries}/{max_retries}): {e}")
            if retries >= max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded. Last error: {e}")
                raise
            time.sleep(5)
    logging.debug(f"Model: {model}\nPrompt:\n {messages}\n Result: {response.choices[0].message.content}")
    return response.choices[0].message.content



def free_model(model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None):
    try:
        model.cpu()
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(e)


def get_embedding_e5mistral(model, tokenizer, sentence, task=None):
    model.eval()
    device = model.device

    if task != None:
        # It's a query to be embed
        sentence = get_detailed_instruct(task, sentence)

    sentence = [sentence]

    max_length = 4096
    # Tokenize the input texts
    batch_dict = tokenizer(
        sentence, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True
    )
    # append eos_token_id to every input_ids
    batch_dict["input_ids"] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors="pt")

    batch_dict.to(device)

    embeddings = model(**batch_dict).detach().cpu()

    assert len(embeddings) == 1

    return embeddings[0]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def get_embedding_sts(model: SentenceTransformer, text: str, prompt_name=None, prompt=None):
    embedding = model.encode(text, prompt_name=prompt_name, prompt=prompt)
    return embedding


def parse_raw_entities(raw_entities: str):
    parsed_entities = []
    left_bracket_idx = raw_entities.index("[")
    right_bracket_idx = raw_entities.index("]")
    try:
        parsed_entities = ast.literal_eval(raw_entities[left_bracket_idx : right_bracket_idx + 1])
    except Exception as e:
        pass
    logging.debug(f"Entities {raw_entities} parsed as {parsed_entities}")
    return parsed_entities


def parse_raw_triplets(raw_triplets: str):
    # Look for enclosing brackets
    unmatched_left_bracket_indices = []
    matched_bracket_pairs = []

    collected_triples = []
    for c_idx, c in enumerate(raw_triplets):
        if c == "[":
            unmatched_left_bracket_indices.append(c_idx)
        if c == "]":
            if len(unmatched_left_bracket_indices) == 0:
                continue
            # Found a right bracket, match to the last found left bracket
            matched_left_bracket_idx = unmatched_left_bracket_indices.pop()
            matched_bracket_pairs.append((matched_left_bracket_idx, c_idx))
    for l, r in matched_bracket_pairs:
        bracketed_str = raw_triplets[l : r + 1]
        try:
            parsed_triple = ast.literal_eval(bracketed_str)
            if len(parsed_triple) == 3 and all([isinstance(t, str) for t in parsed_triple]):
                if all([e != "" and e != "_" for e in parsed_triple]):
                    collected_triples.append(parsed_triple)
            elif not all([type(x) == type(parsed_triple[0]) for x in parsed_triple]):
                for e_idx, e in enumerate(parsed_triple):
                    if isinstance(e, list):
                        parsed_triple[e_idx] = ", ".join(e)
                collected_triples.append(parsed_triple)
        except Exception as e:
            pass
    logger.debug(f"Triplets {raw_triplets} parsed as {collected_triples}")
    return collected_triples


def parse_relation_definition(raw_definitions: str):
    descriptions = raw_definitions.split("\n")
    relation_definition_dict = {}

    for description in descriptions:
        if ":" not in description:
            continue
        index_of_colon = description.index(":")
        relation = description[:index_of_colon].strip()

        relation_description = description[index_of_colon + 1 :].strip()

        if relation == "Answer":
            continue

        relation_definition_dict[relation] = relation_description
    logger.debug(f"Relation Definitions {raw_definitions} parsed as {relation_definition_dict}")
    return relation_definition_dict


def is_model_openai(model_name):
    return "gpt" in model_name


def generate_completion_transformers(
    input: list,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_token=256,
    answer_prepend="",
):
    device = model.device
    tokenizer.pad_token = tokenizer.eos_token

    messages = tokenizer.apply_chat_template(input, add_generation_prompt=True, tokenize=False) + answer_prepend

    model_inputs = tokenizer(messages, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    generation = model.generate(**model_inputs, generation_config=generation_config)
    sequences = generation["sequences"]
    generated_ids = sequences[:, model_inputs["input_ids"].shape[1] :]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    logging.debug(f"Prompt:\n {messages}\n Result: {generated_texts}")
    return generated_texts


def openai_chat_completion(system_prompt, history, temperature=0.1, max_tokens=512, max_retries=3):
    """Make a chat completion call using the global client (from environment variables).
    
    Note: For multi-provider parallel crawling, use openai_chat_completion_with_client() instead.
    """
    if client is None:
        raise ValueError(
            "Global OpenAI client not initialized. "
            "Either set OPENAI_KEY and OPENAI_MODEL environment variables, "
            "or use openai_chat_completion_with_client() with a provider config."
        )
    
    response = None
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}] + history
    else:
        messages = history

    retries = 0
    while response is None:
        try:
            response = client.chat.completions.create(
                model=_global_model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as e:
            retries += 1
            logger.warning(f"API call failed (attempt {retries}/{max_retries}): {e}")
            if retries >= max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded. Last error: {e}")
                raise
            time.sleep(5)
    logging.debug(f"Model: {_global_model}\nPrompt:\n {messages}\n Result: {response.choices[0].message.content}")
    return response.choices[0].message.content
