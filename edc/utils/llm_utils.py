import os
import time
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import ast
from sentence_transformers import SentenceTransformer
from typing import List
import gc
import torch
import logging

logger = logging.getLogger(__name__)

# Check if local LLM mode is enabled
USE_LOCAL_LLM = os.environ.get("USE_LOCAL_LLM", "").lower() == "true"

# Only initialize OpenAI client if not using local LLM
client = None
model = None

if not USE_LOCAL_LLM:
    import openai
    
    # Read environment variables for OpenAI
    api_key = os.environ.get("OPENAI_KEY")
    base_url = os.environ.get("OPENAI_API_BASE")
    model = os.environ.get("OPENAI_MODEL")

    if not model:
        raise ValueError(
            "OPENAI_MODEL environment variable is not set. "
            "Run: source export_sambanova.sh or source export_google_ai.sh, "
            "or set USE_LOCAL_LLM=true to use local models."
        )
    if not api_key:
        raise ValueError(
            "OPENAI_KEY environment variable is not set. "
            "Or set USE_LOCAL_LLM=true to use local models."
        )

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    logger.info(f"Initialized OpenAI client with model: {model}")
else:
    logger.info("Local LLM mode enabled. OpenAI client not initialized.")


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
    try:
        left_bracket_idx = raw_entities.index("[")
        right_bracket_idx = raw_entities.index("]")
        parsed_entities = ast.literal_eval(raw_entities[left_bracket_idx : right_bracket_idx + 1])
    except (ValueError, SyntaxError) as e:
        # LLM returned malformed response without proper list format
        logging.warning(f"Could not parse entities from: {raw_entities[:100]}... Error: {e}")
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
    # Handle NaN/None values
    if not isinstance(raw_definitions, str) or not raw_definitions or raw_definitions.strip() == "":
        return {}
    
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

        # Strip number prefix (e.g., "1. relation" -> "relation", "2. relation" -> "relation")
        # This handles cases where the LLM outputs numbered lists like "1. university: ..."
        if re.match(r"^\d+\.\s*", relation):
            relation = re.sub(r"^\d+\.\s*", "", relation)

        relation_definition_dict[relation] = relation_description
    logger.debug(f"Relation Definitions {raw_definitions} parsed as {relation_definition_dict}")
    return relation_definition_dict


def is_local_llm_mode() -> bool:
    """Check if local LLM mode is enabled."""
    return USE_LOCAL_LLM


def is_model_openai(model_name: str) -> bool:
    """
    Check if we should use OpenAI API for the given model.
    
    Returns False if local LLM mode is enabled (USE_LOCAL_LLM=true),
    otherwise returns True.
    
    Args:
        model_name: The model name to check
        
    Returns:
        True if OpenAI API should be used, False for local model
    """
    if USE_LOCAL_LLM:
        return False
    return True


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
    """
    Generate a chat completion using OpenAI API or local LLM.
    
    When USE_LOCAL_LLM=true is set, this function automatically routes
    to the local LLM manager instead of calling OpenAI API.
    
    Args:
        system_prompt: System prompt for the conversation (can be None for local models,
                      or the model name when called from existing code)
        history: List of message dicts with 'role' and 'content' keys
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        max_retries: Maximum retries for API calls (only used with OpenAI)
        
    Returns:
        Generated text response
    """
    # Route to local LLM if enabled
    if USE_LOCAL_LLM:
        from edc.utils.local_llm import local_chat_completion
        return local_chat_completion(
            system_prompt=system_prompt,
            history=history,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    # Original OpenAI implementation
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
    result = response.choices[0].message.content
    logging.debug(f"Model: {model}\nPrompt:\n {messages}\n Result: {result}")
    # Return empty string if content is None to prevent downstream errors
    return result if result is not None else ""
