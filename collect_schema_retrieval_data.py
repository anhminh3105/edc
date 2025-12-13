import pandas as pd
import os
import time
import logging
from tqdm import tqdm
import csv
import json
import random
import edc.utils.llm_utils as llm_utils
import ast
from collections import Counter
from argparse import ArgumentParser
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


def read_tekgen(tekgen_path):
    json_dict_list = []
    with open(tekgen_path, "r") as f:
        lines = f.readlines()
        for l in tqdm(lines):
            line_json_dict = json.loads(l)
            triples = line_json_dict["triples"]
            text = line_json_dict["sentence"]

            skip_flag = False
            for triple in triples:
                # skip quadruples
                if len(triple) != 3:
                    skip_flag = True
                else:
                    subject = triple[0]
                    relation = triple[1]
                    object = triple[2]

                    # Check if subject and object are present in text
                    if subject.lower() not in text.lower() or object.lower() not in text.lower():
                        skip_flag = True
            if not skip_flag:
                json_dict_list.append(line_json_dict)
    return json_dict_list


def load_checkpoint(checkpoint_path):
    """Load checkpoint from file if it exists."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
            return checkpoint.get("last_processed_idx", -1), set(checkpoint.get("collected_relations", []))
    return -1, set()


def save_checkpoint(checkpoint_path, last_processed_idx, collected_relations):
    """Save checkpoint to file."""
    checkpoint = {
        "last_processed_idx": last_processed_idx,
        "collected_relations": list(collected_relations)
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)


def crawl_relation_definitions(json_dict_list, result_csv_path, dataset_size, sleep_duration=1.0):
    """
    Crawl relation definitions from the dataset.
    
    Args:
        json_dict_list: List of json dictionaries containing text and triples
        result_csv_path: Path to save the result CSV
        dataset_size: Target number of unique relations to collect
        sleep_duration: Time to sleep between API calls (in seconds) to prevent rate limiting
    """
    logger.info(f"Starting crawl_relation_definitions with dataset_size={dataset_size}, sleep_duration={sleep_duration}")
    logger.debug(f"Result CSV path: {result_csv_path}")
    logger.debug(f"Total entries in json_dict_list: {len(json_dict_list)}")
    
    schema_definition_prompt_template = open("./prompt_templates/sd_template.txt").read()
    schema_definition_few_shot_examples = open("./few_shot_examples/example/sd_few_shot_examples.txt").read()
    logger.debug("Loaded prompt template and few-shot examples")

    # Checkpoint file path (same directory as result, with .checkpoint.json extension)
    checkpoint_path = result_csv_path + ".checkpoint.json"
    
    # Load checkpoint if available
    last_processed_idx, collected_relations = load_checkpoint(checkpoint_path)
    
    if last_processed_idx >= 0:
        logger.info(f"Resuming from checkpoint: last processed index = {last_processed_idx}, "
                    f"collected relations = {len(collected_relations)}")
    else:
        logger.info("No checkpoint found, starting fresh")

    # Check if file needs header (doesn't exist or is empty)
    needs_header = not os.path.exists(result_csv_path) or os.path.getsize(result_csv_path) == 0
    
    if not os.path.exists(result_csv_path):
        result_csv = open(result_csv_path, "w")
        logger.debug(f"Created new result CSV file: {result_csv_path}")
    else:
        result_csv = open(result_csv_path, "a")
        logger.debug(f"Appending to existing result CSV file: {result_csv_path}")
    
    csv_writer = csv.writer(result_csv)
    
    # Write header if needed
    if needs_header:
        csv_writer.writerow(["text", "triplets", "relations", "definitions"])
        result_csv.flush()
        logger.debug("Wrote CSV header")

    total_entries = len(json_dict_list)
    start_idx = last_processed_idx + 1 if last_processed_idx >= 0 else 0
    progress_bar = tqdm(total=total_entries, initial=start_idx, desc=f"Processing entries (relations: {len(collected_relations)})")
    
    try:
        for idx, json_dict in enumerate(json_dict_list):
            # Skip already processed entries
            if idx <= last_processed_idx:
                logger.debug(f"Skipping index {idx}: already processed")
                continue
                
            if len(collected_relations) >= dataset_size:
                logger.info(f"Reached dataset size {dataset_size}, stopping")
                break
                
            triples = json_dict["triples"]
            skip_flag = False
            skip_reason = None
            
            for triple in triples:
                print(triple)
                # skip quadruples
                if len(triple) != 3:
                    logger.debug(f"Index {idx}: skipping due to quadruple (len={len(triple)})")
                    skip_flag = True
                    skip_reason = "quadruple"
                    break
                relation = triple[1]
                if relation in collected_relations:
                    logger.debug(f"Index {idx}: skipping, relation '{relation}' already collected")
                    skip_flag = True
                    skip_reason = f"relation '{relation}' already collected"
                    break
                    
            if skip_flag:
                logger.debug(f"Skipping index {idx}: {skip_reason}")
                progress_bar.update(1)
                progress_bar.set_description(f"Processing entries (relations: {len(collected_relations)})")
                continue
            
            # Process this entry
            new_relations_count = 0
            for triple in triples:
                relation = triple[1]
                if relation not in collected_relations:
                    collected_relations.add(relation)
                    new_relations_count += 1
                    
            logger.debug(f"Index {idx}: added {new_relations_count} new relation(s), total={len(collected_relations)}")
            
            text = json_dict["sentence"]
            triples = json_dict["triples"]
            present_relations = list(set([t[1] for t in triples]))

            filled_first_prompt = schema_definition_prompt_template.format_map(
                {
                    "few_shot_examples": schema_definition_few_shot_examples,
                    "text": text,
                    "triples": triples,
                    "relations": present_relations,
                }
            )

            logger.debug(f"Index {idx}: calling LLM API for relations {present_relations}")
            output = llm_utils.openai_chat_completion(
                system_prompt=None,
                history=[{"role": "user", "content": filled_first_prompt}],
            )
            logger.debug(f"Index {idx}: LLM API call successful, output length={len(output) if output else 0}")
            
            csv_writer.writerow([text, triples, present_relations, output])
            result_csv.flush()
            
            # Save checkpoint after each successful API call
            save_checkpoint(checkpoint_path, idx, collected_relations)
            logger.debug(f"Index {idx}: checkpoint saved")
            
            progress_bar.update(1)
            progress_bar.set_description(f"Processing entries (relations: {len(collected_relations)})")
            
            # Sleep to prevent rate limiting
            if sleep_duration > 0:
                logger.debug(f"Sleeping for {sleep_duration} seconds")
                time.sleep(sleep_duration)
                
    except Exception as e:
        logger.error(f"Error at index {idx}: {e}")
        progress_bar.close()
        result_csv.close()
        raise
    
    progress_bar.close()
    result_csv.close()
    
    # Remove checkpoint file when completed successfully
    if len(collected_relations) >= dataset_size and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info(f"Crawling completed successfully. Checkpoint file removed.")
    else:
        logger.info(f"Crawling finished. Collected {len(collected_relations)} relations.")


def collect_samples(df, dataset_size):
    # entries: list of dicts containing text and triples
    # relation_definitions: dict from relation to definitions
    collected_samples = []

    # Skip incomplete rows (missing/blank definitions). We do not retry them.
    if "definitions" in df.columns:
        df = df[
            df["definitions"].notna()
            & df["definitions"].astype(str).str.strip().ne("")
        ].reset_index(drop=True)

    relation_definition_dict_list = []
    aggregated_relation_definition_dict = {}

    for idx, row in df.iterrows():
        raw_definitions = row["definitions"]
        relation_definition_dict = llm_utils.parse_relation_definition(raw_definitions)
        
        # Skip rows with no valid definitions (e.g., API failures during crawling)
        if not relation_definition_dict:
            relation_definition_dict_list.append({})
            continue
            
        relation_definition_dict_list.append(relation_definition_dict)
        for relation, definition in relation_definition_dict.items():
            if relation not in aggregated_relation_definition_dict:
                aggregated_relation_definition_dict[relation] = [definition]
            else:
                aggregated_relation_definition_dict[relation].append(definition)

    for row_idx, row in df.iterrows():
        # Skip rows with no valid definitions
        if not relation_definition_dict_list[row_idx]:
            continue
            
        text = row["text"]
        triples = ast.literal_eval(row["triplets"])

        positive_relations = set()

        relation_triple_dict = {}

        for triple in triples:
            subject = triple[0]
            relation = triple[1]
            object = triple[2]

            # Check if subject and object are present in text
            if subject.lower() not in text.lower() or object.lower() not in text.lower():
                print(f"{triple} not explicitly in {text}")
                continue

            if relation in relation_definition_dict_list[row_idx]:
                positive_relations.add(relation)
                if relation not in relation_triple_dict:
                    relation_triple_dict[relation] = [triple]
                else:
                    relation_triple_dict[relation].append(triple)
        negative_relations = set(aggregated_relation_definition_dict.keys()) - positive_relations
        negative_relations = random.sample(list(negative_relations), len(positive_relations))

        positive_relations = list(positive_relations)
        negative_relations = list(negative_relations)

        assert len(positive_relations) == len(negative_relations)

        for idx in range(len(negative_relations)):
            if idx >= 2:
                # Max 3 samples per sentence to ensure diversity of datasets
                break
            sample = {
                "sentence": text,
                "positive": f"{positive_relations[idx]}: {relation_definition_dict_list[row_idx][positive_relations[idx]]}",
                "negative": f"{negative_relations[idx]}: {random.choice(aggregated_relation_definition_dict[negative_relations[idx]])}",
                "positive_relation": positive_relations[idx],
                "negative_relation": negative_relations[idx],
                "positive_triple": relation_triple_dict[positive_relations[idx]],
            }
            collected_samples.append(sample)
            if len(collected_samples) >= dataset_size:
                return collected_samples
    return collected_samples


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tekgen_path", help="Path to tekgen path")
    parser.add_argument("--relation_definition_csv_path", help="Output path of relation definition of tekgen")
    parser.add_argument("--dataset_size", default=796982, type=int)
    parser.add_argument("--output_path", default="./schema_retriever_dataset")
    parser.add_argument("--sleep_duration", default=1.0, type=float, 
                        help="Time to sleep between API calls in seconds to prevent rate limiting (default: 1.0)")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    tekgen_path = args.tekgen_path
    relation_definition_csv_path = args.relation_definition_csv_path
    dataset_size = args.dataset_size
    output_path = args.output_path
    sleep_duration = args.sleep_duration

    entries = read_tekgen(tekgen_path)

    # Run crawl if: CSV doesn't exist, CSV is empty, OR a checkpoint file exists (indicating incomplete crawl)
    checkpoint_path = relation_definition_csv_path + ".checkpoint.json"
    if not os.path.exists(relation_definition_csv_path) or os.path.getsize(relation_definition_csv_path) == 0 or os.path.exists(checkpoint_path):
        crawl_relation_definitions(entries, relation_definition_csv_path, dataset_size, sleep_duration)

    collected_samples = collect_samples(pd.read_csv(relation_definition_csv_path), dataset_size)

    data = Dataset.from_list(collected_samples)

    train_test_split = data.train_test_split()
    test_valid = train_test_split["test"].train_test_split(test_size=0.5)
    train_test_valid_dataset = DatasetDict(
        {"train": train_test_split["train"], "test": test_valid["test"], "valid": test_valid["train"]}
    )

    train_test_valid_dataset.save_to_disk(output_path)
