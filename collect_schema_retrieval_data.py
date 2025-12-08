import pandas as pd
import os
import time
import glob
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import csv
import json
import random
import edc.utils.llm_utils as llm_utils
import ast
from collections import Counter
from argparse import ArgumentParser
from datasets import Dataset, DatasetDict


def parse_export_scripts(project_root):
    """Parse export_*.sh files and return list of provider configs.
    
    Args:
        project_root: Root directory containing export_*.sh files
        
    Returns:
        List of dicts, each containing OPENAI_KEY, OPENAI_API_BASE, OPENAI_MODEL
    """
    providers = []
    for sh_file in sorted(glob.glob(os.path.join(project_root, "export_*.sh"))):
        config = {"source_file": os.path.basename(sh_file)}
        with open(sh_file) as f:
            for line in f:
                if line.startswith("export OPENAI_"):
                    # Parse: export OPENAI_KEY="value"
                    line = line.replace("export ", "").strip()
                    if "=" in line:
                        key, value = line.split("=", 1)
                        config[key.strip()] = value.strip().strip('"')
        # Only add if we have the required keys
        if "OPENAI_KEY" in config and "OPENAI_MODEL" in config:
            providers.append(config)
    return providers


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
    """Load checkpoint from file if it exists.
    
    Returns:
        Tuple of (processed_indices set, collected_relations set)
    """
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
            processed_indices = set(checkpoint.get("processed_indices", []))
            collected_relations = set(checkpoint.get("collected_relations", []))
            return processed_indices, collected_relations
    return set(), set()


def save_checkpoint(checkpoint_path, processed_indices, collected_relations, lock=None):
    """Save checkpoint to file (thread-safe if lock provided).
    
    Args:
        checkpoint_path: Path to save checkpoint file
        processed_indices: Set of processed entry indices
        collected_relations: Set of collected relation names
        lock: Optional threading.Lock for thread-safe writes
    """
    checkpoint = {
        "processed_indices": sorted(list(processed_indices)),
        "collected_relations": sorted(list(collected_relations))
    }
    
    def _write():
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
    
    if lock:
        with lock:
            _write()
    else:
        _write()


def crawl_worker(
    worker_id,
    provider_config,
    work_queue,
    result_lock,
    checkpoint_lock,
    progress_bar,
    csv_writer,
    result_csv,
    collected_relations,
    processed_indices,
    checkpoint_path,
    schema_definition_prompt_template,
    schema_definition_few_shot_examples,
    dataset_size,
    sleep_duration,
    stop_event,
):
    """Worker function for parallel crawling.
    
    Args:
        worker_id: Unique identifier for this worker
        provider_config: Dict with OPENAI_KEY, OPENAI_API_BASE, OPENAI_MODEL
        work_queue: Queue of (idx, json_dict) items to process
        result_lock: Lock for CSV writes and collected_relations updates
        checkpoint_lock: Lock for checkpoint file writes
        progress_bar: Shared tqdm progress bar
        csv_writer: CSV writer object
        result_csv: CSV file handle (for flushing)
        collected_relations: Shared set of collected relations
        processed_indices: Shared set of processed indices
        checkpoint_path: Path to checkpoint file
        schema_definition_prompt_template: Prompt template string
        schema_definition_few_shot_examples: Few-shot examples string
        dataset_size: Target number of relations to collect
        sleep_duration: Sleep duration between API calls
        stop_event: Threading event to signal workers to stop
    """
    # Create client for this worker's provider
    client, model = llm_utils.create_openai_client(provider_config)
    provider_name = provider_config.get("source_file", "unknown")
    
    while not stop_event.is_set():
        try:
            # Get item from queue with timeout to check stop_event periodically
            item = work_queue.get(timeout=0.5)
        except:
            continue
            
        if item is None:  # Poison pill
            work_queue.task_done()
            break
            
        idx, json_dict = item
        
        # Check if we should stop (dataset_size reached)
        with result_lock:
            if len(collected_relations) >= dataset_size:
                work_queue.task_done()
                stop_event.set()
                break
        
        triples = json_dict["triples"]
        text = json_dict["sentence"]
        present_relations = list(set([t[1] for t in triples]))
        
        # Check if this entry has new relations worth collecting
        with result_lock:
            new_relations = [r for r in present_relations if r not in collected_relations]
            if not new_relations:
                work_queue.task_done()
                continue
        
        # Prepare the prompt
        filled_first_prompt = schema_definition_prompt_template.format_map(
            {
                "few_shot_examples": schema_definition_few_shot_examples,
                "text": text,
                "triples": triples,
                "relations": present_relations,
            }
        )
        
        try:
            # Make API call using this worker's client
            output = llm_utils.openai_chat_completion_with_client(
                client=client,
                model=model,
                system_prompt=None,
                history=[{"role": "user", "content": filled_first_prompt}],
            )
            
            # Update shared state with lock
            with result_lock:
                # Double-check relations are still new after API call
                new_relations_count = 0
                for relation in present_relations:
                    if relation not in collected_relations:
                        collected_relations.add(relation)
                        new_relations_count += 1
                
                # Write to CSV
                csv_writer.writerow([text, triples, present_relations, output])
                result_csv.flush()
                
                # Track processed index
                processed_indices.add(idx)
                
                # Update progress bar
                if new_relations_count > 0:
                    progress_bar.update(new_relations_count)
            
            # Save checkpoint (with its own lock)
            save_checkpoint(checkpoint_path, processed_indices, collected_relations, checkpoint_lock)
            
        except Exception as e:
            print(f"Worker {worker_id} ({provider_name}) error processing idx {idx}: {e}")
        
        work_queue.task_done()
        
        # Sleep to prevent rate limiting
        if sleep_duration > 0:
            time.sleep(sleep_duration)


def crawl_relation_definitions(json_dict_list, result_csv_path, dataset_size, providers, sleep_duration=1.0):
    """
    Crawl relation definitions from the dataset using parallel workers.
    
    Args:
        json_dict_list: List of json dictionaries containing text and triples
        result_csv_path: Path to save the result CSV
        dataset_size: Target number of unique relations to collect
        providers: List of provider config dicts (from parse_export_scripts)
        sleep_duration: Time to sleep between API calls (in seconds) to prevent rate limiting
    """
    num_workers = len(providers)
    print(f"Starting parallel crawl with {num_workers} workers:")
    for i, p in enumerate(providers):
        print(f"  Worker {i}: {p.get('source_file', 'unknown')} -> {p.get('OPENAI_MODEL', 'unknown')}")
    
    schema_definition_prompt_template = open("./prompt_templates/sd_template.txt").read()
    schema_definition_few_shot_examples = open("./few_shot_examples/example/sd_few_shot_examples.txt").read()

    # Checkpoint file path (same directory as result, with .checkpoint.json extension)
    checkpoint_path = result_csv_path + ".checkpoint.json"
    
    # Load checkpoint if available
    processed_indices, collected_relations = load_checkpoint(checkpoint_path)
    
    if processed_indices:
        print(f"Resuming from checkpoint: {len(processed_indices)} entries processed, "
              f"{len(collected_relations)} relations collected")

    # Open CSV file
    if not os.path.exists(result_csv_path):
        result_csv = open(result_csv_path, "w", newline='')
        csv_writer = csv.writer(result_csv)
        csv_writer.writerow(["text", "triplets", "relations", "definitions"])
    else:
        result_csv = open(result_csv_path, "a", newline='')
        csv_writer = csv.writer(result_csv)

    # Create thread-safe primitives
    work_queue = Queue()
    result_lock = threading.Lock()
    checkpoint_lock = threading.Lock()
    stop_event = threading.Event()
    
    # Progress bar
    progress_bar = tqdm(total=dataset_size, initial=len(collected_relations), desc="Collecting relations")
    
    # Pre-filter and enqueue work items
    work_items = []
    for idx, json_dict in enumerate(json_dict_list):
        # Skip already processed entries
        if idx in processed_indices:
            continue
        
        triples = json_dict["triples"]
        skip_flag = False
        for triple in triples:
            # skip quadruples
            if len(triple) != 3:
                skip_flag = True
                break
        
        if not skip_flag:
            work_items.append((idx, json_dict))
    
    print(f"Queued {len(work_items)} work items for processing")
    
    # Add work items to queue
    for item in work_items:
        work_queue.put(item)
    
    # Add poison pills (one per worker)
    for _ in range(num_workers):
        work_queue.put(None)
    
    # Start worker threads
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for worker_id, provider_config in enumerate(providers):
            future = executor.submit(
                crawl_worker,
                worker_id,
                provider_config,
                work_queue,
                result_lock,
                checkpoint_lock,
                progress_bar,
                csv_writer,
                result_csv,
                collected_relations,
                processed_indices,
                checkpoint_path,
                schema_definition_prompt_template,
                schema_definition_few_shot_examples,
                dataset_size,
                sleep_duration,
                stop_event,
            )
            futures.append(future)
        
        # Wait for all workers to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Worker thread error: {e}")
    
    progress_bar.close()
    result_csv.close()
    
    # Remove checkpoint file when completed successfully
    if len(collected_relations) >= dataset_size and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Crawling completed successfully with {len(collected_relations)} relations. Checkpoint file removed.")
    else:
        print(f"Crawling paused with {len(collected_relations)} relations collected. Checkpoint saved.")


def collect_samples(df, dataset_size):
    # entries: list of dicts containing text and triples
    # relation_definitions: dict from relation to definitions
    collected_samples = []

    relation_definition_dict_list = []
    aggregated_relation_definition_dict = {}

    for idx, row in df.iterrows():
        raw_definitions = row["definitions"]
        relation_definition_dict = llm_utils.parse_relation_definition(raw_definitions)
        relation_definition_dict_list.append(relation_definition_dict)
        for relation, definition in relation_definition_dict.items():
            if relation not in aggregated_relation_definition_dict:
                aggregated_relation_definition_dict[relation] = [definition]
            else:
                aggregated_relation_definition_dict[relation].append(definition)

    for row_idx, row in df.iterrows():
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
        # print(len(aggregated_relation_definition_dict))
        negative_relations = set(aggregated_relation_definition_dict.keys()) - positive_relations
        # print(positive_relations)
        # print(negative_relations)
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
            # print(sample)
            collected_samples.append(sample)
            print(sample)
            if len(collected_samples) >= dataset_size:
                return collected_samples
    return collected_samples


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tekgen_path", help="Path to tekgen path")
    parser.add_argument("--relation_definition_csv_path", help="Output path of relation definition of tekgen")
    parser.add_argument("--dataset_size", default=50000, type=int)
    parser.add_argument("--output_path", default="./schema_retriever_dataset")
    parser.add_argument("--sleep_duration", default=1.0, type=float, 
                        help="Time to sleep between API calls in seconds to prevent rate limiting (default: 1.0)")
    parser.add_argument("--project_root", default=".", 
                        help="Root directory containing export_*.sh files (default: current directory)")

    args = parser.parse_args()

    tekgen_path = args.tekgen_path
    relation_definition_csv_path = args.relation_definition_csv_path
    dataset_size = args.dataset_size
    output_path = args.output_path
    sleep_duration = args.sleep_duration
    project_root = args.project_root

    # Parse export_*.sh files to get provider configurations
    providers = parse_export_scripts(project_root)
    
    if not providers:
        print(f"Error: No export_*.sh files found in {project_root}")
        print("Please create at least one export_*.sh file with OPENAI_KEY and OPENAI_MODEL exports.")
        exit(1)
    
    print(f"Found {len(providers)} API provider(s): {[p.get('source_file') for p in providers]}")

    entries = read_tekgen(tekgen_path)

    # Run crawl if: CSV doesn't exist, CSV is empty, OR a checkpoint file exists (indicating incomplete crawl)
    checkpoint_path = relation_definition_csv_path + ".checkpoint.json"
    if not os.path.exists(relation_definition_csv_path) or os.path.getsize(relation_definition_csv_path) == 0 or os.path.exists(checkpoint_path):
        crawl_relation_definitions(entries, relation_definition_csv_path, dataset_size, providers, sleep_duration)

    collected_samples = collect_samples(pd.read_csv(relation_definition_csv_path), dataset_size)

    data = Dataset.from_list(collected_samples)

    train_test_split = data.train_test_split()
    test_valid = train_test_split["test"].train_test_split(test_size=0.5)
    train_test_valid_dataset = DatasetDict(
        {"train": train_test_split["train"], "test": test_valid["test"], "valid": test_valid["train"]}
    )

    train_test_valid_dataset.save_to_disk(output_path)
