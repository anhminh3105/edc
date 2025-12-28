from argparse import ArgumentParser
from edc.edc_framework import EDC
import os
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check for local LLM mode and set appropriate defaults
USE_LOCAL_LLM = os.environ.get("USE_LOCAL_LLM", "").lower() == "true"

# Default models - configured via environment variables when in local LLM mode
if USE_LOCAL_LLM:
    DEFAULT_LLM = os.environ.get("LOCAL_LLM_MODEL")
    DEFAULT_EMBEDDER = os.environ.get("LOCAL_EMBEDDER_MODEL")
    
    if not DEFAULT_LLM:
        raise ValueError(
            "LOCAL_LLM_MODEL environment variable is not set. "
            "Run: source export_local_llm.sh"
        )
    if not DEFAULT_EMBEDDER:
        raise ValueError(
            "LOCAL_EMBEDDER_MODEL environment variable is not set. "
            "Run: source export_local_llm.sh"
        )
else:
    DEFAULT_LLM = os.environ.get("OPENAI_MODEL")
    # Embedder is still a local sentence transformer, use a sensible default
    DEFAULT_EMBEDDER = os.environ.get("EMBEDDER_MODEL")
    if not DEFAULT_EMBEDDER:
        raise ValueError(
            "EMBEDDER_MODEL environment variable is not set. "
        )

if __name__ == "__main__":
    parser = ArgumentParser()
    # OIE module setting
    parser.add_argument(
        "--oie_llm", default=DEFAULT_LLM, help="LLM used for open information extraction."
    )
    parser.add_argument(
        "--oie_prompt_template_file_path",
        default="./prompt_templates/oie_template.txt",
        help="Promp template used for open information extraction.",
    )
    parser.add_argument(
        "--oie_few_shot_example_file_path",
        default="./few_shot_examples/example/oie_few_shot_examples.txt",
        help="Few shot examples used for open information extraction.",
    )

    # Schema Definition setting
    parser.add_argument(
        "--sd_llm", default=DEFAULT_LLM, help="LLM used for schema definition."
    )
    parser.add_argument(
        "--sd_prompt_template_file_path",
        default="./prompt_templates/sd_template.txt",
        help="Prompt template used for schema definition.",
    )
    parser.add_argument(
        "--sd_few_shot_example_file_path",
        default="./few_shot_examples/example/sd_few_shot_examples.txt",
        help="Few shot examples used for schema definition.",
    )

    # Schema Canonicalization setting
    parser.add_argument(
        "--sc_llm",
        default=DEFAULT_LLM,
        help="LLM used for schema canonicaliztion verification.",
    )
    parser.add_argument(
        "--sc_embedder", default=DEFAULT_EMBEDDER,
        help="Embedder used for schema canonicalization. Has to be a sentence transformer. Please refer to https://sbert.net/"
    )
    parser.add_argument(
        "--sc_prompt_template_file_path",
        default="./prompt_templates/sc_template.txt",
        help="Prompt template used for schema canonicalization verification.",
    )

    # Refinement setting
    parser.add_argument("--sr_adapter_path", default=None, help="Path to adapter of schema retriever.")
    parser.add_argument(
        "--sr_embedder", default=DEFAULT_EMBEDDER,
        help="Embedding model used for schema retriever. Has to be a sentence transformer. Please refer to https://sbert.net/"
    )
    parser.add_argument(
        "--oie_refine_prompt_template_file_path",
        default="./prompt_templates/oie_r_template.txt",
        help="Prompt template used for refined open information extraction.",
    )
    parser.add_argument(
        "--oie_refine_few_shot_example_file_path",
        default="./few_shot_examples/example/oie_few_shot_refine_examples.txt",
        help="Few shot examples used for refined open information extraction.",
    )
    parser.add_argument(
        "--ee_llm", default=DEFAULT_LLM, help="LLM used for entity extraction."
    )
    parser.add_argument(
        "--ee_prompt_template_file_path",
        default="./prompt_templates/ee_template.txt",
        help="Prompt templated used for entity extraction.",
    )
    parser.add_argument(
        "--ee_few_shot_example_file_path",
        default="./few_shot_examples/example/ee_few_shot_examples.txt",
        help="Few shot examples used for entity extraction.",
    )
    parser.add_argument(
        "--em_prompt_template_file_path",
        default="./prompt_templates/em_template.txt",
        help="Prompt template used for entity merging.",
    )

    # Input setting
    parser.add_argument(
        "--input_text_file_path",
        default="./datasets/example.txt",
        help="File containing input texts to extract KG from, each line contains one piece of text.",
    )
    parser.add_argument(
        "--target_schema_path",
        default="./schemas/example_schema.csv",
        help="File containing the target schema to align to.",
    )
    parser.add_argument(
        "--max_input_texts",
        default=None,
        type=int,
        help="Maximum number of input texts to process. If not set, all texts are processed.",
    )
    parser.add_argument(
        "--start_index",
        default=0,
        type=int,
        help="Start index for input text processing (0-based, inclusive).",
    )
    parser.add_argument(
        "--end_index",
        default=None,
        type=int,
        help="End index for input text processing (exclusive). If not set, uses start_index + max_input_texts, or processes to end if max_input_texts is also not set.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output files instead of failing if output directory exists. Use for resuming interrupted runs.",
    )
    parser.add_argument("--refinement_iterations", default=0, type=int, help="Number of iteration to run.")
    parser.add_argument(
        "--enrich_schema",
        action="store_true",
        help="Whether un-canonicalizable relations should be added to the schema.",
    )

    # Output setting
    parser.add_argument("--output_dir", default=None, help="Directory to output to. Defaults to ./output_<dataset_name>.")
    parser.add_argument("--logging_verbose", action="store_const", dest="loglevel", const=logging.INFO)
    parser.add_argument("--logging_debug", action="store_const", dest="loglevel", const=logging.DEBUG)

    args = parser.parse_args()
    args = vars(args)

    # Set output_dir based on dataset name if not explicitly provided
    dataset_name = os.path.splitext(os.path.basename(args["input_text_file_path"]))[0]
    if args["output_dir"] is None:
        args["output_dir"] = f"./output_{dataset_name}"

    edc = EDC(**args)
    

    input_text_list = open(args["input_text_file_path"], "r").readlines()
    start_idx = args["start_index"]
    end_idx = args["end_index"]

    if end_idx is None:
        if args["max_input_texts"] is not None:
            end_idx = start_idx + args["max_input_texts"]
        # else: end_idx stays None, meaning slice to end

    input_text_list = input_text_list[start_idx:end_idx]
    output_kg = edc.extract_kg(
        input_text_list,
        args["output_dir"],
        refinement_iterations=args["refinement_iterations"],
        append=args["append"],
    )
