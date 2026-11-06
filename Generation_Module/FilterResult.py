#!/usr/bin/env python3
import json
import os
import argparse

def filter_planning_results(input_path, output_path):
    """
    Read a JSON file and remove items whose filter_result is "NO".

    Args:
        input_path: input JSON file path
        output_path: output JSON file path
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        original_count = len(data)
        print(f"Original data contains {original_count} items")

        filtered_data = [item for item in data if not (item.get('filter_result') == "NO" or
                                                    item.get('is_valid') is False)]

        filtered_count = len(filtered_data)
        print(f"Filtered data contains {filtered_count} items")
        print(f"Removed {original_count - filtered_count} items with filter_result 'NO'")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)

        print(f"Filtered data saved to {output_path}")

    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--output_file", type=str, default="",
                      help="Specify the output directory (default: reward_results)")

    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    filter_planning_results(input_file, output_file)
