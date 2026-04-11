import os
import sys
import bpy
import bmesh
import mathutils
import logging
from datetime import datetime

# ... existing imports and constants ...

def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'processing_log.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def check_incomplete_processing(log_file):
    incomplete_files = set()
    with open(log_file, 'r') as f:
        for line in f:
            if 'START' in line:
                file_name = line.split(' - ')[-1].strip()
                incomplete_files.add(file_name)
            elif 'END' in line:
                file_name = line.split(' - ')[-1].strip()
                incomplete_files.discard(file_name)
    return incomplete_files

def process_obj_files(input_dir, output_dir, handle_errors=False):
    setup_logging(output_dir)
    log_file = os.path.join(output_dir, 'processing_log.txt')
    incomplete_files = check_incomplete_processing(log_file)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.obj'):
            if file_name in incomplete_files:
                logging.info(f"Skipping previously incomplete file: {file_name}")
                continue

            input_file = os.path.join(input_dir, file_name)
            output_file = os.path.join(output_dir, file_name)

            logging.info(f"START - {file_name}")

            try:
                # ... existing processing code ...

                logging.info(f"END - {file_name}")
            except Exception as e:
                logging.error(f"Error processing {file_name}: {str(e)}")
                if not handle_errors:
                    raise

def main(handle_errors=False):
    # ... existing code to get input_dir and output_dir ...

    try:
        process_obj_files(input_dir, output_dir, handle_errors)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main(handle_errors=True)  # Set to False if you want the script to stop on errors