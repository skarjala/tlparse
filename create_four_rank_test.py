#!/usr/bin/env python3
"""
Create working 4-rank log files by duplicating and modifying existing working 2-rank logs
"""

import shutil
import json
from pathlib import Path

def modify_chromium_events_for_rank(content, new_rank):
    """Modify chromium events in log content to use new rank number"""
    lines = content.split('\n')
    modified_lines = []

    for line in lines:
        if line.strip().startswith('{') and '"rank":' in line:
            # This is a chromium event JSON line, modify the rank
            try:
                event = json.loads(line.strip())
                if 'args' in event and 'rank' in event['args']:
                    event['args']['rank'] = str(new_rank)
                line = '\t' + json.dumps(event)
            except:
                pass  # If parsing fails, keep original line
        modified_lines.append(line)

    return '\n'.join(modified_lines)

def create_four_rank_logs():
    """Create 4-rank log files from existing 2-rank logs"""
    source_dir = Path("test_multi_rank")
    dest_dir = Path("test_four_ranks")

    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} not found")
        return False

    # Remove and recreate destination directory
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir()

    # Copy and modify existing rank files
    for rank in range(4):
        # Use rank 0 or 1 as source (cycle between them)
        source_rank = rank % 2
        source_file = source_dir / f"rank_{source_rank}.log"
        dest_file = dest_dir / f"rank_{rank}.log"

        if not source_file.exists():
            print(f"Error: Source file {source_file} not found")
            return False

        # Read source content
        with open(source_file, 'r') as f:
            content = f.read()

        # Modify rank information in chromium events
        modified_content = modify_chromium_events_for_rank(content, rank)

        # Write to destination
        with open(dest_file, 'w') as f:
            f.write(modified_content)

        print(f"Created {dest_file} from {source_file}")

    print(f"\nCreated 4-rank test logs in {dest_dir}")
    return True

def main():
    """Main function"""
    print("Creating 4-rank test logs from existing working 2-rank logs...")

    if create_four_rank_logs():
        print("\n✅ Success! Now run:")
        print("tlparse --all-ranks test_four_ranks -o four_rank_output --overwrite")
    else:
        print("\n❌ Failed to create 4-rank logs")

if __name__ == "__main__":
    main()
