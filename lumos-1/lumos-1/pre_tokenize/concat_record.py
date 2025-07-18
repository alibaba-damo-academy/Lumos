from argparse import ArgumentParser
import json
import os
import re
import warnings


def find_sub_records(directory: str):
    pattern = re.compile(r"\d+-of-\d+-record\.json(l)?")

    sub_record_files = [f for f in os.listdir(directory) if pattern.match(f)]
    sorted_files = sorted(sub_record_files, key=lambda filename: int(filename.split("-of")[0]))
    return sorted_files


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--sub_record_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--merge_sub_tasks", 
        action="store_true", 
        default=False
    )
    args = parser.parse_args()

    if args.merge_sub_tasks:
        sub_tasks_path = [p for p in os.listdir(args.sub_record_dir) if "-in-" in p]
        l_sub_records = []
        for s_path in sub_tasks_path:
            s_path_records = find_sub_records(os.path.join(args.sub_record_dir, s_path))
            s_path_records = [os.path.join(s_path, s) for s in s_path_records]
            l_sub_records += s_path_records
    else:
        l_sub_records = find_sub_records(args.sub_record_dir)
    
    # import ipdb
    # ipdb.set_trace()

    print(f"find {len(l_sub_records)} sub-records in {args.sub_record_dir}")
    print(str(l_sub_records) + "\n\n")

    complete_record = []
    for sub_record in l_sub_records:
        with open(os.path.join(args.sub_record_dir, sub_record)) as f:
            lines = f.readlines()
            for i, l in enumerate(lines):
                try:
                    l_item = json.loads(l)
                    complete_record.append(l_item)
                except:
                    if i == len(lines) - 1:
                        print(f"{sub_record} seems still writing, skip last incomplete record")
                    else:
                        warnings.warn(f"read line failed: {l}")
    
    print(f"Total items: {len(complete_record)}")
    with open(args.save_path, "w") as f:
        json.dump(complete_record, f)
