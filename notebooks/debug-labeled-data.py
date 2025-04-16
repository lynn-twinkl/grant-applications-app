import sys
import json
import re


json_file_path = sys.argv[1]
text_key = sys.argv[2]
out_path = sys.argv[3]

# ------------ FUNCTION ------------

def trim_and_fix_offsets(raw_data, context_key=text_key):
    """
    Attempt to fix leading/trailing whitespace in spans and recalc offsets.
    Then do a local substring search to fix minor misalignments.
    """
    fixed_data = []
    for i, record in enumerate(raw_data):
        text = record[context_key]
        new_labels = []
        for ann in record["label"]:
            label = ann["labels"][0]
            old_start, old_end = ann["start"], ann["end"]
            original_substring = text[old_start:old_end]
            trimmed_substring = original_substring.strip()
            
            # 1) Trim leading/trailing whitespace offsets
            # Move start forward while it points to space
            start = old_start
            while start < old_end and text[start].isspace():
                start += 1
            # Move end backward while it points to space
            end = old_end
            while end > start and text[end - 1].isspace():
                end -= 1
            
            # After naive trimming, see if the substring still matches
            new_substring = text[start:end]
            if new_substring == trimmed_substring:
                # Great, we can trust these offsets directly
                pass
            else:
                # Possibly there's hidden Unicode or the original offset was off.
                # We'll do a local substring search around `old_start`.
                # We'll search for `trimmed_substring` in a window of +/- 30 chars.
                window_size = 30
                
                # Define a safe search window in the text
                search_start = max(0, old_start - window_size)
                search_end = min(len(text), old_end + window_size)
                window_text = text[search_start:search_end]
                
                # Try to find the first occurrence of trimmed_substring in that window
                local_pos = window_text.find(trimmed_substring)
                if local_pos != -1:
                    # Recalc absolute offset
                    start = search_start + local_pos
                    end = start + len(trimmed_substring)
                    new_substring = text[start:end]
                else:
                    # We failed to find it in the local region
                    print(f"[Record {i}] Can't find '{trimmed_substring}' near offset {old_start}-{old_end}")
                    # We'll leave this annotation as-is or skip it
                    start, end = old_start, old_end
                    new_substring = original_substring

            new_labels.append({
                "start": start,
                "end": end,
                "text": new_substring,
                "labels": [label]
            })
        
        # Update the record with the new label data
        new_record = dict(record)
        new_record["label"] = new_labels
        fixed_data.append(new_record)
    
    return fixed_data


# ----------------- USAGE ----------------
with open(json_file_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

fixed_data = trim_and_fix_offsets(raw_data, context_key=text_key)

with open(out_path, "w", encoding="utf-8") as out:
    json.dump(fixed_data, out, indent=2, ensure_ascii=False)
