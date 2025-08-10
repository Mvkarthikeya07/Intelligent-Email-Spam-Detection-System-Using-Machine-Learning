# make_emails_csv.py
import os, pandas as pd
rows = []
root_folder = "enron_maildir"  # change this path
for subdir, dirs, files in os.walk(root_folder):
    for f in files:
        path = os.path.join(subdir, f)
        label = 1 if 'spam' in path.lower() else 0
        try:
            with open(path, 'r', errors='ignore') as fh:
                text = fh.read()
            rows.append({'message': text, 'label': label})
        except:
            pass
pd.DataFrame(rows).to_csv("emails.csv", index=False)
print("Saved emails.csv with", len(rows), "rows")
