# Phase 1: Data audit and split (target: ~1.5 h)

Data locations (I will put them here manually, never commit them):
- data/deeplure/   (Drive corpus, proprietary)
- data/kaggle/     (Indian Saree Patterns)

Tasks:
1. Write scripts/audit_data.py that prints: folder structure, image counts, resolution stats, how labels are encoded (folder names? filenames? CSV?), and whether the same design appears in multiple colorways. Show me ~10 example file paths per source (paths only, never copy images anywhere).
2. Explain to me what a "design ID" should be for each source and how confident we are in it. Flag label noise or duplicates (use perceptual hash on grayscale to find near-duplicates).
3. Propose the split: train / val / test BY DESIGN ID (no design shared across splits), plus gallery/query rules inside test. Give me 2 options with trade-offs. Wait for my choice.
4. Implement src/data/splits.py that writes a deterministic splits CSV (path, design_id, colorway_id, source, split, role) to outputs/splits.csv.
5. Add a DECISIONS.md entry for the labeling and split.

Quiz me with 3 interview questions about data and splits at the end.
