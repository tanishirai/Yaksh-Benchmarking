import pandas as pd
import re
from pathlib import Path

print("="*70)
print("🔄 MERGING API RESULTS INTO yaksh_100_qns.csv")
print("="*70)

# Load base CSV
df = pd.read_csv("data/raw/yaksh_100_qns.csv")
print(f"\n✅ Loaded base CSV: {df.shape}")
print(f"   Columns: {list(df.columns)[:5]}...")

# Define paths
runs = ['run1', 'run2']
modes = ['single', 'multi']
models = [
    'anthropic_claude-sonnet-4.5',
    'deepseek_deepseek-v3.2',
    'google_gemini-2.5-flash',
    'openai_gpt-5.2',
    'openai_gpt-oss-120b',
    'qwen_qwen3-coder'
]

def clean_prediction(line):
    """
    Remove question numbers from predictions
    Examples:
      "1 D" -> "D"
      "2 C,D" -> "C,D"
      "1. NONE" -> "NONE"
      "100 G" -> "G"
      "D" -> "D" (already clean)
    """
    line = line.strip()
    
    # Pattern 1: "number. prediction" (e.g., "1. D")
    # Pattern 2: "number prediction" (e.g., "1 D")
    # Pattern 3: "number) prediction" (e.g., "1) D")
    match = re.match(r'^\d+[.\):\s]\s*(.+)$', line)
    if match:
        return match.group(1).strip()
    
    # If no number prefix, return as is
    return line


print("\n📁 Scanning folders: results_run1/, results_run2/")

# Process each combination
added_columns = 0
missing_files = []

for run in runs:
    for mode in modes:
        folder_path = Path(f"results_{run}") / mode
        
        if not folder_path.exists():
            print(f"\n❌ Folder not found: {folder_path}")
            continue
        
        print(f"\n📂 Processing: {folder_path}")
        
        for model in models:
            # Filename pattern: {model}_{mode}.txt
            file_name = f"{model}_{mode}.txt"
            file_path = folder_path / file_name
            
            col_name = f"results_{run}_{mode}_{model}"
            
            if file_path.exists():
                print(f"   ✅ {file_name}")
                
                # Read predictions
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Clean predictions (remove question numbers)
                    predictions = []
                    for line in lines:
                        cleaned = clean_prediction(line)
                        if cleaned:  # Skip empty lines
                            predictions.append(cleaned)
                    
                    # Verify count
                    if len(predictions) != len(df):
                        print(f"      ⚠️  Has {len(predictions)} predictions, expected {len(df)}")
                        # Pad or trim
                        if len(predictions) < len(df):
                            predictions.extend([''] * (len(df) - len(predictions)))
                        else:
                            predictions = predictions[:len(df)]
                    
                    # Add to dataframe
                    df[col_name] = predictions
                    added_columns += 1
                    
                    # Show sample (before and after cleaning)
                    if lines:
                        raw_sample = lines[0].strip()
                        clean_sample = predictions[0]
                        print(f"      Raw: '{raw_sample}' → Clean: '{clean_sample}'")
                    
                except Exception as e:
                    print(f"      ❌ Error reading file: {e}")
                    missing_files.append(f"{run}/{mode}/{file_name}")
                
            else:
                print(f"   ❌ NOT FOUND: {file_name}")
                missing_files.append(f"{run}/{mode}/{file_name}")

print("\n" + "="*70)
print(f"📊 SUMMARY")
print("="*70)
print(f"✅ Successfully added: {added_columns}/24 result columns")
print(f"❌ Failed/Missing: {len(missing_files)} files")

if missing_files:
    print(f"\n⚠️  Missing/Failed files:")
    for mf in missing_files:
        print(f"   - {mf}")

print(f"\nFinal DataFrame shape: {df.shape}")
print(f"Total columns: {len(df.columns)}")

# Show added columns
result_cols = [c for c in df.columns if c.startswith('results_')]
print(f"\n📋 Result columns added ({len(result_cols)}):")
for i, col in enumerate(result_cols, 1):
    print(f"   {i}. {col}")

# Save combined CSV
output_path = "data/processed/yaksh_100_qns_with_api_results.csv"
Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"\n💾 Saved: {output_path}")

# Show preview
print("\n📋 Preview (first 3 rows, select columns):")
preview_cols = ['Sr. no.']
if result_cols:
    preview_cols.extend(result_cols[:3])  # First 3 result columns
    if all(c in df.columns for c in preview_cols):
        print(df[preview_cols].head(3).to_string(index=False, max_colwidth=30))

print("\n" + "="*70)
print("🎉 MERGE COMPLETE!")
print("="*70)
print(f"\nNow you can use: data/processed/yaksh_100_qns_with_api_results.csv")
print("This CSV contains all 100 questions + 24 model prediction columns!")
