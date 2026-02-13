from pathlib import Path

print("="*70)
print("📁 LISTING ACTUAL FILENAMES")
print("="*70)

folders = [
    "results_run1/single",
    "results_run1/multi",
    "results_run2/single",
    "results_run2/multi"
]

for folder in folders:
    folder_path = Path(folder)
    if folder_path.exists():
        print(f"\n📂 {folder}:")
        txt_files = list(folder_path.glob("*.txt"))
        for f in txt_files:
            print(f"   - {f.name}")
    else:
        print(f"\n❌ {folder}: NOT FOUND")
