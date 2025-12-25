import nltk
import os

NLTK_DIR = "/opt/render/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)

# Download all required resources
resources = [
    'punkt',
    'punkt_tab',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng',
    'wordnet',
    'omw-eng',
    'tagsets',
    'universal_tagset'
]

for resource in resources:
    print(f"Downloading {resource}...")
    try:
        nltk.download(resource, download_dir=NLTK_DIR)
        print(f"  {resource} downloaded successfully")
    except Exception as e:
        print(f"  Error downloading {resource}: {e}")

print("All downloads attempted!")
