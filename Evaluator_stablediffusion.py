import pandas as pd
from difflib import get_close_matches
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support, classification_report


INPUT_CSV_PATH = r"XXXXX"
OUTPUT_CSV_PATH = r"XXXXX"
GROUND_TRUTH_PATH = r"XXXX"
CONFUSION_MATRIX_PNG = r"XXXX"

SYNONYMS = {
    "airplane": ["airplane", "plane", "aeroplane", "jet", "aircraft", "airliner", "propeller plane", "glider", "fighter jet"],
    "chair": ["chair", "seat", "stool", "armchair", "rocking chair", "dining chair", "office chair"],
    "table": ["table", "desk", "workbench", "dining table", "kitchen table", "coffee table"],
    "sofa": ["sofa", "couch", "loveseat", "settee", "sectional"],
    "bed": ["bed", "bunk", "mattress", "double bed", "single bed", "king size bed"],
    "lamp": ["lamp", "light", "lantern", "table lamp", "floor lamp", "ceiling light"],
    "car": ["car", "automobile", "vehicle", "sedan", "coupe", "sports car", "hatchback"],
    "bottle": ["bottle", "flask", "jug", "container", "vessel", "canister"],
    "bookshelf": ["bookshelf", "bookcase", "shelf", "library shelf", "storage shelf"],
    "tv_stand": ["tv", "television", "screen", "monitor", "flat screen", "lcd", "led tv"],
    "keyboard": ["keyboard", "keypad", "piano keys", "input device", "computer keyboard", "qwerty"],
    "laptop": ["laptop", "notebook", "computer", "macbook", "portable computer", "personal computer", "device"],
    "range_hood": ["range", "stove", "cooktop", "oven", "burner", "kitchen range", "cooking appliance"],
    "mantel": ["mantel", "mantelpiece", "fireplace shelf", "fireplace ledge", "chimney shelf", "above fireplace"],
    "piano": ["piano", "grand piano", "upright piano", "music instrument", "instrument with keys"],
    "radio": ["radio", "transistor", "boombox", "receiver", "audio device", "radio player", "speaker"],
    "sink": ["sink", "washbasin", "basin", "kitchen sink", "bathroom sink", "utility sink", "cleaning station"],
    "stairs": ["stairs", "staircase", "steps", "stairway", "flight of stairs", "ladder"],
    "tent": ["tent", "canopy", "camp", "camping shelter", "fabric shelter", "outdoor shelter", "pavilion"],
    "toilet": ["toilet", "wc", "lavatory", "restroom", "commode", "urinal"],
    "glass_box": ["glass", "cup", "tumbler", "goblet", "drinking glass", "beaker"],
    "xbox": ["xbox", "console", "gaming console", "game system", "xbox one", "xbox series x", "video game console", "controller"],
    "cone": ["cone", "traffic cone", "ice cream cone", "orange cone"],
    "wardrobe": ["wardrobe", "closet", "dresser", "clothes cabinet", "armoire", "storage for clothes"],
    "curtain": ["curtain", "drape", "drapery", "window covering", "window curtain"],
    "flower_pot": ["flower", "blossom", "bloom", "plant", "floral"],
    "cup": ["cup", "mug", "glass", "teacup", "coffee cup"],
    "door": ["door", "entrance", "gate", "entryway", "wooden door"],
    "bench": ["bench", "seat", "pew", "workbench", "outdoor bench"],
    "stool": ["stool", "barstool", "seat", "step stool"],
    "plant": ["plant", "potted plant", "greenery", "foliage", "indoor plant"],
    "person": ["person", "human", "figure", "man", "woman", "individual"],
    "guitar": ["guitar", "instrument", "electric guitar", "acoustic guitar", "six string"],
    "vase": ["vase", "flower holder", "urn", "ceramic vase", "glass vase"],
    "bowl": ["bowl", "dish", "basin", "mixing bowl", "salad bowl"]
}

def extract_expected_label(filename):
    return filename.split('_')[0].lower()

def is_prediction_correct(prompt, expected_label, synonyms_dict, similarity_threshold=0.8):
    prompt_words = prompt.lower().replace(",", "").replace(".", "").split()
    candidates = synonyms_dict.get(expected_label, [expected_label])
    for word in prompt_words:
        close_matches = get_close_matches(word, candidates, n=1, cutoff=similarity_threshold)
        if close_matches:
            return True
    return False

df = pd.read_csv(INPUT_CSV_PATH)
df['expected_label'] = df['filename'].apply(extract_expected_label)
df['pred'] = df['prompt'].str.lower().str.strip()
df['correct'] = df.apply(lambda row: is_prediction_correct(row['prompt'], row['expected_label'], SYNONYMS), axis=1)

df.to_csv(OUTPUT_CSV_PATH, index=False)

class_report = df.groupby('expected_label')['correct'].agg(['mean', 'count']).reset_index()
class_report.rename(columns={'mean': 'accuracy', 'count': 'total_samples'}, inplace=True)
class_report.to_csv(GROUND_TRUTH_PATH, index=False)

df['true'] = df['expected_label']
df['pred_label'] = df.apply(lambda row: row['expected_label'] if row['correct'] else "unknown", axis=1)
classes = sorted(set(df['true']) | set(df['pred_label']))
cm = confusion_matrix(df["true"], df["pred_label"], labels=classes)

fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(include_values=True, cmap="Blues", ax=ax, xticks_rotation=90)
plt.title("Confusion Matrix (Synonym-Based Classification)")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PNG, dpi=300)
plt.show()

accuracy = accuracy_score(df["true"], df["pred_label"])
print(f"Accuracy: {accuracy:.2%}")

precision, recall, f1, support = precision_recall_fscore_support(
    df["true"], df["pred_label"], labels=classes, zero_division=0
)

print("Per-Class")
for cls, p, r, f, s in zip(classes, precision, recall, f1, support):
    print(f"- {cls:<12} | Precision: {p:.2%} | Recall: {r:.2%} | F1 Score: {f:.2%} | Support: {s}")

print("Classification")
print(classification_report(df["true"], df["pred_label"], labels=classes, zero_division=0))
