import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

# === CONFIGURATION ===
CSV_PATH = "../../data/annotated_dataset.csv"
IMAGE_ROOT = "../../data/inat"

# === LOAD CSV ===
df = pd.read_csv(CSV_PATH)

# Ensure bounding box columns exist
for col in ['x_exuviae', 'y_exuviae', 'w_exuviae', 'h_exuviae',
            'x_organism', 'y_organism', 'w_organism', 'h_organism']:
    if col not in df.columns:
        df[col] = pd.NA

# === VALIDATION LOOP ===
idx = 0
while idx < len(df):
    row = df.iloc[idx]
    split = row['split']
    stage = row['stage']
    filename = row['filename']
    img_path = os.path.join(IMAGE_ROOT, split, stage, filename)

    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        idx += 1
        continue

    img = mpimg.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img)

    # Draw exuviae box in red
    if pd.notna(row['x_exuviae']):
        rect = patches.Rectangle(
            (row['x_exuviae'], row['y_exuviae']),
            row['w_exuviae'], row['h_exuviae'],
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

    # Draw organism box in blue
    if pd.notna(row['x_organism']):
        rect = patches.Rectangle(
            (row['x_organism'], row['y_organism']),
            row['w_organism'], row['h_organism'],
            linewidth=2, edgecolor='blue', facecolor='none'
        )
        ax.add_patch(rect)

    ax.set_title(
        f"[{stage.upper()}] {filename}\nKeys: 'd' = delete, 'b' = back, 'n' = skip, 'q' = quit",
        fontsize=10
    )

    pressed_key = []

    def on_key(event):
        if event.key in ['n', 'b', 'q', 'd']:
            pressed_key.append(event.key)
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    if not pressed_key:
        continue

    key = pressed_key[0]

    if key == 'q':
        break
    elif key == 'n':
        idx += 1
    elif key == 'b':
        idx = max(0, idx - 1)
    elif key == 'd':
        for col in ['x_exuviae', 'y_exuviae', 'w_exuviae', 'h_exuviae',
                    'x_organism', 'y_organism', 'w_organism', 'h_organism']:
            df.at[df.index[idx], col] = pd.NA
        print(f"Annotations deleted for: {filename}")
        idx += 1

    # Save after each modification
    df.to_csv(CSV_PATH, index=False)

print(f"\nValidation completed. Annotations saved to: {CSV_PATH}")
