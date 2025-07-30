import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# === CONFIGURATION ===
CSV_PATH = "../../data/annotated_dataset.csv"
IMAGE_ROOT = "../../data/inat"

# === LOAD METADATA ===
df = pd.read_csv(CSV_PATH)

# Ensure suture location columns exist
if 'x_suture' not in df.columns:
    df['x_suture'] = pd.NA
if 'y_suture' not in df.columns:
    df['y_suture'] = pd.NA

# Identify unannotated rows
idx_list = df.index[df['x_suture'].isna() | df['y_suture'].isna()].tolist()

print(f"Ready to annotate {len(idx_list)} suture points.")
i = 0

while i < len(idx_list):
    idx = idx_list[i]
    row = df.loc[idx]
    split = row['split']
    stage = row['stage']
    filename = row['filename']
    img_path = os.path.join(IMAGE_ROOT, split, stage, filename)

    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        i += 1
        continue

    img = mpimg.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(
        f"[{split}/{stage}] {filename}\nClick on the suture location. Keys: 'n' = skip, 'b' = back, 'q' = quit",
        fontsize=10
    )

    clicked = []
    key_pressed = []

    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            clicked.append((int(event.xdata), int(event.ydata)))
            plt.close()

    def on_key(event):
        key_pressed.append(event.key)
        plt.close()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    if key_pressed:
        key = key_pressed[0]
        if key == 'n':
            print(f"Skipped: {filename}")
            i += 1
            continue
        elif key == 'b':
            i = max(i - 1, 0)
            prev_idx = idx_list[i]
            df.at[prev_idx, 'x_suture'] = pd.NA
            df.at[prev_idx, 'y_suture'] = pd.NA
            continue
        elif key == 'q':
            print("Exiting annotation.")
            break
    elif clicked:
        x, y = clicked[0]
        df.at[idx, 'x_suture'] = x
        df.at[idx, 'y_suture'] = y
        print(f"Annotated: {filename} → ({x}, {y})")
        i += 1
    else:
        print(f"No click registered. Skipping: {filename}")
        i += 1

    # Save metadata after each annotation
    df.to_csv(CSV_PATH, index=False)

print("\nSuture annotation completed.")
print(f"File updated: {CSV_PATH}")
