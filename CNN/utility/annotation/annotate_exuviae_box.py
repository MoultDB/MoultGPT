import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

# === CONFIGURATION ===
CSV_PATH = "../../data/annotated_dataset.csv"
IMAGE_ROOT = "../../data/inat"
TARGET_STAGES = {"moulting", "post-moult", "exuviae"}

# === LOAD METADATA ===
df = pd.read_csv(CSV_PATH)

# Ensure exuviae box columns exist
for col in ["x_exuviae", "y_exuviae", "w_exuviae", "h_exuviae"]:
    if col not in df.columns:
        df[col] = pd.NA

# Check that organism box columns exist, otherwise exit with message
required_organism_cols = ["x_organism", "y_organism", "w_organism", "h_organism"]
missing_organism_cols = [col for col in required_organism_cols if col not in df.columns]
if missing_organism_cols:
    print(f"Missing columns for organism box: {missing_organism_cols}")
    print("Please run annotate_organism_box.py first to annotate organism bounding boxes.")
    sys.exit(1)

# Filter: only target stages and missing exuviae box
df_indices = df[
    (df['stage'].isin(TARGET_STAGES)) &
    ((~df['x_organism'].isna()) | (df['stage'] == 'exuviae')) &
    (df['x_exuviae'].isna() | df['y_exuviae'].isna() |
     df['w_exuviae'].isna() | df['h_exuviae'].isna())
].index.tolist()

print(f"Annotating {len(df_indices)} images from stages: {TARGET_STAGES}")
i = 0

while i < len(df_indices):
    idx = df_indices[i]
    row = df.loc[idx]
    split = row['split']
    stage = row['stage']
    fname = row['filename']
    img_path = os.path.join(IMAGE_ROOT, split, stage, fname)

    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        i += 1
        continue

    img = mpimg.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"[{stage.upper()}] {fname}\nDraw box for EXUVIAE ←→. Press 'n'=skip, 'b'=back, 'q'=quit.", fontsize=10)

    rect_coords = []
    current_rect = [None]
    skip_flag = [False]
    back_flag = [False]
    quit_flag = [False]

    def on_press(event):
        rect_coords.clear()
        rect_coords.append((event.xdata, event.ydata))
        current_rect[0] = Rectangle((event.xdata, event.ydata), 1, 1,
                                    linewidth=1.5, edgecolor='r', facecolor='none')
        ax.add_patch(current_rect[0])
        fig.canvas.draw()

    def on_motion(event):
        if current_rect[0] is not None and event.xdata and event.ydata:
            x0, y0 = rect_coords[0]
            x1, y1 = event.xdata, event.ydata
            x_min = min(x0, x1)
            y_min = min(y0, y1)
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            current_rect[0].set_bounds(x_min, y_min, w, h)
            fig.canvas.draw_idle()

    def on_release(event):
        if rect_coords and not skip_flag[0] and not back_flag[0] and not quit_flag[0]:
            x0, y0 = rect_coords[0]
            x1, y1 = event.xdata, event.ydata
            x_min = int(min(x0, x1))
            y_min = int(min(y0, y1))
            w = int(abs(x1 - x0))
            h = int(abs(y1 - y0))
            df.at[idx, 'x_exuviae'] = x_min
            df.at[idx, 'y_exuviae'] = y_min
            df.at[idx, 'w_exuviae'] = w
            df.at[idx, 'h_exuviae'] = h
            print(f"Annotated {fname} → box: ({x_min}, {y_min}, {w}, {h})")
            plt.close()

    def on_key(event):
        if event.key == 'n':
            skip_flag[0] = True
            print(f"Skipped {fname}")
            plt.close()
        elif event.key == 'b':
            back_flag[0] = True
            print("Going back to previous image")
            plt.close()
        elif event.key == 'q':
            quit_flag[0] = True
            print("Quitting annotation.")
            plt.close()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

    if quit_flag[0]:
        break
    elif back_flag[0]:
        if i > 0:
            i -= 1
            b_idx = df_indices[i]
            df.at[b_idx, 'x_exuviae'] = pd.NA
            df.at[b_idx, 'y_exuviae'] = pd.NA
            df.at[b_idx, 'w_exuviae'] = pd.NA
            df.at[b_idx, 'h_exuviae'] = pd.NA
        continue
    else:
        i += 1
        df.to_csv(CSV_PATH, index=False)

print("\nAnnotation session ended.")
print(f"All changes saved to: {CSV_PATH}")
