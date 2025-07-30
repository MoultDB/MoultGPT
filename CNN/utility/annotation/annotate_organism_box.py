import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

# === CONFIGURATION ===
CSV_PATH = "../../data/annotated_dataset.csv"
IMAGE_ROOT = "../../data/inat"
TARGET_STAGES = {"moulting", "post-moult", "pre-moult"}

# === LOAD METADATA ===
df = pd.read_csv(CSV_PATH)

# Ensure organism + exuviae box columns exist
for col in ["x_exuviae", "y_exuviae", "w_exuviae", "h_exuviae",
            "x_organism", "y_organism", "w_organism", "h_organism"]:
    if col not in df.columns:
        df[col] = pd.NA

# === FILTER ===
df_indices = df[
    (df['stage'].isin(TARGET_STAGES)) &
    (df['x_organism'].isna() | df['y_organism'].isna() |
     df['w_organism'].isna() | df['h_organism'].isna())
].index.tolist()

print(f"Ready to annotate {len(df_indices)} organism boxes.")

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
    ax.set_title(f"[{stage.upper()}] {fname}\nExuvia = red | Draw organism box (← drag). Press 'n'=skip, 'b'=back, 'q'=quit", fontsize=10)

    # Draw existing exuvia box if present
    ex_x = row['x_exuviae']
    ex_y = row['y_exuviae']
    ex_w = row['w_exuviae']
    ex_h = row['h_exuviae']
    if pd.notna(ex_x) and pd.notna(ex_y) and pd.notna(ex_w) and pd.notna(ex_h):
        ex_rect = Rectangle((ex_x, ex_y), ex_w, ex_h,
                            linewidth=1.5, edgecolor='r', facecolor='none')
        ax.add_patch(ex_rect)

    # Prepare organism box drawing
    rect_coords = []
    organism_rect = [None]
    skip_flag = [False]
    back_flag = [False]
    quit_flag = [False]

    def on_press(event):
        rect_coords.clear()
        rect_coords.append((event.xdata, event.ydata))
        organism_rect[0] = Rectangle((event.xdata, event.ydata), 1, 1,
                                     linewidth=1.5, edgecolor='blue', facecolor='none')
        ax.add_patch(organism_rect[0])
        fig.canvas.draw()

    def on_motion(event):
        if organism_rect[0] is not None and event.xdata and event.ydata:
            x0, y0 = rect_coords[0]
            x1, y1 = event.xdata, event.ydata
            x_min = min(x0, x1)
            y_min = min(y0, y1)
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            organism_rect[0].set_bounds(x_min, y_min, w, h)
            fig.canvas.draw_idle()

    def on_release(event):
        if rect_coords and not skip_flag[0] and not back_flag[0] and not quit_flag[0]:
            x0, y0 = rect_coords[0]
            x1, y1 = event.xdata, event.ydata
            x_min = int(min(x0, x1))
            y_min = int(min(y0, y1))
            w = int(abs(x1 - x0))
            h = int(abs(y1 - y0))
            df.at[idx, 'x_organism'] = x_min
            df.at[idx, 'y_organism'] = y_min
            df.at[idx, 'w_organism'] = w
            df.at[idx, 'h_organism'] = h
            print(f"{fname} → organism box: ({x_min}, {y_min}, {w}, {h})")
            plt.close()

    def on_key(event):
        if event.key == 'n':
            skip_flag[0] = True
            print(f"⏭️ Skipped: {fname}")
            plt.close()
        elif event.key == 'b':
            back_flag[0] = True
            print("↩️ Going back.")
            plt.close()
        elif event.key == 'q':
            quit_flag[0] = True
            print("Exiting annotation.")
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
            prev_idx = df_indices[i]
            for col in ["x_organism", "y_organism", "w_organism", "h_organism"]:
                df.at[prev_idx, col] = pd.NA
        continue
    else:
        i += 1
        df.to_csv(CSV_PATH, index=False)

print("\nFinished organism annotation.")
print(f"File updated: {CSV_PATH}")
