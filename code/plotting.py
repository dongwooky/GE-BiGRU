import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def plot_anomaly_scores(xs, att, num_segments=2, THRESHOLD=None, filename='anomaly_plot.png'):
    num_data_points = xs.shape[0]
    segment_size = num_data_points // num_segments
    fig, axs = plt.subplots(num_segments, figsize=(20, 4 * num_segments))
    sns.set(style="whitegrid")

    if num_segments == 1:
        axs = [axs]  # Make axs iterable for single subplot

    for i in range(num_segments):
        L, R = i * segment_size, min((i + 1) * segment_size, num_data_points)
        xticks = range(L, R)

        if len(xs[L:R]) > 0:
            peak = max(xs[L:R])
            # Don't add a label here as we'll create a custom legend
            anomaly_line = sns.lineplot(x=xticks, y=xs[L:R], ax=axs[i], color='dodgerblue')
            axs[i].fill_between(xticks, att[L:R] * peak, color='crimson', alpha=0.5)

        if THRESHOLD is not None:
            axs[i].axhline(y=THRESHOLD, color='r', linestyle='--')

        if i < num_segments - 1:
            axs[i].set_xlabel('', fontsize=28, fontweight='bold')
        else:
            axs[i].set_xlabel('Time(sec)', fontsize=28, fontweight='bold')

        axs[i].set_ylabel('')
        axs[i].tick_params(axis='both', which='major', labelsize=26)

        # Create custom legend handles
        anomaly_line_handle = mlines.Line2D([], [], color='dodgerblue', label='Anomaly Score')
        anomaly_patch = mpatches.Patch(color='crimson', alpha=0.5, label='Abnormal')
        threshold_line = mlines.Line2D([], [], color='red', linestyle='--', label='Threshold')
        
        # Create legend with custom handles - all explicitly defined
        axs[i].legend(handles=[anomaly_line_handle, threshold_line, anomaly_patch], 
                     loc='upper left', fontsize=24)

    fig.text(0.06, 0.5, 'Anomaly Score', va='center', rotation='vertical', fontsize=28, fontweight='bold')

    plt.tight_layout(rect=[0.05, 0, 1, 1])  # Adjust layout so y-label fits
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save as PNG
    plt.close(fig)  # Close the figure to free up memory


def plot_channel_heatmap(
    per_channel_score,
    attack_labels,
    sensor_names,
    filename='channel_heatmap.png',
    downsample=60,
    apply_log=True,
    percentile_clip=99,
):
    """Plot a heatmap of per-channel anomaly scores.

    Args:
        per_channel_score: np.ndarray [T, 79], per-channel |P(t) - R(t)|
        attack_labels:     np.ndarray [T], 0/1 attack labels
        sensor_names:      sequence of the 79 sensor names
        filename:          output path
        downsample:        keep one sample every N steps (default 60)
        apply_log:         apply a log1p transform
        percentile_clip:   clip values above the N-th percentile (None disables)
    """
    T = per_channel_score.shape[0]
    idx = np.arange(0, T, downsample)
    data = per_channel_score[idx, :].T   # [79, T_ds]
    att_ds = attack_labels[idx]

    if apply_log:
        data = np.log1p(data)
    if percentile_clip is not None:
        data = np.clip(data, 0, np.percentile(data, percentile_clip))

    sensor_names = list(sensor_names)
    separator_rows = [38, 60, 67]   # P1/P2, P2/P3, P3/P4 boundaries

    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 20], hspace=0.03)
    ax_att = fig.add_subplot(gs[0])
    ax_hm  = fig.add_subplot(gs[1])

    # Top: attack bar
    ax_att.imshow(att_ds[np.newaxis, :], aspect='auto',
                  cmap='RdYlGn_r', vmin=0, vmax=1, interpolation='nearest')
    ax_att.set_xticks([])
    ax_att.set_yticks([0])
    ax_att.set_yticklabels(['Attack'], fontsize=18, fontweight='bold')
    ax_att.tick_params(left=False)

    # Bottom: heatmap
    sns.set(style='white')
    sns.heatmap(data, ax=ax_hm, cmap='viridis', vmin=0,
                cbar=True,
                cbar_kws={'label': 'log(1+|P-R|)' if apply_log else '|P-R|',
                          'shrink': 0.6, 'pad': 0.01},
                xticklabels=False, yticklabels=sensor_names)

    # Process group separators
    for row in separator_rows:
        ax_hm.axhline(y=row, color='white', linewidth=2.5, linestyle='--')

    # Group labels (right side)
    T_ds = data.shape[1]
    group_info = {
        'P1': ((0  + 38) / 2, 'salmon'),
        'P2': ((38 + 60) / 2, 'skyblue'),
        'P3': ((60 + 67) / 2, 'lightgreen'),
        'P4': ((67 + 79) / 2, 'orchid'),
    }
    for label, (cy, color) in group_info.items():
        ax_hm.text(T_ds * 1.01, cy, label, va='center', ha='left',
                   fontsize=20, fontweight='bold', color=color, clip_on=False)

    ax_hm.set_xlabel(f'Time (every {downsample}s)', fontsize=22, fontweight='bold')
    ax_hm.set_ylabel('')
    ax_hm.tick_params(axis='y', labelsize=10)

    cbar = ax_hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('log(1+|P-R|)' if apply_log else '|P-R|', fontsize=16)

    fig.suptitle('Per-Channel Anomaly Score Heatmap',
                 fontsize=24, fontweight='bold', y=1.01)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_topk_channel_heatmap(
    per_channel_score,
    attack_labels,
    sensor_names,
    gat_nodes=None,
    top_k=20,
    filename='topk_channel_heatmap.png',
    downsample=60,
    apply_log=True,
    percentile_clip=99,
    exclude_channels=None,
    time_window=None,
    force_include_channels=None,
):
    """Plot a heatmap of the top-K channels that respond most to attacks.

    Channels with high GAT attention are highlighted in tomato color with a ★ mark.

    Args:
        per_channel_score: np.ndarray [T, N], per-channel |P(t) - R(t)|
        attack_labels:     np.ndarray [T], 0/1 ground-truth labels
        sensor_names:      N sensor names (list / Index)
        gat_nodes:         channel names to highlight (None for no highlighting)
        top_k:             number of channels to select
        filename:          output path
        downsample:        keep one sample every N steps
        apply_log:         apply a log1p transform
        percentile_clip:   clip values above the N-th percentile
    """
    sensor_names = list(sensor_names)
    gat_nodes = set(gat_nodes) if gat_nodes else set()
    attack_labels = np.asarray(attack_labels, dtype=int)

    # Drop excluded channels
    if exclude_channels:
        exclude_set = set(exclude_channels)
        keep = [i for i, n in enumerate(sensor_names) if n not in exclude_set]
        per_channel_score = per_channel_score[:, keep]
        sensor_names = [sensor_names[i] for i in keep]

    # Ratio of mean error during attacks vs. normal operation, per channel
    attack_mask = attack_labels == 1
    normal_mask = attack_labels == 0
    mean_attack = per_channel_score[attack_mask].mean(axis=0) if attack_mask.any() else np.zeros(per_channel_score.shape[1])
    mean_normal = per_channel_score[normal_mask].mean(axis=0) if normal_mask.any() else np.ones(per_channel_score.shape[1])
    ratio = mean_attack / (mean_normal + 1e-10)

    # Select the top_k channels (descending ratio)
    topk_idx = np.argsort(ratio)[::-1][:top_k]

    # Add forced channels (even outside top_k), sorted by ratio
    if force_include_channels:
        force_set = set(force_include_channels)
        topk_set  = set(topk_idx.tolist())
        extra_idx = [i for i, n in enumerate(sensor_names)
                     if n in force_set and i not in topk_set]
        if extra_idx:
            all_idx  = sorted(list(topk_idx) + extra_idx,
                              key=lambda i: ratio[i], reverse=True)
            topk_idx = np.array(all_idx)

    topk_names = [sensor_names[i] for i in topk_idx]
    topk_ratio = ratio[topk_idx]

    # Slice the plotted time range (channel selection already used the full data)
    if time_window is not None:
        w_start, w_end = time_window
        vis_score  = per_channel_score[w_start:w_end]
        vis_labels = attack_labels[w_start:w_end]
    else:
        vis_score  = per_channel_score
        vis_labels = attack_labels

    # Downsample
    T = vis_score.shape[0]
    ds_idx = np.arange(0, T, downsample)
    data = vis_score[ds_idx][:, topk_idx].T   # [top_k, T_ds]
    att_ds = vis_labels[ds_idx]

    if apply_log:
        data = np.log1p(data)
    if percentile_clip is not None:
        data = np.clip(data, 0, np.percentile(data, percentile_clip))

    import matplotlib as mpl

    T_ds = data.shape[1]
    fig_height = max(10, top_k * 0.75)
    fig = plt.figure(figsize=(26, fig_height))

    # 2-column gridspec: [heatmap | colorbar]
    # the attack bar spans column 0 only, so it has exactly the heatmap's width
    gs = fig.add_gridspec(2, 2,
                          height_ratios=[3, top_k],
                          width_ratios=[20, 1],
                          hspace=0.0, wspace=0.05)
    ax_att = fig.add_subplot(gs[0, 0])   # attack bar
    ax_hm  = fig.add_subplot(gs[1, 0])   # heatmap
    ax_cb  = fig.add_subplot(gs[1, 1])   # colorbar
    fig.add_subplot(gs[0, 1]).set_visible(False)  # empty top-right cell

    # Top: attack bar - binary colormap (white = normal, red = attack), pixel-exact via imshow
    attack_cmap = mpl.colors.ListedColormap(['white', 'crimson'])
    ax_att.imshow(att_ds[np.newaxis, :], aspect='auto',
                  cmap=attack_cmap, vmin=0, vmax=1, interpolation='nearest')
    ax_att.set_xticks([])
    ax_att.set_yticks([0])
    ax_att.set_yticklabels(['Ground\nTruth'], fontsize=30, fontweight='bold')
    ax_att.tick_params(left=False)
    for spine in ax_att.spines.values():
        spine.set_edgecolor('#cccccc')
    from matplotlib.patches import Patch
    ax_att.legend(
        handles=[Patch(facecolor='crimson', label='Attack')],
        loc='upper right', fontsize=28, framealpha=0.85,
        handlelength=1.5, handleheight=1.5
    )

    # Bottom: heatmap
    sns.set(style='white')
    sns.heatmap(data, ax=ax_hm, cmap='viridis', vmin=0,
                cbar=False,
                xticklabels=False, yticklabels=topk_names)

    # Colorbar
    vmax = data.max()
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=vmax),
                              cmap='viridis'),
        cax=ax_cb)
    cb.ax.tick_params(labelleft=False, labelright=False, left=False, right=False)
    cb.set_label('')

    # y-axis label color + ★ mark
    for tick_label in ax_hm.get_yticklabels():
        ch = tick_label.get_text()
        if ch in gat_nodes:
            tick_label.set_color('tomato')
            tick_label.set_fontweight('bold')
            tick_label.set_text(ch + ' ★')
    ax_hm.set_yticklabels(ax_hm.get_yticklabels(), fontsize=26)

    ax_hm.set_xlabel(f'Time (every {downsample}s)', fontsize=32, fontweight='bold')
    ax_hm.set_ylabel('')

    fig.subplots_adjust(top=0.99)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)