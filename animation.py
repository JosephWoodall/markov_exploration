import math
import random
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manim
import matplotlib.patches as mpatch
from matplotlib.gridspec import GridSpec

random.seed(42)
np.random.seed(42)

# ── config ────────────────────────────────────────────────────────────────────
LABELS  = ['A', 'B', 'C', 'D']
COLORS  = ['#4C9BE8', '#5DBB7A', '#E8614C', '#A97FCC']
BG      = '#0f1117'
PANEL   = '#1a1d27'
BORDER  = '#2a2d3e'
TEXT    = '#e0e0e0'
DIM     = '#555577'

K       = 3
LR      = 0.18
PATTERN = [0, 1, 2, 3, 2, 1]
N_TOTAL = 80
N_ANIM  = 24   # steps to animate
FPS     = 8
SUBFRAMES = 5  # phases per step


# ── data + instrumented predictor ────────────────────────────────────────────
def gen_seq(n):
    s = []
    for i in range(n):
        v = PATTERN[i % len(PATTERN)]
        if random.random() < 0.12:
            v = random.randint(0, 3)
        s.append(v)
    return s

SEQ = gen_seq(N_TOTAL)

def hamming(a, b):
    return sum(x == y for x, y in zip(a, b)) / len(a)

def collect_states(seq, k, lr):
    history, credibility, quality_hist, states = [], [], [], []

    for actual in seq:
        n = len(history)
        if n > k:
            context   = history[-k:]
            ctx_start = n - k
            win_data  = {}
            weights   = defaultdict(float)
            total_w   = total_wsq = 0.0
            contribs  = {}

            for i in range(n - k - 1):
                window = history[i:i+k]
                succ   = history[i+k]
                sim    = hamming(window, context)
                cred   = credibility[i]
                w      = sim * cred
                win_data[i] = (sim, cred)
                if w > 1e-10:
                    weights[succ] += w
                    total_w       += w
                    total_wsq     += w * w
                    contribs[i]    = (w, succ)

            dist = {v: w/total_w for v, w in weights.items()} if total_w > 0 else {}
            pred = max(dist, key=dist.get) if dist else None

            states.append({
                'history':    list(history),
                'credibility':list(credibility),
                'context':    list(context),
                'ctx_start':  ctx_start,
                'win_data':   dict(win_data),
                'dist':       dict(dist),
                'prediction': pred,
                'actual':     actual,
                'quality_hist': list(quality_hist),
            })

            # feedback
            if contribs:
                for i, (w, succ) in contribs.items():
                    c = w / total_w
                    credibility[i] *= (1 + lr*c) if succ == actual else (1 - lr*c)
                    credibility[i] = max(0.01, credibility[i])
                mc = sum(credibility) / len(credibility)
                credibility = [c/mc for c in credibility]

            if credibility:
                sc = sorted(credibility, reverse=True)
                q  = sum(sc[:max(1,len(sc)//4)]) / max(1,len(sc)//4)
                quality_hist.append(q)

        history.append(actual)
        credibility.append(1.0)

    return states

ALL_STATES = collect_states(SEQ, K, LR)
ANIM_STATES = [s for s in ALL_STATES if len(s['history']) >= 18][-N_ANIM:]

PHASE_LABELS = [
    "Step 1 — Identify the current context (last {} values)".format(K),
    "Step 2 — Scan history for similar patterns",
    "Step 3 — Collect weighted votes from each match",
    "Step 4 — Make prediction",
    "Step 5 — Observe actual value → update credibility",
]


# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 11), facecolor=BG)
fig.suptitle('Universal Sequence Predictor', color=TEXT, fontsize=20,
             fontweight='bold', y=0.97, family='monospace')

gs = GridSpec(3, 2, figure=fig,
              height_ratios=[2.2, 0.15, 2.2],
              hspace=0.55, wspace=0.35,
              left=0.05, right=0.97, top=0.91, bottom=0.07)

ax_hist    = fig.add_subplot(gs[0, :])
ax_phase   = fig.add_subplot(gs[1, :])
ax_votes   = fig.add_subplot(gs[2, 0])
ax_quality = fig.add_subplot(gs[2, 1])

for ax in [ax_hist, ax_votes, ax_quality]:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.tick_params(colors=DIM, labelsize=8)

ax_phase.set_facecolor(BG)
ax_phase.axis('off')

phase_text = ax_phase.text(
    0.5, 0.5, '', ha='center', va='center',
    color='#aaaacc', fontsize=13, family='monospace',
    transform=ax_phase.transAxes
)

legend_patches = [mpatch.Patch(color=COLORS[i], label=LABELS[i]) for i in range(4)]
ax_hist.legend(handles=legend_patches, loc='upper left',
               framealpha=0.2, labelcolor=TEXT, fontsize=8,
               facecolor=PANEL, edgecolor=BORDER)


# ── draw helpers ──────────────────────────────────────────────────────────────
MAX_SHOW = 32  # max history boxes to display

def draw_history(state, phase):
    ax_hist.clear()
    ax_hist.set_facecolor(PANEL)
    for sp in ax_hist.spines.values():
        sp.set_color(BORDER)
    ax_hist.axis('off')
    ax_hist.set_title('History  ·  box brightness = credibility',
                      color=DIM, fontsize=10, pad=5, loc='right')

    hist  = state['history']
    cred  = state['credibility']
    ctx_s = state['ctx_start']
    n     = len(hist)

    start = max(0, n - MAX_SHOW)
    h_show = hist[start:]
    c_show = cred[start:]
    n_show = len(h_show)

    # which windows are "similar" (top 5 by sim, phase ≥ 1)
    sim_idxs = set()
    if phase >= 1:
        top = sorted(state['win_data'].items(),
                     key=lambda x: x[1][0]*x[1][1], reverse=True)[:5]
        for wi, _ in top:
            for off in range(K):
                gi = wi + off
                if gi >= start:
                    sim_idxs.add(gi - start)

    ax_hist.set_xlim(-1, n_show)
    ax_hist.set_ylim(-0.05, 1.35)

    for i, (v, c) in enumerate(zip(h_show, c_show)):
        gi = i + start
        alpha = min(1.0, max(0.15, (c - 0.5) / 1.5 + 0.4))

        is_ctx = (ctx_s - start <= i < ctx_s - start + K)
        is_sim = (i in sim_idxs) and (phase >= 1)

        ec = '#66bbff' if is_ctx else ('#ffcc44' if is_sim else BORDER)
        lw = 2.5 if (is_ctx or is_sim) else 0.5

        rect = mpatch.FancyBboxPatch(
            (i - 0.42, 0.18), 0.84, 0.72,
            boxstyle='round,pad=0.04',
            facecolor=COLORS[v], alpha=alpha,
            edgecolor=ec, linewidth=lw,
            zorder=2
        )
        ax_hist.add_patch(rect)
        ax_hist.text(i, 0.54, LABELS[v], ha='center', va='center',
                     color='white', fontsize=11, fontweight='bold', zorder=3)

    # credibility bar underneath each box
    for i, c in enumerate(c_show):
        bar_h = min(0.14, 0.14 * c / 2.0)
        bar = mpatch.Rectangle(
            (i - 0.38, 0.02), 0.76, bar_h,
            facecolor='#66ccaa', alpha=0.6, zorder=1
        )
        ax_hist.add_patch(bar)

    # label context bracket
    c_disp_start = ctx_s - start
    if 0 <= c_disp_start < n_show:
        ax_hist.annotate('', xy=(c_disp_start + K - 0.5 + 0.05, 1.2),
                         xytext=(c_disp_start - 0.5 - 0.05, 1.2),
                         arrowprops=dict(arrowstyle='-', color='#66bbff', lw=1.5))
        ax_hist.text(c_disp_start + K/2 - 0.5, 1.27, 'context',
                     ha='center', va='bottom', color='#66bbff', fontsize=7)


def draw_votes(state, phase):
    ax_votes.clear()
    ax_votes.set_facecolor(PANEL)
    for sp in ax_votes.spines.values():
        sp.set_color(BORDER)
    ax_votes.tick_params(colors=DIM, labelsize=8)
    ax_votes.set_title('Weighted Votes', color=DIM, fontsize=11, pad=5)
    ax_votes.set_ylim(0, 1.1)
    ax_votes.set_xticks(range(4))
    ax_votes.set_xticklabels(LABELS, color=TEXT, fontsize=13)
    ax_votes.set_yticks([0, 0.5, 1.0])
    ax_votes.yaxis.set_tick_params(labelcolor=DIM)

    dist = state['dist']
    pred = state['prediction']

    for i, label in enumerate(LABELS):
        val = dist.get(i, 0.0) if phase >= 2 else 0.0
        color = COLORS[i]
        alpha = 0.9

        if phase >= 3 and i == pred:
            ec, lw = '#ffffff', 2.0
        else:
            ec, lw = BORDER, 0.5

        bar = ax_votes.bar(i, val, color=color, alpha=alpha,
                           edgecolor=ec, linewidth=lw, width=0.6)

        if phase >= 2 and val > 0.05:
            ax_votes.text(i, val + 0.03, f'{val:.2f}',
                          ha='center', va='bottom', color=TEXT, fontsize=10)

    if phase >= 3 and pred is not None:
        ax_votes.text(0.5, 1.05,
                      f'Prediction → {LABELS[pred]}',
                      ha='center', va='top', color='#ffff88',
                      fontsize=11, fontweight='bold',
                      transform=ax_votes.transAxes)

    if phase >= 4:
        actual = state['actual']
        correct = (pred == actual)
        color = '#66ee88' if correct else '#ee6644'
        ax_votes.text(0.5, 0.95,
                      f'Actual: {LABELS[actual]}  {"✓" if correct else "✗"}',
                      ha='center', va='top', color=color,
                      fontsize=11, transform=ax_votes.transAxes)


def draw_quality(state, phase):
    ax_quality.clear()
    ax_quality.set_facecolor(PANEL)
    for sp in ax_quality.spines.values():
        sp.set_color(BORDER)
    ax_quality.tick_params(colors=DIM, labelsize=8)
    ax_quality.set_title('Similarity Quality  (convergence curve)', color=DIM, fontsize=11, pad=5)
    ax_quality.set_ylabel('quality', color=DIM, fontsize=10)
    ax_quality.set_xlabel('observations', color=DIM, fontsize=10)

    qh = state['quality_hist']
    if not qh:
        ax_quality.text(0.5, 0.5, 'building…', ha='center', va='center',
                        color=DIM, transform=ax_quality.transAxes)
        return

    xs = list(range(len(qh)))
    ax_quality.plot(xs, qh, color='#66aaff', lw=1.5, zorder=2)
    ax_quality.fill_between(xs, qh, alpha=0.15, color='#66aaff', zorder=1)
    ax_quality.set_xlim(0, max(10, len(qh)))
    ymax = max(max(qh) * 1.15, 1.1)
    ax_quality.set_ylim(0.9, ymax)

    # fit convergence if enough data
    if len(qh) >= 6:
        L, tau = _fit(qh)
        if math.isfinite(tau) and 0 < tau < 1e4:
            x_ext = np.linspace(0, len(qh) + 40, 200)
            y_ext = L * (1 - np.exp(-x_ext / tau))
            ax_quality.plot(x_ext, y_ext, color='#ffaa44',
                            lw=1.2, linestyle='--', alpha=0.7, zorder=3)
            ax_quality.axhline(L, color='#ffaa44', lw=0.8,
                               linestyle=':', alpha=0.5, zorder=2)
            ax_quality.text(0.02, 0.96, f'Plateau L = {L:.3f}',
                            ha='left', va='top', color='#ffaa44',
                            fontsize=10, transform=ax_quality.transAxes)

    ax_quality.text(0.02, 0.86, f'Quality now = {qh[-1]:.3f}',
                    ha='left', va='top', color='#66aaff',
                    fontsize=10, transform=ax_quality.transAxes)


def _fit(qh):
    deltas = [qh[i] - qh[i-1] for i in range(1, len(qh))]
    prev   = qh[:-1]
    n      = len(deltas)
    xbar   = sum(prev) / n
    ybar   = sum(deltas) / n
    num    = sum((x-xbar)*(y-ybar) for x, y in zip(prev, deltas))
    den    = sum((x-xbar)**2 for x in prev)
    if abs(den) < 1e-12 or abs(num) < 1e-12:
        return qh[-1], float('inf')
    slope = num / den
    intcp = ybar - slope * xbar
    if slope >= 0 or abs(slope) < 1e-6:
        return qh[-1], float('inf')
    alpha = -slope
    L     = -intcp / slope
    tau   = (-1/math.log(1-alpha)) if 0 < alpha < 1 else 1/alpha
    return max(L, qh[-1]), max(tau, 1.0)


# ── animation function ────────────────────────────────────────────────────────
def update(frame):
    state_idx = frame // SUBFRAMES
    phase     = frame % SUBFRAMES

    state = ANIM_STATES[min(state_idx, len(ANIM_STATES)-1)]

    draw_history(state, phase)
    draw_votes(state, phase)
    draw_quality(state, phase)

    phase_text.set_text(PHASE_LABELS[phase])

    step_label = (
        f"Step {state_idx+1}/{len(ANIM_STATES)}  ·  "
        f"History: {len(state['history'])} obs  ·  "
        f"Context: [{' '.join(LABELS[v] for v in state['context'])}]  ·  "
        f"Predicted: {LABELS[state['prediction']] if state['prediction'] is not None else '?'}  ·  "
        f"Actual: {LABELS[state['actual']]}"
    )
    fig.texts[1].set_text(step_label) if len(fig.texts) > 1 else \
        fig.text(0.5, 0.01, step_label, ha='center', color=DIM,
                 fontsize=8, family='monospace')

    return []


# ── render ────────────────────────────────────────────────────────────────────
total_frames = len(ANIM_STATES) * SUBFRAMES
ani = manim.FuncAnimation(fig, update, frames=total_frames,
                          interval=1000//FPS, blit=False)

out_mp4 = 'predictor_animation.mp4'
out_gif = 'predictor_animation.gif'

saved = False
try:
    writer = manim.FFMpegWriter(fps=FPS, bitrate=3200,
                                metadata={'title': 'Universal Sequence Predictor'})
    ani.save(out_mp4, writer=writer, dpi=160)
    print(f'Saved: {out_mp4}')
    saved = True
except Exception as e:
    print(f'ffmpeg unavailable ({e}), trying GIF...')

if not saved:
    try:
        ani.save(out_gif, writer='pillow', fps=FPS, dpi=100)
        print(f'Saved: {out_gif}')
    except Exception as e:
        print(f'GIF failed ({e}). Saving static summary instead.')
        update(total_frames - SUBFRAMES)   # draw final state
        fig.savefig('predictor_summary.png', dpi=130, facecolor=BG)
        print('Saved: predictor_summary.png')
