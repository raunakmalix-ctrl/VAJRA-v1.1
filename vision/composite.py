"""
Frame compositing for windowed lip-sync.

The problem this solves. Lip-syncing a ten-minute clip because four words changed
is wasteful twice over: it burns GPU time proportional to the whole video, and it
passes every frame through a generative model even though the overwhelming
majority of them should be untouched. Any drift the model introduces -- a slight
colour shift, softened skin texture, a jitter in head pose -- is then applied to
footage that had no reason to change.

So lip-sync runs only on the time windows that actually changed, and the result
is composited back into the ORIGINAL frames through a soft mask covering just the
mouth region. Two consequences:

  spatial   Outside the mask, every pixel is the original frame. The model
            cannot alter hair, background, shoulders or eyes.
  temporal  Outside the windows, frames are the original frames untouched.

Both boundaries need feathering for the same reason the audio splice does. A hard
mask edge shows as a visible seam around the mouth; a hard window edge makes the
mouth visibly "pop" as it switches between generated and original on one frame.
The spatial feather is a distance ramp at the mask edge; the temporal one ramps
the mask's strength across the first and last few frames of each window.

Pure numpy -- no model, no GPU. Everything here is unit-testable, which matters
because the surrounding video plumbing is not.
"""
import numpy as np

# Fraction of the face box height, measured from the chin upward, treated as
# "mouth region". Wav2Lip and LatentSync both operate on roughly the lower half
# of the face; this covers the jaw and lips with margin without reaching the eyes.
DEFAULT_MOUTH_TOP = 0.42        # start this far down the face box
DEFAULT_FEATHER_FRAC = 0.14     # feather width as a fraction of region size
DEFAULT_RAMP_FRAMES = 3         # temporal fade at each window edge


# ── time / frame arithmetic ────────────────────────────────────────────────
def frame_range(t_start, t_end, fps, n_frames=None):
    """Convert a time range to a half-open frame index range [i0, i1).

    Rounds outward so the returned frames fully cover the requested time -- a
    frame only partially inside the window still shows part of the edit, so it
    must be included or the mouth changes mid-frame.
    """
    if fps <= 0:
        return 0, 0
    i0 = int(np.floor(float(t_start) * fps))
    i1 = int(np.ceil(float(t_end) * fps))
    i0 = max(0, i0)
    i1 = max(i0, i1)
    if n_frames is not None:
        i0 = min(i0, int(n_frames))
        i1 = min(i1, int(n_frames))
    return i0, i1


def merge_windows(windows, pad_sec=0.0, min_gap_sec=0.0, duration_sec=None):
    """Pad, sort and coalesce time windows.

    pad_sec       widen each window. Lip-sync models need surrounding context to
                  produce a stable mouth, and the audio splice's own crossfade
                  extends slightly past the edit, so the visual window must cover
                  a little more than the audio edit did.
    min_gap_sec   windows closer than this are merged. Two adjacent windows mean
                  two model invocations and two pairs of temporal seams; one
                  slightly longer window is cheaper and cleaner.
    """
    out = []
    for t0, t1 in windows or []:
        a = float(t0) - float(pad_sec)
        b = float(t1) + float(pad_sec)
        if duration_sec is not None:
            a = min(max(a, 0.0), float(duration_sec))
            b = min(max(b, 0.0), float(duration_sec))
        else:
            a = max(a, 0.0)
        if b > a:
            out.append([a, b])
    if not out:
        return []
    out.sort(key=lambda w: w[0])
    merged = [out[0]]
    for a, b in out[1:]:
        if a - merged[-1][1] <= float(min_gap_sec):
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(round(a, 4), round(b, 4)) for a, b in merged]


def coverage(windows, duration_sec):
    """Fraction of the video the windows cover -- the GPU work actually saved."""
    if duration_sec <= 0:
        return 0.0
    total = sum(max(0.0, b - a) for a, b in windows or [])
    return float(min(total / float(duration_sec), 1.0))


# ── masks ──────────────────────────────────────────────────────────────────
def _ramp(d, feather):
    """0 at the edge, 1 at `feather` inside, smooth (raised cosine) between."""
    if feather <= 0:
        return (d > 0).astype(np.float32)
    t = np.clip(d / float(feather), 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


def feathered_box_mask(h, w, box, feather=None):
    """Soft-edged rectangular mask. box = (x0, y0, x1, y1) in pixels."""
    x0, y0, x1, y1 = [float(v) for v in box]
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    if feather is None:
        feather = DEFAULT_FEATHER_FRAC * max(min(x1 - x0, y1 - y0), 1.0)
    feather = max(float(feather), 0.0)

    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    # Signed distance inside each edge; the minimum is distance to the nearest.
    dx = np.minimum(xs - x0, x1 - xs)
    dy = np.minimum(ys - y0, y1 - ys)
    d = np.minimum(dx, dy)
    return np.clip(_ramp(d, feather), 0.0, 1.0)


def feathered_ellipse_mask(h, w, box, feather=None):
    """Soft-edged ellipse inscribed in `box`.

    Preferred over a rectangle for a mouth: a rectangular seam runs straight
    across the cheeks and chin where skin tone is uniform, which is exactly
    where a discontinuity is most visible. An ellipse follows the shape of the
    region being replaced, so what seam remains falls on contours that already
    vary.
    """
    x0, y0, x1, y1 = [float(v) for v in box]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = max((x1 - x0) / 2.0, 1e-3), max((y1 - y0) / 2.0, 1e-3)
    if feather is None:
        feather = DEFAULT_FEATHER_FRAC * min(rx, ry) * 2.0
    feather = max(float(feather), 0.0)

    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    # Normalised radius; convert to an approximate pixel distance inside the edge.
    r = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
    d = (1.0 - r) * min(rx, ry)
    return np.clip(_ramp(d, feather), 0.0, 1.0)


def mouth_box(face_box, mouth_top=DEFAULT_MOUTH_TOP, expand=0.06,
              frame_shape=None):
    """Lower-face box from a face bounding box.

    face_box = (x0, y0, x1, y1). Returns the region a lip-sync model actually
    changes, so the composite touches nothing above it.
    """
    x0, y0, x1, y1 = [float(v) for v in face_box]
    fh, fw = (y1 - y0), (x1 - x0)
    top = y0 + mouth_top * fh
    ex, ey = expand * fw, expand * fh
    bx0, by0 = x0 - ex, top - ey
    bx1, by1 = x1 + ex, y1 + ey
    if frame_shape is not None:
        h, w = frame_shape[0], frame_shape[1]
        bx0, bx1 = max(0.0, bx0), min(float(w), bx1)
        by0, by1 = max(0.0, by0), min(float(h), by1)
    return (bx0, by0, bx1, by1)


def mouth_box_from_landmarks(points, frame_shape=None, pad=0.55):
    """Mouth box from facial landmarks -- tighter than deriving it from a bbox.

    points: (N, 2) array. Any landmark set works; the lower-third centroid
    spread is used rather than specific indices, so this does not depend on a
    particular model's point ordering.
    """
    p = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if p.shape[0] < 3:
        return None
    y_sorted = np.sort(p[:, 1])
    y_cut = y_sorted[int(0.6 * (y_sorted.size - 1))]
    lower = p[p[:, 1] >= y_cut]
    if lower.shape[0] < 3:
        lower = p
    cx, cy = float(np.mean(lower[:, 0])), float(np.mean(lower[:, 1]))
    hw = float(np.max(lower[:, 0]) - np.min(lower[:, 0])) / 2.0
    hh = float(np.max(lower[:, 1]) - np.min(lower[:, 1])) / 2.0
    hw = max(hw, 4.0) * (1.0 + pad)
    hh = max(hh, 4.0) * (1.0 + pad)
    box = (cx - hw, cy - hh, cx + hw, cy + hh)
    if frame_shape is not None:
        h, w = frame_shape[0], frame_shape[1]
        box = (max(0.0, box[0]), max(0.0, box[1]),
               min(float(w), box[2]), min(float(h), box[3]))
    return box


def changed_region_box(base_frames, synced_frames, percentile=99.0,
                       min_area_frac=1e-4, max_area_frac=0.40, margin=0.10):
    """Bounding box of where `synced_frames` actually differ from `base_frames`.

    Used instead of a face detector to decide what the mouth mask should cover.
    Reasons that is the better choice here:

      - It targets what the model genuinely altered, rather than a guess derived
        from a face bounding box. If a model changes more or less than the
        canonical "lower half", the mask follows it.
      - No extra model to load, no VRAM, nothing to fail. A face detector in this
        path would be a second thing that can mis-detect and a second thing
        competing for the GPU.

    Returns None -- meaning "do not mask, use the synced frame as-is within the
    window" -- when the difference is negligible (nothing to composite) or when
    it spans most of the frame. A global difference means the model re-encoded or
    shifted the whole image rather than editing a region, and masking part of
    that would leave a visible discontinuity between the two versions.
    """
    n = min(len(base_frames), len(synced_frames))
    if n == 0:
        return None

    acc = None
    for i in range(n):
        b = np.asarray(base_frames[i]).astype(np.float32)
        s = np.asarray(synced_frames[i]).astype(np.float32)
        if b.shape != s.shape:
            return None
        d = np.abs(s - b)
        if d.ndim == 3:
            d = d.max(axis=2)
        acc = d if acc is None else np.maximum(acc, d)

    h, w = acc.shape
    hi = float(np.percentile(acc, percentile))
    # A fraction of the strongest change, floored so sensor noise or codec
    # dither cannot define the region.
    thresh = max(hi * 0.25, 2.0)
    ys, xs = np.nonzero(acc >= thresh)
    if ys.size == 0:
        return None

    area_frac = ys.size / float(h * w)
    if area_frac < min_area_frac or area_frac > max_area_frac:
        return None

    y0, y1 = float(ys.min()), float(ys.max() + 1)
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    mw, mh = (x1 - x0) * margin, (y1 - y0) * margin
    return (max(0.0, x0 - mw), max(0.0, y0 - mh),
            min(float(w), x1 + mw), min(float(h), y1 + mh))


def temporal_weight(index, n, ramp=DEFAULT_RAMP_FRAMES):
    """Mask strength for frame `index` of an n-frame window, 0..1.

    Ramps in and out so the generated mouth does not appear or vanish on a
    single frame. Without this the swap is visible as a pop at each window edge
    even when every individual frame is perfect.
    """
    n = int(n)
    if n <= 0:
        return 0.0
    ramp = int(max(0, min(ramp, n // 2)))
    if ramp == 0:
        return 1.0
    i = int(np.clip(index, 0, n - 1))
    if i < ramp:
        t = (i + 1) / float(ramp + 1)
    elif i >= n - ramp:
        t = (n - i) / float(ramp + 1)
    else:
        return 1.0
    return float(0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0)))


# ── compositing ────────────────────────────────────────────────────────────
def blend(base, overlay, mask, strength=1.0):
    """Alpha-composite `overlay` onto `base` using `mask` (h, w) in [0, 1].

    Returns the same dtype as `base`, so an 8-bit frame in gives an 8-bit frame
    out with no surprise promotion. Blending is done in float to avoid the
    banding that integer arithmetic produces on a soft gradient.
    """
    b = np.asarray(base)
    o = np.asarray(overlay)
    if b.shape != o.shape:
        raise ValueError(f"shape mismatch: base {b.shape} vs overlay {o.shape}")
    m = np.asarray(mask, dtype=np.float32)
    if m.shape[:2] != b.shape[:2]:
        raise ValueError(f"mask {m.shape[:2]} does not match frame {b.shape[:2]}")
    m = np.clip(m * float(strength), 0.0, 1.0)
    if b.ndim == 3:
        m = m[:, :, None]

    out = b.astype(np.float32) * (1.0 - m) + o.astype(np.float32) * m
    if np.issubdtype(b.dtype, np.integer):
        info = np.iinfo(b.dtype)
        return np.clip(np.rint(out), info.min, info.max).astype(b.dtype)
    return out.astype(b.dtype)


def composite_window(base_frames, synced_frames, mask, ramp=DEFAULT_RAMP_FRAMES):
    """Composite a whole window, applying the temporal ramp per frame.

    base_frames / synced_frames: equal-length sequences of same-shape frames.
    Returns a list of blended frames.
    """
    n = min(len(base_frames), len(synced_frames))
    out = []
    for i in range(n):
        w = temporal_weight(i, n, ramp)
        out.append(blend(base_frames[i], synced_frames[i], mask, strength=w))
    return out
