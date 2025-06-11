#!/usr/bin/env python3
"""
shuffle_picker_resizable.py
Visible shuffle → fade-in → stays up until you close (✕ or Esc).

✱  WINDOW_FRACTION sets how much of the desktop to occupy:
       1.0 = full screen, 0.75 = ¾ screen, 0.5 = half screen, …
✱  Images keep their original aspect ratio and are centred.
✱  Requires pygame >= 2.0        pip install pygame
"""

import os, sys, random, time, pygame

SUPPORTED = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
WINDOW_FRACTION = 0.75        #  ← tweak this for higher / lower resolution
SHUFFLE_SECONDS = (3, 5)
FPS_SHUFFLE      = 15
ALPHA_STEP       = 8          # 256 / 8 ≈ 32 steps ≈ 1-s fade

# ─── helpers ───────────────────────────────────────────────────────────────────
def image_paths(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(SUPPORTED)
    ]

def desktop_size():
    pygame.display.init()
    try:
        return pygame.display.get_desktop_sizes()[0]      # (w, h)
    except Exception:
        info = pygame.display.Info()
        return info.current_w, info.current_h

def load_scaled_center(path, win_size):
    """Load path → scale to *fit* win_size, return (surface, blit_rect)."""
    surf = pygame.image.load(path).convert_alpha()
    iw, ih               = surf.get_size()
    ww, wh               = win_size
    scale                = min(ww / iw, wh / ih)
    new_size             = (int(iw * scale), int(ih * scale))
    surf                 = pygame.transform.smoothscale(surf, new_size)
    rect                 = surf.get_rect(center=(ww // 2, wh // 2))
    return surf, rect
# ───────────────────────────────────────────────────────────────────────────────

def shuffle_picker(folder):
    pygame.init()

    paths = image_paths(folder)
    if not paths:
        raise SystemExit(
            f"No images with extensions {SUPPORTED} found in {folder!r}"
        )

    # Choose window size
    dw, dh  = desktop_size()
    win_w   = int(dw * WINDOW_FRACTION)
    win_h   = int(dh * WINDOW_FRACTION)
    size    = (win_w, win_h)

    # Create window
    screen  = pygame.display.set_mode(size)
    pygame.display.set_caption(
        f"Image Shuffle Picker ({int(WINDOW_FRACTION*100)}% of screen)"
    )
    clock   = pygame.time.Clock()

    # ─── Phase 1: visible shuffle ──────────────────────────────────────────────
    shuffle_end = time.time() + random.uniform(*SHUFFLE_SECONDS)
    current_surf, current_rect = load_scaled_center(random.choice(paths), size)

    while time.time() < shuffle_end:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (
                ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE
            ):
                pygame.quit(); return

        current_surf, current_rect = load_scaled_center(random.choice(paths), size)
        screen.fill((0, 0, 0))
        screen.blit(current_surf, current_rect)
        pygame.display.flip()
        clock.tick(FPS_SHUFFLE)

    # ─── Phase 2: smooth fade-in of chosen image ──────────────────────────────
    final_surf, final_rect = current_surf, current_rect
    for alpha in range(0, 256, ALPHA_STEP):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (
                ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE
            ):
                pygame.quit(); return

        screen.fill((0, 0, 0))
        temp = final_surf.copy()
        temp.set_alpha(alpha)
        screen.blit(temp, final_rect)
        pygame.display.flip()
        clock.tick(60)

    # ─── Phase 3: hold until user closes ──────────────────────────────────────
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (
                ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE
            ):
                pygame.quit(); return
        clock.tick(30)      # idle, but keep event loop responsive

if __name__ == "__main__":
    folder_arg = sys.argv[1] if len(sys.argv) > 1 else "images"
    shuffle_picker(folder_arg)