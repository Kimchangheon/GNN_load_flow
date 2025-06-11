#!/usr/bin/env python3
"""
shuffle_picker_halfscreen_ar.py
• Window = half of primary monitor.
• Images keep original aspect ratio (scaled to fit, then centered).
• Visible shuffle (3–5 s) → smooth fade-in → stays until you close or press Esc.
Requires:  pygame >= 2.0      pip install pygame
"""

import os, sys, random, time, pygame

SUPPORTED = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

# ---------- helpers ----------------------------------------------------------
def image_paths(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(SUPPORTED)
    ]

def half_screen():
    """Return (width//2, height//2) for primary display."""
    pygame.display.init()
    try:                                 # PyGame ≥ 2.1
        w, h = pygame.display.get_desktop_sizes()[0]
    except Exception:                    # fallback
        info = pygame.display.Info()
        w, h = info.current_w, info.current_h
    return w // 2, h // 2

def load_scaled_center(path, win_size):
    """
    Load *path*, scale to fit inside *win_size* while preserving aspect,
    and return (Surface, rect-to-blit).
    """
    surf = pygame.image.load(path).convert_alpha()
    iw, ih = surf.get_size()
    ww, wh = win_size
    scale = min(ww / iw, wh / ih)            # keep ratio
    new_size = (int(iw * scale), int(ih * scale))
    surf = pygame.transform.smoothscale(surf, new_size)
    rect = surf.get_rect(center=(ww // 2, wh // 2))
    return surf, rect
# -----------------------------------------------------------------------------

def shuffle_picker(folder, shuffle_seconds=(3, 5), fps=15):
    pygame.init()

    paths = image_paths(folder)
    if not paths:
        raise SystemExit(
            f"No images with extensions {SUPPORTED} found in {folder!r}"
        )

    size = half_screen()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Image Shuffle Picker (½-screen, aspect-kept)")
    clock = pygame.time.Clock()

    # ---------- Phase 1 – visible shuffle ----------
    shuffle_end = time.time() + random.uniform(*shuffle_seconds)
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
        clock.tick(fps)

    # ---------- Phase 2 – fade-in reveal ----------
    final_surf, final_rect = current_surf, current_rect
    for alpha in range(0, 256, 8):            # ≈1-second fade
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

    # ---------- Phase 3 – stay until user closes ----------
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (
                ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE
            ):
                pygame.quit(); return
        clock.tick(30)

if __name__ == "__main__":
    folder_arg = sys.argv[1] if len(sys.argv) > 1 else "images"
    shuffle_picker(folder_arg)