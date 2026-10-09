"""Build the 1200x630 social cards (og:image / twitter:image) of the site.

    python3 tools/make_cards.py            # every card
    python3 tools/make_cards.py mhc pdf-rag

Run from the repository root. Needs Pillow and the Inter font. Three cards start from
sources outside this repository (EXTERNAL below); a card whose source is missing is
skipped with a warning, and the committed card is left as it is.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
SS = 4  # supersampling for anti-aliased strokes
FONTS = Path("/usr/share/fonts/opentype/inter")
POSTS = Path("posts")
HOME = Path.home()
EXTERNAL = {
    "logo": HOME / "Documents/crest/human-ai-mathematics/logo.png",
    "ab_figure": HOME / "PycharmProjects/haiku-shunt/docs/ab_figure.py",
    "teaser": HOME / "PycharmProjects/postdoc/edm-error-propagation/aistats/figures/generated"
              / "numerical_teaser.pdf",
}

NAVY = (11, 31, 58)
BLUE = (26, 92, 255)
MID = (70, 120, 225)
LIGHT = (140, 172, 238)
GREY = (110, 116, 128)
BG = (250, 250, 247)


def bezier(p0, p1, p2, p3, n=400):
    pts = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        pts.append(tuple(u**3 * a + 3 * u * u * t * b + 3 * u * t * t * c + t**3 * d
                         for a, b, c, d in zip(p0, p1, p2, p3)))
    return pts


def tracked(draw, xy, text, font, fill, tracking):
    """Draw text with extra letter spacing; xy is the top-centre."""
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x, y = xy[0] - total / 2, xy[1]
    for ch, w in zip(text, widths):
        draw.text((x, y), ch, font=font, fill=fill)
        x += w + tracking


def default_card(out):
    im = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(im)
    s = SS
    # Four trajectories leaving a common horizontal band and fanning out, as in the
    # "Flow" proposal: (start y, end point, colour, width).
    x0 = 395
    lines = [
        (262, (790, 118), BLUE, 7),
        (276, (790, 176), NAVY, 7),
        (290, (790, 226), MID, 6),
        (304, (745, 272), LIGHT, 6),
    ]
    for y0, (x1, y1), col, w in lines:
        pts = bezier((x0, y0), (x0 + 0.45 * (x1 - x0), y0),
                     (x0 + 0.55 * (x1 - x0), y1), (x1, y1))
        d.line([(x * s, y * s) for x, y in pts], fill=col, width=w * s, joint="curve")
        r = 10 if col in (BLUE, NAVY) else 9
        d.ellipse([(x1 - r) * s, (y1 - r) * s, (x1 + r) * s, (y1 + r) * s], fill=col)
        r0 = w / 2
        d.ellipse([(x0 - r0) * s, (y0 - r0) * s, (x0 + r0) * s, (y0 + r0) * s], fill=col)
    name = ImageFont.truetype(str(FONTS / "Inter-Light.otf"), 64 * s)
    tag = ImageFont.truetype(str(FONTS / "Inter-Regular.otf"), 26 * s)
    tracked(d, (W / 2 * s, 370 * s), "Nicolas Brosse", name, NAVY, 9 * s)
    d.line([(570 * s, 477 * s), (630 * s, 477 * s)], fill=(200, 203, 210), width=2 * s)
    tracked(d, (W / 2 * s, 505 * s), "AI research & development", tag, GREY, 2 * s)
    im.resize((W, H), Image.LANCZOS).save(out, optimize=True)


def font(name, size):
    return ImageFont.truetype(str(FONTS / f"Inter-{name}.otf"), size * SS)


def heading(d, text, sub):
    d.text((80 * SS, 62 * SS), text, font=font("Medium", 40), fill=NAVY)
    d.text((80 * SS, 118 * SS), sub, font=font("Regular", 24), fill=GREY)


def slides_card(out):
    """llm-slides: the post's headline result, lines of source per format."""
    im = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(im)
    s = SS
    heading(d, "Same five slides, four formats", "Lines of code Gemini 2.5 Pro wrote for each")
    rows = [("Quarto / Reveal.js", 130, BLUE), ("python-pptx", 270, NAVY),
            ("HTML / CSS", 492, NAVY), ("Google Slides API", 977, NAVY)]
    x0, x1, y, step, bh = 400, 1020, 215, 92, 50
    for label, n, col in rows:
        d.text(((x0 - 24) * s, (y + bh / 2) * s), label, font=font("Regular", 28),
               fill=NAVY, anchor="rm")
        w = (x1 - x0) * n / 977
        d.rounded_rectangle([x0 * s, y * s, (x0 + w) * s, (y + bh) * s], radius=6 * s, fill=col)
        d.text(((x0 + w + 16) * s, (y + bh / 2) * s), str(n), font=font("Medium", 28),
               fill=col, anchor="lm")
        y += step
    im.resize((W, H), Image.LANCZOS).save(out, optimize=True)


def tree_card(out):
    """pdf-rag: page-by-page fragments on the left, the recovered document tree on the right."""
    im = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(im)
    s = SS
    heading(d, "From pages to a document tree", "Recovering the structure a reader relies on")
    # A fanned stack of pages, each with heading bars of arbitrary sizes.
    for k in range(3):
        x, y = 110 + 34 * k, 205 + 26 * k
        d.rounded_rectangle([x * s, y * s, (x + 220) * s, (y + 290) * s], radius=8 * s,
                            fill=(255, 255, 255), outline=(200, 205, 215), width=2 * s)
    x, y = 178, 257
    for w, h, col in [(120, 14, NAVY), (150, 7, LIGHT), (140, 7, LIGHT), (90, 11, MID),
                      (150, 7, LIGHT), (130, 7, LIGHT), (100, 14, NAVY), (150, 7, LIGHT),
                      (110, 7, LIGHT)]:
        d.rounded_rectangle([x * s, y * s, (x + w) * s, (y + h) * s], radius=3 * s, fill=col)
        y += h + 12
    # Arrow.
    d.line([(470 * s, 360 * s), (560 * s, 360 * s)], fill=GREY, width=4 * s)
    d.polygon([(572 * s, 360 * s), (556 * s, 350 * s), (556 * s, 370 * s)], fill=GREY)
    # Tree: root, three sections, subsections.
    root = (850, 205)
    sections = [(690, 330), (850, 330), (1010, 330)]
    leaves = {0: [(650, 450), (730, 450)], 1: [(850, 450)], 2: [(970, 450), (1050, 450)]}
    edge = (175, 185, 200)
    for i, sec in enumerate(sections):
        d.line([(root[0] * s, root[1] * s), (sec[0] * s, sec[1] * s)], fill=edge, width=4 * s)
        for leaf in leaves[i]:
            d.line([(sec[0] * s, sec[1] * s), (leaf[0] * s, leaf[1] * s)], fill=edge,
                   width=3 * s)
    def node(c, r, col):
        d.ellipse([(c[0] - r) * s, (c[1] - r) * s, (c[0] + r) * s, (c[1] + r) * s], fill=col)
    node(root, 22, BLUE)
    for sec in sections:
        node(sec, 17, NAVY)
    for ls in leaves.values():
        for leaf in ls:
            node(leaf, 13, MID)
    im.resize((W, H), Image.LANCZOS).save(out, optimize=True)


def pad_card(src, out, bg=None, margin=40, scale=None):
    """Fit an existing figure inside a 1200x630 card without cropping it."""
    im = Image.open(src).convert("RGB")
    bg = bg or im.getpixel((2, 2))  # extend the figure's own background
    box = (W - 2 * margin, H - 2 * margin)
    f = min(box[0] / im.width, box[1] / im.height)
    if scale:
        f = min(f, scale)
    im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
    card = Image.new("RGB", (W, H), bg)
    card.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    card.save(out, optimize=True)


def ab_cost_png(tmp):
    """Rerun haiku-shunt's figure script, writing a PNG instead of its SVG."""
    script = EXTERNAL["ab_figure"].read_text().replace(
        'fig.savefig("docs/ab-cost.svg")', f'fig.savefig("{tmp}/ab.png", dpi=300)')
    (tmp / "ab_figure.py").write_text(script)
    subprocess.run(["uv", "run", "-q", "--no-project", "--with", "matplotlib", "python",
                    str(tmp / "ab_figure.py")], check=True)
    return tmp / "ab.png"


def teaser_png(tmp):
    """Rasterize the vector teaser of the EDM paper."""
    subprocess.run(["pdftoppm", "-r", "600", "-png", "-singlefile", str(EXTERNAL["teaser"]),
                    str(tmp / "teaser")], check=True)
    return tmp / "teaser.png"


CARDS = {
    "default": (None, lambda tmp: default_card(Path("social-card.png"))),
    "llm-slides": (None, lambda tmp: slides_card(POSTS / "llm-slides/figures/card.png")),
    "pdf-rag": (None, lambda tmp: tree_card(POSTS / "pdf-rag/figures/card.png")),
    "human-ai-mathematics": ("logo", lambda tmp: pad_card(
        EXTERNAL["logo"], POSTS / "human-ai-mathematics/figures/card.png", margin=45)),
    "haiku-shunt": ("ab_figure", lambda tmp: pad_card(
        ab_cost_png(tmp), POSTS / "haiku-shunt/figures/card.png")),
    "edm-error-propagation": ("teaser", lambda tmp: pad_card(
        teaser_png(tmp), POSTS / "edm-error-propagation/figures/card.png", margin=30)),
    "posterior-averaging": (None, lambda tmp: pad_card(
        POSTS / "posterior-averaging/figures/loss_decomposition.png",
        POSTS / "posterior-averaging/figures/card.png", margin=30)),
    "mhc": (None, lambda tmp: pad_card(
        POSTS / "mhc/figures/MHC_expression.png", POSTS / "mhc/figures/card.png", margin=0)),
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(CARDS)
    with tempfile.TemporaryDirectory() as d:
        for name in names:
            source, build = CARDS[name]
            if source and not EXTERNAL[source].exists():
                print(f"skip {name}: {EXTERNAL[source]} not found", file=sys.stderr)
                continue
            build(Path(d))
            print(f"built {name}")
