#!/usr/bin/env python3
"""
Complete A0 Academic Poster Generator for ICCAS 2026
Paper Title: Bridging the Sim-to-Real Gap in Quadrupedal Visual Navigation via 3D Gaussian Splatting
Authors: Minseok Lee, Seongyu Han, Heejae Moon, and Sung Soo Hwang*
Organization: CGV Lab, Handong Global University
"""

import os
import sys
import math
import re
from PIL import Image
from lxml import etree
import qrcode

from pptx import Presentation
from pptx.util import Inches, Pt, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# --- Paths ---
BASE_DIR = "/root/isaac-go2-ros2/ICCAS"
FIG_DIR = os.path.join(BASE_DIR, "figures")
ASSETS_DIR = os.path.join(BASE_DIR, "poster", "assets")
TEMPLATE_PATH = os.path.join(BASE_DIR, "ICCAS2026_Poster-Sample.pptx")
OUTPUT_PPTX = os.path.join(BASE_DIR, "poster", "ICCAS2026_Poster_Quadruped.pptx")
os.makedirs(ASSETS_DIR, exist_ok=True)

# --- Configuration & Theme Colors ---
FONT_FAMILY = "Arial"
SESSION_CODE = "Session Code [TuP...]"
REPO_URL = "https://github.com/CGV-HGU/isaac-go2-ros2"

# Color Palette (Matching Fig 1 & Academic Standard)
COLOR_NAVY_DARK = RGBColor(0x07, 0x1A, 0x33)
COLOR_NAVY      = RGBColor(0x0B, 0x25, 0x45)
COLOR_BLUE      = RGBColor(0x15, 0x65, 0xC0)  # Sim / Virtual
COLOR_BLUE_BG   = RGBColor(0xF0, 0xF6, 0xFE)
COLOR_GREEN     = RGBColor(0x16, 0x65, 0x34)  # Real / Physical
COLOR_GREEN_BG  = RGBColor(0xF0, 0xFD, 0xF4)
COLOR_GRAY      = RGBColor(0x33, 0x41, 0x55)  # Motivation / Challenges
COLOR_GRAY_BG   = RGBColor(0xF8, 0xFA, 0xFC)
COLOR_GOLD      = RGBColor(0xF5, 0x9E, 0x0B)  # Accent / Badges
COLOR_GOLD_LIGHT= RGBColor(0xFE, 0xF3, 0xC7)
COLOR_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
COLOR_TEXT_MAIN = RGBColor(0x0F, 0x17, 0x2A)  # Slate 900
COLOR_TEXT_SUB  = RGBColor(0x47, 0x55, 0x69)  # Slate 600
COLOR_LIGHT_BLUE= RGBColor(0xBF, 0xDB, 0xFE)

# --- Helper Functions ---

def strip_shape_style(shape):
    sp = shape._element
    st = sp.find(qn("p:style"))
    if st is not None:
        sp.remove(st)

def add_rect(slide, x, y, w, h, fill_color, line_color=None, line_width=1, shape_type=MSO_SHAPE.RECTANGLE, radius=None):
    shp = slide.shapes.add_shape(shape_type, Cm(x), Cm(y), Cm(w), Cm(h))
    strip_shape_style(shp)
    if fill_color is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill_color
    if line_color is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line_color
        shp.line.width = Pt(line_width)
    if radius is not None and len(shp.adjustments) > 0:
        shp.adjustments[0] = radius
    return shp

def parse_runs(text):
    """Parses simple inline tags: **bold**, *italic*, {{c:HEX}}color{{/c}}, ^sup^, ~sub~"""
    pattern = re.compile(r"(\*\*.+?\*\*|\*.+?\*|\^.+?\^|~.+?~|\{\{c:[0-9A-Fa-f]{6}\}\}.+?\{\{/c\}\})")
    runs = []
    for part in pattern.split(text):
        if not part:
            continue
        r = {"text": part, "bold": False, "italic": False, "color": None, "base": 0}
        if part.startswith("**") and part.endswith("**"):
            r["text"] = part[2:-2]
            r["bold"] = True
        elif part.startswith("{{c:"):
            # {{c:HEX}} -> length of prefix is 12 chars
            r["color"] = RGBColor.from_string(part[4:10])
            r["text"] = part[12:-6]
            r["bold"] = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r["text"] = part[1:-1]
            r["italic"] = True
        elif part.startswith("^") and part.endswith("^"):
            r["text"] = part[1:-1]
            r["base"] = 30000
        elif part.startswith("~") and part.endswith("~"):
            r["text"] = part[1:-1]
            r["base"] = -25000
        runs.append(r)
    return runs

def add_paragraph(tf, text, pt=28, bold=False, color=COLOR_TEXT_MAIN, align=PP_ALIGN.LEFT, 
                  space_after=4, bullet=False, bullet_color=COLOR_BLUE, is_first=False):
    p = tf.paragraphs[0] if is_first else tf.add_paragraph()
    p.alignment = align
    p.line_spacing = 1.05
    p.space_before = Pt(0)
    p.space_after = Pt(space_after)
    
    if bullet:
        ind = 0.8
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(Cm(ind))))
        pPr.set("indent", str(int(-Cm(ind))))
        buClr = etree.SubElement(pPr, qn("a:buClr"))
        etree.SubElement(buClr, qn("a:srgbClr"), val=str(bullet_color))
        etree.SubElement(pPr, qn("a:buSzPct"), val="85000")
        etree.SubElement(pPr, qn("a:buFont"), typeface=FONT_FAMILY)
        etree.SubElement(pPr, qn("a:buChar"), char="\u25A0") # Square bullet
        
    runs_data = parse_runs(text)
    for rd in runs_data:
        r = p.add_run()
        r.text = rd["text"]
        r.font.name = FONT_FAMILY
        r.font.size = Pt(pt)
        r.font.bold = rd["bold"] or bold
        r.font.italic = rd["italic"]
        r.font.color.rgb = rd["color"] if rd["color"] else color
        if rd["base"] != 0:
            r._r.get_or_add_rPr().set("baseline", str(rd["base"]))
            
        rPr = r._r.get_or_add_rPr()
        for tag in ("a:ea", "a:cs"):
            el = rPr.find(qn(tag))
            if el is None:
                el = etree.SubElement(rPr, qn(tag))
            el.set("typeface", FONT_FAMILY)
    return p

def create_textbox(slide, x, y, w, h, inset=0.0):
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Cm(inset)
    tf.margin_right = Cm(inset)
    tf.margin_top = Cm(inset)
    tf.margin_bottom = Cm(inset)
    return tf

def add_tag_badge(slide, x, y, text, bg_color, pt=22, text_color=COLOR_WHITE):
    approx_w = len(text) * (pt * 0.024) + 0.8
    h = (pt * 0.042) + 0.32
    shp = add_rect(slide, x, y, approx_w, h, bg_color, line_color=None, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.4)
    tf = shp.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Cm(0)
    add_paragraph(tf, text, pt=pt, bold=True, color=text_color, align=PP_ALIGN.CENTER, space_after=0, is_first=True)
    return approx_w, h

# --- Card Component ---
def draw_card(slide, x, y, w, h, title, title_color, bg_color, border_color=None):
    border_col = border_color if border_color else title_color
    add_rect(slide, x, y, w, h, bg_color, line_color=border_col, line_width=2.5, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.035)
    
    header_h = 2.0
    hb = add_rect(slide, x, y, w, header_h, title_color, line_color=None, shape_type=MSO_SHAPE.ROUND_2_SAME_RECTANGLE)
    if len(hb.adjustments) > 0:
        hb.adjustments[0] = 0.3
        if len(hb.adjustments) > 1:
            hb.adjustments[1] = 0.0
            
    tf = hb.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Cm(0.8)
    tf.margin_right = Cm(0.8)
    tf.margin_top = tf.margin_bottom = Cm(0)
    add_paragraph(tf, title, pt=38, bold=True, color=COLOR_WHITE, align=PP_ALIGN.LEFT, space_after=0, is_first=True)
    
    pad = 0.6
    return (x + pad, y + header_h + pad, w - 2*pad, h - header_h - 2*pad)

# --- Asset Processing ---
def prepare_all_assets():
    print("[INFO] Preparing high-resolution poster assets...")
    assets = {}
    
    # 1. System Overview (high resolution)
    src_overview = os.path.join(FIG_DIR, "fig_system_overview.png")
    dst_overview = os.path.join(ASSETS_DIR, "system_overview.png")
    if not os.path.exists(dst_overview):
        im = Image.open(src_overview).convert("RGB")
        im.resize((8000, 3000), Image.LANCZOS).save(dst_overview, optimize=True)
    assets["overview"] = dst_overview

    # 2. 3DGS & Mesh
    assets["mesh"] = os.path.join(FIG_DIR, "fig_3dgs_mesh.png")

    # 3. Nav2 Replanning
    assets["simnav"] = os.path.join(FIG_DIR, "fig_sim_nav2.png")

    # 4. Robot Crop
    src_robot = os.path.join(FIG_DIR, "fig_real_robot.jpg")
    dst_robot = os.path.join(ASSETS_DIR, "real_robot_crop.jpg")
    if not os.path.exists(dst_robot):
        im = Image.open(src_robot).convert("RGB")
        W, H = im.size
        cw = int(H * 0.85)
        cx = int(W * 0.485)
        x0 = max(0, min(W - cw, cx - cw // 2))
        im.crop((x0, 0, x0 + cw, H)).save(dst_robot, quality=95)
    assets["robot"] = dst_robot

    # 5. 3D Point Cloud Map
    assets["map3d"] = os.path.join(FIG_DIR, "fig_3d_map.png")

    # 6. 2D Occupancy Grid Map
    src_map2d = os.path.join(FIG_DIR, "fig_2d_map.png")
    dst_map2d = os.path.join(ASSETS_DIR, "map2d_x8.png")
    if not os.path.exists(dst_map2d):
        im = Image.open(src_map2d).convert("RGB")
        im.resize((im.size[0] * 8, im.size[1] * 8), Image.NEAREST).save(dst_map2d)
    assets["map2d"] = dst_map2d

    # 7. QR Code to Repository
    qr_path = os.path.join(ASSETS_DIR, "qr_repo.png")
    qr = qrcode.QRCode(border=1, box_size=15, error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(REPO_URL)
    qr.make(fit=True)
    qr.make_image(fill_color="#0B2545", back_color="white").save(qr_path)
    assets["qr"] = qr_path

    return assets

# --- Main Poster Assembly ---
def generate_iccas_poster():
    assets = prepare_all_assets()
    
    prs = Presentation(TEMPLATE_PATH)
    slide = prs.slides[0]
    
    # Remove existing template placeholders
    for shp in list(slide.shapes):
        shp._element.getparent().remove(shp._element)

    # Page Geometry: A0 (84.1 cm x 118.0 cm)
    PAGE_W = 84.1
    PAGE_H = 118.0
    MARGIN_X = 2.0
    CONTENT_W = PAGE_W - 2 * MARGIN_X # 80.1 cm
    
    # =========================================================================
    # 1. TOP HEADER BANNER (Y: 0.0 to 16.0 cm)
    # =========================================================================
    HEADER_H = 16.0
    
    # Background Gradient (Deep Navy to Academic Blue)
    hdr_bg = add_rect(slide, 0, 0, PAGE_W, HEADER_H, COLOR_NAVY)
    hdr_bg.fill.gradient()
    hdr_bg.fill.gradient_angle = 0
    hdr_bg.fill.gradient_stops[0].color.rgb = COLOR_NAVY_DARK
    hdr_bg.fill.gradient_stops[0].position = 0.0
    hdr_bg.fill.gradient_stops[1].color.rgb = COLOR_BLUE
    hdr_bg.fill.gradient_stops[1].position = 1.0

    # Top Sub-bar (Session code & Organization)
    SUB_BAR_H = 1.5
    add_rect(slide, 0, 0, PAGE_W, SUB_BAR_H, COLOR_NAVY_DARK)
    
    # Session Badge
    add_tag_badge(slide, MARGIN_X, 0.32, f"ICCAS 2026 Poster  [ {SESSION_CODE} ]", COLOR_GOLD, pt=26, text_color=COLOR_NAVY_DARK)
    
    # Lab Info Right
    tf_top_r = create_textbox(slide, MARGIN_X + 40.0, 0.32, CONTENT_W - 40.0, 1.0)
    add_paragraph(tf_top_r, "Computer Graphics & Virtual Reality Lab (CGV)  \u00b7  Handong Global University", 
                  pt=26, bold=True, color=COLOR_LIGHT_BLUE, align=PP_ALIGN.RIGHT, is_first=True)

    # Paper Title (Arial 74pt Bold, White, 2 Lines, Centered)
    tf_title = create_textbox(slide, MARGIN_X, SUB_BAR_H + 0.3, CONTENT_W, 5.8)
    add_paragraph(tf_title, "Bridging the Sim-to-Real Gap in Quadrupedal Visual Navigation via 3D Gaussian Splatting", 
                  pt=74, bold=True, color=COLOR_WHITE, align=PP_ALIGN.CENTER, is_first=True)

    # Authors (Clear separation from Title)
    tf_authors = create_textbox(slide, MARGIN_X, SUB_BAR_H + 6.6, CONTENT_W, 1.8)
    add_paragraph(tf_authors, "Minseok Lee^1^,  Seongyu Han^1^,  Heejae Moon^1^,  and  Sung Soo Hwang^1*^", 
                  pt=42, bold=True, color=COLOR_WHITE, align=PP_ALIGN.CENTER, space_after=2, is_first=True)
    
    # Affiliations & Contact
    tf_affil = create_textbox(slide, MARGIN_X, SUB_BAR_H + 8.5, CONTENT_W, 1.6)
    add_paragraph(tf_affil, "^1^School of Artificial Intelligence, Computer and Electrical Engineering, Handong Global University, Pohang 37554, Korea   \u00b7   ^*^sshwang@handong.edu", 
                  pt=28, color=COLOR_LIGHT_BLUE, align=PP_ALIGN.CENTER, space_after=0, is_first=True)

    # Gold Accent Stripe
    add_rect(slide, 0, HEADER_H, PAGE_W, 0.35, COLOR_GOLD)

    # =========================================================================
    # 2. HERO SECTION: SYSTEM ARCHITECTURE BANNER (Y: 16.9 to 39.8 cm, H = 22.9 cm)
    # =========================================================================
    HERO_Y = 16.8
    HERO_H = 22.6
    cx, cy, cw, ch = draw_card(slide, MARGIN_X, HERO_Y, CONTENT_W, HERO_H, 
                               "Sim-to-Real Quadrupedal Visual Navigation Architecture", 
                               COLOR_NAVY, COLOR_WHITE, border_color=COLOR_NAVY)

    # Place Fig 1 System Overview Image (Centered)
    # Aspect ratio = 2.67. Height = 15.5 cm -> Width = 15.5 * 2.67 = 41.4 cm.
    img_h = 15.4
    img_w = img_h * 2.67
    img_x = cx + (cw - img_w) / 2.0
    slide.shapes.add_picture(assets["overview"], Cm(img_x), Cm(cy + 0.1), Cm(img_w), Cm(img_h))
    
    # Subtitle Pill / Takeaway Caption
    tf_hero_cap = create_textbox(slide, cx, cy + img_h + 0.3, cw, 1.6)
    add_paragraph(tf_hero_cap, 
                  "**Core Sim-to-Real Methodology:** Virtual parameters optimized under simulated 3DGS photorealism & RL gait bobbing noise (Phase B) transfer directly to the physical Unitree Go2 (Phase C). Hardware control routes commands to the built-in Sport API via a ROS 2 bridge, bypassing heavy RL computation on the Jetson Orin NX.", 
                  pt=25, color=COLOR_TEXT_SUB, align=PP_ALIGN.CENTER, is_first=True)

    # =========================================================================
    # 3. TWO-COLUMN MAIN BODY (Y: 40.2 to 111.4 cm, Total Available Height = 71.2 cm)
    # =========================================================================
    BODY_Y = 40.2
    COL_GAP = 1.1
    COL_W = (CONTENT_W - COL_GAP) / 2.0 # 39.5 cm
    COL_LEFT_X = MARGIN_X
    COL_RIGHT_X = MARGIN_X + COL_W + COL_GAP

    # -------------------------------------------------------------------------
    # LEFT COLUMN: Motivation -> Digital Twin -> Virtual Replanning
    # -------------------------------------------------------------------------
    
    # Card 1: Motivation & Sim-to-Real Gap
    C1_H = 17.6
    c1_x, c1_y, c1_w, c1_h = draw_card(slide, COL_LEFT_X, BODY_Y, COL_W, C1_H,
                                       "1. Motivation & Sim-to-Real Gap", COLOR_GRAY, COLOR_GRAY_BG)
    tf_c1 = create_textbox(slide, c1_x, c1_y, c1_w, c1_h)
    add_paragraph(tf_c1, "**LiDAR-Free Quadruped Autonomy:** Single RGB-D camera replaces heavy, power-hungry LiDAR on legged robots.",
                  pt=28, bullet=True, bullet_color=COLOR_GRAY, space_after=4, is_first=True)
    add_paragraph(tf_c1, "{{c:E66A00}}Perceptual Sim-to-Real Gap:{{/c}} Simulators lack photorealistic textures, causing V-SLAM feature tracking to fail in real environments.",
                  pt=28, bullet=True, bullet_color=COLOR_GRAY, space_after=4)
    add_paragraph(tf_c1, "{{c:E66A00}}Dynamic Gait Bobbing Noise:{{/c}} Locomotion induces periodic camera jitter (pitch/roll/vertical), generating severe motion blur and false phantom obstacles.",
                  pt=28, bullet=True, bullet_color=COLOR_GRAY, space_after=4)
    add_paragraph(tf_c1, "**Hardware Safety Bottleneck:** Physical trial-and-error tuning carries high collision risks and severe battery runtime limits.",
                  pt=28, bullet=True, bullet_color=COLOR_GRAY, space_after=0)

    # Card 2: 3DGS Digital Twin & Collision Mesh
    C2_Y = BODY_Y + C1_H + 0.8
    C2_H = 25.8
    c2_x, c2_y, c2_w, c2_h = draw_card(slide, COL_LEFT_X, C2_Y, COL_W, C2_H,
                                       "2. 3DGS Digital Twin & Collision Mesh", COLOR_BLUE, COLOR_BLUE_BG)
    
    # Place Fig 2: 3DGS & fVDB Mesh
    m_h = 11.5
    m_w = m_h * 1.81
    m_x = c2_x + (c2_w - m_w) / 2.0
    slide.shapes.add_picture(assets["mesh"], Cm(m_x), Cm(c2_y), Cm(m_w), Cm(m_h))
    
    # Badges on image
    add_tag_badge(slide, m_x + 0.35, c2_y + 0.3, "fVDB Collision Mesh", COLOR_GRAY, pt=20)
    add_tag_badge(slide, m_x + m_w / 2.0 + 0.35, c2_y + 0.3, "3DGS Photorealistic Render", COLOR_BLUE, pt=20)

    tf_c2 = create_textbox(slide, c2_x, c2_y + m_h + 0.35, c2_w, c2_h - m_h - 0.35)
    add_paragraph(tf_c2, "**Offline Digital Twin Modeling:** Reconstructed from DSLR images via COLMAP Structure-from-Motion (SfM) and 3D Gaussian Splatting (3DGS) exported to USDZ format.",
                  pt=28, bullet=True, bullet_color=COLOR_BLUE, space_after=4, is_first=True)
    add_paragraph(tf_c2, "**NVIDIA fVDB Sparse Volumetric Mesh:** Integrated into Isaac Sim 5.1.0 to compute accurate contact geometry and collision boundaries for quadruped foot placement.",
                  pt=28, bullet=True, bullet_color=COLOR_BLUE, space_after=4)
    add_paragraph(tf_c2, "**Sensor Profile Alignment:** Virtual camera FOV, resolution (640x480), and optical center strictly match the physical Intel RealSense D435i sensor to eliminate hardware discrepancies.",
                  pt=28, bullet=True, bullet_color=COLOR_BLUE, space_after=0)

    # Card 3: Virtual Validation & Replanning
    C3_Y = C2_Y + C2_H + 0.8
    C3_H = 25.8
    c3_x, c3_y, c3_w, c3_h = draw_card(slide, COL_LEFT_X, C3_Y, COL_W, C3_H,
                                       "3. Virtual Validation & Replanning", COLOR_BLUE, COLOR_BLUE_BG)
    
    # Place Fig 3: Nav2 Obstacle Replanning
    sim_w = c3_w
    sim_h = sim_w / 3.56
    slide.shapes.add_picture(assets["simnav"], Cm(c3_x), Cm(c3_y), Cm(sim_w), Cm(sim_h))
    add_tag_badge(slide, c3_x + 0.35, c3_y + 0.3, "Isaac Sim Quadruped Model", COLOR_BLUE, pt=20)
    add_tag_badge(slide, c3_x + sim_w * 0.5 + 0.35, c3_y + 0.3, "RViz Nav2 Costmap & Global Path", COLOR_NAVY, pt=20)

    tf_c3 = create_textbox(slide, c3_x, c3_y + sim_h + 0.35, c3_w, c3_h - sim_h - 0.35)
    add_paragraph(tf_c3, "**Gait Oscillation Emulation:** RL policy (PPO in Isaac Lab 2.3.2) trained with domain randomization (mass \u00b13kg, friction 0.6\u20130.8) reproduces realistic camera bobbing dynamics.",
                  pt=28, bullet=True, bullet_color=COLOR_BLUE, space_after=4, is_first=True)
    add_paragraph(tf_c3, "**RTAB-Map V-SLAM Optimization:** Point cloud registration and loop closure thresholds tuned to withstand simulated locomotion pitch/roll oscillations without tracking loss.",
                  pt=28, bullet=True, bullet_color=COLOR_BLUE, space_after=4)
    add_paragraph(tf_c3, "**Nav2 Local Replanning:** DWB local planner dynamically computes collision-free trajectories around inserted obstacles in the 80 m \u00d7 2.5 m digital corridor.",
                  pt=28, bullet=True, bullet_color=COLOR_BLUE, space_after=0)

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: Physical Deployment -> Key Results -> Conclusion
    # -------------------------------------------------------------------------
    
    # Card 4: Physical Deployment (Unitree Go2)
    C4_H = 26.5
    c4_x, c4_y, c4_w, c4_h = draw_card(slide, COL_RIGHT_X, BODY_Y, COL_W, C4_H,
                                       "4. Physical Deployment (Unitree Go2)", COLOR_GREEN, COLOR_GREEN_BG)

    # Image Row: Go2 Crop (r=0.8) + 3D Map (r=1.36) + 2D Map (r=0.22)
    row_h = 12.0
    r_rob = row_h * 0.80
    r_3d  = row_h * 1.36
    r_2d  = row_h * 0.22
    total_row_w = r_rob + r_3d + r_2d + 1.2
    start_rx = c4_x + (c4_w - total_row_w) / 2.0
    
    slide.shapes.add_picture(assets["robot"], Cm(start_rx), Cm(c4_y), Cm(r_rob), Cm(row_h))
    add_tag_badge(slide, start_rx + 0.2, c4_y + 0.2, "Go2 + Orin NX", COLOR_GREEN, pt=18)

    slide.shapes.add_picture(assets["map3d"], Cm(start_rx + r_rob + 0.6), Cm(c4_y), Cm(r_3d), Cm(row_h))
    add_tag_badge(slide, start_rx + r_rob + 0.8, c4_y + 0.2, "RTAB-Map 3D", COLOR_GREEN, pt=18)

    slide.shapes.add_picture(assets["map2d"], Cm(start_rx + r_rob + r_3d + 1.2), Cm(c4_y), Cm(r_2d), Cm(row_h))
    add_tag_badge(slide, start_rx + r_rob + r_3d + 1.2, c4_y + 0.2, "2D", COLOR_GREEN, pt=16)

    tf_c4 = create_textbox(slide, c4_x, c4_y + row_h + 0.35, c4_w, c4_h - row_h - 0.35)
    add_paragraph(tf_c4, "**Fully Onboard Edge Computing:** Intel RealSense D435i connected via USB 3.0 to an NVIDIA Jetson Orin NX, executing RTAB-Map and Nav2 autonomously in real time.",
                  pt=28, bullet=True, bullet_color=COLOR_GREEN, space_after=4, is_first=True)
    add_paragraph(tf_c4, "**Decoupled Locomotion Control:** To conserve edge compute for SLAM, the heavy RL policy is bypassed on physical hardware; Nav2 velocity commands route directly to the built-in Sport API via a lightweight ROS 2 bridge.",
                  pt=28, bullet=True, bullet_color=COLOR_GREEN, space_after=0)

    # Card 5: Experimental Results & Quantitative Metrics
    C5_Y = BODY_Y + C4_H + 0.8
    C5_H = 25.0
    c5_x, c5_y, c5_w, c5_h = draw_card(slide, COL_RIGHT_X, C5_Y, COL_W, C5_H,
                                       "5. Experimental Results & Verification", COLOR_GREEN, COLOR_GREEN_BG)

    # 4 Big Metric Stat Tiles (2x2 Grid)
    tile_w = (c5_w - 0.8) / 2.0
    tile_h = 5.0
    
    stats_data = [
        ("80 m", "3DGS Corridor Digital Twin", COLOR_BLUE),
        ("0.4 m/s", "Walking Speed w/ Loop Closure", COLOR_GREEN),
        ("1 RGB-D", "Single D435i Camera (Zero LiDAR)", COLOR_NAVY),
        ("0 Recalib", "Zero-Code Sim-to-Real Transfer", COLOR_GOLD),
    ]

    for idx, (val, lbl, col) in enumerate(stats_data):
        row = idx // 2
        col_idx = idx % 2
        tx = c5_x + col_idx * (tile_w + 0.8)
        ty = c5_y + row * (tile_h + 0.5)
        
        # Tile background box
        add_rect(slide, tx, ty, tile_w, tile_h, COLOR_WHITE, line_color=col, line_width=3, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.14)
        
        tf_stat = create_textbox(slide, tx + 0.2, ty + 0.15, tile_w - 0.4, tile_h - 0.3)
        add_paragraph(tf_stat, val, pt=56, bold=True, color=col, align=PP_ALIGN.CENTER, space_after=1, is_first=True)
        add_paragraph(tf_stat, lbl, pt=24, bold=True, color=COLOR_TEXT_MAIN, align=PP_ALIGN.CENTER, space_after=0)

    # Findings text below tiles
    tf_c5_text = create_textbox(slide, c5_x, c5_y + 2 * tile_h + 1.2, c5_w, c5_h - 2 * tile_h - 1.2)
    add_paragraph(tf_c5_text, "**Direct Parameter Migration:** Parameters optimized under simulated camera bobbing transferred seamlessly across ROS distributions (Jazzy on sim \u2192 Foxy on robot) without physical retuning.",
                  pt=28, bullet=True, bullet_color=COLOR_GREEN, space_after=4, is_first=True)
    add_paragraph(tf_c5_text, "**Robust Visual Mapping:** RTAB-Map successfully extracted features and detected global loop closures throughout the 80 m corridor, eliminating phantom obstacles.",
                  pt=28, bullet=True, bullet_color=COLOR_GREEN, space_after=0)

    # Card 6: Conclusion & Future Outlook
    C6_Y = C5_Y + C5_H + 0.8
    C6_H = 16.9
    c6_x, c6_y, c6_w, c6_h = draw_card(slide, COL_RIGHT_X, C6_Y, COL_W, C6_H,
                                       "6. Conclusion & Future Outlook", COLOR_NAVY, COLOR_GRAY_BG)

    tf_c6 = create_textbox(slide, c6_x, c6_y, c6_w, c6_h)
    add_paragraph(tf_c6, "**Practical Virtual Sandbox:** Demonstrates that pairing a photorealistic 3DGS digital twin with RL-generated gait dynamics provides an effective, risk-free testbed for visual quadruped navigation.",
                  pt=28, bullet=True, bullet_color=COLOR_NAVY, space_after=4, is_first=True)
    add_paragraph(tf_c6, "**Decoupled Edge Efficiency:** Separating simulation-based parameter discovery from onboard execution enables lightweight real-world deployment on resource-constrained Jetson platforms.",
                  pt=28, bullet=True, bullet_color=COLOR_NAVY, space_after=4)
    add_paragraph(tf_c6, "**Future Research:** Expanding from static corridors to dynamic obstacle avoidance benchmarks, multi-floor terrain transitions, and direct neural policy deployment.",
                  pt=28, bullet=True, bullet_color=COLOR_NAVY, space_after=0)

    # =========================================================================
    # 4. BOTTOM FOOTER: REFERENCES, ACKNOWLEDGEMENT & QR (Y: 111.8 to 116.8 cm)
    # =========================================================================
    FOOTER_Y = 111.8
    FOOTER_H = 4.8
    
    # Separator line
    add_rect(slide, MARGIN_X, FOOTER_Y - 0.2, CONTENT_W, 0.08, COLOR_NAVY)

    # Left: References & Acknowledgement
    QR_SIZE = 4.4
    REFS_W = CONTENT_W - QR_SIZE - 1.5
    tf_ftr = create_textbox(slide, MARGIN_X, FOOTER_Y, REFS_W, FOOTER_H)
    
    add_paragraph(tf_ftr, "**References:**  [1] B. Kerbl et al., \u201c3D Gaussian splatting for real-time radiance field rendering,\u201d *ACM TOG*, 2023.  [2] M. Labb\u00e9 & F. Michaud, \u201cRTAB-Map as an open-source lidar and visual SLAM library,\u201d *arXiv:2403.06341*, 2024.  [3] S. Macenski et al., \u201cRobot Operating System 2: Design and architecture,\u201d *Science Robotics*, 2022.  [4] S. Zhu et al., \u201cVR-Robo: Real-to-sim-to-real for visual robot navigation,\u201d *IEEE RA-L*, 2025.",
                  pt=22, color=COLOR_TEXT_SUB, space_after=3, is_first=True)
    add_paragraph(tf_ftr, "**Acknowledgement:** This research was supported by the National Research Foundation of Korea (NRF) grant funded by the Korean government (MSIT) (No. RS-2025-24683458).",
                  pt=22, color=COLOR_NAVY, bold=True, space_after=0)

    # Right: QR Code + Label
    qr_x = MARGIN_X + CONTENT_W - QR_SIZE
    slide.shapes.add_picture(assets["qr"], Cm(qr_x), Cm(FOOTER_Y), Cm(QR_SIZE), Cm(QR_SIZE))
    
    tf_qr_lbl = create_textbox(slide, qr_x - 0.5, FOOTER_Y + QR_SIZE + 0.05, QR_SIZE + 1.0, 0.6)
    add_paragraph(tf_qr_lbl, "GitHub Repo & Code", pt=18, bold=True, color=COLOR_NAVY, align=PP_ALIGN.CENTER, is_first=True)

    # Save PPTX
    prs.save(OUTPUT_PPTX)
    print(f"[SUCCESS] ICCAS 2026 Academic Poster successfully created at:\n  {OUTPUT_PPTX}")

if __name__ == "__main__":
    generate_iccas_poster()
