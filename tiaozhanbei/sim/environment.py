#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
桌面工业零件仿真环境。

桌面上表面与 z=0 对齐；机械臂渲染默认关闭，仅显示桌面与标准件。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from panda3d.core import CardMaker, Texture

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_TB_DIR = os.path.abspath(os.path.join(_SIM_DIR, os.pardir))
_REPO_ROOT = os.path.abspath(os.path.join(_TB_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wrs import wd, rm, mgm, mcm
import wrs.basis.data_adapter as da
# 暂时不渲染机械臂，仅保留桌面 + 零件场景。
# from wrs.robot_sim.robots.panthera_ht.panthera_ht_single_arm import PantheraHTSglArm


# 桌面尺寸（单位：米）
DESK_XY = np.array([1.20, 0.80])
DESK_THICKNESS = 0.04
DESK_HEIGHT = 0.72
LEG_XY = np.array([0.05, 0.05])

# 标准件 STL 多数来自 CAD 模型库，常见单位是毫米；WRS 场景使用米。
MODEL_SCALE = 0.001

# 桌面条纹贴图缓存目录（桌面盒子本身无 UV，条纹贴在 z≈0 的薄卡片上）。
DESK_TEXTURE_DIR = os.path.join(_TB_DIR, "textures", "desk_stripes")

# 可选桌面条纹预设；采集时可用 seed 随机挑一种，或显式指定名称。
DESK_STRIPE_PRESETS = {
    "plain": {
        "orientation": None,
        "stripe_px": 0,
        "color_a": (210, 210, 200),
        "color_b": (210, 210, 200),
    },
    "vert_gray": {
        "orientation": "v",
        "stripe_px": 36,
        "color_a": (200, 200, 195),
        "color_b": (165, 165, 158),
    },
    "horiz_gray": {
        "orientation": "h",
        "stripe_px": 36,
        "color_a": (200, 200, 195),
        "color_b": (165, 165, 158),
    },
    "vert_wide": {
        "orientation": "v",
        "stripe_px": 72,
        "color_a": (215, 210, 200),
        "color_b": (150, 155, 160),
    },
    "horiz_narrow": {
        "orientation": "h",
        "stripe_px": 18,
        "color_a": (220, 218, 210),
        "color_b": (175, 170, 160),
    },
    "diag_soft": {
        "orientation": "diag",
        "stripe_px": 28,
        "color_a": (205, 200, 190),
        "color_b": (160, 165, 170),
    },
    "vert_bluegray": {
        "orientation": "v",
        "stripe_px": 40,
        "color_a": (190, 200, 210),
        "color_b": (145, 155, 165),
    },
    # 更丰富纹理：棋盘 / 网格 / 斜纹变体 / 暖冷色
    "checker_warm": {
        "orientation": "checker",
        "stripe_px": 48,
        "color_a": (215, 205, 190),
        "color_b": (170, 160, 145),
    },
    "checker_cool": {
        "orientation": "checker",
        "stripe_px": 40,
        "color_a": (195, 205, 215),
        "color_b": (150, 160, 175),
    },
    "grid_light": {
        "orientation": "grid",
        "stripe_px": 64,
        "color_a": (220, 220, 215),
        "color_b": (155, 155, 150),
    },
    "diag_dense": {
        "orientation": "diag",
        "stripe_px": 16,
        "color_a": (210, 200, 185),
        "color_b": (155, 150, 145),
    },
    "diag_greengray": {
        "orientation": "diag",
        "stripe_px": 32,
        "color_a": (195, 210, 195),
        "color_b": (145, 160, 150),
    },
    "horiz_warm": {
        "orientation": "h",
        "stripe_px": 28,
        "color_a": (225, 210, 195),
        "color_b": (175, 155, 140),
    },
    "vert_olive": {
        "orientation": "v",
        "stripe_px": 44,
        "color_a": (200, 205, 180),
        "color_b": (155, 160, 130),
    },
    "plain_warm": {
        "orientation": None,
        "stripe_px": 0,
        "color_a": (218, 210, 198),
        "color_b": (218, 210, 198),
    },
    "plain_cool": {
        "orientation": None,
        "stripe_px": 0,
        "color_a": (200, 205, 210),
        "color_b": (200, 205, 210),
    },
    # 第二批纹理：供大批量采集轮换
    "vert_slate": {
        "orientation": "v",
        "stripe_px": 52,
        "color_a": (185, 190, 195),
        "color_b": (135, 140, 148),
    },
    "horiz_slate": {
        "orientation": "h",
        "stripe_px": 52,
        "color_a": (185, 190, 195),
        "color_b": (135, 140, 148),
    },
    "diag_blue": {
        "orientation": "diag",
        "stripe_px": 36,
        "color_a": (200, 210, 220),
        "color_b": (140, 155, 175),
    },
    "diag_amber": {
        "orientation": "diag",
        "stripe_px": 24,
        "color_a": (230, 215, 185),
        "color_b": (180, 150, 110),
    },
    "checker_fine": {
        "orientation": "checker",
        "stripe_px": 24,
        "color_a": (210, 210, 205),
        "color_b": (160, 160, 155),
    },
    "checker_large": {
        "orientation": "checker",
        "stripe_px": 80,
        "color_a": (205, 198, 188),
        "color_b": (155, 148, 138),
    },
    "grid_dense": {
        "orientation": "grid",
        "stripe_px": 32,
        "color_a": (225, 225, 220),
        "color_b": (140, 145, 150),
    },
    "grid_warm": {
        "orientation": "grid",
        "stripe_px": 56,
        "color_a": (230, 220, 205),
        "color_b": (165, 145, 125),
    },
    "dots_cool": {
        "orientation": "dots",
        "stripe_px": 28,
        "color_a": (205, 210, 215),
        "color_b": (145, 155, 165),
    },
    "dots_warm": {
        "orientation": "dots",
        "stripe_px": 36,
        "color_a": (220, 210, 195),
        "color_b": (165, 145, 125),
    },
    "noise_soft": {
        "orientation": "noise",
        "stripe_px": 8,
        "color_a": (210, 208, 200),
        "color_b": (170, 168, 160),
    },
    "plain_sand": {
        "orientation": None,
        "stripe_px": 0,
        "color_a": (225, 215, 195),
        "color_b": (225, 215, 195),
    },
    "plain_steel": {
        "orientation": None,
        "stripe_px": 0,
        "color_a": (190, 195, 200),
        "color_b": (190, 195, 200),
    },
    "vert_burgundy": {
        "orientation": "v",
        "stripe_px": 30,
        "color_a": (210, 195, 195),
        "color_b": (155, 125, 130),
    },
    "horiz_teal": {
        "orientation": "h",
        "stripe_px": 34,
        "color_a": (195, 215, 210),
        "color_b": (120, 155, 150),
    },
    "diag_cross": {
        "orientation": "diag2",
        "stripe_px": 40,
        "color_a": (215, 215, 210),
        "color_b": (150, 155, 160),
    },
}

# 从 Part_Model 的七个分类中各选 1~2 个 STL，作为桌面上的示例零件。
PART_MODEL_DIR = os.path.join(_TB_DIR, "Part_Model")
PART_SPECS = [
    # 轴承类 Bearings
    ("轴承-角接触球轴承", ("Bearings", "Bearings (ball, roller, needle, etc.)", "Ball bearings",
                    "Angular contact ball bearings", "Angular contact ball bearings1.stl")),
    ("轴承-深沟球轴承", ("Bearings", "Bearings (ball, roller, needle, etc.)", "Ball bearings",
                    "Deep groove ball bearings", "Deep groove ball bearings1.stl")),
    # 齿轮类 Gears
    ("齿轮-锥齿轮", ("Gears", "Bevel gears", "Bevel gears1.stl")),
    ("齿轮-斜齿轮", ("Gears", "Helical gears", "Helical gears1.stl")),
    # 螺母类 Nuts
    ("螺母-盖形螺母", ("Nuts", "Acorn nuts", "Acorn nuts1.stl")),
    ("螺母-六角螺母", ("Nuts", "Hex nuts", "Hex nuts1.stl")),
    # 铆钉类 Rivets
    ("铆钉-抽芯铆钉", ("Rivets", "Blind rivets", "Blind rivets1.stl")),
    ("铆钉-压缩铆钉", ("Rivets", "Compression rivets", "Compression rivets1.stl")),
    # 滚子轴承类 Roller bearings
    ("滚子轴承-角接触滚子轴承", ("Roller bearings", "Angular contact roller bearings",
                         "Angular contact roller bearings1.stl")),
    ("滚子轴承-圆柱滚子轴承", ("Roller bearings", "Cylindrical roller bearings",
                         "Cylindrical roller bearings1.stl")),
    # 螺钉和螺栓类 Screws and bolts
    ("螺栓-地脚螺栓", ("Screws and bolts", "bolts", "Anchor Bolts", "Anchor Bolts1.stl")),
    ("螺栓-沉头螺栓", ("Screws and bolts", "bolts", "Countersunk head bolts",
                    "Countersunk head bolts1.stl")),
    # 垫圈类 Washers
    ("垫圈-沉头垫圈", ("Washers", "Countersunk washers", "Countersunk washers1.stl")),
    ("垫圈-杯形垫圈", ("Washers", "Cup washers", "Cup washers1.stl")),
]


def gen_desk():
    """生成桌面，上表面精确位于 z=0。"""
    desk = mcm.gen_box(
        xyz_lengths=rm.vec(DESK_XY[0], DESK_XY[1], DESK_THICKNESS),
        pos=rm.vec(0.0, 0.0, -DESK_THICKNESS / 2.0),
        rgb=rm.vec(0.82, 0.82, 0.78),
    )

    leg_height = DESK_HEIGHT - DESK_THICKNESS
    leg_z = -DESK_THICKNESS - leg_height / 2.0
    leg_offsets = [
        ( DESK_XY[0] / 2.0 - LEG_XY[0],  DESK_XY[1] / 2.0 - LEG_XY[1]),
        ( DESK_XY[0] / 2.0 - LEG_XY[0], -DESK_XY[1] / 2.0 + LEG_XY[1]),
        (-DESK_XY[0] / 2.0 + LEG_XY[0],  DESK_XY[1] / 2.0 - LEG_XY[1]),
        (-DESK_XY[0] / 2.0 + LEG_XY[0], -DESK_XY[1] / 2.0 + LEG_XY[1]),
    ]
    legs = [
        mcm.gen_box(
            xyz_lengths=rm.vec(LEG_XY[0], LEG_XY[1], leg_height),
            pos=rm.vec(x, y, leg_z),
            rgb=rm.vec(0.35, 0.35, 0.35),
        )
        for x, y in leg_offsets
    ]
    return desk, legs


def list_desk_stripe_styles():
    """返回可用桌面条纹预设名称。"""
    return list(DESK_STRIPE_PRESETS.keys())


def resolve_desk_stripe_style(style=None, seed=None):
    """解析条纹样式：显式名称优先，否则按 seed 在预设里选。"""
    names = list_desk_stripe_styles()
    if style is not None:
        if style not in DESK_STRIPE_PRESETS:
            raise ValueError(f"未知桌面条纹样式 '{style}'，可选: {names}")
        return style
    rng = np.random.default_rng(seed)
    return names[int(rng.integers(0, len(names)))]


def make_desk_stripe_image(style_name, size=512):
    """按预设生成条纹 RGB 图（uint8）。"""
    preset = DESK_STRIPE_PRESETS[style_name]
    color_a = np.array(preset["color_a"], dtype=np.uint8)
    color_b = np.array(preset["color_b"], dtype=np.uint8)
    orientation = preset["orientation"]
    stripe_px = max(1, int(preset["stripe_px"]))

    img = np.zeros((size, size, 3), dtype=np.uint8)
    if orientation is None:
        img[:] = color_a
        return img

    yy, xx = np.mgrid[0:size, 0:size]
    if orientation == "v":
        band = (xx // stripe_px) % 2
    elif orientation == "h":
        band = (yy // stripe_px) % 2
    elif orientation == "diag":
        band = ((xx + yy) // stripe_px) % 2
    elif orientation == "checker":
        band = ((xx // stripe_px) + (yy // stripe_px)) % 2
    elif orientation == "diag2":
        # 交叉斜纹感：两方向条带叠加
        band = (((xx + yy) // stripe_px) + ((xx - yy) // stripe_px)) % 2
    elif orientation == "grid":
        # 细线网格：默认底色 a，线为 b
        img[:] = color_a
        img[yy % stripe_px == 0] = color_b
        img[xx % stripe_px == 0] = color_b
        return img
    elif orientation == "dots":
        img[:] = color_a
        cell = max(4, int(stripe_px))
        cy = (yy % cell) - cell // 2
        cx = (xx % cell) - cell // 2
        rad2 = (cell * 0.22) ** 2
        img[(cx * cx + cy * cy) <= rad2] = color_b
        return img
    elif orientation == "noise":
        # 确定性伪随机颗粒（可复现，不依赖全局 RNG）
        h = ((xx * 374761393) ^ (yy * 668265263) ^ (stripe_px * 1274126177)) & 0xFFFFFFFF
        band = (h >> 16) % 2
    else:
        raise ValueError(f"不支持的条纹方向: {orientation}")

    img[band == 0] = color_a
    img[band == 1] = color_b
    return img


def save_desk_stripe_texture(style_name, size=512):
    """生成并缓存条纹贴图，返回 PNG 路径。"""
    os.makedirs(DESK_TEXTURE_DIR, exist_ok=True)
    path = os.path.join(DESK_TEXTURE_DIR, f"{style_name}_{size}.png")
    if not os.path.exists(path):
        img = make_desk_stripe_image(style_name, size=size)
        # Windows 中文路径更稳妥：imencode + tofile
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"无法编码桌面条纹贴图: {style_name}")
        buf.tofile(path)
    return path


def attach_desk_stripe(base, style=None, seed=None, size=512):
    """在桌面上表面贴一张薄卡片条纹纹理（桌面盒子本身无 UV）。

    :param base: WRS World / ShowBase
    :param style: 预设名；None 时按 seed 随机选
    :param seed: 随机种子
    :return: 条纹信息 dict，含 style / texture_path / node
    """
    style_name = resolve_desk_stripe_style(style=style, seed=seed)
    tex_path = save_desk_stripe_texture(style_name, size=size)

    tex = Texture(f"desk_stripe_{style_name}")
    # 直接从内存灌纹理，避免 loader 对路径编码的兼容问题。
    img = make_desk_stripe_image(style_name, size=size)
    tex.setup2dTexture(size, size, Texture.T_unsigned_byte, Texture.F_rgb)
    # Panda 默认按行从下到上读取，这里翻转一下让条纹方向符合直觉。
    tex.setRamImage(np.flipud(img).tobytes())
    tex.setMagfilter(Texture.FTLinear)
    tex.setMinfilter(Texture.FTLinear)

    card_maker = CardMaker(f"desk_stripe_card_{style_name}")
    half_x = float(DESK_XY[0] / 2.0)
    half_y = float(DESK_XY[1] / 2.0)
    card_maker.setFrame(-half_x, half_x, -half_y, half_y)
    card = base.render.attachNewNode(card_maker.generate())
    # CardMaker 默认在 XY 平面；绕 X 转 -90° 铺到桌面（z=0）上。
    card.setP(-90)
    card.setPos(0.0, 0.0, 0.0008)
    card.setTexture(tex)
    card.setColor(1.0, 1.0, 1.0, 1.0)
    card.setLightOff()
    card.setMaterialOff()
    # 略微压低优先级，避免与零件 z-fighting；零件底面在 z=0。
    card.setDepthOffset(1)

    return {
        "style": style_name,
        "texture_path": tex_path,
        "node": card,
        "seed": seed,
    }


def resolve_stl_path(rel_path):
    """把相对 Part_Model 的路径解析为绝对路径。

    rel_path 可为路径元组/列表，或使用 / 或系统分隔符的相对字符串。
    """
    if isinstance(rel_path, (list, tuple)):
        return os.path.join(PART_MODEL_DIR, *rel_path)
    return os.path.join(PART_MODEL_DIR, str(rel_path).replace("/", os.sep))


def normalize_rel_path(rel_path):
    """统一为使用正斜杠的相对路径字符串，便于 manifest 跨平台比较。"""
    if isinstance(rel_path, (list, tuple)):
        return "/".join(str(p) for p in rel_path)
    return str(rel_path).replace("\\", "/")


# 顶层分类与叶子目录的中文名（用于标注预览 / annotations.class）
CATEGORY_ZH = {
    "Bearings": "轴承",
    "Gears": "齿轮",
    "Machines & Tools": "机器工具",
    "Nuts": "螺母",
    "Rivets": "铆钉",
    "Roller bearings": "滚子轴承",
    "Screws and bolts": "螺钉螺栓",
    "Washers": "垫圈",
}

LEAF_DIR_ZH = {
    # Bearings
    "Angular contact ball bearings": "角接触球轴承",
    "Cam rollers": "凸轮滚轮",
    "Combined bearings": "组合轴承",
    "Deep groove ball bearings": "深沟球轴承",
    "Four point contact balls bearings": "四点接触球轴承",
    "Journal bearings": "滑动轴承",
    "Needle bearings": "滚针轴承",
    "Radial insert ball bearings": "外球面球轴承",
    "Rod ends": "杆端关节轴承",
    "Roller bearings": "滚子轴承",
    "Rotary table bearings": "转盘轴承",
    "Self aligning ball bearings": "调心球轴承",
    "Slewing rings": "回转支承",
    "Spindle ball bearings": "主轴球轴承",
    "Thrust bearings": "推力轴承",
    # Gears
    "Bevel gears": "锥齿轮",
    "Helical gears": "斜齿轮",
    "Herring bone gears (double helical)": "人字齿轮",
    "Hypoid gears": "准双曲面齿轮",
    "Internal gears": "内齿轮",
    "Racks": "齿条",
    "Spiral bevel gears": "螺旋锥齿轮",
    "Spiroid gears": "蜗杆面齿轮",
    "Spur gears": "直齿轮",
    "Worm gears": "蜗轮",
    # Machines & Tools
    "Bits": "钻头批头",
    "Brushes": "刷子",
    "Cutting": "切削工具",
    "Drilling": "钻孔工具",
    "Grinding": "磨削工具",
    "Hammers": "锤子",
    "Keys": "键",
    "Pliers": "钳子",
    "Ratchets and sockets": "棘轮套筒",
    "Riveting": "铆接工具",
    "Sanders": "砂光工具",
    "Screwdrivers": "螺丝刀",
    "Torque control": "扭矩工具",
    "Vice": "台钳",
    "Wrenches": "扳手",
    # Nuts
    "Acorn nuts": "盖形螺母",
    "Arm nuts": "手柄螺母",
    "Barrel nuts": "筒形螺母",
    "Castle nuts, KM slotted nuts, slotted nuts": "槽形螺母",
    "Crimp Nuts": "压接螺母",
    "Flange nuts": "法兰螺母",
    "G-nuts": "G型螺母",
    "Hex nuts": "六角螺母",
    "Insert nuts": "嵌装螺母",
    "Knurled nuts": "滚花螺母",
    "Rivet nuts": "铆螺母",
    "Self-locking nuts": "自锁螺母",
    "Sleeves": "衬套",
    "Square nuts": "方螺母",
    "Square style nuts": "方形螺母",
    "T-nuts": "T型螺母",
    "Weld nuts": "焊接螺母",
    "Wing nuts": "蝶形螺母",
    # Rivets
    "Blind rivets": "抽芯铆钉",
    "Brake lining  Clutch facing rivets": "制动离合器铆钉",
    "Compression rivets": "压缩铆钉",
    "Semi-tubular rivets": "半空心铆钉",
    "Snap rivets": "按扣铆钉",
    "Solid rivets": "实心铆钉",
    # Roller bearings
    "Angular contact roller bearings": "角接触滚子轴承",
    "Cylindrical roller bearings": "圆柱滚子轴承",
    "Spherical roller bearings": "调心滚子轴承",
    "Tapered roller bearings": "圆锥滚子轴承",
    # Screws and bolts
    "Anchor Bolts": "地脚螺栓",
    "Binding screws": "装订螺丝",
    "Binding tapping screws": "装订自攻螺丝",
    "Bugle screws": "喇叭头螺丝",
    "Button screws": "圆头螺丝",
    "Button tapping screws": "圆头自攻螺丝",
    "Cheese tapping screws": "圆柱头自攻螺丝",
    "Cone point set screws": "尖端紧定螺丝",
    "Countersunk head bolts": "沉头螺栓",
    "Countersunk or flat screws": "沉头螺丝",
    "Countersunk or flat tapping screws": "沉头自攻螺丝",
    "Cup point set screws": "凹端紧定螺丝",
    "Cylindrical head bolts": "圆柱头螺栓",
    "Dog point set screws": "圆柱端紧定螺丝",
    "Drywall bugle screws": "石膏板喇叭头螺丝",
    "Drywall button screws": "石膏板圆头螺丝",
    "Drywall various screws": "石膏板螺丝",
    "Drywall washer screws": "石膏板垫圈螺丝",
    "Eye screws": "吊环螺丝",
    "Fillister screws": "圆柱头螺丝",
    "Fillister tapping screws": "圆柱头自攻螺丝",
    "Flat point set screws": "平端紧定螺丝",
    "Hex head bolts": "六角头螺栓",
    "Hexagonal screws": "六角螺丝",
    "Hexagonal tapping screws": "六角自攻螺丝",
    "Knurled screws": "滚花螺丝",
    "Mushroom or truss screws": "蘑菇头螺丝",
    "Mushroom or truss tapping screws": "蘑菇头自攻螺丝",
    "Oval or raised screws": "半沉头螺丝",
    "Oval or raised tapping screws": "半沉头自攻螺丝",
    "Pan screws": "盘头螺丝",
    "Pan tapping screws": "盘头自攻螺丝",
    "Plastic cheese head screws": "塑料圆柱头螺丝",
    "Plastic countersunk or flat screws": "塑料沉头螺丝",
    "Plastic fillister screws": "塑料圆柱头螺丝",
    "Plastic hexagonal screws": "塑料六角螺丝",
    "Plastic knurled screws": "塑料滚花螺丝",
    "Plastic socket screws": "塑料内六角螺丝",
    "Round head bolts": "圆头螺栓",
    "Round screws": "圆头螺丝",
    "Socket screws, Socket head screws": "内六角螺丝",
    "Square screws": "方头螺丝",
    "T-Bolts": "T型螺栓",
    "U-Bolts": "U型螺栓",
    "Washer screws": "垫圈螺丝",
    # Washers
    "Countersunk washers": "沉头垫圈",
    "Cup washers": "杯形垫圈",
    "Fender washers": "加大垫圈",
    "Lock washers": "锁紧垫圈",
    "Plain or flat washers": "平垫圈",
    "Sealing washers": "密封垫圈",
    "Shoulder washers": "台阶垫圈",
    "Slotted washers": "开口垫圈",
    "Spherical seat washers": "球面垫圈",
    "Spring washers": "弹簧垫圈",
    "Square washers": "方垫圈",
    "Tab washers": "止动垫圈",
    "Taper washers": "锥形垫圈",
}


def class_name_from_rel_path(rel_path):
    """由相对路径生成中文 class 名：顶层分类-叶子类型。"""
    norm = normalize_rel_path(rel_path)
    parts = [p for p in norm.split("/") if p]
    if not parts:
        return "未知"
    top = parts[0]
    stem = os.path.splitext(parts[-1])[0]
    leaf_dir = parts[-2] if len(parts) >= 2 else stem
    top_zh = CATEGORY_ZH.get(top, top)
    leaf_zh = LEAF_DIR_ZH.get(leaf_dir, leaf_dir)
    return f"{top_zh}-{leaf_zh}"


def scan_part_library(root_dir=None):
    """扫描 Part_Model 下全部 STL，返回稳定排序的零件描述列表。

    每项：
      {
        "rel_path": "Bearings/.../xxx.stl",  # 正斜杠
        "category": "Bearings",
        "class": "Bearings-Angular contact ball bearings",
        "stl_path": 绝对路径,
      }
    """
    root = root_dir or PART_MODEL_DIR
    entries = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if not filename.lower().endswith(".stl"):
                continue
            abs_path = os.path.join(dirpath, filename)
            rel = normalize_rel_path(os.path.relpath(abs_path, root))
            parts = rel.split("/")
            category = parts[0] if parts else "unknown"
            entries.append(
                {
                    "rel_path": rel,
                    "category": category,
                    "class": class_name_from_rel_path(rel),
                    "stl_path": abs_path,
                }
            )
    entries.sort(key=lambda e: e["rel_path"].lower())
    return entries


def load_part_model(name, rel_path, pos, rgb, rotmat=np.eye(3)):
    """加载一个 STL 零件，并把模型底面放到桌面 z=0 上。

    处理流程：
    1. 按相对路径找到 Part_Model 中的 STL；
    2. 将毫米制 STL 缩放到米制；
    3. 把模型局部原点移动到“底面中心”，方便用 pos 摆放。
    4. 根据 rotmat 设置零件放置角度。
    """
    stl_path = resolve_stl_path(rel_path)
    mesh = da.trm.load(stl_path)
    mesh.apply_scale(np.array([MODEL_SCALE, MODEL_SCALE, MODEL_SCALE]))

    # 先把模型 XY 中心移到局部原点
    bounds = mesh.bounds
    xy_center = (bounds[0, :2] + bounds[1, :2]) / 2.0
    mesh.apply_translation(np.array([-xy_center[0], -xy_center[1], 0.0]))
    # 把旋转直接应用到 mesh 顶点上
    homomat = np.eye(4)
    homomat[:3, :3] = rotmat
    mesh.apply_transform(homomat)
    # 旋转后重新计算最低点，让最低点贴到桌面 z=0
    bounds_after_rot = mesh.bounds
    z_min_after_rot = bounds_after_rot[0, 2]
    mesh.apply_translation(np.array([0.0, 0.0, -z_min_after_rot]))
    part = mcm.CollisionModel(mesh, name=name, rgb=rgb)
    part.pos = pos
    return {
        "name": name,
        "rel_path": normalize_rel_path(rel_path),
        "stl_path": stl_path,
        "model": part,
        "pos": pos,
        "rgb": rgb,
        "rotmat": rotmat,
    }


def sample_desk_positions(n, seed=None, margin=0.10, min_dist=0.08, max_tries=2000):
    """在桌面范围内采样 n 个互不重叠的摆放点（z=0）。

    兼容旧接口；新采集请用 ``sample_safe_part_positions``（避开臂/收纳盒）。
    """
    rng = np.random.default_rng(seed)
    half_x = DESK_XY[0] / 2.0 - margin
    half_y = DESK_XY[1] / 2.0 - margin
    positions = []
    for _ in range(n):
        placed = False
        for _try in range(max_tries):
            x = float(rng.uniform(-half_x, half_x))
            y = float(rng.uniform(-half_y, half_y))
            ok = True
            for px, py, _pz in positions:
                if (x - px) ** 2 + (y - py) ** 2 < min_dist ** 2:
                    ok = False
                    break
            if ok:
                positions.append(rm.vec(x, y, 0.0))
                placed = True
                break
        if not placed:
            # 放宽失败时退化为网格点，保证总能放下。
            gx = -half_x + (len(positions) % 5) * (2 * half_x / 4.0)
            gy = -half_y + (len(positions) // 5) * (2 * half_y / 4.0)
            positions.append(rm.vec(float(gx), float(gy), 0.0))
    return positions


def sample_safe_part_positions(n, seed=None, min_dist=None):
    """在 ``scene_layout`` 安全区内采样：IK 友好，避开机械臂基座与收纳盒。"""
    from tiaozhanbei.sim.scene_layout import (
        PART_MIN_CENTER_DIST,
        sample_part_positions,
    )

    md = PART_MIN_CENTER_DIST if min_dist is None else float(min_dist)
    raw = sample_part_positions(n, seed=seed, min_dist=md)
    return [rm.vec(float(p[0]), float(p[1]), 0.0) for p in raw]


def load_parts_from_entries(part_entries, seed=None, use_safe_layout=True):
    """按零件描述列表加载并随机摆放到桌面。

    part_entries: scan_part_library() 的子集，或含 rel_path/class 的 dict 列表。
    use_safe_layout: True 时避开机械臂/收纳盒固定区（推荐采集用）。
    """
    colors = [
        rm.vec(0.80, 0.42, 0.36),
        rm.vec(0.42, 0.62, 0.86),
        rm.vec(0.48, 0.75, 0.46),
        rm.vec(0.82, 0.70, 0.38),
        rm.vec(0.65, 0.50, 0.75),
        rm.vec(0.40, 0.70, 0.70),
    ]
    rng = np.random.default_rng(seed)
    pos_seed = None if seed is None else seed + 1
    if use_safe_layout:
        positions = sample_safe_part_positions(len(part_entries), seed=pos_seed)
    else:
        positions = sample_desk_positions(len(part_entries), seed=pos_seed)
    part_infos = []
    for i, entry in enumerate(part_entries):
        rel_path = entry["rel_path"] if isinstance(entry, dict) else entry[1]
        name = (
            entry.get("class")
            if isinstance(entry, dict)
            else entry[0]
        )
        if not name:
            name = class_name_from_rel_path(rel_path)
        yaw = float(rng.uniform(0.0, 2.0 * np.pi))
        rotmat = rm.rotmat_from_euler(0.0, 0.0, yaw)
        info = load_part_model(
            name,
            rel_path,
            positions[i],
            colors[i % len(colors)],
            rotmat=rotmat,
        )
        info["category"] = entry.get("category") if isinstance(entry, dict) else None
        part_infos.append(info)
    return part_infos


def load_table_parts():
    """兼容旧预览：加载 PART_SPECS 中的固定示例零件。"""
    entries = [
        {"rel_path": normalize_rel_path(rel_path), "class": name, "category": rel_path[0]}
        for name, rel_path in PART_SPECS
    ]
    return load_parts_from_entries(entries, seed=0)


def build_scene(
    show_frame=True,
    seed=None,
    desk_stripe=None,
    part_entries=None,
    use_safe_layout=True,
):
    """构建仿真场景，并返回后续采集需要的对象句柄。

    :param show_frame: 是否显示世界坐标轴
    :param seed: 随机种子；desk_stripe 为 None 时用来挑选条纹样式；零件摆位也用它
    :param desk_stripe: 桌面条纹预设名（见 DESK_STRIPE_PRESETS）；None 则按 seed 选
    :param part_entries: 本次要摆放的零件列表（scan_part_library 子集）；None 则用 PART_SPECS
    :param use_safe_layout: 零件避开机械臂/收纳盒固定区（采集推荐 True；不渲染盒/臂）
    """
    base = wd.World(cam_pos=[1.4, 1.2, 0.9], lookat_pos=[0.0, 0.0, 0.15])
    if show_frame:
        mgm.gen_frame(ax_length=0.15).attach_to(base)

    desk, legs = gen_desk()
    desk.attach_to(base)
    for leg in legs:
        leg.attach_to(base)

    # 桌面盒子无 UV，条纹用 z≈0 薄卡片贴图实现，可按 seed/样式切换。
    desk_stripe_info = attach_desk_stripe(base, style=desk_stripe, seed=seed)

    if part_entries is None:
        part_infos = load_table_parts()
    else:
        part_infos = load_parts_from_entries(
            part_entries, seed=seed, use_safe_layout=use_safe_layout
        )
    for part_info in part_infos:
        part_info["model"].attach_to(base)

    # -------------------------------------------------------------------------
    # 机械臂暂时不渲染；需要时取消下面注释即可恢复。
    # -------------------------------------------------------------------------
    # robot = PantheraHTSglArm(pos=rm.vec(0.0, 0.0, 0.0), enable_cc=False)
    # robot.goto_given_conf(np.zeros(6))
    # robot_mesh = robot.gen_meshmodel(toggle_tcp_frame=True, toggle_jnt_frames=False)
    # robot_mesh.attach_to(base)
    robot = None
    robot_mesh = None

    return {
        "base": base,
        "desk": desk,
        "legs": legs,
        "part_infos": part_infos,
        "desk_stripe": desk_stripe_info,
        "robot": robot,
        "robot_mesh": robot_mesh,
    }


def main():
    # 预览时可用：python environment.py --stripe vert_gray
    import argparse

    parser = argparse.ArgumentParser(description="预览桌面零件场景（可切换条纹）")
    parser.add_argument(
        "--stripe",
        default=None,
        choices=list_desk_stripe_styles(),
        help="桌面条纹样式；不指定则按 seed 随机",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子（条纹等）")
    args = parser.parse_args()

    scene = build_scene(show_frame=True, seed=args.seed, desk_stripe=args.stripe)
    print(f"[env] desk_stripe = {scene['desk_stripe']['style']}")
    scene["base"].run()


if __name__ == "__main__":
    main()
