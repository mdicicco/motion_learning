"""
Scene Generator - Creates random MJCF XML scenes with articulated objects.

Generates scenes with:
- Multiple objects with random shapes (sphere, cylinder, box, capsule)
- Articulated joints (hinge, slide, ball, free)
- Random colorization
- Optional world-fixed bases
"""

import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path
import numpy as np


ShapeType = Literal["sphere", "cylinder", "box", "capsule"]
JointType = Literal["hinge", "slide", "ball", "free"]


@dataclass
class BodyConfig:
    """Configuration for a single body in the scene."""
    name: str
    shape: ShapeType
    size: tuple[float, ...]
    pos: tuple[float, float, float]
    color: tuple[float, float, float, float]
    joint_type: JointType | None = None
    joint_axis: tuple[float, float, float] = (1, 0, 0)
    joint_range: tuple[float, float] = (-1.57, 1.57)


@dataclass
class ObjectConfig:
    """Configuration for an object (potentially articulated) in the scene."""
    name: str
    base_pos: tuple[float, float, float]
    bodies: list[BodyConfig] = field(default_factory=list)
    fixed_base: bool = True


@dataclass
class EnvironmentConfig:
    """Configuration for floor and sky textures."""
    # Floor settings
    floor_type: Literal["checker", "flat", "gradient"] = "checker"
    floor_color1: tuple[float, float, float] = (0.2, 0.2, 0.25)
    floor_color2: tuple[float, float, float] = (0.3, 0.3, 0.35)
    floor_size: float = 10.0
    
    # Sky settings
    sky_type: Literal["gradient", "flat"] = "gradient"
    sky_color1: tuple[float, float, float] = (0.4, 0.6, 0.9)  # horizon
    sky_color2: tuple[float, float, float] = (0.1, 0.2, 0.4)  # zenith
    sky_radius: float = 50.0


def random_environment() -> EnvironmentConfig:
    """Generate random environment settings."""
    # Random floor style
    floor_type = random.choice(["checker", "flat", "gradient"])
    
    # Generate floor colors with earthy/neutral tones
    floor_hue = random.uniform(0.0, 0.15)  # browns, grays, tans
    floor_sat = random.uniform(0.1, 0.4)
    floor_val1 = random.uniform(0.15, 0.35)
    floor_val2 = random.uniform(0.25, 0.45)
    
    floor_color1 = _hsv_to_rgb(floor_hue, floor_sat, floor_val1)
    floor_color2 = _hsv_to_rgb(floor_hue, floor_sat * 0.8, floor_val2)
    
    # Random sky style
    sky_type = random.choice(["gradient", "flat"])
    
    # Generate sky colors - blues, oranges, purples
    sky_hue = random.choice([
        random.uniform(0.55, 0.65),  # blue sky
        random.uniform(0.05, 0.12),  # sunset orange
        random.uniform(0.75, 0.85),  # purple dusk
        random.uniform(0.45, 0.50),  # teal
    ])
    sky_sat1 = random.uniform(0.4, 0.7)
    sky_sat2 = random.uniform(0.5, 0.8)
    sky_val1 = random.uniform(0.6, 0.95)  # horizon (brighter)
    sky_val2 = random.uniform(0.2, 0.5)   # zenith (darker)
    
    sky_color1 = _hsv_to_rgb(sky_hue, sky_sat1, sky_val1)
    sky_color2 = _hsv_to_rgb(sky_hue * 1.05, sky_sat2, sky_val2)
    
    return EnvironmentConfig(
        floor_type=floor_type,
        floor_color1=floor_color1,
        floor_color2=floor_color2,
        sky_type=sky_type,
        sky_color1=sky_color1,
        sky_color2=sky_color2,
    )


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Convert HSV (0-1 range) to RGB (0-1 range)."""
    h = h % 1.0
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    if h_i == 0:
        return (v, t, p)
    elif h_i == 1:
        return (q, v, p)
    elif h_i == 2:
        return (p, v, t)
    elif h_i == 3:
        return (p, q, v)
    elif h_i == 4:
        return (t, p, v)
    else:
        return (v, p, q)


def random_color(alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Generate a random RGBA color with good saturation."""
    # Use HSV-like approach for vibrant colors
    hue = random.random()
    saturation = random.uniform(0.6, 1.0)
    value = random.uniform(0.5, 1.0)
    
    # Convert HSV to RGB (simplified)
    h_i = int(hue * 6)
    f = hue * 6 - h_i
    p = value * (1 - saturation)
    q = value * (1 - f * saturation)
    t = value * (1 - (1 - f) * saturation)
    
    if h_i == 0:
        r, g, b = value, t, p
    elif h_i == 1:
        r, g, b = q, value, p
    elif h_i == 2:
        r, g, b = p, value, t
    elif h_i == 3:
        r, g, b = p, q, value
    elif h_i == 4:
        r, g, b = t, p, value
    else:
        r, g, b = value, p, q
    
    return (r, g, b, alpha)


def random_shape() -> tuple[ShapeType, tuple[float, ...]]:
    """Generate a random shape type with appropriate size."""
    shape = random.choice(["sphere", "cylinder", "box", "capsule"])
    
    if shape == "sphere":
        size = (random.uniform(0.05, 0.15),)
    elif shape == "cylinder":
        size = (random.uniform(0.03, 0.08), random.uniform(0.1, 0.3))
    elif shape == "box":
        size = (
            random.uniform(0.05, 0.15),
            random.uniform(0.05, 0.15),
            random.uniform(0.05, 0.15),
        )
    else:  # capsule
        size = (random.uniform(0.03, 0.08), random.uniform(0.1, 0.25))
    
    return shape, size


def random_joint_type(exclude_free: bool = True) -> JointType:
    """Generate a random joint type."""
    if exclude_free:
        return random.choice(["hinge", "slide", "ball"])
    return random.choice(["hinge", "slide", "ball", "free"])


def random_axis() -> tuple[float, float, float]:
    """Generate a random unit axis vector."""
    axis = random.choice([
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
    ])
    return axis


def generate_articulated_object(
    name: str,
    base_pos: tuple[float, float, float],
    num_bodies: int,
    fixed_base: bool = True,
) -> ObjectConfig:
    """Generate an articulated object with multiple linked bodies."""
    obj = ObjectConfig(name=name, base_pos=base_pos, fixed_base=fixed_base)
    
    # Generate chain of bodies
    current_offset = [0.0, 0.0, 0.0]
    
    for i in range(num_bodies):
        shape, size = random_shape()
        color = random_color()
        
        # Determine joint type for this body
        if i == 0:
            joint_type = None if fixed_base else "free"
        else:
            joint_type = random_joint_type(exclude_free=True)
        
        # Calculate position offset based on previous body
        if i > 0:
            # Offset along a random axis
            offset_axis = random.choice([(0.2, 0, 0), (0, 0.2, 0), (0, 0, 0.2)])
            current_offset = [
                current_offset[0] + offset_axis[0],
                current_offset[1] + offset_axis[1],
                current_offset[2] + offset_axis[2],
            ]
        
        # Joint range depends on type
        if joint_type == "hinge":
            joint_range = (-np.pi * 0.75, np.pi * 0.75)
        elif joint_type == "slide":
            joint_range = (-0.3, 0.3)
        elif joint_type == "ball":
            joint_range = (-np.pi * 0.5, np.pi * 0.5)
        else:
            joint_range = (-1.57, 1.57)
        
        body = BodyConfig(
            name=f"{name}_body_{i}",
            shape=shape,
            size=size,
            pos=tuple(current_offset),
            color=color,
            joint_type=joint_type,
            joint_axis=random_axis(),
            joint_range=joint_range,
        )
        obj.bodies.append(body)
    
    return obj


def generate_random_scene(
    num_objects: int = 3,
    max_bodies_per_object: int = 4,
    min_bodies_per_object: int = 1,
    fixed_base_probability: float = 0.5,
) -> tuple[list[ObjectConfig], EnvironmentConfig]:
    """Generate a random scene with multiple articulated objects and environment."""
    objects = []
    
    # Spread objects in a grid pattern
    grid_size = int(np.ceil(np.sqrt(num_objects)))
    spacing = 0.8
    
    for i in range(num_objects):
        row = i // grid_size
        col = i % grid_size
        
        # Center the grid
        x = (col - grid_size / 2 + 0.5) * spacing
        y = (row - grid_size / 2 + 0.5) * spacing
        z = 0.3  # Above ground
        
        num_bodies = random.randint(min_bodies_per_object, max_bodies_per_object)
        fixed_base = random.random() < fixed_base_probability
        
        obj = generate_articulated_object(
            name=f"object_{i}",
            base_pos=(x, y, z),
            num_bodies=num_bodies,
            fixed_base=fixed_base,
        )
        objects.append(obj)
    
    # Generate random environment
    environment = random_environment()
    
    return objects, environment


def scene_to_mjcf(
    objects: list[ObjectConfig], 
    scene_name: str = "generated_scene",
    environment: EnvironmentConfig | None = None,
) -> str:
    """Convert scene configuration to MJCF XML string."""
    if environment is None:
        environment = EnvironmentConfig()
    
    # Root mujoco element
    mujoco = ET.Element("mujoco", model=scene_name)
    
    # Compiler settings
    ET.SubElement(mujoco, "compiler", angle="radian", autolimits="true")
    
    # Visual settings for better rendering
    visual = ET.SubElement(mujoco, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "quality", shadowsize="4096")
    
    # Options
    ET.SubElement(mujoco, "option", gravity="0 0 -9.81", timestep="0.002")
    
    # Assets (materials for colors)
    asset = ET.SubElement(mujoco, "asset")
    
    # Floor texture
    floor_rgb1 = " ".join(f"{c:.3f}" for c in environment.floor_color1)
    floor_rgb2 = " ".join(f"{c:.3f}" for c in environment.floor_color2)
    
    if environment.floor_type == "checker":
        ET.SubElement(asset, "texture", name="floor_tex", type="2d", builtin="checker",
                      width="512", height="512", rgb1=floor_rgb1, rgb2=floor_rgb2)
        ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex", 
                      texrepeat="8 8", reflectance="0.1")
    elif environment.floor_type == "gradient":
        ET.SubElement(asset, "texture", name="floor_tex", type="2d", builtin="gradient",
                      width="512", height="512", rgb1=floor_rgb1, rgb2=floor_rgb2)
        ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex", 
                      texrepeat="1 1", reflectance="0.05")
    else:  # flat
        floor_rgba = floor_rgb1 + " 1.0"
        ET.SubElement(asset, "material", name="floor_mat", rgba=floor_rgba,
                      reflectance="0.1", specular="0.2")
    
    # Sky texture (gradient from horizon to zenith)
    sky_rgb1 = " ".join(f"{c:.3f}" for c in environment.sky_color1)
    sky_rgb2 = " ".join(f"{c:.3f}" for c in environment.sky_color2)
    
    if environment.sky_type == "gradient":
        ET.SubElement(asset, "texture", name="sky_tex", type="skybox", builtin="gradient",
                      width="512", height="512", rgb1=sky_rgb1, rgb2=sky_rgb2)
    else:  # flat
        ET.SubElement(asset, "texture", name="sky_tex", type="skybox", builtin="flat",
                      width="512", height="512", rgb1=sky_rgb1, rgb2=sky_rgb2)
    
    # Sky material for the dome
    ET.SubElement(asset, "material", name="sky_mat", texture="sky_tex",
                  emission="1", specular="0", shininess="0")
    
    # Create materials for each body
    mat_idx = 0
    for obj in objects:
        for body in obj.bodies:
            rgba = " ".join(f"{c:.3f}" for c in body.color)
            ET.SubElement(asset, "material", name=f"mat_{mat_idx}", rgba=rgba,
                         specular="0.5", shininess="0.5")
            mat_idx += 1
    
    # Worldbody
    worldbody = ET.SubElement(mujoco, "worldbody")
    
    # Sky dome (large inverted sphere)
    sky_size = f"{environment.sky_radius:.1f}"
    ET.SubElement(worldbody, "geom", name="sky_dome", type="sphere",
                  size=sky_size, pos="0 0 0", material="sky_mat",
                  contype="0", conaffinity="0")  # No collision
    
    # Ground plane
    floor_size = f"{environment.floor_size:.1f} {environment.floor_size:.1f} 0.1"
    ET.SubElement(worldbody, "geom", name="ground", type="plane", 
                  size=floor_size, material="floor_mat")
    
    # Lighting
    ET.SubElement(worldbody, "light", name="top_light", pos="0 0 3", 
                  dir="0 0 -1", diffuse="0.8 0.8 0.8", specular="0.3 0.3 0.3")
    ET.SubElement(worldbody, "light", name="side_light", pos="2 2 2",
                  dir="-1 -1 -1", diffuse="0.4 0.4 0.4")
    
    # Add objects
    mat_idx = 0
    for obj in objects:
        # Create base body at object position
        pos_str = " ".join(f"{p:.3f}" for p in obj.base_pos)
        
        current_parent = worldbody
        
        for i, body in enumerate(obj.bodies):
            body_pos = " ".join(f"{p:.3f}" for p in body.pos)
            body_elem = ET.SubElement(current_parent, "body", 
                                     name=body.name, pos=body_pos)
            
            # Add joint if specified
            if body.joint_type:
                joint_attrs = {"name": f"{body.name}_joint", "type": body.joint_type}
                
                if body.joint_type in ["hinge", "slide"]:
                    joint_attrs["axis"] = " ".join(str(a) for a in body.joint_axis)
                    joint_attrs["range"] = f"{body.joint_range[0]:.3f} {body.joint_range[1]:.3f}"
                elif body.joint_type == "ball":
                    joint_attrs["range"] = f"0 {body.joint_range[1]:.3f}"
                # free joints don't need axis or range
                
                ET.SubElement(body_elem, "joint", **joint_attrs)
            
            # Add geometry
            geom_attrs = {
                "name": f"{body.name}_geom",
                "type": body.shape,
                "size": " ".join(f"{s:.3f}" for s in body.size),
                "material": f"mat_{mat_idx}",
            }
            ET.SubElement(body_elem, "geom", **geom_attrs)
            
            # Next body is child of this one (for articulated chains)
            current_parent = body_elem
            mat_idx += 1
    
    # Pretty print
    rough_string = ET.tostring(mujoco, encoding="unicode")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def generate_and_save_scene(
    output_path: Path | str,
    num_objects: int = 3,
    max_bodies_per_object: int = 4,
    seed: int | None = None,
) -> list[ObjectConfig]:
    """Generate a random scene and save it to an XML file."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    objects = generate_random_scene(
        num_objects=num_objects,
        max_bodies_per_object=max_bodies_per_object,
    )
    
    xml_content = scene_to_mjcf(objects, scene_name=output_path.stem)
    
    with open(output_path, "w") as f:
        f.write(xml_content)
    
    print(f"Scene saved to: {output_path}")
    return objects


# =============================================================================
# Compact Scene Encoding
# =============================================================================
#
# Format: One line per object, bodies separated by |
#
# Object line: bx,by,bz,F|body1|body2|...
#   - bx,by,bz = base position (floats)
#   - F = fixed_base (0 or 1)
#
# Body: S:sizes@px,py,pz#RRGGBBAA^J:ax,ay,az:r1,r2
#   - S = shape: s(sphere), c(cylinder), b(box), p(capsule)
#   - sizes = comma-separated size values (1-3 depending on shape)
#   - px,py,pz = position offset
#   - RRGGBBAA = 8-char hex color
#   - J = joint: n(none), h(hinge), l(slide), a(ball), f(free)
#   - ax,ay,az = axis (for h/l only)
#   - r1,r2 = range (for h/l/a)
#
# Example:
# 0.000,0.000,0.300,1|s:0.134@0.000,0.000,0.000#93661827^f|c:0.062,0.193@0.000,0.000,0.200#f67c2cff^a:0.000,1.571
# =============================================================================

SHAPE_TO_CHAR = {"sphere": "s", "cylinder": "c", "box": "b", "capsule": "p"}
CHAR_TO_SHAPE = {v: k for k, v in SHAPE_TO_CHAR.items()}

JOINT_TO_CHAR = {"hinge": "h", "slide": "l", "ball": "a", "free": "f", None: "n"}
CHAR_TO_JOINT = {"h": "hinge", "l": "slide", "a": "ball", "f": "free", "n": None}


def color_to_hex(rgba: tuple[float, float, float, float]) -> str:
    """Convert RGBA (0-1 floats) to 8-char hex string."""
    r, g, b, a = rgba
    return f"{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}{int(a*255):02x}"


def hex_to_color(hex_str: str) -> tuple[float, float, float, float]:
    """Convert 8-char hex string to RGBA (0-1 floats)."""
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    a = int(hex_str[6:8], 16) / 255.0
    return (r, g, b, a)


def encode_body(body: BodyConfig) -> str:
    """Encode a single body to compact string format (structure only, no color).
    
    Format: shape:sizes@pos^joint
    """
    # Shape and sizes
    shape_char = SHAPE_TO_CHAR[body.shape]
    sizes_str = ",".join(f"{s:.3f}" for s in body.size)
    
    # Position
    pos_str = ",".join(f"{p:.3f}" for p in body.pos)
    
    # Joint
    joint_char = JOINT_TO_CHAR[body.joint_type]
    
    if body.joint_type in ("hinge", "slide"):
        axis_str = ",".join(str(int(a)) for a in body.joint_axis)
        range_str = f"{body.joint_range[0]:.3f},{body.joint_range[1]:.3f}"
        joint_str = f"{joint_char}:{axis_str}:{range_str}"
    elif body.joint_type == "ball":
        range_str = f"{body.joint_range[0]:.3f},{body.joint_range[1]:.3f}"
        joint_str = f"{joint_char}:{range_str}"
    elif body.joint_type == "free":
        joint_str = joint_char
    else:  # None
        joint_str = joint_char
    
    return f"{shape_char}:{sizes_str}@{pos_str}^{joint_str}"


def decode_body(encoded: str, body_name: str) -> BodyConfig:
    """Decode a compact string to BodyConfig (assigns default grey color)."""
    # Split: shape:sizes@pos^joint
    shape_part, rest = encoded.split("@")
    pos_part, joint_part = rest.split("^")
    
    # Shape and sizes
    shape_char, sizes_str = shape_part.split(":")
    shape = CHAR_TO_SHAPE[shape_char]
    size = tuple(float(s) for s in sizes_str.split(","))
    
    # Position
    pos = tuple(float(p) for p in pos_part.split(","))
    
    # Default grey color for decoded bodies
    color = (0.5, 0.5, 0.5, 1.0)
    
    # Joint
    joint_parts = joint_part.split(":")
    joint_char = joint_parts[0]
    joint_type = CHAR_TO_JOINT[joint_char]
    
    joint_axis = (1, 0, 0)  # default
    joint_range = (-1.57, 1.57)  # default
    
    if joint_type in ("hinge", "slide"):
        axis_str = joint_parts[1]
        range_str = joint_parts[2]
        joint_axis = tuple(int(a) for a in axis_str.split(","))
        joint_range = tuple(float(r) for r in range_str.split(","))
    elif joint_type == "ball":
        range_str = joint_parts[1]
        joint_range = tuple(float(r) for r in range_str.split(","))
    
    return BodyConfig(
        name=body_name,
        shape=shape,
        size=size,
        pos=pos,
        color=color,
        joint_type=joint_type,
        joint_axis=joint_axis,
        joint_range=joint_range,
    )


def encode_object(obj: ObjectConfig) -> str:
    """Encode an object (with all bodies) to a single line."""
    # Header: bx,by,bz,F
    header = ",".join(f"{p:.3f}" for p in obj.base_pos)
    header += f",{1 if obj.fixed_base else 0}"
    
    # Bodies
    body_strs = [encode_body(body) for body in obj.bodies]
    
    return header + "|" + "|".join(body_strs)


def decode_object(encoded: str, obj_name: str) -> ObjectConfig:
    """Decode a single line to ObjectConfig."""
    parts = encoded.split("|")
    header = parts[0]
    body_strs = parts[1:]
    
    # Parse header: bx,by,bz,F
    header_parts = header.split(",")
    base_pos = (float(header_parts[0]), float(header_parts[1]), float(header_parts[2]))
    fixed_base = header_parts[3] == "1"
    
    # Parse bodies
    bodies = []
    for i, body_str in enumerate(body_strs):
        body_name = f"{obj_name}_body_{i}"
        bodies.append(decode_body(body_str, body_name))
    
    return ObjectConfig(
        name=obj_name,
        base_pos=base_pos,
        fixed_base=fixed_base,
        bodies=bodies,
    )


def skeleton_environment() -> EnvironmentConfig:
    """Create a neutral grey skeleton environment for visualization."""
    return EnvironmentConfig(
        floor_type="flat",
        floor_color1=(0.3, 0.3, 0.3),
        floor_color2=(0.3, 0.3, 0.3),
        sky_type="flat",
        sky_color1=(0.15, 0.15, 0.15),
        sky_color2=(0.15, 0.15, 0.15),
    )


def encode_scene(objects: list[ObjectConfig]) -> str:
    """Encode scene structure to compact multi-line string (no color/texture).
    
    One line per object, containing only structural information:
    shape, size, position, joint type, axis, range.
    """
    lines = [encode_object(obj) for obj in objects]
    return "\n".join(lines)


def decode_scene(encoded: str) -> list[ObjectConfig]:
    """Decode compact string back to list of ObjectConfig with grey colors."""
    lines = [line.strip() for line in encoded.strip().split("\n") if line.strip()]
    
    objects = []
    for i, line in enumerate(lines):
        obj_name = f"object_{i}"
        objects.append(decode_object(line, obj_name))
    
    return objects


def save_scene_encoding(objects: list[ObjectConfig], output_path: Path | str) -> None:
    """Save compact scene encoding to a text file (structure only)."""
    output_path = Path(output_path)
    encoded = encode_scene(objects)
    with open(output_path, "w") as f:
        f.write(encoded)


def load_scene_encoding(input_path: Path | str) -> list[ObjectConfig]:
    """Load scene from compact encoding file."""
    input_path = Path(input_path)
    with open(input_path, "r") as f:
        encoded = f.read()
    return decode_scene(encoded)


def scene_to_skeleton_mjcf(objects: list[ObjectConfig], scene_name: str = "skeleton") -> str:
    """Generate MJCF XML for a skeleton scene with joint visualizations.
    
    Features:
    - Semi-transparent grey objects (alpha=0.4)
    - Red joint visualizations:
      - Hinge: red cylinder aligned with rotation axis
      - Slide: red box aligned with slide axis
      - Ball: red sphere
      - Free: red octahedron (two pyramids)
    - Neutral grey floor and sky
    """
    env = skeleton_environment()
    
    # Joint visualization sizes
    JOINT_VIS_SIZE = 0.025  # Base size for joint markers
    JOINT_COLOR = "0.9 0.1 0.1 1.0"  # Bright red
    BODY_ALPHA = 0.4  # Semi-transparent bodies
    
    # Root mujoco element
    mujoco_elem = ET.Element("mujoco", model=scene_name)
    
    # Compiler settings
    ET.SubElement(mujoco_elem, "compiler", angle="radian", autolimits="true")
    
    # Visual settings
    visual = ET.SubElement(mujoco_elem, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "quality", shadowsize="4096")
    
    # Options
    ET.SubElement(mujoco_elem, "option", gravity="0 0 -9.81", timestep="0.002")
    
    # Assets
    asset = ET.SubElement(mujoco_elem, "asset")
    
    # Floor material (grey, no reflection)
    floor_rgb = " ".join(f"{c:.3f}" for c in env.floor_color1)
    ET.SubElement(asset, "material", name="floor_mat", rgba=f"{floor_rgb} 1.0",
                  reflectance="0", specular="0")
    
    # Sky texture
    sky_rgb = " ".join(f"{c:.3f}" for c in env.sky_color1)
    ET.SubElement(asset, "texture", name="sky_tex", type="skybox", builtin="flat",
                  width="512", height="512", rgb1=sky_rgb, rgb2=sky_rgb)
    ET.SubElement(asset, "material", name="sky_mat", texture="sky_tex",
                  emission="1", specular="0", shininess="0")
    
    # Semi-transparent grey material for bodies
    ET.SubElement(asset, "material", name="body_mat", 
                  rgba=f"0.5 0.5 0.5 {BODY_ALPHA}", specular="0.3", shininess="0.3")
    
    # Joint visualization materials
    ET.SubElement(asset, "material", name="joint_hinge_mat", rgba=JOINT_COLOR,
                  specular="0.8", shininess="0.8")
    ET.SubElement(asset, "material", name="joint_slide_mat", rgba="0.1 0.9 0.1 1.0",
                  specular="0.8", shininess="0.8")  # Green for slide
    ET.SubElement(asset, "material", name="joint_ball_mat", rgba="0.1 0.1 0.9 1.0",
                  specular="0.8", shininess="0.8")  # Blue for ball
    ET.SubElement(asset, "material", name="joint_free_mat", rgba="0.9 0.9 0.1 1.0",
                  specular="0.8", shininess="0.8")  # Yellow for free
    
    # Worldbody
    worldbody = ET.SubElement(mujoco_elem, "worldbody")
    
    # Sky dome
    ET.SubElement(worldbody, "geom", name="sky_dome", type="sphere",
                  size=f"{env.sky_radius:.1f}", pos="0 0 0", material="sky_mat",
                  contype="0", conaffinity="0")
    
    # Ground plane
    ET.SubElement(worldbody, "geom", name="ground", type="plane",
                  size=f"{env.floor_size:.1f} {env.floor_size:.1f} 0.1", material="floor_mat")
    
    # Diffuse ambient lighting (no shadows)
    ET.SubElement(worldbody, "light", name="ambient1", pos="0 0 5",
                  dir="0 0 -1", diffuse="0.6 0.6 0.6", specular="0 0 0",
                  castshadow="false", directional="true")
    ET.SubElement(worldbody, "light", name="ambient2", pos="0 0 -5",
                  dir="0 0 1", diffuse="0.4 0.4 0.4", specular="0 0 0",
                  castshadow="false", directional="true")
    ET.SubElement(worldbody, "light", name="ambient3", pos="5 0 0",
                  dir="-1 0 0", diffuse="0.3 0.3 0.3", specular="0 0 0",
                  castshadow="false", directional="true")
    ET.SubElement(worldbody, "light", name="ambient4", pos="-5 0 0",
                  dir="1 0 0", diffuse="0.3 0.3 0.3", specular="0 0 0",
                  castshadow="false", directional="true")
    
    # Add objects with joint visualizations
    joint_idx = 0
    for obj in objects:
        current_parent = worldbody
        
        for i, body in enumerate(obj.bodies):
            body_pos = " ".join(f"{p:.3f}" for p in body.pos)
            body_elem = ET.SubElement(current_parent, "body",
                                     name=body.name, pos=body_pos)
            
            # Add joint if specified
            if body.joint_type:
                joint_attrs = {"name": f"{body.name}_joint", "type": body.joint_type}
                
                if body.joint_type in ["hinge", "slide"]:
                    joint_attrs["axis"] = " ".join(str(a) for a in body.joint_axis)
                    joint_attrs["range"] = f"{body.joint_range[0]:.3f} {body.joint_range[1]:.3f}"
                elif body.joint_type == "ball":
                    joint_attrs["range"] = f"0 {body.joint_range[1]:.3f}"
                
                ET.SubElement(body_elem, "joint", **joint_attrs)
                
                # Add joint visualization geom (decorative, no collision)
                jv_name = f"joint_vis_{joint_idx}"
                jv_attrs = {
                    "name": jv_name,
                    "contype": "0",
                    "conaffinity": "0",
                    "pos": "0 0 0",
                }
                
                if body.joint_type == "hinge":
                    # Cylinder aligned with rotation axis
                    ax = body.joint_axis
                    # Compute euler angles to align cylinder with axis
                    # For simplicity, use quat or just set size
                    jv_attrs["type"] = "cylinder"
                    jv_attrs["size"] = f"{JOINT_VIS_SIZE} {JOINT_VIS_SIZE * 2}"
                    jv_attrs["material"] = "joint_hinge_mat"
                    # Align with axis using fromto instead
                    axis_len = JOINT_VIS_SIZE * 2.5
                    fromto = f"{-ax[0]*axis_len} {-ax[1]*axis_len} {-ax[2]*axis_len} {ax[0]*axis_len} {ax[1]*axis_len} {ax[2]*axis_len}"
                    jv_attrs["fromto"] = fromto
                    jv_attrs.pop("size")  # fromto defines size
                    jv_attrs["size"] = f"{JOINT_VIS_SIZE}"
                    
                elif body.joint_type == "slide":
                    # Box aligned with slide axis
                    ax = body.joint_axis
                    jv_attrs["type"] = "box"
                    # Make box elongated along slide axis
                    sx = JOINT_VIS_SIZE * (3 if abs(ax[0]) > 0.5 else 1)
                    sy = JOINT_VIS_SIZE * (3 if abs(ax[1]) > 0.5 else 1)
                    sz = JOINT_VIS_SIZE * (3 if abs(ax[2]) > 0.5 else 1)
                    jv_attrs["size"] = f"{sx:.3f} {sy:.3f} {sz:.3f}"
                    jv_attrs["material"] = "joint_slide_mat"
                    
                elif body.joint_type == "ball":
                    # Sphere for ball joint
                    jv_attrs["type"] = "sphere"
                    jv_attrs["size"] = f"{JOINT_VIS_SIZE * 1.5}"
                    jv_attrs["material"] = "joint_ball_mat"
                    
                elif body.joint_type == "free":
                    # Larger sphere for free joint (6 DOF)
                    jv_attrs["type"] = "sphere"
                    jv_attrs["size"] = f"{JOINT_VIS_SIZE * 2}"
                    jv_attrs["material"] = "joint_free_mat"
                
                ET.SubElement(body_elem, "geom", **jv_attrs)
                joint_idx += 1
            
            # Add body geometry (semi-transparent)
            geom_attrs = {
                "name": f"{body.name}_geom",
                "type": body.shape,
                "size": " ".join(f"{s:.3f}" for s in body.size),
                "material": "body_mat",
            }
            ET.SubElement(body_elem, "geom", **geom_attrs)
            
            # Next body is child of this one
            current_parent = body_elem
    
    # Pretty print
    rough_string = ET.tostring(mujoco_elem, encoding="unicode")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


if __name__ == "__main__":
    # Test generation
    output = Path("data/test_scene/scene.xml")
    generate_and_save_scene(output, num_objects=4, max_bodies_per_object=3, seed=42)

