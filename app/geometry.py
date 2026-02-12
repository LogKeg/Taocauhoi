"""
Geometry drawing module for educational questions.
Supports: triangle, rectangle, square, circle, parallelogram, trapezoid.
"""
import io
import math
import re
from typing import Optional, Tuple, List, Dict, Any

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Arc
import numpy as np


# Patterns to detect geometry shapes in questions
GEOMETRY_PATTERNS = {
    "triangle": re.compile(
        r"tam giác\s*([A-Z]{3})?|triangle\s*([A-Z]{3})?|△\s*([A-Z]{3})?",
        re.IGNORECASE
    ),
    "rectangle": re.compile(
        r"hình chữ nhật\s*([A-Z]{4})?|rectangle\s*([A-Z]{4})?",
        re.IGNORECASE
    ),
    "square": re.compile(
        r"hình vuông\s*([A-Z]{4})?|square\s*([A-Z]{4})?",
        re.IGNORECASE
    ),
    "circle": re.compile(
        r"đường tròn|hình tròn|circle|tâm\s*([A-Z])|bán kính",
        re.IGNORECASE
    ),
    "parallelogram": re.compile(
        r"hình bình hành\s*([A-Z]{4})?|parallelogram\s*([A-Z]{4})?",
        re.IGNORECASE
    ),
    "trapezoid": re.compile(
        r"hình thang\s*([A-Z]{4})?|trapezoid\s*([A-Z]{4})?",
        re.IGNORECASE
    ),
}

# Pattern to extract measurements
MEASURE_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(cm|m|dm|mm|km)?",
    re.IGNORECASE
)


def detect_geometry(text: str) -> Optional[Dict[str, Any]]:
    """Detect geometry shape and extract labels from question text."""
    text_lower = text.lower()

    for shape, pattern in GEOMETRY_PATTERNS.items():
        match = pattern.search(text)
        if match:
            # Extract vertex labels if present
            labels = None
            for group in match.groups():
                if group and group.isupper():
                    labels = list(group)
                    break

            # Extract measurements
            measures = MEASURE_PATTERN.findall(text)
            measurements = []
            for val, unit in measures:
                try:
                    measurements.append(float(val.replace(",", ".")))
                except ValueError:
                    pass

            return {
                "shape": shape,
                "labels": labels,
                "measurements": measurements[:4],  # Max 4 measurements
                "text": text
            }

    return None


def draw_triangle(
    labels: Optional[List[str]] = None,
    measurements: Optional[List[float]] = None,
    right_angle: bool = False
) -> bytes:
    """Draw a triangle and return PNG bytes."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    ax.set_aspect('equal')
    ax.axis('off')

    labels = labels or ['A', 'B', 'C']

    if right_angle:
        # Right triangle
        points = np.array([[0, 0], [4, 0], [0, 3]])
    else:
        # General triangle
        points = np.array([[0, 0], [4, 0], [1.5, 3]])

    triangle = Polygon(points, fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(triangle)

    # Add vertex labels
    offsets = [(-0.3, -0.3), (0.2, -0.3), (-0.1, 0.2)]
    for i, (point, label) in enumerate(zip(points, labels[:3])):
        ox, oy = offsets[i] if i < len(offsets) else (0, 0.2)
        ax.annotate(label, point, fontsize=12, fontweight='bold',
                   xytext=(point[0] + ox, point[1] + oy))

    # Add measurements if provided
    if measurements:
        mid_points = [
            ((points[0] + points[1]) / 2, measurements[0] if len(measurements) > 0 else None),
            ((points[1] + points[2]) / 2, measurements[1] if len(measurements) > 1 else None),
            ((points[2] + points[0]) / 2, measurements[2] if len(measurements) > 2 else None),
        ]
        for (mid, m) in mid_points:
            if m is not None:
                ax.annotate(f"{m}", mid, fontsize=10, ha='center',
                           xytext=(mid[0], mid[1] - 0.3))

    # Draw right angle marker if applicable
    if right_angle:
        square_size = 0.3
        square = plt.Rectangle((0, 0), square_size, square_size,
                               fill=False, edgecolor='black', linewidth=1)
        ax.add_patch(square)

    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 4)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def draw_rectangle(
    labels: Optional[List[str]] = None,
    measurements: Optional[List[float]] = None
) -> bytes:
    """Draw a rectangle and return PNG bytes."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.set_aspect('equal')
    ax.axis('off')

    labels = labels or ['A', 'B', 'C', 'D']
    width, height = 4, 2.5

    if measurements and len(measurements) >= 2:
        # Scale based on ratio
        ratio = measurements[0] / measurements[1] if measurements[1] != 0 else 1.6
        if ratio > 3:
            ratio = 3
        elif ratio < 0.33:
            ratio = 0.33
        height = width / ratio

    points = np.array([
        [0, 0], [width, 0], [width, height], [0, height]
    ])

    rect = Polygon(points, fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)

    # Add vertex labels
    offsets = [(-0.3, -0.3), (0.15, -0.3), (0.15, 0.15), (-0.3, 0.15)]
    for i, (point, label) in enumerate(zip(points, labels[:4])):
        ox, oy = offsets[i]
        ax.annotate(label, point, fontsize=12, fontweight='bold',
                   xytext=(point[0] + ox, point[1] + oy))

    # Add measurements
    if measurements:
        if len(measurements) > 0:
            ax.annotate(f"{measurements[0]}", (width/2, -0.4), fontsize=10, ha='center')
        if len(measurements) > 1:
            ax.annotate(f"{measurements[1]}", (width + 0.3, height/2), fontsize=10, va='center')

    # Draw right angle markers
    square_size = 0.2
    for corner in points:
        sq = plt.Rectangle(corner, square_size, square_size,
                          fill=False, edgecolor='black', linewidth=0.5)
        # Adjust position for each corner

    ax.set_xlim(-0.8, width + 0.8)
    ax.set_ylim(-0.8, height + 0.8)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def draw_square(
    labels: Optional[List[str]] = None,
    measurements: Optional[List[float]] = None
) -> bytes:
    """Draw a square and return PNG bytes."""
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    ax.set_aspect('equal')
    ax.axis('off')

    labels = labels or ['A', 'B', 'C', 'D']
    side = 3

    points = np.array([
        [0, 0], [side, 0], [side, side], [0, side]
    ])

    square = Polygon(points, fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(square)

    # Add vertex labels
    offsets = [(-0.3, -0.3), (0.15, -0.3), (0.15, 0.15), (-0.3, 0.15)]
    for i, (point, label) in enumerate(zip(points, labels[:4])):
        ox, oy = offsets[i]
        ax.annotate(label, point, fontsize=12, fontweight='bold',
                   xytext=(point[0] + ox, point[1] + oy))

    # Add measurement
    if measurements and len(measurements) > 0:
        ax.annotate(f"{measurements[0]}", (side/2, -0.4), fontsize=10, ha='center')

    ax.set_xlim(-0.8, side + 0.8)
    ax.set_ylim(-0.8, side + 0.8)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def draw_circle(
    labels: Optional[List[str]] = None,
    measurements: Optional[List[float]] = None
) -> bytes:
    """Draw a circle and return PNG bytes."""
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    ax.set_aspect('equal')
    ax.axis('off')

    center_label = labels[0] if labels else 'O'
    radius = 1.5

    circle = Circle((0, 0), radius, fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)

    # Draw center point
    ax.plot(0, 0, 'ko', markersize=4)
    ax.annotate(center_label, (0, 0), fontsize=12, fontweight='bold',
               xytext=(-0.25, -0.25))

    # Draw radius line
    ax.plot([0, radius], [0, 0], 'k-', linewidth=1)

    # Add radius measurement
    if measurements and len(measurements) > 0:
        ax.annotate(f"r={measurements[0]}", (radius/2, 0.15), fontsize=10, ha='center')
    else:
        ax.annotate("r", (radius/2, 0.15), fontsize=10, ha='center')

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def draw_parallelogram(
    labels: Optional[List[str]] = None,
    measurements: Optional[List[float]] = None
) -> bytes:
    """Draw a parallelogram and return PNG bytes."""
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    ax.set_aspect('equal')
    ax.axis('off')

    labels = labels or ['A', 'B', 'C', 'D']

    # Parallelogram vertices
    points = np.array([
        [0, 0], [4, 0], [5, 2.5], [1, 2.5]
    ])

    para = Polygon(points, fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(para)

    # Add vertex labels
    offsets = [(-0.3, -0.3), (0.15, -0.3), (0.15, 0.15), (-0.3, 0.15)]
    for i, (point, label) in enumerate(zip(points, labels[:4])):
        ox, oy = offsets[i]
        ax.annotate(label, point, fontsize=12, fontweight='bold',
                   xytext=(point[0] + ox, point[1] + oy))

    # Add measurements
    if measurements:
        if len(measurements) > 0:
            ax.annotate(f"{measurements[0]}", (2, -0.4), fontsize=10, ha='center')
        if len(measurements) > 1:
            ax.annotate(f"{measurements[1]}", (4.7, 1.25), fontsize=10, va='center')

    ax.set_xlim(-0.8, 6)
    ax.set_ylim(-0.8, 3.3)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def draw_trapezoid(
    labels: Optional[List[str]] = None,
    measurements: Optional[List[float]] = None
) -> bytes:
    """Draw a trapezoid and return PNG bytes."""
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    ax.set_aspect('equal')
    ax.axis('off')

    labels = labels or ['A', 'B', 'C', 'D']

    # Trapezoid vertices (parallel sides: bottom and top)
    points = np.array([
        [0, 0], [5, 0], [4, 2.5], [1, 2.5]
    ])

    trap = Polygon(points, fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(trap)

    # Add vertex labels
    offsets = [(-0.3, -0.3), (0.15, -0.3), (0.15, 0.15), (-0.3, 0.15)]
    for i, (point, label) in enumerate(zip(points, labels[:4])):
        ox, oy = offsets[i]
        ax.annotate(label, point, fontsize=12, fontweight='bold',
                   xytext=(point[0] + ox, point[1] + oy))

    # Add measurements
    if measurements:
        if len(measurements) > 0:
            ax.annotate(f"{measurements[0]}", (2.5, -0.4), fontsize=10, ha='center')
        if len(measurements) > 1:
            ax.annotate(f"{measurements[1]}", (2.5, 2.7), fontsize=10, ha='center')
        if len(measurements) > 2:
            ax.annotate(f"h={measurements[2]}", (5.3, 1.25), fontsize=10, va='center')

    # Draw height line (dashed)
    ax.plot([1, 1], [0, 2.5], 'k--', linewidth=0.8)

    ax.set_xlim(-0.8, 6)
    ax.set_ylim(-0.8, 3.3)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# Map shape names to drawing functions
SHAPE_DRAWERS = {
    "triangle": draw_triangle,
    "rectangle": draw_rectangle,
    "square": draw_square,
    "circle": draw_circle,
    "parallelogram": draw_parallelogram,
    "trapezoid": draw_trapezoid,
}


def draw_geometry_for_question(question: str) -> Optional[bytes]:
    """
    Analyze question text and draw appropriate geometry figure.
    Returns PNG bytes or None if no geometry detected.
    """
    geo = detect_geometry(question)
    if not geo:
        return None

    shape = geo["shape"]
    drawer = SHAPE_DRAWERS.get(shape)
    if not drawer:
        return None

    # Check for right angle triangle
    kwargs = {
        "labels": geo.get("labels"),
        "measurements": geo.get("measurements"),
    }

    if shape == "triangle":
        # Check if it's a right triangle
        if re.search(r"vuông|right|90°|90 độ", question, re.IGNORECASE):
            kwargs["right_angle"] = True

    return drawer(**kwargs)


def save_geometry_image(question: str, output_path: str) -> bool:
    """Save geometry image for a question to file."""
    img_bytes = draw_geometry_for_question(question)
    if img_bytes:
        with open(output_path, 'wb') as f:
            f.write(img_bytes)
        return True
    return False
