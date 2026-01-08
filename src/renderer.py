"""
Renderer - Load MJCF scenes, render images, perturb joints, and compute motion deltas.

Features:
- Load and render MJCF XML scenes
- Perturb joint positions (including free joints)
- Motion detection using optical flow and frame differencing
"""

import mujoco
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from PIL import Image


@dataclass
class RenderConfig:
    """Configuration for rendering."""
    width: int = 1280
    height: int = 960
    camera_distance: float = 2.5
    camera_azimuth: float = 135.0
    camera_elevation: float = -25.0
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.3)


class SceneRenderer:
    """Handles loading, rendering, and manipulating MJCF scenes."""
    
    def __init__(self, xml_path: Path | str, config: RenderConfig | None = None):
        self.xml_path = Path(xml_path)
        self.config = config or RenderConfig()
        
        # Load model
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        
        # Setup renderer
        self.renderer = mujoco.Renderer(
            self.model, 
            height=self.config.height, 
            width=self.config.width
        )
        
        # Setup camera
        self.camera = mujoco.MjvCamera()
        self.camera.distance = self.config.camera_distance
        self.camera.azimuth = self.config.camera_azimuth
        self.camera.elevation = self.config.camera_elevation
        self.camera.lookat[:] = self.config.camera_lookat
        
        # Forward simulation to settle
        mujoco.mj_forward(self.model, self.data)
    
    def get_joint_info(self) -> list[dict]:
        """Get information about all joints in the scene."""
        joints = []
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_type = self.model.jnt_type[i]
            
            # Get joint type name
            type_names = {
                mujoco.mjtJoint.mjJNT_FREE: "free",
                mujoco.mjtJoint.mjJNT_BALL: "ball",
                mujoco.mjtJoint.mjJNT_SLIDE: "slide",
                mujoco.mjtJoint.mjJNT_HINGE: "hinge",
            }
            type_name = type_names.get(joint_type, "unknown")
            
            # Get qpos indices for this joint
            qpos_adr = self.model.jnt_qposadr[i]
            
            # Number of qpos elements depends on joint type
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                nq = 7  # 3 pos + 4 quat
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                nq = 4  # quaternion
            else:
                nq = 1  # single value
            
            # Get joint limits
            limited = self.model.jnt_limited[i]
            if limited:
                range_low = self.model.jnt_range[i, 0]
                range_high = self.model.jnt_range[i, 1]
            else:
                range_low, range_high = None, None
            
            joints.append({
                "id": i,
                "name": joint_name,
                "type": type_name,
                "type_id": joint_type,
                "qpos_adr": qpos_adr,
                "nq": nq,
                "limited": bool(limited),
                "range": (range_low, range_high) if limited else None,
            })
        
        return joints
    
    def render(self) -> np.ndarray:
        """Render the current scene state and return RGB image."""
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()
    
    def save_qpos(self) -> np.ndarray:
        """Save current joint positions."""
        return self.data.qpos.copy()
    
    def restore_qpos(self, qpos: np.ndarray):
        """Restore joint positions."""
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
    
    def perturb_joints(
        self,
        position_scale: float = 0.05,
        rotation_scale: float = 0.1,
        joint_angle_scale: float = 0.15,
    ) -> dict[str, np.ndarray]:
        """
        Perturb all joints by small random amounts.
        
        Returns a dict mapping joint names to their perturbation values.
        """
        perturbations = {}
        joints = self.get_joint_info()
        
        for joint in joints:
            qpos_adr = joint["qpos_adr"]
            nq = joint["nq"]
            joint_type = joint["type_id"]
            
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                # Free joint: perturb position and rotation
                # Position (first 3 elements)
                pos_delta = np.random.randn(3) * position_scale
                self.data.qpos[qpos_adr:qpos_adr + 3] += pos_delta
                
                # Rotation (quaternion, elements 3-6)
                # Small rotation perturbation
                axis = np.random.randn(3)
                axis = axis / (np.linalg.norm(axis) + 1e-8)
                angle = np.random.randn() * rotation_scale
                
                # Convert axis-angle to quaternion delta
                half_angle = angle / 2
                dq = np.array([
                    np.cos(half_angle),
                    axis[0] * np.sin(half_angle),
                    axis[1] * np.sin(half_angle),
                    axis[2] * np.sin(half_angle),
                ])
                
                # Multiply quaternions
                q_orig = self.data.qpos[qpos_adr + 3:qpos_adr + 7].copy()
                q_new = self._quat_multiply(q_orig, dq)
                q_new = q_new / (np.linalg.norm(q_new) + 1e-8)  # Normalize
                self.data.qpos[qpos_adr + 3:qpos_adr + 7] = q_new
                
                perturbations[joint["name"]] = np.concatenate([pos_delta, [angle]])
                
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                # Ball joint: perturb quaternion
                axis = np.random.randn(3)
                axis = axis / (np.linalg.norm(axis) + 1e-8)
                angle = np.random.randn() * rotation_scale
                
                half_angle = angle / 2
                dq = np.array([
                    np.cos(half_angle),
                    axis[0] * np.sin(half_angle),
                    axis[1] * np.sin(half_angle),
                    axis[2] * np.sin(half_angle),
                ])
                
                q_orig = self.data.qpos[qpos_adr:qpos_adr + 4].copy()
                q_new = self._quat_multiply(q_orig, dq)
                q_new = q_new / (np.linalg.norm(q_new) + 1e-8)
                self.data.qpos[qpos_adr:qpos_adr + 4] = q_new
                
                perturbations[joint["name"]] = np.array([angle])
                
            else:
                # Hinge or slide joint: perturb single value
                delta = np.random.randn() * joint_angle_scale
                
                # Respect joint limits if they exist
                if joint["limited"] and joint["range"] is not None:
                    low, high = joint["range"]
                    new_val = np.clip(
                        self.data.qpos[qpos_adr] + delta,
                        low, high
                    )
                    delta = new_val - self.data.qpos[qpos_adr]
                
                self.data.qpos[qpos_adr] += delta
                perturbations[joint["name"]] = np.array([delta])
        
        # Update simulation state
        mujoco.mj_forward(self.model, self.data)
        
        return perturbations
    
    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions (w, x, y, z format)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
    
    def close(self):
        """Clean up renderer resources."""
        self.renderer.close()


def compute_motion_delta(
    image1: np.ndarray,
    image2: np.ndarray,
    method: str = "combined",
) -> np.ndarray:
    """
    Compute motion delta between two images.
    
    Methods:
    - "difference": Simple frame difference
    - "optical_flow": Dense optical flow visualization
    - "combined": Combination of both methods
    """
    # Convert to grayscale for processing
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    
    if method == "difference":
        return _compute_difference(gray1, gray2)
    elif method == "optical_flow":
        return _compute_optical_flow(gray1, gray2)
    else:  # combined
        diff = _compute_difference(gray1, gray2)
        flow = _compute_optical_flow(gray1, gray2)
        
        # Blend the two visualizations
        combined = cv2.addWeighted(diff, 0.5, flow, 0.5, 0)
        return combined


def _compute_difference(gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
    """Compute absolute frame difference with enhancement."""
    # Compute absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply Gaussian blur to reduce noise
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    
    # Enhance contrast
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply colormap for visualization
    colored = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)
    
    # Convert BGR to RGB
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    return colored


def _compute_optical_flow(gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
    """Compute dense optical flow and visualize it."""
    # Compute Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    
    # Compute magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV image
    hsv = np.zeros((*gray1.shape, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue = direction
    hsv[..., 1] = 255  # Saturation = max
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude
    
    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb


def save_image(image: np.ndarray, path: Path | str, verbose: bool = False):
    """Save an image to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    img = Image.fromarray(image)
    img.save(path)
    if verbose:
        print(f"Image saved to: {path}")


def render_and_perturb_to_arrays(
    xml_path: Path | str,
    perturbation_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load scene, render, perturb, render again, compute motion.
    
    Returns tuple of (original_image, perturbed_image, motion_delta) as numpy arrays.
    Does not save to disk.
    """
    xml_path = Path(xml_path)
    
    # Create renderer
    renderer = SceneRenderer(xml_path)
    
    try:
        # Render original state
        original_image = renderer.render()
        
        # Perturb joints
        renderer.perturb_joints(
            position_scale=0.03 * perturbation_scale,
            rotation_scale=0.08 * perturbation_scale,
            joint_angle_scale=0.12 * perturbation_scale,
        )
        
        # Render perturbed state
        perturbed_image = renderer.render()
        
        # Compute motion delta
        motion_delta = compute_motion_delta(original_image, perturbed_image)
        
        return original_image, perturbed_image, motion_delta
        
    finally:
        renderer.close()


def render_and_perturb(
    xml_path: Path | str,
    output_dir: Path | str,
    perturbation_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Main pipeline: load scene, render, perturb, render again, compute motion.
    
    Returns tuple of (original_image, perturbed_image, motion_delta).
    """
    xml_path = Path(xml_path)
    output_dir = Path(output_dir)
    
    # Create renderer
    renderer = SceneRenderer(xml_path)
    
    try:
        # Render original state
        original_image = renderer.render()
        
        # Save original qpos
        original_qpos = renderer.save_qpos()
        
        # Perturb joints
        perturbations = renderer.perturb_joints(
            position_scale=0.03 * perturbation_scale,
            rotation_scale=0.08 * perturbation_scale,
            joint_angle_scale=0.12 * perturbation_scale,
        )
        
        print("Applied perturbations:")
        for name, delta in perturbations.items():
            print(f"  {name}: {delta}")
        
        # Render perturbed state
        perturbed_image = renderer.render()
        
        # Compute motion delta
        motion_delta = compute_motion_delta(original_image, perturbed_image)
        
        # Save images
        save_image(original_image, output_dir / "original" / "render.png")
        save_image(perturbed_image, output_dir / "perturbed" / "render.png")
        save_image(motion_delta, output_dir / "motion_delta" / "delta.png")
        
        return original_image, perturbed_image, motion_delta
        
    finally:
        renderer.close()


if __name__ == "__main__":
    # Test rendering with an existing scene
    test_scene = Path("data/test_scene/scene.xml")
    
    if test_scene.exists():
        render_and_perturb(test_scene, Path("data/test_scene"))
    else:
        print(f"Scene file not found: {test_scene}")
        print("Run scene_generator.py first to create a test scene.")

