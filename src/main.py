"""
Main entry point for motion learning data generation.

This script:
1. Creates a date-stamped run folder
2. Generates multiple random articulated scenes (MJCF XML)
3. For each scene: renders original, perturbs joints, renders perturbed, computes motion delta
4. Saves all outputs in organized folders with sequential scene IDs
"""

import argparse
from datetime import datetime
from pathlib import Path
import json
import random
import numpy as np

from scene_generator import (
    generate_random_scene, 
    scene_to_mjcf, 
    save_scene_encoding,
    EnvironmentConfig,
)
from renderer import render_and_perturb_to_arrays, save_image


def generate_run_id() -> str:
    """Generate a date-stamped run identifier."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_generation_pipeline(
    output_base: Path,
    num_scenes: int = 50,
    min_objects: int = 1,
    max_objects: int = 5,
    min_bodies: int = 1,
    max_bodies: int = 5,
    perturbation_scale: float = 1.0,
    seed: int | None = None,
) -> dict:
    """
    Run the complete generation pipeline for multiple scenes.
    
    Returns metadata about the generated run.
    """
    # Set seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate unique run ID (date-stamped)
    run_id = generate_run_id()
    run_dir = output_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'=' * 60}")
    print(f"Starting generation run: {run_id}")
    print(f"Generating {num_scenes} scenes")
    print(f"Objects per scene: {min_objects}-{max_objects}")
    print(f"Bodies per object: {min_bodies}-{max_bodies}")
    print(f"{'=' * 60}")
    
    all_scene_metadata = []
    
    for scene_idx in range(num_scenes):
        scene_id = f"{scene_idx:04d}"
        scene_dir = run_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[Scene {scene_id}] Generating...")
        
        # Randomize number of objects (flat distribution)
        num_objects = random.randint(min_objects, max_objects)
        
        # Generate scene - each object gets its own random body count
        objects, environment = generate_random_scene(
            num_objects=num_objects,
            min_bodies_per_object=min_bodies,
            max_bodies_per_object=max_bodies,
        )
        
        xml_content = scene_to_mjcf(objects, scene_name=f"scene_{scene_id}", environment=environment)
        xml_path = scene_dir / f"{scene_id}_scene.xml"
        
        with open(xml_path, "w") as f:
            f.write(xml_content)
        
        # Save compact scene encoding
        encoding_path = scene_dir / f"{scene_id}_scene.txt"
        save_scene_encoding(objects, encoding_path, environment)
        
        body_counts = [len(obj.bodies) for obj in objects]
        print(f"  Objects: {num_objects}, Bodies per object: {body_counts}")
        
        # Render and perturb
        try:
            original_img, perturbed_img, motion_img = render_and_perturb_to_arrays(
                xml_path,
                perturbation_scale=perturbation_scale,
            )
            
            # Save images with scene ID in filename
            save_image(original_img, scene_dir / f"{scene_id}_original.png")
            save_image(perturbed_img, scene_dir / f"{scene_id}_perturbed.png")
            save_image(motion_img, scene_dir / f"{scene_id}_motion_delta.png")
            
            scene_metadata = {
                "scene_id": scene_id,
                "num_objects": num_objects,
                "objects": [
                    {
                        "name": obj.name,
                        "base_pos": obj.base_pos,
                        "fixed_base": obj.fixed_base,
                        "num_bodies": len(obj.bodies),
                        "bodies": [
                            {
                                "name": b.name,
                                "shape": b.shape,
                                "joint_type": b.joint_type,
                            }
                            for b in obj.bodies
                        ],
                    }
                    for obj in objects
                ],
                "files": {
                    "xml": f"{scene_id}_scene.xml",
                    "encoding": f"{scene_id}_scene.txt",
                    "original": f"{scene_id}_original.png",
                    "perturbed": f"{scene_id}_perturbed.png",
                    "motion_delta": f"{scene_id}_motion_delta.png",
                },
            }
            
            print(f"  ✓ Complete")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            scene_metadata = {
                "scene_id": scene_id,
                "num_objects": num_objects,
                "skipped": True,
                "reason": str(e),
            }
        
        all_scene_metadata.append(scene_metadata)
    
    # Save run metadata
    run_metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "parameters": {
            "num_scenes": num_scenes,
            "min_objects": min_objects,
            "max_objects": max_objects,
            "min_bodies": min_bodies,
            "max_bodies": max_bodies,
            "perturbation_scale": perturbation_scale,
        },
        "scenes": all_scene_metadata,
        "stats": {
            "total_scenes": num_scenes,
            "successful": sum(1 for s in all_scene_metadata if not s.get("skipped", False)),
            "skipped": sum(1 for s in all_scene_metadata if s.get("skipped", False)),
        },
    }
    
    metadata_path = run_dir / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Generation complete!")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {run_dir}")
    print(f"Successful scenes: {run_metadata['stats']['successful']}/{num_scenes}")
    print(f"{'=' * 60}")
    
    return run_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate articulated scenes and motion delta images"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("data"),
        help="Base output directory for all runs (default: data)",
    )
    
    parser.add_argument(
        "-n", "--num-scenes",
        type=int,
        default=50,
        help="Number of scenes to generate (default: 50)",
    )
    
    parser.add_argument(
        "--min-objects",
        type=int,
        default=1,
        help="Minimum number of objects per scene (default: 1)",
    )
    
    parser.add_argument(
        "--max-objects",
        type=int,
        default=5,
        help="Maximum number of objects per scene (default: 5)",
    )
    
    parser.add_argument(
        "--min-bodies",
        type=int,
        default=1,
        help="Minimum bodies per object (default: 1)",
    )
    
    parser.add_argument(
        "--max-bodies",
        type=int,
        default=5,
        help="Maximum bodies per object (default: 5)",
    )
    
    parser.add_argument(
        "-p", "--perturbation-scale",
        type=float,
        default=1.0,
        help="Scale factor for joint perturbations (default: 1.0)",
    )
    
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    
    args = parser.parse_args()
    
    run_generation_pipeline(
        output_base=args.output,
        num_scenes=args.num_scenes,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        min_bodies=args.min_bodies,
        max_bodies=args.max_bodies,
        perturbation_scale=args.perturbation_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
