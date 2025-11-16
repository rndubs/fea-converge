"""
Smith Model Runner

Common utility for building and running Smith contact models from the
fea-converge/smith-models directory. This module provides a unified interface
for all optimization methods (config, fr_bo, gp-classification, shebo) to
interact with Smith models.

Usage:
    from smith_runner import SmithModelRunner

    runner = SmithModelRunner()
    result = runner.run_model("die-on-slab")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
"""

import subprocess
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json


class BuildMode(Enum):
    """Build mode for Smith models."""
    AUTO = "auto"
    DOCKER = "docker"
    LOCAL = "local"


@dataclass
class SmithModelResult:
    """
    Results from running a Smith model.

    Attributes:
        model_name: Name of the model that was run
        converged: Whether the simulation converged
        success: Whether the model ran without errors (returncode == 0)
        iterations: Total number of nonlinear solver iterations
        timesteps_completed: Number of timesteps completed
        final_residual: Final residual norm (if available)
        solve_time: Total solve time in seconds (if available)
        output_files: List of generated output files (ParaView, etc.)
        stdout: Standard output from the model
        stderr: Standard error from the model
        returncode: Process return code
        error_message: Error message if failed
    """
    model_name: str
    converged: bool
    success: bool
    iterations: int
    timesteps_completed: int
    final_residual: Optional[float] = None
    solve_time: Optional[float] = None
    output_files: List[Path] = None
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'converged': self.converged,
            'success': self.success,
            'iterations': self.iterations,
            'timesteps_completed': self.timesteps_completed,
            'final_residual': self.final_residual,
            'solve_time': self.solve_time,
            'output_files': [str(f) for f in self.output_files],
            'returncode': self.returncode,
            'error_message': self.error_message
        }


class SmithModelRunner:
    """
    Runner for Smith contact models.

    This class provides a Python interface to the ./run_model bash script
    and parses the output to extract convergence metrics.
    """

    # All available Smith models
    AVAILABLE_MODELS = [
        "block-on-slab",
        "concentric-spheres",
        "deep-indentation",
        "die-on-slab",
        "hemisphere-twisting",
        "hollow-sphere-pinching",
        "sphere-in-sphere",
        "stacked-blocks",
    ]

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        build_mode: BuildMode = BuildMode.AUTO,
        verbose: bool = False
    ):
        """
        Initialize Smith model runner.

        Args:
            repo_root: Path to fea-converge repository root
                      (auto-detected if not provided)
            build_mode: Whether to use Docker, local build, or auto-detect
            verbose: Print detailed output
        """
        if repo_root is None:
            # Auto-detect repo root (search upward from this file)
            current = Path(__file__).resolve().parent
            while current != current.parent:
                if (current / "run_model").exists():
                    repo_root = current
                    break
                current = current.parent
            else:
                raise RuntimeError(
                    "Could not find fea-converge repository root. "
                    "Please specify repo_root explicitly."
                )

        self.repo_root = Path(repo_root)
        self.build_mode = build_mode
        self.verbose = verbose

        # Validate repository structure
        self.run_model_script = self.repo_root / "run_model"
        self.models_dir = self.repo_root / "smith-models"

        if not self.run_model_script.exists():
            raise FileNotFoundError(
                f"run_model script not found at {self.run_model_script}"
            )

        if not self.models_dir.exists():
            raise FileNotFoundError(
                f"smith-models directory not found at {self.models_dir}"
            )

    def list_models(self) -> List[str]:
        """
        List all available Smith models.

        Returns:
            List of model names
        """
        return self.AVAILABLE_MODELS.copy()

    def model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists.

        Args:
            model_name: Name of the model

        Returns:
            True if model exists
        """
        return (self.models_dir / model_name).exists()

    def build_model(
        self,
        model_name: str,
        clean: bool = False,
        timeout: int = 600
    ) -> bool:
        """
        Build a Smith model without running it.

        Args:
            model_name: Name of the model to build
            clean: Clean build directory before building
            timeout: Build timeout in seconds

        Returns:
            True if build succeeded

        Raises:
            ValueError: If model doesn't exist
        """
        if not self.model_exists(model_name):
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {', '.join(self.AVAILABLE_MODELS)}"
            )

        # Build command
        cmd = [str(self.run_model_script)]

        # Add mode flag
        if self.build_mode == BuildMode.DOCKER:
            cmd.append("--docker")
        elif self.build_mode == BuildMode.LOCAL:
            cmd.append("--local")

        # Add clean flag
        if clean:
            cmd.append("--clean")

        # Add model name
        cmd.append(model_name)

        if self.verbose:
            print(f"Building {model_name}...")
            print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if self.verbose:
                print(result.stdout)
                if result.stderr:
                    print("Stderr:", result.stderr)

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"Build timed out after {timeout}s")
            return False
        except Exception as e:
            if self.verbose:
                print(f"Build error: {e}")
            return False

    def run_model(
        self,
        model_name: str,
        clean: bool = False,
        timeout: int = 600
    ) -> SmithModelResult:
        """
        Run a Smith model and parse the results.

        Args:
            model_name: Name of the model to run
            clean: Clean build directory before running
            timeout: Execution timeout in seconds

        Returns:
            SmithModelResult with parsed convergence metrics

        Raises:
            ValueError: If model doesn't exist
        """
        if not self.model_exists(model_name):
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {', '.join(self.AVAILABLE_MODELS)}"
            )

        # Build command
        cmd = [str(self.run_model_script)]

        # Add mode flag
        if self.build_mode == BuildMode.DOCKER:
            cmd.append("--docker")
        elif self.build_mode == BuildMode.LOCAL:
            cmd.append("--local")

        # Add clean flag
        if clean:
            cmd.append("--clean")

        # Add model name
        cmd.append(model_name)

        if self.verbose:
            print(f"Running {model_name}...")
            print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Parse output
            parsed_result = self._parse_output(
                model_name=model_name,
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode
            )

            if self.verbose:
                print(f"Result: {parsed_result.to_dict()}")

            return parsed_result

        except subprocess.TimeoutExpired as e:
            return SmithModelResult(
                model_name=model_name,
                converged=False,
                success=False,
                iterations=0,
                timesteps_completed=0,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
                returncode=-1,
                error_message=f"Timeout after {timeout}s"
            )
        except Exception as e:
            return SmithModelResult(
                model_name=model_name,
                converged=False,
                success=False,
                iterations=0,
                timesteps_completed=0,
                error_message=str(e),
                returncode=-1
            )

    def run_all_models(
        self,
        clean: bool = False,
        timeout: int = 600
    ) -> Dict[str, SmithModelResult]:
        """
        Run all available Smith models.

        Args:
            clean: Clean build directories before running
            timeout: Execution timeout per model in seconds

        Returns:
            Dictionary mapping model name to result
        """
        results = {}

        for model_name in self.AVAILABLE_MODELS:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Running model: {model_name}")
                print(f"{'='*60}")

            results[model_name] = self.run_model(
                model_name=model_name,
                clean=clean,
                timeout=timeout
            )

        return results

    def _parse_output(
        self,
        model_name: str,
        stdout: str,
        stderr: str,
        returncode: int
    ) -> SmithModelResult:
        """
        Parse Smith model output to extract metrics.

        Args:
            model_name: Name of the model
            stdout: Standard output
            stderr: Standard error
            returncode: Process return code

        Returns:
            SmithModelResult with parsed metrics
        """
        # Initialize result
        result = SmithModelResult(
            model_name=model_name,
            converged=False,
            success=(returncode == 0),
            iterations=0,
            timesteps_completed=0,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode
        )

        # Parse convergence status
        # Look for patterns like "converged", "CONVERGED", "Solution converged"
        if re.search(
            r'(converged|CONVERGED|Solution converged|Successfully completed)',
            stdout,
            re.IGNORECASE
        ):
            result.converged = True

        # Look for failure indicators
        if re.search(
            r'(failed|FAILED|diverged|DIVERGED|ERROR|Exception)',
            stdout + stderr,
            re.IGNORECASE
        ):
            result.converged = False
            result.success = False

        # Parse iteration count
        # Look for patterns like "iteration 15", "Newton iteration: 42", etc.
        iter_matches = re.findall(
            r'(?:iteration|iter|Newton iteration)[:\s]+(\d+)',
            stdout,
            re.IGNORECASE
        )
        if iter_matches:
            # Take the maximum iteration number found
            result.iterations = max(int(m) for m in iter_matches)

        # Parse timestep count
        # Look for patterns like "step 35", "timestep 20", "time step: 15"
        step_matches = re.findall(
            r'(?:step|timestep|time step)[:\s]+(\d+)',
            stdout,
            re.IGNORECASE
        )
        if step_matches:
            result.timesteps_completed = max(int(m) for m in step_matches)

        # Parse final residual
        # Look for patterns like "residual: 1.23e-08", "final residual = 5.6e-10"
        residual_matches = re.findall(
            r'(?:final\s+)?residual[:\s=]+([0-9.eE+-]+)',
            stdout,
            re.IGNORECASE
        )
        if residual_matches:
            # Take the last residual found (likely the final one)
            result.final_residual = float(residual_matches[-1])

        # Parse solve time
        # Look for patterns like "solve time: 12.34", "total time = 56.78 s"
        time_matches = re.findall(
            r'(?:solve\s+time|total\s+time|elapsed\s+time)[:\s=]+([0-9.]+)',
            stdout,
            re.IGNORECASE
        )
        if time_matches:
            result.solve_time = float(time_matches[-1])

        # Find output files
        build_dir = self.repo_root / f"build_{model_name}"
        if build_dir.exists():
            # Look for ParaView files
            pvd_files = list(build_dir.glob("*.pvd"))
            vtu_files = list(build_dir.glob("*.vtu"))
            result.output_files = pvd_files + vtu_files

        # Check for specific error messages
        if returncode != 0 and not result.error_message:
            if "No such file or directory" in stderr:
                result.error_message = "Build or executable not found"
            elif "Smith not found" in stdout + stderr:
                result.error_message = "Smith not built or installed"
            else:
                result.error_message = f"Process exited with code {returncode}"

        return result


def main():
    """Example usage of SmithModelRunner."""
    print("Smith Model Runner - Example Usage")
    print("=" * 60)

    # Initialize runner
    runner = SmithModelRunner(verbose=True)

    # List available models
    print("\nAvailable models:")
    for model in runner.list_models():
        exists = "✓" if runner.model_exists(model) else "✗"
        print(f"  {exists} {model}")

    # Run a single model
    print("\n" + "=" * 60)
    print("Running die-on-slab model...")
    print("=" * 60)

    result = runner.run_model("die-on-slab", clean=False)

    print("\nResults:")
    print(f"  Model: {result.model_name}")
    print(f"  Success: {result.success}")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Timesteps: {result.timesteps_completed}")
    if result.final_residual:
        print(f"  Final Residual: {result.final_residual:.2e}")
    if result.solve_time:
        print(f"  Solve Time: {result.solve_time:.2f} s")
    if result.output_files:
        print(f"  Output Files: {len(result.output_files)}")
        for f in result.output_files[:3]:  # Show first 3
            print(f"    - {f.name}")
    if result.error_message:
        print(f"  Error: {result.error_message}")

    # Save results to JSON
    output_file = Path("smith_results.json")
    with open(output_file, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
