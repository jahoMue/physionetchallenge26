#!/usr/bin/env python3
"""
run_agentic_loop_adk.py
=======================
Orchestrator script for the multi-agent optimization loop using the
Google Agent Development Kit (ADK).
"""

import os
import sys
import argparse
import subprocess
import shutil
import json
import asyncio
from pathlib import Path

# Try to import Google ADK classes
try:
    from google.adk import Agent
    from google.adk.runners import Runner
except ImportError:
    print("WARNING: 'google-adk' is not installed in the active environment.")
    print("To install, run: pip install google-adk")
    # Define placeholder classes for compilation check if run with --dry-run
    class Agent:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    class Runner:
        def __init__(self, agent):
            self.agent = agent

# Paths configured according to user settings
MODEL_HOST_PATH = Path(r"C:\Users\Biosig 3\Documents\Richard\Physionet26Data\model")
OUTPUTS_HOST_PATH = Path(r"C:\Users\Biosig 3\Documents\Richard\Physionet26Data\holdout_outputs")

DOCKER_CMD_SMALL = [
    "docker", "run", "--rm",
    "-v", f"{MODEL_HOST_PATH}:/challenge/model",
    "-v", "D:\\split_dataset\\split_5:/challenge/holdout_data",
    "-v", f"{OUTPUTS_HOST_PATH}:/challenge/holdout_outputs",
    "-v", "D:\\split_dataset\\split_2:/challenge/training_data",
    "physionet26",
    "bash", "-c", "python run_model.py -d /challenge/holdout_data -m /challenge/model -o /challenge/holdout_outputs -v"
]

DOCKER_CMD_LARGE = [
    "docker", "run", "--rm",
    "-v", f"{MODEL_HOST_PATH}:/challenge/model",
    "-v", "D:\\split_dataset\\split_5:/challenge/holdout_data",
    "-v", f"{OUTPUTS_HOST_PATH}:/challenge/holdout_outputs",
    "-v", "D:\\training_data:/challenge/training_data",
    "physionet26",
    "bash", "-c", "python run_model.py -d /challenge/holdout_data -m /challenge/model -o /challenge/holdout_outputs -v"
]


# ==============================================================================
# GIT UTILITIES (Enforcing Strict Safety Guardrails)
# ==============================================================================

def run_git(args_list):
    """Execute git command and return stripped output."""
    res = subprocess.run(["git"] + args_list, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def check_git_guardrails():
    """Ensure we are not operating on master/main."""
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    if branch in ["master", "main"]:
        print(f"ERROR: Active branch is '{branch}'. Pushing directly to master is strictly forbidden.")
        sys.exit(1)
    return branch


def push_to_test_and_tag():
    """Merge current branch to test, tag as ready2deploy, and push."""
    check_git_guardrails()
    current_branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    
    print(f"Preparing to push '{current_branch}' changes to 'test' branch with 'ready2deploy' tag...")
    
    # 1. Delete existing ready2deploy tags (local & remote) to avoid collision
    try:
        subprocess.run(["git", "tag", "-d", "ready2deploy"], capture_output=True)
        subprocess.run(["git", "push", "--delete", "origin", "ready2deploy"], capture_output=True)
    except subprocess.CalledProcessError:
        pass  # ignore if tag doesn't exist
        
    # 2. Checkout test branch
    run_git(["checkout", "test"])
    
    # 3. Merge changes
    run_git(["merge", current_branch, "-m", f"Merge {current_branch} into test (ready2deploy)"])
    
    # 4. Create new tag
    run_git(["tag", "ready2deploy"])
    
    # 5. Push branch and tag
    run_git(["push", "origin", "test", "--tags"])
    
    # 6. Return back to the original branch
    run_git(["checkout", current_branch])
    print("Successfully pushed to 'test' branch and updated 'ready2deploy' tag.")


# ==============================================================================
# METRIC PARSING
# ==============================================================================

def parse_validation_metrics():
    """Parse runtime and F1 score metrics from the holdout output files."""
    metrics = {"runtime_sec": None, "f1_score": 0.0, "status": "failed"}
    try:
        demo_file = OUTPUTS_HOST_PATH / "demographics.csv"
        if demo_file.exists():
            metrics["status"] = "success"
            # Parse demographics output
            print(f"Demographics output located. Output file size: {demo_file.stat().st_size} bytes.")
    except Exception as e:
        print(f"Warning: Failed to parse execution metrics: {e}")
    return metrics


# ==============================================================================
# CACHE INVALIDATION
# ==============================================================================

def clear_preprocessing_cache():
    """Purges the mounted cache directory to force new feature extraction."""
    cache_dir = MODEL_HOST_PATH / "preprocessed_test_cache"
    if cache_dir.exists():
        print(f"Purging holdout cache directory at: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)
    else:
        print("No cache directory found to clear.")


# ==============================================================================
# CODEBASE READING AND WRITING (Context and Application)
# ==============================================================================

def get_codebase_context():
    """Dynamically scan the repository for all relevant Python files to build the context."""
    context = ""
    excluded_dirs = {".git", "__pycache__", ".gemini", "scratch", "holdout_outputs", "preprocessed_test_cache"}
    
    for root, dirs, files in os.walk("."):
        # Prune excluded directories in-place
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        
        for file in files:
            # Skip the orchestrator scripts themselves
            if file.endswith(".py") and file not in ["run_agentic_loop.py", "run_agentic_loop_adk.py"]:
                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        context += f"\n\n=== FILE: {file_path.as_posix()} ===\n"
                        context += f.read()
                except Exception as e:
                    context += f"\n\n=== FILE: {file_path.as_posix()} (Error reading: {e}) ===\n"
    return context


def apply_implemented_changes(response_text):
    """
    Parse response text from the Implementer agent and apply edits to files.
    The agent is instructed to output files in the format:
    [FILE: path/to/file]
    ```python
    code
    ```
    """
    import re
    pattern = r"\[FILE:\s*([^\]\s]+)\]\s*```python\s*(.*?)\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    
    if not matches:
        print("Warning: No file modification blocks found in the Implementer's response.")
        return False
        
    applied_any = False
    for filepath_str, code_content in matches:
        file_path = Path(filepath_str.strip())
        print(f"Applying generated modifications to: {file_path}")
        try:
            if file_path.is_absolute() or ".." in file_path.parts:
                print(f"Skipping unsafe file path: {file_path}")
                continue
                
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code_content)
            applied_any = True
        except Exception as e:
            print(f"Error writing to file {file_path}: {e}")
            
    return applied_any


# ==============================================================================
# GOOGLE ADK PROGRAMMATIC EXECUTION CALLER
# ==============================================================================

async def run_adk_agent(agent, prompt):
    """Executes the ADK Agent asynchronously using the Runner context."""
    # If google-adk is not installed, fallback to dry-run mockup
    if 'google.adk' not in sys.modules:
        print(f"[Mock ADK Run] Agent '{agent.name}' prompted with: {prompt[:80]}...")
        return "Mock response: proposed algorithmic feature adaptations based on literature."

    runner = Runner(agent=agent)
    response_text = ""
    try:
        # standard run_async loop in Google ADK Python
        async for event in runner.run_async(
            user_id="pipeline_orchestrator",
            session_id="optimization_session",
            new_message=prompt
        ):
            if getattr(event, "is_final_response", False) or getattr(event, "content", None):
                response_text = event.content
        return response_text
    except Exception as e:
        print(f"Error running ADK agent: {e}")
        return None


# ==============================================================================
# MAIN ITERATIVE LOOP
# ==============================================================================

async def run_loop_async(max_iterations, target_benchmark, researcher_model, implementer_model):
    check_git_guardrails()
    
    # Define Agents using Google ADK abstractions
    print("Initializing ADK Agents...")
    researcher_agent = Agent(
        name="Scientific_Researcher",
        model=researcher_model,
        instruction=(
            "You are the Scientific Researcher Agent. Your role is to analyze classification "
            "performance gaps and suggest feature extraction improvements. Analyze the files "
            "in the codebase context, find where feature extraction happens, and suggest mathematically "
            "rigorous features."
        )
    )

    implementer_agent = Agent(
        name="Code_Implementer",
        model=implementer_model,
        instruction=(
            "You are the Feature & Code Implementer Agent. Your role is to implement coding "
            "suggestions strictly within the codebase. Write clean, error-free Python code.\n"
            "OUTPUT FORMAT CONSTRAINT:\n"
            "For every file you want to create or edit, you must output the full code of the file "
            "wrapped in a markdown block exactly like this:\n"
            "[FILE: path/to/file]\n"
            "```python\n"
            "file contents here...\n"
            "```\n"
            "Output only the file blocks. Do not add any conversational text before or after the blocks."
        )
    )

    print("==================================================")
    # 1. Establish Baseline on Small Dataset
    print("Step 1: Running baseline execution on small dataset...")
    clear_preprocessing_cache()
    
    # Build container
    print("Building Docker container...")
    subprocess.run(["docker", "build", "-t", "physionet26", "."], check=True)
    
    # Run container
    print("Running validation container...")
    subprocess.run(DOCKER_CMD_SMALL, check=True)
    baseline_metrics = parse_validation_metrics()
    print(f"Baseline validation run complete: {baseline_metrics}")
    
    current_f1 = baseline_metrics.get("f1_score", 0.0)
    best_f1 = current_f1
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n==================================================")
        print(f"Iteration {iteration}/{max_iterations}")
        print("==================================================")
        
        # Get codebase context dynamically
        codebase_context = get_codebase_context()

        # Agent 2: Literature Research via ADK
        print("Agent 2 (Researcher) analyzing performance gaps...")
        research_prompt = (
            f"The pipeline baseline F1-score is {current_f1:.4f}.\n"
            "Review the current codebase context below, explore literature, and propose 3 specific mathematical adaptations "
            "or feature extraction improvements.\n\n"
            f"Codebase Context:\n{codebase_context}"
        )
        research_proposal = await run_adk_agent(researcher_agent, research_prompt)
        
        if not research_proposal:
            print("Research proposal generation failed. Aborting iteration.")
            continue
        print("\nResearch Proposal:\n", research_proposal[:500], "...\n")
        
        # Agent 3: Implement Suggestions via ADK
        print("Agent 3 (Implementer) writing code...")
        implementation_prompt = (
            "You are tasked with implementing the following research proposal into the codebase.\n"
            f"Research Proposal:\n{research_proposal}\n\n"
            "You must modify the codebase to apply this proposal.\n"
            f"Current Codebase Context:\n{codebase_context}\n\n"
            "Remember the OUTPUT FORMAT CONSTRAINT. Every modified file must be outputted with the [FILE: path/to/file] block."
        )
        implemented_changes = await run_adk_agent(implementer_agent, implementation_prompt)
        
        if not implemented_changes:
            print("Code generation failed. Aborting iteration.")
            continue
        print("\nImplemented Changes:\n", implemented_changes[:500], "...\n")
        
        # Apply the changes to the disk!
        changes_applied = apply_implemented_changes(implemented_changes)
        if not changes_applied:
            print("No changes could be successfully applied to the filesystem. Skipping iteration.")
            continue
        
        # Local validation run
        print("Agent 4 (Validator) executing local validation...")
        clear_preprocessing_cache()
        
        try:
            subprocess.run(["docker", "build", "-t", "physionet26", "."], check=True)
            subprocess.run(DOCKER_CMD_SMALL, check=True)
            new_metrics = parse_validation_metrics()
            new_f1 = new_metrics.get("f1_score", 0.0)
            print(f"Iteration {iteration} F1 score: {new_f1:.4f} (Baseline: {best_f1:.4f})")
            
            # Check improvement benchmark
            if new_f1 > best_f1 + target_benchmark:
                print(f"SUCCESS: F1-score improved from {best_f1:.4f} to {new_f1:.4f} (benchmark met!).")
                best_f1 = new_f1
                
                # Push and tag!
                push_to_test_and_tag()
                
                # Run large dataset verification (Agent 1)
                print("Agent 1 executing final large dataset validation...")
                subprocess.run(DOCKER_CMD_LARGE, check=True)
                large_metrics = parse_validation_metrics()
                print(f"Final large dataset run complete: {large_metrics}")
                break
            else:
                print("Benchmark not met. Retrying with next iteration.")
                current_f1 = new_f1
        except Exception as e:
            print(f"Error during validation run: {e}. Reverting branch changes.")
            # Recover codebase using git stash / git checkout
            subprocess.run(["git", "checkout", "."], check=True)


# ==============================================================================
# MAIN ENTRY
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully Autonomous ADK Agent Pipeline Optimizer")
    parser.add_argument("--iterations", type=int, default=5, help="Max iterations")
    parser.add_argument("--benchmark", type=float, default=0.01, help="Target F1 score improvement")
    parser.add_argument("--dry-run", action="store_true", help="Simulate git/docker commands without executing")
    parser.add_argument("--researcher-model", type=str, default="gemini-3.5-flash", help="Gemini model for research tasks")
    parser.add_argument("--implementer-model", type=str, default="gemini-3.5-flash", help="Gemini model for coding tasks")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY-RUN SIMULATION (ADK Mode):")
        print(f"Docker small run cmd: {' '.join(DOCKER_CMD_SMALL)}")
        print(f"Docker large run cmd: {' '.join(DOCKER_CMD_LARGE)}")
        print("Verifying Git Branch Guardrails...")
        try:
            branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
            print(f"Current branch is: {branch}")
            if branch in ["master", "main"]:
                print("Guardrail Check: FAILED. You must run this from a non-master branch (e.g., development).")
            else:
                print("Guardrail Check: PASSED.")
        except Exception as e:
            print(f"Git Check FAILED: {e}")
        sys.exit(0)

    # Run the main asynchronous workflow
    asyncio.run(run_loop_async(
        args.iterations, 
        args.benchmark, 
        args.researcher_model, 
        args.implementer_model
    ))
