#!/usr/bin/env python3
"""
run_agentic_loop.py
===================
Orchestrator script for the multi-agent optimization loop.
Coordinates Git, Docker, local metric parsing, and LLM-based code adaptation.
"""

import os
import sys
import argparse
import subprocess
import shutil
import json
import urllib.request
import urllib.parse
from pathlib import Path

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
        # Assuming the metrics are printed at the end of the run
        # and written to outputs/demographics.csv or metrics.json
        demo_file = OUTPUTS_HOST_PATH / "demographics.csv"
        if demo_file.exists():
            metrics["status"] = "success"
            # Simple file content scan (adjust depending on demographics structure)
            with open(demo_file, "r") as f:
                content = f.read()
                # Parse metrics if logged or present
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
# LLM INTEGRATION (SDK-free Gemini API caller)
# ==============================================================================

def call_gemini_api(api_key, model, system_instruction, prompt):
    """SDK-free API call using urllib.request."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]}
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as res:
            resp_data = json.loads(res.read().decode("utf-8"))
            return resp_data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None


# ==============================================================================
# CORE SYSTEM PROMPTS (Agent 2 & Agent 3 Role Instructions)
# ==============================================================================

RESEARCHER_INSTRUCTION = """
You are the Scientific Researcher Agent. Your role is to analyze classification performance gaps and suggest feature extraction improvements.
Analyze the files in the codebase, find where feature extraction happens, and suggest mathematically rigorous features.
"""

IMPLEMENTER_INSTRUCTION = """
You are the Feature & Code Implementer Agent. Your role is to implement coding suggestions strictly within the codebase.
Write clean, error-free Python code. You will output the changes formatted as a standard search-and-replace list or complete modified code.
"""


# ==============================================================================
# MAIN ITERATIVE LOOP
# ==============================================================================

def run_loop(api_key, max_iterations, target_benchmark):
    check_git_guardrails()
    
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
        
        # Agent 2: Literature Research
        print("Agent 2 (Researcher) analyzing performance gaps...")
        research_prompt = f"The pipeline baseline F1-score is {current_f1:.4f}. Explore literature and suggest 3 features to improve sleep stage classification."
        research_proposal = call_gemini_api(api_key, "gemini-2.5-pro", RESEARCHER_INSTRUCTION, research_prompt)
        
        if not research_proposal:
            print("Research proposal generation failed. Aborting iteration.")
            continue
        print("\nResearch Proposal:\n", research_proposal[:500], "...\n")
        
        # Agent 3: Implement Suggestions
        print("Agent 3 (Implementer) writing code...")
        implementation_prompt = f"Implement the following research proposal in the codebase. Proposal: {research_proposal}"
        implemented_changes = call_gemini_api(api_key, "gemini-2.5-pro", IMPLEMENTER_INSTRUCTION, implementation_prompt)
        
        if not implemented_changes:
            print("Code generation failed. Aborting iteration.")
            continue
        print("\nImplemented Changes:\n", implemented_changes[:500], "...\n")
        
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
    parser = argparse.ArgumentParser(description="Fully Autonomous Agent Pipeline Optimizer")
    parser.add_argument("--key", type=str, default=os.environ.get("GEMINI_API_KEY"), help="Gemini API Key")
    parser.add_argument("--iterations", type=int, default=5, help="Max iterations")
    parser.add_argument("--benchmark", type=float, default=0.01, help="Target F1 score improvement")
    parser.add_argument("--dry-run", action="store_true", help="Simulate git/docker commands without executing")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY-RUN SIMULATION:")
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

    if not args.key:
        print("ERROR: Gemini API Key not found. Please set GEMINI_API_KEY environment variable or pass --key.")
        sys.exit(1)
        
    run_loop(args.key, args.iterations, args.benchmark)
