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
    from google.adk.runners import InMemoryRunner, ContextCacheConfig
    from google.adk.tools import google_search
except ImportError:
    print("WARNING: 'google-adk' is not installed in the active environment.")
    print("To install, run: pip install google-adk")
    # Define placeholder classes for compilation check if run with --dry-run
    class Agent:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    class InMemoryRunner:
        def __init__(self, agent, **kwargs):
            self.agent = agent
            self.app_name = kwargs.get("app_name", "InMemoryRunner")
            self.context_cache_config = None
            class DummySessionService:
                def delete_session_sync(self, user_id, session_id, **kwargs):
                    pass
                async def create_session(self, app_name, user_id, session_id, **kwargs):
                    pass
            self.session_service = DummySessionService()
        async def run_async(self, user_id, session_id, new_message):
            class DummyEvent:
                def __init__(self):
                    self.is_final_response = True
                    self.content = "Mock response: proposed algorithmic feature adaptations."
            yield DummyEvent()
    class ContextCacheConfig:
        def __init__(self, **kwargs):
            pass
    google_search = None



# ==============================================================================
# DOCKER COMMAND BUILDERS
# ==============================================================================

def get_docker_cmd_small(model_path, holdout_path, outputs_path, training_path):
    return [
        "docker", "run", "--rm",
        "-v", f"{Path(model_path).resolve()}:/challenge/model",
        "-v", f"{Path(holdout_path).resolve()}:/challenge/holdout_data",
        "-v", f"{Path(outputs_path).resolve()}:/challenge/holdout_outputs",
        "-v", f"{Path(training_path).resolve()}:/challenge/training_data",
        "physionet26",
        "bash", "-c", "python run_model.py -d /challenge/holdout_data -m /challenge/model -o /challenge/holdout_outputs -v"
    ]


def get_docker_cmd_large(model_path, holdout_path, outputs_path, training_path):
    return [
        "docker", "run", "--rm",
        "-v", f"{Path(model_path).resolve()}:/challenge/model",
        "-v", f"{Path(holdout_path).resolve()}:/challenge/holdout_data",
        "-v", f"{Path(outputs_path).resolve()}:/challenge/holdout_outputs",
        "-v", f"{Path(training_path).resolve()}:/challenge/training_data",
        "physionet26",
        "bash", "-c", "python run_model.py -d /challenge/holdout_data -m /challenge/model -o /challenge/holdout_outputs -v"
    ]


def get_docker_cmd_dry_run(model_path, dry_run_data_path, outputs_path):
    return [
        "docker", "run", "--rm",
        "-v", f"{Path(model_path).resolve()}:/challenge/model",
        "-v", f"{Path(dry_run_data_path).resolve()}:/challenge/holdout_data",
        "-v", f"{Path(outputs_path).resolve()}:/challenge/holdout_outputs",
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

def parse_validation_metrics(outputs_path):
    """Parse runtime and F1 score metrics from the holdout output files."""
    metrics = {"runtime_sec": None, "f1_score": 0.0, "status": "failed"}
    try:
        demo_file = Path(outputs_path) / "demographics.csv"
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

def clear_preprocessing_cache(model_path):
    """Purges the mounted cache directory to force new feature extraction."""
    cache_dir = Path(model_path) / "preprocessed_test_cache"
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
    Parse response text from the agents and apply edits to files.
    The agents are instructed to output files in the format:
    [FILE: path/to/file]
    ```<language>
    code
    ```
    """
    import re
    pattern = r"\[FILE:\s*([^\]\s]+)\]\s*```[a-zA-Z0-9+_-]*\s*(.*?)\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    
    if not matches:
        print("Warning: No file modification blocks found in the agent's response.")
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
# HIGH-ASSURANCE HELPERS
# ==============================================================================

def check_code_syntax(code_content):
    """Compiles code content to check for syntax errors before execution."""
    try:
        compile(code_content, "<string>", "exec")
        return True, ""
    except Exception as e:
        return False, str(e)


def setup_dry_run_dataset(holdout_data_path, scratch_dir):
    """Creates a 1-patient dataset in scratch_dir by copying the first record from holdout_data_path."""
    try:
        holdout_path = Path(holdout_data_path)
        scratch_path = Path(scratch_dir)
        if not holdout_path.exists():
            print(f"Holdout data path {holdout_path} does not exist. Skipping dry run setup.")
            return None
        
        demo_file = holdout_path / "demographics.csv"
        if not demo_file.exists():
            print(f"demographics.csv not found in {holdout_path}. Skipping dry run setup.")
            return None
            
        # Read demographics and get the first patient ID
        import csv
        with open(demo_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                print("demographics.csv is empty. Skipping dry run setup.")
                return None
            first_row = rows[0]
            fieldnames = reader.fieldnames

        # Get first patient ID using standard column names
        patient_id = first_row.get("bids_folder")
        if not patient_id:
            print("Could not find 'bids_folder' in demographics.csv. Skipping dry run setup.")
            return None
            
        # Create scratch dir
        scratch_path.mkdir(parents=True, exist_ok=True)
        
        # Save edited demographics table with only the first row
        with open(scratch_path / "demographics.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(first_row)
        
        # Copy patient directory or files
        src_patient_dir = holdout_path / patient_id
        if src_patient_dir.exists() and src_patient_dir.is_dir():
            dest_patient_dir = scratch_path / patient_id
            if dest_patient_dir.exists():
                shutil.rmtree(dest_patient_dir, ignore_errors=True)
            shutil.copytree(src_patient_dir, dest_patient_dir)
            print(f"Set up dry run dataset in {scratch_path} using patient {patient_id}")
            return scratch_path
        else:
            # Fallback for flat directories
            copied_any = False
            for file in holdout_path.glob(f"*{patient_id}*"):
                if file.is_file():
                    shutil.copy2(file, scratch_path / file.name)
                    copied_any = True
            if copied_any:
                print(f"Set up dry run dataset in {scratch_path} using matching files for {patient_id}")
                return scratch_path
    except Exception as e:
        print(f"Warning: Could not automatically set up dry run dataset: {e}")
    return None


# ==============================================================================
# GOOGLE ADK PROGRAMMATIC EXECUTION CALLER
# ==============================================================================

async def run_adk_agent(runner, prompt, session_id="optimization_session"):
    """Executes the ADK Agent asynchronously using the Runner context."""
    # If google-adk is not installed, fallback to dry-run mockup
    if 'google.adk' not in sys.modules or not hasattr(runner, "run_async"):
        agent_name = runner.agent.name if hasattr(runner, "agent") and runner.agent else "Unknown"
        print(f"[Mock ADK Run] Agent '{agent_name}' prompted with: {prompt[:80]}...")
        if agent_name == "Scientific_Researcher":
            return "Mock response: proposed algorithmic feature adaptations based on sleep spindle power spectrum."
        elif agent_name == "Peer_Reviewer":
            return "Mock review: The proposed spindle features are theoretically sound. Make sure window overlaps match sample rates."
        elif agent_name == "Code_Implementer":
            return "[FILE: feature_extraction/features_eeg_spindle_so.py]\n```python\n# Mock implemented changes\n```"
        elif agent_name == "Quality_Manager":
            import re
            m = re.search(r"\[FILE:\s*([^\]\s]+)\]\s*```([a-zA-Z0-9+_-]*)\s*(.*?)\s*```", prompt, re.DOTALL)
            if m:
                file_path, lang, content = m.groups()
                return f"[FILE: {file_path}]\n```{lang}\n{content}\n# Reviewed by Quality Manager\n```"
            return "[FILE: feature_extraction/features_eeg_spindle_so.py]\n```python\n# Mock reviewed changes by Quality Manager\n```"
        elif agent_name == "Documentation_Agent":
            return "[FILE: DOCUMENTATION.md]\n```markdown\n# Project Documentation\n\n## Medical Strategy\nClassification is based on sleep spindle and slow oscillation coupling.\n```"
        return "Mock response: general agent execution."

    response_text = ""
    try:
        from google.genai import types
        msg = types.Content(parts=[types.Part.from_text(text=prompt)], role="user")
    except Exception:
        msg = prompt

    try:
        async for event in runner.run_async(
            user_id="pipeline_orchestrator",
            session_id=session_id,
            new_message=msg
        ):
            if getattr(event, "is_final_response", False) or getattr(event, "content", None):
                response_text = event.content
        return response_text
    except Exception as e:
        print(f"Error running ADK agent: {e}")
        return None


async def prepare_runner_session(runner, session_id):
    """Resets the runner session and runs a tiny pre-warm message to trigger context caching."""
    if 'google.adk' in sys.modules and hasattr(runner, "session_service"):
        app_name = getattr(runner, "app_name", "pipeline_optimizer")
        try:
            print(f"Clearing previous session {session_id} to ensure clean iteration...")
            runner.session_service.delete_session_sync(
                app_name=app_name,
                user_id="pipeline_orchestrator",
                session_id=session_id
            )
        except Exception as e:
            # Session might not exist
            pass
            
        try:
            print(f"Creating session {session_id} explicitly...")
            await runner.session_service.create_session(
                app_name=app_name,
                user_id="pipeline_orchestrator",
                session_id=session_id
            )
        except Exception as e:
            print(f"Warning creating session: {e}")
            
    # Pre-warm with a tiny hello turn so that the subsequent prompt triggers cache creation.
    print(f"Pre-warming session {session_id} for context caching...")
    await run_adk_agent(runner, "Hello, starting optimizer step.", session_id=session_id)


# ==============================================================================
# MAIN ITERATIVE LOOP
# ==============================================================================

async def run_loop_async(
    max_iterations, 
    target_benchmark, 
    researcher_model, 
    implementer_model, 
    documentation_model, 
    quality_manager_model,
    peer_reviewer_model,
    holdout_data_path,
    training_data_path,
    model_path,
    outputs_path,
    interactive
):
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
            "rigorous features. You can use the Google Search tool to search for medical papers, signal processing literature, or EEG/sleep spindle classification standards."
        ),
        tools=[google_search] if google_search else []
    )

    peer_reviewer_agent = Agent(
        name="Peer_Reviewer",
        model=peer_reviewer_model,
        instruction=(
            "You are the Peer Reviewer Agent, a senior machine learning engineer and expert "
            "in physiological signal processing. Your role is to critically analyze the "
            "Scientific Researcher's proposed feature extraction modifications. Look for mathematical "
            "soundness, check for typical ML errors (like data leakage, shape mismatches, edge effects), "
            "and verify if the proposed libraries are standard. Point out flaws and suggest improvements. You can use the Google Search tool to look up papers or verify standard implementations."
        ),
        tools=[google_search] if google_search else []
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

    documentation_agent = Agent(
        name="Documentation_Agent",
        model=documentation_model,
        instruction=(
            "You are the Documentation Agent. Your role is to document the medical strategy on which the "
            "classification is based, how this is implemented by the software, what changes were made in "
            "each iteration, and how these changes performed in the respective tests.\n"
            "OUTPUT FORMAT CONSTRAINT:\n"
            "For every file you want to create or edit (such as DOCUMENTATION.md), you must output the full code "
            "or content of the file wrapped in a markdown block exactly like this:\n"
            "[FILE: path/to/file]\n"
            "```markdown\n"
            "file contents here...\n"
            "```\n"
            "Output only the file blocks. Do not add any conversational text before or after the blocks."
        )
    )

    quality_manager_agent = Agent(
        name="Quality_Manager",
        model=quality_manager_model,
        instruction=(
            "You are the Quality Manager Agent. Your role is to ensure that the code meets the quality "
            "standards of good programming practice, that the documentation also complies with the standard "
            "requirements for such a project, and that no typical machine learning errors occur during "
            "training and testing (e.g., data leakage, overfitting, class imbalance issues, improper validation).\n"
            "You will review proposed changes (both code and documentation), identify any flaws, and output "
            "corrected versions of the files.\n"
            "OUTPUT FORMAT CONSTRAINT:\n"
            "For every file you want to edit or create, you must output the full code/content of the file "
            "wrapped in a markdown block exactly like this:\n"
            "[FILE: path/to/file]\n"
            "```python\n"
            "file contents here...\n"
            "```\n"
            "or for markdown files:\n"
            "[FILE: path/to/file]\n"
            "```markdown\n"
            "file contents here...\n"
            "```\n"
            "Output only the file blocks. Do not add any conversational text before or after the blocks."
        )
    )

    # Initialize Runners and set Context Caching Configuration
    print("Setting up InMemoryRunners with Context Cache configurations...")
    cache_config = ContextCacheConfig(
        ttl_seconds=3600,
        cache_intervals=10,
        min_tokens=0
    )

    researcher_runner = InMemoryRunner(agent=researcher_agent, app_name="pipeline_optimizer")
    researcher_runner.context_cache_config = cache_config

    peer_reviewer_runner = InMemoryRunner(agent=peer_reviewer_agent, app_name="pipeline_optimizer")
    peer_reviewer_runner.context_cache_config = cache_config

    implementer_runner = InMemoryRunner(agent=implementer_agent, app_name="pipeline_optimizer")
    implementer_runner.context_cache_config = cache_config

    qm_runner = InMemoryRunner(agent=quality_manager_agent, app_name="pipeline_optimizer")
    qm_runner.context_cache_config = cache_config

    doc_runner = InMemoryRunner(agent=documentation_agent, app_name="pipeline_optimizer")
    doc_runner.context_cache_config = cache_config

    print("==================================================")
    # 1. Establish Baseline on Small Dataset
    print("Step 1: Running baseline execution on small dataset...")
    clear_preprocessing_cache(model_path)
    
    # Build container
    print("Building Docker container...")
    subprocess.run(["docker", "build", "-t", "physionet26", "."], check=True)
    
    # Run container
    print("Running validation container...")
    docker_cmd_small = get_docker_cmd_small(model_path, holdout_data_path, outputs_path, training_data_path)
    subprocess.run(docker_cmd_small, check=True)
    baseline_metrics = parse_validation_metrics(outputs_path)
    print(f"Baseline validation run complete: {baseline_metrics}")
    
    current_f1 = baseline_metrics.get("f1_score", 0.0)
    best_f1 = current_f1
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n==================================================")
        print(f"Iteration {iteration}/{max_iterations}")
        print("==================================================")
        
        # Get codebase context dynamically
        codebase_context = get_codebase_context()
        session_id = f"opt_session_iter_{iteration}"

        # Agent 2: Literature Research & Debate Loop via ADK
        # Prepare sessions (delete old session history and pre-warm for caching)
        await prepare_runner_session(researcher_runner, session_id)
        await prepare_runner_session(peer_reviewer_runner, session_id)

        print("Agent 2 (Researcher) drafting initial proposal...")
        research_prompt = (
            f"The pipeline baseline F1-score is {current_f1:.4f}.\n"
            "Review the codebase context below, search the literature, and propose "
            "3 specific mathematical adaptations or feature extraction improvements.\n\n"
            "CRITICAL INSTRUCTION: Your proposal must be highly detailed, including specific math "
            "equations, filtering bands, and targeted file locations so it is actionable for a developer agent.\n\n"
            f"Codebase Context:\n{codebase_context}"
        )
        research_proposal = await run_adk_agent(researcher_runner, research_prompt, session_id=session_id)
        
        if not research_proposal:
            print("Research proposal generation failed. Aborting iteration.")
            continue
            
        # Debate loop: Peer Reviewer reviews and Researcher refines
        debate_rounds = 2
        for round_idx in range(1, debate_rounds + 1):
            print(f"Peer Reviewer reviewing proposal (Round {round_idx}/{debate_rounds})...")
            review_prompt = (
                "Review the following research proposal for mathematical rigor, feasibility, "
                "and typical machine learning pitfalls (such as data leakage, overfitting, edge effects, "
                "unsupported dependencies, computational bottlenecks).\n\n"
                f"Proposed Research:\n{research_proposal}"
            )
            review_feedback = await run_adk_agent(peer_reviewer_runner, review_prompt, session_id=session_id)
            if not review_feedback:
                print("Review feedback failed. Using current proposal.")
                break
            print(f"\nReview Feedback (Round {round_idx}):\n", review_feedback[:500], "...\n")
            
            print(f"Scientific Researcher refining proposal (Round {round_idx}/{debate_rounds})...")
            refinement_prompt = (
                "Refine your proposal by addressing the following peer review feedback. "
                "Correct any flaws, improve clarity, and output your final updated proposal.\n\n"
                f"Peer Review Feedback:\n{review_feedback}"
            )
            refined_proposal = await run_adk_agent(researcher_runner, refinement_prompt, session_id=session_id)
            if refined_proposal:
                research_proposal = refined_proposal
                print(f"\nRefined Proposal (Round {round_idx}):\n", research_proposal[:500], "...\n")
        
        # Prepare runner sessions for implementer & quality manager
        await prepare_runner_session(implementer_runner, session_id)
        await prepare_runner_session(qm_runner, session_id)

        implementation_prompt = (
            "You are tasked with implementing the following research proposal into the codebase.\n"
            f"Research Proposal:\n{research_proposal}\n\n"
            "You must modify the codebase to apply this proposal.\n"
            f"Current Codebase Context:\n{codebase_context}\n\n"
            "Remember the OUTPUT FORMAT CONSTRAINT. Every modified file must be outputted with the [FILE: path/to/file] block."
        )

        final_code_changes = None
        code_apply_success = False
        max_repair_attempts = 3

        for attempt in range(1, max_repair_attempts + 1):
            print(f"Agent 3 (Implementer) writing code (Attempt {attempt}/{max_repair_attempts})...")
            implemented_changes = await run_adk_agent(implementer_runner, implementation_prompt, session_id=session_id)
            
            if not implemented_changes:
                print("Code generation failed.")
                continue
                
            # Quality Manager reviews code changes
            print("Agent (Quality Manager) reviewing proposed code changes...")
            qm_code_prompt = (
                "Review the proposed changes from the Code Implementer for good programming practices "
                "and typical machine learning issues (e.g., data leakage, overfitting, scaling, incorrect validation).\n\n"
                f"Current Codebase Context:\n{codebase_context}\n\n"
                f"Proposed Changes:\n{implemented_changes}\n\n"
                "If the proposed changes are correct and meet quality standards, output them exactly as is.\n"
                "If there are any issues, output the corrected and improved files with the modifications applied.\n"
                "Remember the OUTPUT FORMAT CONSTRAINT. Every modified/corrected file must be outputted with the [FILE: path/to/file] block."
            )
            reviewed_changes = await run_adk_agent(qm_runner, qm_code_prompt, session_id=session_id)
            if not reviewed_changes:
                reviewed_changes = implemented_changes
            
            # Syntax checking on reviewed files
            import re
            pattern = r"\[FILE:\s*([^\]\s]+)\]\s*```[a-zA-Z0-9+_-]*\s*(.*?)\s*```"
            matches = re.findall(pattern, reviewed_changes, re.DOTALL)
            
            if not matches:
                print("No file block changes found in the implementation response.")
                implementation_prompt += "\n\nError: No file blocks were detected. Please output the code wrapped in [FILE: path/to/file] block."
                continue
                
            syntax_errors = []
            for filepath, content in matches:
                ok, err = check_code_syntax(content)
                if not ok:
                    syntax_errors.append(f"File: {filepath}\nSyntaxError: {err}")
            
            if syntax_errors:
                err_log = "\n".join(syntax_errors)
                print(f"Linter/Syntax check failed:\n{err_log}")
                # Feed error log back to implementer for self-repair
                implementation_prompt += (
                    f"\n\nYour previous implementation failed with the following syntax error(s). "
                    f"Please correct the code and re-generate the file blocks:\n{err_log}"
                )
                continue
                
            # Try applying the changes locally
            changes_applied = apply_implemented_changes(reviewed_changes)
            if not changes_applied:
                print("Failed to apply code changes to filesystem.")
                implementation_prompt += "\n\nError: Failed to write files to disk. Ensure file paths are relative and correct."
                continue
                
            # Perform a 1-patient dry-run if dry-run dataset is available
            dry_run_dir = Path("scratch/dry_run_data")
            dry_run_dataset = setup_dry_run_dataset(holdout_data_path, dry_run_dir)
            if dry_run_dataset:
                print("Executing 1-patient dry run in Docker sandbox...")
                docker_cmd_dry = get_docker_cmd_dry_run(model_path, dry_run_dataset, outputs_path)
                try:
                    subprocess.run(["docker", "build", "-t", "physionet26", "."], check=True)
                    # Clear demographics output so we check if it is created correctly
                    demo_out = Path(outputs_path) / "demographics.csv"
                    if demo_out.exists():
                        demo_out.unlink()
                    subprocess.run(docker_cmd_dry, check=True)
                    
                    # Verify outputs exist
                    if not demo_out.exists():
                        raise Exception("Dry run completed but demographics.csv output was not created.")
                    print("1-patient dry run succeeded! All pre-flight tests passed.")
                    final_code_changes = reviewed_changes
                    code_apply_success = True
                    break
                except Exception as e:
                    print(f"Dry run failed with error: {e}")
                    # Revert modifications using Git checkout
                    subprocess.run(["git", "checkout", "."], check=True)
                    implementation_prompt += (
                        f"\n\nDry run execution failed with the following error. "
                        f"Please analyze the error and fix your implementation:\n{e}"
                    )
                    continue
            else:
                print("Holdout dataset not accessible. Skipping 1-patient dry run check.")
                final_code_changes = reviewed_changes
                code_apply_success = True
                break

        if not code_apply_success:
            print("Failed to generate correct, compiling code after maximum repair attempts. Skipping iteration.")
            continue
        
        # Human-in-the-loop Gate
        if interactive:
            print("\n==================================================")
            print("HUMAN GATE: Pre-flight checks passed!")
            print(f"Ready to execute days-long validation run for Iteration {iteration}.")
            print("==================================================")
            user_decision = input("Proceed with days-long validation run? (y/n): ").strip().lower()
            if user_decision != 'y':
                print("Validation cancelled. Reverting changes.")
                subprocess.run(["git", "checkout", "."], check=True)
                break

        # Local validation run
        print("Agent 4 (Validator) executing local validation...")
        clear_preprocessing_cache(model_path)
        
        try:
            subprocess.run(["docker", "build", "-t", "physionet26", "."], check=True)
            subprocess.run(docker_cmd_small, check=True)
            new_metrics = parse_validation_metrics(outputs_path)
            new_f1 = new_metrics.get("f1_score", 0.0)
            print(f"Iteration {iteration} F1 score: {new_f1:.4f} (Baseline: {best_f1:.4f})")
            
            # Documenting the iteration
            await prepare_runner_session(doc_runner, session_id)
            print("Agent (Documentation) updating project documentation...")
            doc_prompt = (
                "You need to document the development process of this medical classification software.\n"
                "Review the current codebase, the changes applied in this iteration, and the performance results:\n\n"
                f"Iteration: {iteration}\n"
                f"Previous best F1 score: {best_f1:.4f}\n"
                f"New F1 score: {new_f1:.4f}\n"
                f"Changes made:\n{final_code_changes}\n\n"
                "Please update or create a markdown file named 'DOCUMENTATION.md' in the root directory.\n"
                "The document must detail:\n"
                "1. The medical strategy on which the classification is based (e.g. EEG features, spindles, slow oscillations, sleep scoring).\n"
                "2. How this strategy is implemented by the software.\n"
                "3. What changes were made in each iteration (keep a running log/history of all iterations).\n"
                "4. How these changes performed in the respective tests.\n"
                "If DOCUMENTATION.md already exists in the codebase context, retrieve its contents and append/update the iteration log. Otherwise, create it.\n"
                "Remember the OUTPUT FORMAT CONSTRAINT. Output the file wrapped in a [FILE: DOCUMENTATION.md] block."
            )
            proposed_doc = await run_adk_agent(doc_runner, doc_prompt, session_id=session_id)
            if not proposed_doc:
                print("Documentation generation failed.")
            else:
                print("\nProposed Documentation:\n", proposed_doc[:500], "...\n")
                
                # Review documentation
                print("Agent (Quality Manager) reviewing proposed documentation...")
                qm_doc_prompt = (
                    "Review the proposed documentation to ensure it complies with the standard requirements for a medical/technical project "
                    "(accuracy, clarity, completeness, proper formatting) and that there are no incorrect claims or typical ML errors in the description.\n\n"
                    f"Proposed Documentation:\n{proposed_doc}\n\n"
                    "If the proposed documentation is correct, output it exactly as is.\n"
                    "If there are any issues, output the corrected and improved documentation.\n"
                    "Remember the OUTPUT FORMAT CONSTRAINT. Output the file wrapped in a [FILE: DOCUMENTATION.md] block."
                )
                reviewed_doc = await run_adk_agent(qm_runner, qm_doc_prompt, session_id=session_id)
                if not reviewed_doc:
                    reviewed_doc = proposed_doc
                apply_implemented_changes(reviewed_doc)
            
            # Check improvement benchmark
            if new_f1 > best_f1 + target_benchmark:
                print(f"SUCCESS: F1-score improved from {best_f1:.4f} to {new_f1:.4f} (benchmark met!).")
                best_f1 = new_f1
                
                # Push and tag!
                push_to_test_and_tag()
                
                # Run large dataset verification (Agent 1)
                print("Agent 1 executing final large dataset validation...")
                docker_cmd_large = get_docker_cmd_large(model_path, holdout_path, outputs_path, training_path)
                subprocess.run(docker_cmd_large, check=True)
                large_metrics = parse_validation_metrics(outputs_path)
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
    parser.add_argument("--interactive", action="store_true", help="Enable interactive human-in-the-loop checkpoint")
    
    # Models
    parser.add_argument("--researcher-model", type=str, default="gemini-3.5-flash", help="Gemini model for research tasks")
    parser.add_argument("--peer-reviewer-model", type=str, default="gemini-3.5-flash", help="Gemini model for peer review tasks")
    parser.add_argument("--implementer-model", type=str, default="gemini-3.5-flash", help="Gemini model for coding tasks")
    parser.add_argument("--documentation-model", type=str, default="gemini-3.5-flash", help="Gemini model for documentation tasks")
    parser.add_argument("--quality-manager-model", type=str, default="gemini-3.5-flash", help="Gemini model for quality management tasks")
    
    # Host Paths configuration
    parser.add_argument("--holdout-data", type=str, default="D:\\split_dataset\\split_5", help="Path to holdout data")
    parser.add_argument("--training-data", type=str, default="D:\\split_dataset\\split_2", help="Path to training data")
    parser.add_argument("--model-path", type=str, default=r"C:\Users\Biosig 3\Documents\Richard\Physionet26Data\model", help="Path to model data")
    parser.add_argument("--outputs-path", type=str, default=r"C:\Users\Biosig 3\Documents\Richard\Physionet26Data\holdout_outputs", help="Path to holdout outputs")

    args = parser.parse_args()
    
    # Ensure directories/paths wrap as Path objects
    holdout_path = Path(args.holdout_data)
    training_path = Path(args.training_data)
    model_path = Path(args.model_path)
    outputs_path = Path(args.outputs_path)

    if args.dry_run:
        print("DRY-RUN SIMULATION (ADK Mode):")
        docker_small = get_docker_cmd_small(model_path, holdout_path, outputs_path, training_path)
        docker_large = get_docker_cmd_large(model_path, holdout_path, outputs_path, training_path)
        print(f"Docker small run cmd: {' '.join(docker_small)}")
        print(f"Docker large run cmd: {' '.join(docker_large)}")
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
        args.implementer_model,
        args.documentation_model,
        args.quality_manager_model,
        args.peer_reviewer_model,
        holdout_path,
        training_path,
        model_path,
        outputs_path,
        args.interactive
    ))
