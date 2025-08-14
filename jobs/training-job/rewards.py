"""
The author of this project does not claim ownership to these reward functions.
They are sourced from: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
Credits to the creators at hugging face and perhaps DeepSeek team (?)
Reference: Hugging Face Open-Source AI Cookbook
"""

import re
import logging
from typing import Any, Dict, List, Optional
from schema import RewardConfig


def format_reward(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
    """
    Format Enforcement: Ensures that the generation follows a specific format
    using <think> </think> <answer> </answer> tags for reasoning.

    Args:
        completions: List of completion messages for each example
        **kwargs: Additional arguments (unused)

    Returns:
        List of rewards (1.0 if format correct, 0.0 otherwise)
    """
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    rewards = []

    for completion in completions:
        # Extract text content from completion
        if isinstance(completion, list) and len(completion) > 0:
            content = completion[0].get("content", "")
        else:
            content = str(completion)

        # Check if content matches the required format
        match = re.match(pattern, content.strip(), re.DOTALL | re.MULTILINE)
        rewards.append(1.0 if match else 0.0)

    return rewards


def accuracy_reward(
    completions: List[List[Dict[str, str]]], solution: List[str], **kwargs
) -> List[Optional[float]]:
    """
    Solution Accuracy: Verifies whether the solution to the problem is correct,
    comparing it to the solution column in the dataset.

    Uses math verification for mathematical expressions when available,
    falls back to normalized text comparison.

    Args:
        completions: List of completion messages for each example
        solution: List of ground truth solutions from dataset
        **kwargs: Additional arguments (unused)

    Returns:
        List of rewards (1.0 if correct, 0.0 if incorrect, None if verification failed)
    """
    try:
        from math_verify import LatexExtractionConfig, parse, verify
        from latex2sympy2_extended import NormalizationConfig

        math_verify_available = True
    except ImportError:
        logging.warning(
            "math_verify not available, falling back to text comparison only"
        )
        math_verify_available = False

    rewards = []

    for completion, sol in zip(completions, solution):
        try:
            # Extract content from completion
            if isinstance(completion, list) and len(completion) > 0:
                content = completion[0].get("content", "")
            else:
                content = str(completion)

            if math_verify_available:
                # Try parsing ground truth
                try:
                    gold_parsed = parse(sol, extraction_mode="first_match")
                except Exception:
                    gold_parsed = []

                if len(gold_parsed) != 0:
                    # Try parsing predicted answer with robust config
                    try:
                        answer_parsed = parse(
                            content,
                            extraction_config=[
                                LatexExtractionConfig(
                                    normalization_config=NormalizationConfig(
                                        nits=False,
                                        malformed_operators=False,
                                        basic_latex=True,
                                        boxed="all",
                                        units=True,
                                    ),
                                    boxed_match_priority=0,
                                    try_extract_without_anchor=False,
                                )
                            ],
                            extraction_mode="first_match",
                        )
                        reward = float(verify(gold_parsed, answer_parsed))
                    except Exception as e:
                        logging.debug(
                            f"Math verification failed: {e}, falling back to text comparison"
                        )
                        reward = None
                else:
                    # Fallback to text match
                    reward = float(content.strip().lower() == sol.strip().lower())
            else:
                # Simple text comparison fallback
                reward = float(content.strip().lower() == sol.strip().lower())

        except Exception as e:
            logging.warning(f"Error in accuracy reward calculation: {e}")
            reward = None

        rewards.append(reward)

    return rewards


# Registry of available reward functions
BUILT_IN_REWARDS = {
    "format": format_reward,
    "accuracy": accuracy_reward,
}


def load_reward_functions_from_config(reward_config: RewardConfig) -> List[Any]:
    """
    Load reward functions from configuration.

    Only supports the two built-in reward functions: 'format' and 'accuracy'.

    Args:
        reward_config: List of reward function specifications with 'name' field

    Returns:
        List of callable reward functions
    """
    if not reward_config:
        return []

    reward_functions = []

    # 1. Load built in functions
    for function_name in reward_config.built_in_func:
        if function_name in BUILT_IN_REWARDS:
            reward_functions.append(BUILT_IN_REWARDS[function_name])
            logging.info(f"Loaded reward function: {function_name}")
        else:
            logging.error(
                f"Unknown reward function: {function_name}. "
                f"Available functions: {list(BUILT_IN_REWARDS.keys())}"
            )

    # 2. will add in future...

    return reward_functions
