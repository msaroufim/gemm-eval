"""
Code-based challenges that can be validated by running.

These are simple, focused coding tasks extracted from NVFP4 GEMM patterns.
Each challenge has a template, expected behavior, and validation function.
"""

import re
from typing import Callable, Dict, Any, Tuple

CODE_CHALLENGES = {
    "scale_factor_permute": {
        "id": "code_sf_permute",
        "skill": "scale_factors",
        "difficulty": "hard",
        "description": "Implement the scale factor permutation for tcgen05",
        "prompt": """
Implement a function that permutes scale factors from shape (L, M, K//16) to the
layout expected by tcgen05 MMA.

The transformation should:
1. Reshape M dimension: M -> (M//128, 32, 4)
2. Reshape K//16 dimension: K//16 -> (K//64, 4)
3. Permute to final layout: (32, 4, M//128, 4, K//64, L)

Complete this function:

```python
import torch

def permute_scale_factors(sf: torch.Tensor, m: int, k: int, l: int) -> torch.Tensor:
    '''
    Permute scale factors from (L, M, K//16) to tcgen05 layout.

    Args:
        sf: Scale factor tensor of shape (L, M, K//16)
        m: M dimension (must be divisible by 128)
        k: K dimension (must be divisible by 64)
        l: Batch dimension

    Returns:
        Permuted tensor of shape (32, 4, M//128, 4, K//64, L)
    '''
    # YOUR CODE HERE
    pass
```
""",
        "template": """
import torch

def permute_scale_factors(sf: torch.Tensor, m: int, k: int, l: int) -> torch.Tensor:
    # YOUR CODE HERE
    pass
""",
        "solution": """
import torch

def permute_scale_factors(sf: torch.Tensor, m: int, k: int, l: int) -> torch.Tensor:
    sf_vec_size = 16
    rest_m = m // 128
    rest_k = (k // sf_vec_size) // 4

    # Reshape: (L, M, K//16) -> (L, M//128, 32, 4, K//64, 4)
    sf = sf.view(l, rest_m, 32, 4, rest_k, 4)

    # Permute: (L, M//128, 32, 4, K//64, 4) -> (32, 4, M//128, 4, K//64, L)
    sf = sf.permute(2, 3, 1, 5, 4, 0).contiguous()

    return sf
""",
        "test_cases": [
            {"m": 128, "k": 64, "l": 1, "expected_shape": (32, 4, 1, 4, 1, 1)},
            {"m": 256, "k": 128, "l": 2, "expected_shape": (32, 4, 2, 4, 2, 2)},
            {"m": 512, "k": 256, "l": 4, "expected_shape": (32, 4, 4, 4, 4, 4)},
        ],
        "validation": """
import torch

def validate(permute_scale_factors):
    test_cases = [
        {"m": 128, "k": 64, "l": 1, "expected_shape": (32, 4, 1, 4, 1, 1)},
        {"m": 256, "k": 128, "l": 2, "expected_shape": (32, 4, 2, 4, 2, 2)},
        {"m": 512, "k": 256, "l": 4, "expected_shape": (32, 4, 4, 4, 4, 4)},
    ]

    results = []
    for tc in test_cases:
        m, k, l = tc["m"], tc["k"], tc["l"]
        sf = torch.randn(l, m, k // 16)

        try:
            result = permute_scale_factors(sf, m, k, l)
            if result.shape == tc["expected_shape"]:
                results.append({"case": tc, "passed": True})
            else:
                results.append({"case": tc, "passed": False,
                               "error": f"Shape mismatch: {result.shape} != {tc['expected_shape']}"})
        except Exception as e:
            results.append({"case": tc, "passed": False, "error": str(e)})

    return all(r["passed"] for r in results), results
"""
    },

    "barrier_pattern": {
        "id": "code_barrier",
        "skill": "async_copy",
        "difficulty": "medium",
        "description": "Implement producer-consumer barrier pattern",
        "prompt": """
Complete the producer-consumer synchronization pattern using mbarrier.

The producer:
1. Signals how many bytes it will produce (expect_tx)
2. Produces the data
3. Signals completion (arrive)

The consumer:
1. Waits for data to be ready (wait)
2. Consumes the data
3. Signals it's done (release for next iteration)

Complete this pseudocode:

```python
class PipelineStage:
    def __init__(self, num_bytes: int):
        self.num_bytes = num_bytes
        self.barrier = create_mbarrier()

    def producer_begin(self):
        '''Called before producer starts writing data'''
        # YOUR CODE: What barrier operation goes here?
        pass

    def producer_end(self):
        '''Called after producer finishes writing data'''
        # YOUR CODE: What barrier operation goes here?
        pass

    def consumer_wait(self):
        '''Called before consumer starts reading data'''
        # YOUR CODE: What barrier operation goes here?
        pass

    def consumer_release(self):
        '''Called after consumer finishes reading data'''
        # YOUR CODE: What barrier operation goes here?
        pass
```
""",
        "solution": """
class PipelineStage:
    def __init__(self, num_bytes: int):
        self.num_bytes = num_bytes
        self.barrier = create_mbarrier()

    def producer_begin(self):
        '''Called before producer starts writing data'''
        self.barrier.expect_tx(self.num_bytes)

    def producer_end(self):
        '''Called after producer finishes writing data'''
        self.barrier.arrive()

    def consumer_wait(self):
        '''Called before consumer starts reading data'''
        self.barrier.wait()

    def consumer_release(self):
        '''Called after consumer finishes reading data'''
        # Reset barrier for next iteration (or arrive if double-buffered)
        self.barrier.arrive()
""",
        "validation_type": "pattern_match",
        "expected_patterns": [
            r"expect_tx.*num_bytes",  # producer_begin should have expect_tx
            r"producer_end.*arrive",   # producer_end should have arrive
            r"consumer_wait.*wait",    # consumer_wait should have wait
        ]
    },

    "warp_assignment": {
        "id": "code_warp_assign",
        "skill": "warp_specialization",
        "difficulty": "medium",
        "description": "Calculate warp roles from thread ID",
        "prompt": """
Given 192 threads (6 warps), implement the warp role assignment.

Warp assignment for NVFP4 GEMM:
- Warps 0-3 (threads 0-127): EPILOGUE
- Warp 4 (threads 128-159): MMA
- Warp 5 (threads 160-191): TMA

Complete this function:

```python
from enum import Enum

class WarpRole(Enum):
    EPILOGUE = 0
    MMA = 1
    TMA = 2

def get_warp_role(thread_id: int) -> WarpRole:
    '''
    Given a thread ID (0-191), return the warp's role.

    Args:
        thread_id: Thread ID within the CTA (0-191)

    Returns:
        WarpRole enum value
    '''
    # YOUR CODE HERE
    pass

def is_leader_thread(thread_id: int) -> bool:
    '''
    Return True if this thread is lane 0 of its warp (warp leader).
    '''
    # YOUR CODE HERE
    pass
```
""",
        "solution": """
from enum import Enum

class WarpRole(Enum):
    EPILOGUE = 0
    MMA = 1
    TMA = 2

def get_warp_role(thread_id: int) -> WarpRole:
    warp_id = thread_id // 32

    if warp_id < 4:
        return WarpRole.EPILOGUE
    elif warp_id == 4:
        return WarpRole.MMA
    else:  # warp_id == 5
        return WarpRole.TMA

def is_leader_thread(thread_id: int) -> bool:
    return thread_id % 32 == 0
""",
        "test_cases": [
            {"thread_id": 0, "expected_role": "EPILOGUE", "expected_leader": True},
            {"thread_id": 31, "expected_role": "EPILOGUE", "expected_leader": False},
            {"thread_id": 128, "expected_role": "MMA", "expected_leader": True},
            {"thread_id": 160, "expected_role": "TMA", "expected_leader": True},
            {"thread_id": 191, "expected_role": "TMA", "expected_leader": False},
        ],
        "validation": """
def validate(get_warp_role, is_leader_thread, WarpRole):
    test_cases = [
        {"thread_id": 0, "expected_role": "EPILOGUE", "expected_leader": True},
        {"thread_id": 31, "expected_role": "EPILOGUE", "expected_leader": False},
        {"thread_id": 128, "expected_role": "MMA", "expected_leader": True},
        {"thread_id": 160, "expected_role": "TMA", "expected_leader": True},
        {"thread_id": 191, "expected_role": "TMA", "expected_leader": False},
    ]

    results = []
    for tc in test_cases:
        tid = tc["thread_id"]
        try:
            role = get_warp_role(tid)
            leader = is_leader_thread(tid)

            role_correct = role.name == tc["expected_role"]
            leader_correct = leader == tc["expected_leader"]

            if role_correct and leader_correct:
                results.append({"case": tc, "passed": True})
            else:
                results.append({"case": tc, "passed": False,
                               "error": f"role={role.name} (expected {tc['expected_role']}), leader={leader} (expected {tc['expected_leader']})"})
        except Exception as e:
            results.append({"case": tc, "passed": False, "error": str(e)})

    return all(r["passed"] for r in results), results
"""
    },

    "pipeline_index": {
        "id": "code_pipeline_idx",
        "skill": "pipeline",
        "difficulty": "easy",
        "description": "Calculate pipeline stage index",
        "prompt": """
Implement pipeline stage index calculation for double-buffering.

With N stages, the producer and consumer each track their own index.
The stage index wraps around: 0, 1, 2, ..., N-1, 0, 1, 2, ...

Complete this:

```python
class PipelineState:
    def __init__(self, num_stages: int):
        self.num_stages = num_stages
        self.index = 0

    def current_stage(self) -> int:
        '''Return current stage index (0 to num_stages-1)'''
        # YOUR CODE HERE
        pass

    def advance(self):
        '''Advance to next stage (wrapping around)'''
        # YOUR CODE HERE
        pass

    def is_full(self, consumer_index: int) -> bool:
        '''
        Check if pipeline is full (producer would overwrite unread data).
        Producer is full when it's num_stages ahead of consumer.
        '''
        # YOUR CODE HERE
        pass
```
""",
        "solution": """
class PipelineState:
    def __init__(self, num_stages: int):
        self.num_stages = num_stages
        self.index = 0

    def current_stage(self) -> int:
        return self.index % self.num_stages

    def advance(self):
        self.index += 1

    def is_full(self, consumer_index: int) -> bool:
        return (self.index - consumer_index) >= self.num_stages
""",
        "test_cases": [
            {"num_stages": 2, "advances": 0, "expected_stage": 0},
            {"num_stages": 2, "advances": 1, "expected_stage": 1},
            {"num_stages": 2, "advances": 2, "expected_stage": 0},
            {"num_stages": 4, "advances": 5, "expected_stage": 1},
        ],
        "validation": """
def validate(PipelineState):
    test_cases = [
        {"num_stages": 2, "advances": 0, "expected_stage": 0},
        {"num_stages": 2, "advances": 1, "expected_stage": 1},
        {"num_stages": 2, "advances": 2, "expected_stage": 0},
        {"num_stages": 4, "advances": 5, "expected_stage": 1},
    ]

    results = []
    for tc in test_cases:
        try:
            state = PipelineState(tc["num_stages"])
            for _ in range(tc["advances"]):
                state.advance()

            stage = state.current_stage()
            if stage == tc["expected_stage"]:
                results.append({"case": tc, "passed": True})
            else:
                results.append({"case": tc, "passed": False,
                               "error": f"Got {stage}, expected {tc['expected_stage']}"})
        except Exception as e:
            results.append({"case": tc, "passed": False, "error": str(e)})

    # Also test is_full
    state = PipelineState(2)
    state.advance()
    state.advance()
    if not state.is_full(0):
        results.append({"case": "is_full test", "passed": False, "error": "Should be full"})
    else:
        results.append({"case": "is_full test", "passed": True})

    return all(r["passed"] for r in results), results
"""
    },

    "tmem_layout": {
        "id": "code_tmem_layout",
        "skill": "tmem",
        "difficulty": "medium",
        "description": "Calculate TMEM region offsets",
        "prompt": """
Implement TMEM layout calculation.

TMEM regions are laid out as:
1. Accumulators (FP32) - num_acc_cols columns
2. SFA (FP8) - num_sfa_cols columns
3. SFB (FP8) - num_sfb_cols columns

Each column is 32 bytes. Given tile sizes, calculate the byte offsets.

```python
def calculate_tmem_layout(
    acc_m: int,      # Accumulator M tiles
    acc_n: int,      # Accumulator N tiles
    sf_m: int,       # Scale factor M elements
    sf_k_blocks: int # Number of K blocks (K/16)
) -> dict:
    '''
    Calculate TMEM byte offsets for each region.

    Accumulator uses FP32 (4 bytes), scale factors use FP8 (1 byte).
    TMEM column = 32 bytes.

    Returns:
        dict with keys: acc_offset, sfa_offset, sfb_offset, total_bytes
    '''
    # YOUR CODE HERE
    pass
```
""",
        "solution": """
def calculate_tmem_layout(
    acc_m: int,
    acc_n: int,
    sf_m: int,
    sf_k_blocks: int
) -> dict:
    TMEM_COL_BYTES = 32

    # Accumulators: M * N elements, each FP32 (4 bytes)
    acc_elements = acc_m * acc_n
    acc_bytes = acc_elements * 4
    num_acc_cols = (acc_bytes + TMEM_COL_BYTES - 1) // TMEM_COL_BYTES

    # SFA: M * K_blocks elements, each FP8 (1 byte)
    sfa_elements = sf_m * sf_k_blocks
    sfa_bytes = sfa_elements * 1
    num_sfa_cols = (sfa_bytes + TMEM_COL_BYTES - 1) // TMEM_COL_BYTES

    # SFB: Same size as SFA for symmetric case
    num_sfb_cols = num_sfa_cols

    return {
        "acc_offset": 0,
        "sfa_offset": num_acc_cols * TMEM_COL_BYTES,
        "sfb_offset": (num_acc_cols + num_sfa_cols) * TMEM_COL_BYTES,
        "total_bytes": (num_acc_cols + num_sfa_cols + num_sfb_cols) * TMEM_COL_BYTES
    }
""",
        "validation": """
def validate(calculate_tmem_layout):
    # Test basic case
    result = calculate_tmem_layout(128, 128, 128, 8)

    checks = []
    checks.append(result["acc_offset"] == 0)  # Acc starts at 0
    checks.append(result["sfa_offset"] > result["acc_offset"])  # SFA after acc
    checks.append(result["sfb_offset"] > result["sfa_offset"])  # SFB after SFA
    checks.append(result["total_bytes"] > result["sfb_offset"])  # Total > SFB start

    return all(checks), [{"case": "layout order", "passed": all(checks)}]
"""
    }
}


def get_code_challenges():
    """Return list of all code challenges."""
    return list(CODE_CHALLENGES.values())


def get_challenge_by_id(challenge_id: str):
    """Get a specific challenge by ID."""
    for c in CODE_CHALLENGES.values():
        if c["id"] == challenge_id:
            return c
    return None


def validate_solution(challenge_id: str, submitted_code: str) -> Tuple[bool, list]:
    """
    Validate a submitted solution.

    Returns:
        (passed: bool, details: list of test results)
    """
    challenge = get_challenge_by_id(challenge_id)
    if not challenge:
        return False, [{"error": f"Unknown challenge: {challenge_id}"}]

    if "validation" not in challenge:
        return None, [{"error": "No automated validation available"}]

    # Create execution environment
    exec_globals = {}

    try:
        # Execute submitted code
        exec(submitted_code, exec_globals)

        # Execute validation
        exec(challenge["validation"], exec_globals)

        # Call validate function with extracted functions
        validate_fn = exec_globals["validate"]

        # Get the function(s) that were defined
        if challenge_id == "code_sf_permute":
            return validate_fn(exec_globals["permute_scale_factors"])
        elif challenge_id == "code_warp_assign":
            return validate_fn(
                exec_globals["get_warp_role"],
                exec_globals["is_leader_thread"],
                exec_globals["WarpRole"]
            )
        elif challenge_id == "code_pipeline_idx":
            return validate_fn(exec_globals["PipelineState"])
        elif challenge_id == "code_tmem_layout":
            return validate_fn(exec_globals["calculate_tmem_layout"])
        else:
            return False, [{"error": "Validation not implemented for this challenge"}]

    except Exception as e:
        return False, [{"error": f"Execution error: {str(e)}"}]
