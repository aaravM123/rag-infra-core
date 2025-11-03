"""
Wrapper module for vector_search C++ extension.
The actual module is compiled in the build/ directory.
"""
import sys
import os
from typing import List

# Add build directory to path
build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build')
if build_dir not in sys.path and os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

try:
    # Import the actual C++ compiled module
    from vector_search import *  # type: ignore
except ImportError:
    # Fallback stub for IDE when module not yet built
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        raise NotImplementedError("Module not built. Run: mkdir build && cd build && cmake .. && make")
    
    def l2_distance(a: List[float], b: List[float]) -> float:
        """Compute L2 distance between two vectors."""
        raise NotImplementedError("Module not built. Run: mkdir build && cd build && cmake .. && make")
    
    def search_topk(vectors: List[List[float]], query: List[float], k: int) -> List[int]:
        """Find top-k most similar vectors to query."""
        raise NotImplementedError("Module not built. Run: mkdir build && cd build && cmake .. && make")

