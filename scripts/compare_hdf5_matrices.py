#!/usr/bin/env python3
"""
Compare a2aMatrix datasets between two HDF5 files.

Takes two file paths as command line arguments and compares the 
"a2aMatrix" datasets in each file to verify they are identical.
"""

import argparse
import h5py
import numpy as np
import sys
from pathlib import Path


def load_a2a_matrix(filepath):
    """Load a2aMatrix dataset from HDF5 file."""
    try:
        with h5py.File(filepath, 'r') as f:
            # Get first group key
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"No groups found in {filepath}")
            
            a_group_key = keys[0]
            group_obj = f[a_group_key]
            
            # Access a2aMatrix from the group
            dataset = group_obj["a2aMatrix"]  # type: ignore
            matrix = np.array(dataset)
            return matrix
    except KeyError as e:
        raise ValueError(f"Dataset structure not found in {filepath}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load {filepath}: {e}")


def compare_matrices(matrix1, matrix2, filepath1, filepath2):
    """Compare two matrices and return comparison results."""
    # Check shapes
    if matrix1.shape != matrix2.shape:
        return False, f"Shape mismatch: {matrix1.shape} vs {matrix2.shape}"
    
    # Check data types
    if matrix1.dtype != matrix2.dtype:
        print(f"Warning: dtype mismatch ({matrix1.dtype} vs {matrix2.dtype}), comparing values anyway")
    
    # Check if all values are the same
    if np.array_equal(matrix1, matrix2):
        return True, f"Matrices are identical (shape: {matrix1.shape}, dtype: {matrix1.dtype})"
    
    # If not identical, provide more details
    diff_mask = matrix1 != matrix2
    num_diff = np.sum(diff_mask)
    max_abs_diff = np.max(np.abs(matrix1 - matrix2))
    
    return False, (f"Matrices differ in {num_diff}/{matrix1.size} elements "
                  f"(max absolute difference: {max_abs_diff:.2e})")


def main():
    parser = argparse.ArgumentParser(
        description="Compare a2aMatrix datasets between two HDF5 files"
    )
    parser.add_argument("file1", help="Path to first HDF5 file")
    parser.add_argument("file2", help="Path to second HDF5 file")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Print additional information")
    
    args = parser.parse_args()
    
    # Validate file paths
    file1_path = Path(args.file1)
    file2_path = Path(args.file2)
    
    if not file1_path.exists():
        print(f"Error: File not found: {file1_path}", file=sys.stderr)
        sys.exit(1)
    
    if not file2_path.exists():
        print(f"Error: File not found: {file2_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Load matrices
        if args.verbose:
            print(f"Loading {file1_path}...")
        matrix1 = load_a2a_matrix(file1_path)
        
        if args.verbose:
            print(f"Loading {file2_path}...")
        matrix2 = load_a2a_matrix(file2_path)
        
        # Compare matrices
        are_same, message = compare_matrices(matrix1, matrix2, file1_path, file2_path)
        
        if are_same:
            print(f"IDENTICAL: {message}")
            sys.exit(0)
        else:
            print(f"DIFFERENT: {message}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()