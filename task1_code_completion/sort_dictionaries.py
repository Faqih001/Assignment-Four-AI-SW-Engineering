"""
Task 1: AI-Powered Code Completion
Comparing AI-suggested vs Manual implementation for sorting dictionaries
"""
import time
import random
from typing import List, Dict, Any

# AI-Suggested Implementation (GitHub Copilot style)
def sort_dict_list_ai_suggested(data: List[Dict[str, Any]], key: str, reverse: bool = False) -> List[Dict[str, Any]]:
    """
    AI-suggested function to sort a list of dictionaries by a specific key.
    
    Args:
        data: List of dictionaries to sort
        key: Key to sort by
        reverse: Whether to sort in descending order
    
    Returns:
        Sorted list of dictionaries
    """
    return sorted(data, key=lambda x: x.get(key, 0), reverse=reverse)

# Manual Implementation
def sort_dict_list_manual(data: List[Dict[str, Any]], key: str, reverse: bool = False) -> List[Dict[str, Any]]:
    """
    Manual implementation to sort a list of dictionaries by a specific key.
    
    Args:
        data: List of dictionaries to sort
        key: Key to sort by
        reverse: Whether to sort in descending order
    
    Returns:
        Sorted list of dictionaries
    """
    def get_sort_value(item):
        value = item.get(key)
        if value is None:
            return float('-inf') if not reverse else float('inf')
        return value
    
    # Create a copy to avoid modifying original list
    sorted_data = data.copy()
    
    # Use bubble sort for demonstration (less efficient but more explicit)
    n = len(sorted_data)
    for i in range(n):
        for j in range(0, n - i - 1):
            val1 = get_sort_value(sorted_data[j])
            val2 = get_sort_value(sorted_data[j + 1])
            
            if (not reverse and val1 > val2) or (reverse and val1 < val2):
                sorted_data[j], sorted_data[j + 1] = sorted_data[j + 1], sorted_data[j]
    
    return sorted_data

# Optimized Manual Implementation (using built-in sort)
def sort_dict_list_manual_optimized(data: List[Dict[str, Any]], key: str, reverse: bool = False) -> List[Dict[str, Any]]:
    """
    Optimized manual implementation using list.sort() method.
    """
    sorted_data = data.copy()
    sorted_data.sort(key=lambda x: x.get(key, float('-inf') if not reverse else float('inf')), reverse=reverse)
    return sorted_data

# Performance Testing Function
def performance_test():
    """Test performance of different implementations"""
    
    # Generate test data
    test_data = []
    for i in range(1000):
        test_data.append({
            'id': i,
            'score': random.randint(1, 100),
            'name': f'item_{i}',
            'priority': random.choice(['high', 'medium', 'low'])
        })
    
    # Test AI-suggested implementation
    start_time = time.time()
    result_ai = sort_dict_list_ai_suggested(test_data, 'score')
    ai_time = time.time() - start_time
    
    # Test manual implementation (bubble sort)
    start_time = time.time()
    result_manual = sort_dict_list_manual(test_data, 'score')
    manual_time = time.time() - start_time
    
    # Test optimized manual implementation
    start_time = time.time()
    result_optimized = sort_dict_list_manual_optimized(test_data, 'score')
    optimized_time = time.time() - start_time
    
    print("Performance Comparison:")
    print(f"AI-suggested (sorted()): {ai_time:.6f} seconds")
    print(f"Manual (bubble sort): {manual_time:.6f} seconds")
    print(f"Optimized manual (list.sort()): {optimized_time:.6f} seconds")
    
    # Verify results are identical
    print(f"\nResults identical (AI vs Optimized): {result_ai == result_optimized}")
    print(f"Results identical (AI vs Manual): {result_ai == result_manual}")
    
    return {
        'ai_time': ai_time,
        'manual_time': manual_time,
        'optimized_time': optimized_time
    }

# Demo function
def demo_sorting():
    """Demonstrate the sorting functionality"""
    
    sample_data = [
        {'name': 'Alice', 'age': 30, 'salary': 70000},
        {'name': 'Bob', 'age': 25, 'salary': 50000},
        {'name': 'Charlie', 'age': 35, 'salary': 80000},
        {'name': 'Diana', 'age': 28, 'salary': 60000}
    ]
    
    print("Original data:")
    for item in sample_data:
        print(f"  {item}")
    
    print("\nSorted by age (AI-suggested):")
    sorted_by_age = sort_dict_list_ai_suggested(sample_data, 'age')
    for item in sorted_by_age:
        print(f"  {item}")
    
    print("\nSorted by salary, descending (Manual):")
    sorted_by_salary = sort_dict_list_manual_optimized(sample_data, 'salary', reverse=True)
    for item in sorted_by_salary:
        print(f"  {item}")

if __name__ == "__main__":
    print("=" * 60)
    print("TASK 1: AI-POWERED CODE COMPLETION DEMO")
    print("=" * 60)
    
    # Run demo
    demo_sorting()
    
    print("\n" + "=" * 60)
    print("PERFORMANCE TESTING")
    print("=" * 60)
    
    # Run performance test
    performance_test()
