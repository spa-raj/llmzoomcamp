"""Utilities for handling rate limits when calling OpenAI APIs."""

from tqdm.auto import tqdm
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import RateLimitError


def map_progress_with_rate_limit(pool, seq, f, batch_size=10, delay_between_batches=2):
    """Process sequences with rate limiting and exponential backoff.
    
    Args:
        pool: ThreadPoolExecutor instance
        seq: Sequence to process
        f: Function to apply to each element
        batch_size: Number of items to process before pausing
        delay_between_batches: Seconds to wait between batches
    """
    results = []
    
    with tqdm(total=len(seq)) as progress:
        # Process in batches to avoid hitting rate limits
        for i in range(0, len(seq), batch_size):
            batch = seq[i:i+batch_size]
            futures = {}
            
            for el in batch:
                future = pool.submit(f, el)
                futures[future] = el
            
            # Process completed futures with retry logic
            for future in as_completed(futures):
                retries = 0
                max_retries = 5
                
                while retries < max_retries:
                    try:
                        result = future.result()
                        results.append(result)
                        progress.update()
                        break
                    except RateLimitError as e:
                        retries += 1
                        wait_time = min(2 ** retries + random.uniform(0, 1), 60)
                        print(f"\nRate limit hit. Waiting {wait_time:.1f}s (retry {retries}/{max_retries})...")
                        time.sleep(wait_time)
                        
                        # Resubmit the task
                        if retries < max_retries:
                            el = futures[future]
                            future = pool.submit(f, el)
                    except Exception as e:
                        print(f"\nError processing item: {e}")
                        results.append(None)
                        progress.update()
                        break
            
            # Add delay between batches to avoid rate limits
            if i + batch_size < len(seq):
                time.sleep(delay_between_batches)
    
    return results


def map_progress_sequential(seq, f, delay_between_requests=0.5):
    """Process sequences sequentially with delays to avoid rate limits.
    
    Args:
        seq: Sequence to process
        f: Function to apply to each element
        delay_between_requests: Seconds to wait between requests
    """
    results = []
    
    for el in tqdm(seq):
        retries = 0
        max_retries = 5
        
        while retries < max_retries:
            try:
                result = f(el)
                results.append(result)
                time.sleep(delay_between_requests)  # Add delay between requests
                break
            except RateLimitError as e:
                retries += 1
                wait_time = min(2 ** retries + random.uniform(0, 1), 60)
                print(f"\nRate limit hit. Waiting {wait_time:.1f}s (retry {retries}/{max_retries})...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"\nError processing item: {e}")
                results.append(None)
                break
    
    return results


def map_progress_adaptive(pool, seq, f, initial_batch_size=10, min_batch_size=1):
    """Process sequences with adaptive batch sizing based on rate limit responses.
    
    This function automatically adjusts batch size when rate limits are encountered.
    
    Args:
        pool: ThreadPoolExecutor instance
        seq: Sequence to process
        f: Function to apply to each element
        initial_batch_size: Starting batch size
        min_batch_size: Minimum batch size to use
    """
    results = []
    batch_size = initial_batch_size
    rate_limit_hits = 0
    
    with tqdm(total=len(seq)) as progress:
        i = 0
        while i < len(seq):
            # Adjust batch size based on rate limit hits
            if rate_limit_hits > 2:
                batch_size = max(min_batch_size, batch_size // 2)
                rate_limit_hits = 0
                print(f"\nReducing batch size to {batch_size}")
            
            batch = seq[i:i+batch_size]
            futures = {}
            
            for el in batch:
                future = pool.submit(f, el)
                futures[future] = el
            
            batch_results = []
            batch_had_rate_limit = False
            
            for future in as_completed(futures):
                retries = 0
                max_retries = 5
                success = False
                
                while retries < max_retries:
                    try:
                        result = future.result()
                        batch_results.append(result)
                        progress.update()
                        success = True
                        break
                    except RateLimitError as e:
                        batch_had_rate_limit = True
                        rate_limit_hits += 1
                        retries += 1
                        wait_time = min(2 ** retries + random.uniform(0, 1), 60)
                        print(f"\nRate limit hit. Waiting {wait_time:.1f}s (retry {retries}/{max_retries})...")
                        time.sleep(wait_time)
                        
                        if retries < max_retries:
                            el = futures[future]
                            future = pool.submit(f, el)
                    except Exception as e:
                        print(f"\nError processing item: {e}")
                        batch_results.append(None)
                        progress.update()
                        break
                
                if not success and retries >= max_retries:
                    batch_results.append(None)
                    progress.update()
            
            results.extend(batch_results)
            
            # If we hit rate limits, add extra delay
            if batch_had_rate_limit:
                time.sleep(5)
            else:
                # Small delay between batches
                time.sleep(1)
            
            i += batch_size
    
    return results


def process_with_token_limit(seq, f, tokens_per_minute=10000, estimated_tokens_per_request=500):
    """Process sequences while respecting token-per-minute limits.
    
    Args:
        seq: Sequence to process
        f: Function to apply to each element
        tokens_per_minute: Your API token limit per minute
        estimated_tokens_per_request: Estimated tokens used per request
    """
    results = []
    requests_per_minute = tokens_per_minute // estimated_tokens_per_request
    delay_between_requests = 60 / requests_per_minute
    
    print(f"Processing with {requests_per_minute} requests per minute (delay: {delay_between_requests:.2f}s)")
    
    for el in tqdm(seq):
        start_time = time.time()
        retries = 0
        max_retries = 5
        
        while retries < max_retries:
            try:
                result = f(el)
                results.append(result)
                
                # Calculate time to wait
                elapsed = time.time() - start_time
                remaining_delay = max(0, delay_between_requests - elapsed)
                if remaining_delay > 0:
                    time.sleep(remaining_delay)
                break
                
            except RateLimitError as e:
                retries += 1
                # Extract wait time from error if available
                wait_time = min(2 ** retries + random.uniform(0, 1), 60)
                print(f"\nRate limit hit. Waiting {wait_time:.1f}s (retry {retries}/{max_retries})...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"\nError processing item: {e}")
                results.append(None)
                break
    
    return results


# Example usage functions
def example_usage():
    """Example of how to use these functions in your notebook."""
    
    # Example 1: With ThreadPoolExecutor and batching
    from concurrent.futures import ThreadPoolExecutor
    
    pool = ThreadPoolExecutor(max_workers=2)  # Reduced workers
    
    # Use with your existing process_record function
    # results = map_progress_with_rate_limit(
    #     pool, 
    #     ground_truth, 
    #     process_record,
    #     batch_size=5,  # Process 5 items at a time
    #     delay_between_batches=3  # Wait 3 seconds between batches
    # )
    
    # Example 2: Sequential processing (slowest but most reliable)
    # results = map_progress_sequential(
    #     ground_truth, 
    #     process_record, 
    #     delay_between_requests=1  # 1 second between each request
    # )
    
    # Example 3: Adaptive batch sizing
    # results = map_progress_adaptive(
    #     pool,
    #     ground_truth,
    #     process_record,
    #     initial_batch_size=10,
    #     min_batch_size=1
    # )
    
    # Example 4: Token-based rate limiting
    # results = process_with_token_limit(
    #     ground_truth,
    #     process_record,
    #     tokens_per_minute=10000,  # Your actual limit
    #     estimated_tokens_per_request=500  # Estimate based on your prompts
    # )
    
    pass