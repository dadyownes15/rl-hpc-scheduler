import pandas as pd
import numpy as np
import os
import datetime as dt
import random
from typing import Dict, List, Tuple, Optional
from scipy import stats
from collections import Counter


class JobsetGenerator:
    """Generator for creating jobsets from raw CSV traces"""
    
    def __init__(self, column_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize with column mapping for different CSV formats
        
        Args:
            column_mapping: Dict mapping standard names to actual column names
                          If None, uses Theta default mapping
        """
        # Default Theta column mapping
        self.default_theta_mapping = {
            'job_id': 'COBALT_JOBID',
            'queued_timestamp': 'QUEUED_TIMESTAMP', 
            'start_timestamp': 'START_TIMESTAMP',
            'end_timestamp': 'END_TIMESTAMP',
            'runtime_seconds': 'RUNTIME_SECONDS',
            'walltime_seconds': 'WALLTIME_SECONDS',
            'nodes_used': 'NODES_USED',
            'nodes_requested': 'NODES_REQUESTED',
            'cores_used': 'CORES_USED',
            'cores_requested': 'CORES_REQUESTED',
            'queue_name': 'QUEUE_NAME',
            'username': 'USERNAME_GENID',
            'project': 'PROJECT_NAME_GENID',
            'exit_status': 'EXIT_STATUS'
        }
        
        self.column_mapping = column_mapping or self.default_theta_mapping
        
    def generate_carbon_consideration_index(self, size: int, distribution_type: str = 'inverse_normal') -> np.ndarray:
        """
        Generate carbon consideration index with configurable distribution
        
        Args:
            size: Number of values to generate
            distribution_type: Type of distribution ('inverse_normal', 'uniform', 'bimodal')
            
        Returns:
            Array of carbon consideration indices between 0 and 1
        """
        if distribution_type == 'inverse_normal':
            # Generate normal distribution, then invert to get high mass at edges
            normal_vals = np.random.normal(0.5, 0.15, size)
            # Clip to [0,1] range
            normal_vals = np.clip(normal_vals, 0, 1)
            # Invert: values close to 0.5 become close to 0, values at edges stay high
            inverted = 1 - 2 * np.abs(normal_vals - 0.5)
            return np.clip(inverted, 0, 1)
        elif distribution_type == 'bimodal':
            # Mix of two distributions at the edges
            left_mode = np.random.beta(2, 8, size // 2)  # Skewed towards 0
            right_mode = np.random.beta(8, 2, size - size // 2)  # Skewed towards 1
            return np.concatenate([left_mode, right_mode])
        elif distribution_type == 'uniform':
            return np.random.uniform(0, 1, size)
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    def filter_debug_jobs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out debugging jobs based on queue name
        
        Args:
            df: Input dataframe
            
        Returns:
            Filtered dataframe with debug jobs removed
        """
        queue_col = self.column_mapping['queue_name']
        if queue_col in df.columns:
            # Filter out jobs with 'debug-cache-quad' in queue name
            mask = ~df[queue_col].str.contains('debug-cache-quad', na=False, case=False)
            mask_2 = ~df[queue_col].str.contains('R.pm2', na=False, case=False) 
            return df[mask & mask_2]
        return df
    
    def filter_by_date(self, df: pd.DataFrame, start_year: int = 2023) -> pd.DataFrame:
        """
        Filter jobs to only include those starting after specified year
        
        Args:
            df: Input dataframe
            start_year: Minimum year for job start times
            
        Returns:
            Filtered dataframe
        """
        start_col = self.column_mapping['start_timestamp']
        if start_col in df.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[start_col]):
                # Try parsing as datetime string first, then as unix timestamp
                try:
                    df[start_col] = pd.to_datetime(df[start_col], utc=True)
                except:
                    try:
                        df[start_col] = pd.to_datetime(df[start_col], unit='s', utc=True)
                    except:
                        print(f"Warning: Could not parse {start_col} as datetime")
                        return df
            
            # Filter by year
            mask = df[start_col].dt.year >= start_year
            return df[mask]
        return df
    
    def filter_large_jobs(self, df: pd.DataFrame, max_nodes: int = 4360) -> pd.DataFrame:
        """
        Filter out jobs that request more than the maximum number of nodes
        
        Args:
            df: Input dataframe
            max_nodes: Maximum number of nodes allowed (default: 4360)
            
        Returns:
            Filtered dataframe with large jobs removed
        """
        nodes_col = self.column_mapping['nodes_requested']
        if nodes_col in df.columns:
            # Create mask to keep jobs with nodes_requested <= max_nodes
            mask = df[nodes_col] <= max_nodes
            return df[mask]
        return df
    
    def csv_to_swf_format(self, df: pd.DataFrame) -> List[str]:
        """
        Convert CSV dataframe to SWF format lines
        
        Args:
            df: Input dataframe with job data
            
        Returns:
            List of SWF-formatted strings
        """
        swf_lines = []
        
        for idx, row in df.iterrows():
            # Map CSV columns to SWF format
            job_id = row[self.column_mapping['job_id']]
            
            # Handle timestamp conversion
            queued_ts = row[self.column_mapping['queued_timestamp']]
            start_ts = row[self.column_mapping['start_timestamp']]
            
            # Convert to unix timestamp if needed
            if isinstance(queued_ts, str):
                queued_ts = pd.to_datetime(queued_ts, utc=True).timestamp()
            elif hasattr(queued_ts, 'timestamp'):  # pandas Timestamp
                queued_ts = queued_ts.timestamp()
            
            if isinstance(start_ts, str):
                start_ts = pd.to_datetime(start_ts, utc=True).timestamp()
            elif hasattr(start_ts, 'timestamp'):  # pandas Timestamp
                start_ts = start_ts.timestamp()
            
            submit_time = int(queued_ts)
            wait_time = int(start_ts - queued_ts)
            run_time = int(row[self.column_mapping['runtime_seconds']])
            
            # Use actual values where available, -1 for missing
            # Ensure processor counts are integers
            used_proc = int(row.get(self.column_mapping['nodes_used'], -1))
            used_avg_cpu = -1  # Not available in CSV
            used_mem = -1      # Not available in CSV
            req_proc = int(row[self.column_mapping['nodes_requested']])
            req_time = int(row[self.column_mapping['walltime_seconds']])
            req_mem = -1       # Not available in CSV
            
            status = 1 if row.get(self.column_mapping['exit_status'], 0) == 0 else 0
            user_id = hash(str(row.get(self.column_mapping['username'], 'unknown'))) % 10000
            group_id = hash(str(row.get(self.column_mapping['project'], 'unknown'))) % 1000
            
            # SWF fields we don't have data for
            num_exe = -1
            num_queue = -1
            num_part = -1
            num_pre = -1
            think_time = -1
            
            # Carbon consideration index (will be added later)
            carbon_idx = 0  # Placeholder, will be set when generating jobsets
            
            # Format as SWF line (space-separated)
            swf_line = f"{job_id} {submit_time} {wait_time} {run_time} {used_proc} {used_avg_cpu} {used_mem} {req_proc} {req_time} {req_mem} {status} {user_id} {group_id} {num_exe} {num_queue} {num_part} {num_pre} {think_time} {carbon_idx}"
            swf_lines.append(swf_line)
            
        return swf_lines
    
    def create_swf_header(self, start_timestamp: int) -> List[str]:
        """
        Create SWF header with metadata
        
        Args:
            start_timestamp: Unix timestamp of first job
            
        Returns:
            List of header lines
        """
        max_nodes = 4360
        max_procs = 4360
        header = [
            "; Version: 2.2",
            "; Computer: Theta Supercomputer",
            "; Installation: Argonne Leadership Computing Facility (ALCF)",
            "; Preemption: No",
            f"; UnixStartTime: {start_timestamp}",
            "; TimeZone: 0",
            "; TimeZoneString: UTC",
            f"; MaxNodes: {max_nodes}",
            f"; MaxProcs: {max_procs}",
            "; Note: Generated jobset for DRAS training",
            ";"
        ]
        return header
    
    def create_real_jobsets(self, 
                           csv_file: str, 
                           output_dir: str, 
                           jobs_per_set: int = 3200,
                           num_sets: int = 9,
                           reverse: bool = False) -> List[str]:
        """
        Create real jobsets by splitting the trace into contiguous time periods
        
        Args:
            csv_file: Path to input CSV file
            output_dir: Directory to save jobsets
            jobs_per_set: Number of jobs per jobset
            num_sets: Number of jobsets to create
            reverse: If True, sample from end of year backwards
            
        Returns:
            List of created jobset file paths
        """
        print(f"Loading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} jobs")
        
        # Filter debug jobs
        df = self.filter_debug_jobs(df)
        print(f"After filtering debug jobs: {len(df)} jobs")
        
        # Filter by date (2022+ for training data)
        df = self.filter_by_date(df, start_year=2022)
        print(f"After filtering by date (2022+): {len(df)} jobs")
        
        # Filter large jobs (> 4360 nodes)
        df = self.filter_large_jobs(df)
        print(f"After filtering large jobs (> 4360 nodes): {len(df)} jobs")
        
        if len(df) == 0:
            raise ValueError("No jobs remaining after filtering")
        
        # Convert timestamp columns to datetime for sorting and processing
        queued_col = self.column_mapping['queued_timestamp']
        if not pd.api.types.is_datetime64_any_dtype(df[queued_col]):
            df[queued_col] = pd.to_datetime(df[queued_col], utc=True)
        
        # Sort by queued timestamp
        df = df.sort_values(queued_col).reset_index(drop=True)
        
        # Create output directory
        real_dir = os.path.join(output_dir, 'real')
        os.makedirs(real_dir, exist_ok=True)
        
        created_files = []
        
        # Split into contiguous chunks
        total_jobs = len(df)
        jobs_per_chunk = total_jobs // num_sets
        
        for i in range(num_sets):
            if reverse:
                # Sample from end backwards: latest week = real_week_1
                start_idx = total_jobs - (i + 1) * jobs_per_chunk
                end_idx = total_jobs - i * jobs_per_chunk
                start_idx = max(0, start_idx)
            else:
                # Sample from beginning forwards
                start_idx = i * jobs_per_chunk
                end_idx = min((i + 1) * jobs_per_chunk, total_jobs)
            
            # Take up to jobs_per_set from this chunk
            chunk_df = df.iloc[start_idx:end_idx].head(jobs_per_set)
            
            if len(chunk_df) == 0:
                continue
                
            # Add carbon consideration index
            carbon_indices = self.generate_carbon_consideration_index(len(chunk_df))
            
            # Convert to SWF format
            swf_lines = self.csv_to_swf_format(chunk_df)
            
            # Update carbon consideration index in SWF lines
            updated_lines = []
            for j, line in enumerate(swf_lines):
                parts = line.split()
                parts[-1] = f"{carbon_indices[j]:.3f}"  # Replace last column
                updated_lines.append(' '.join(parts))
            
            # Create header
            first_queued = chunk_df.iloc[0][queued_col]
            if isinstance(first_queued, str):
                start_timestamp = int(pd.to_datetime(first_queued, utc=True).timestamp())
            elif hasattr(first_queued, 'timestamp'):  # pandas Timestamp
                start_timestamp = int(first_queued.timestamp())
            else:
                start_timestamp = int(first_queued)  # assume it's already a unix timestamp
            
            header = self.create_swf_header(start_timestamp)
            
            # Write jobset file
            output_file = os.path.join(real_dir, f"real_week_{i+1}.swf")
            with open(output_file, 'w') as f:
                f.write('\n'.join(header + updated_lines))
            
            created_files.append(output_file)
            print(f"Created {output_file} with {len(updated_lines)} jobs")
        
        return created_files
    
    def create_single_swf(self, df: pd.DataFrame, output_file: str) -> str:
        """
        Create a single SWF file from a dataframe (for validation/test sets)
        
        Args:
            df: Input dataframe with job data
            output_file: Path to output SWF file
            
        Returns:
            Path to created SWF file
        """
        # Add carbon consideration index
        carbon_indices = self.generate_carbon_consideration_index(len(df))
        
        # Convert to SWF format
        swf_lines = self.csv_to_swf_format(df)
        
        # Update carbon consideration index in SWF lines
        updated_lines = []
        for j, line in enumerate(swf_lines):
            parts = line.split()
            parts[-1] = f"{carbon_indices[j]:.3f}"
            updated_lines.append(' '.join(parts))
        
        # Create header
        queued_col = self.column_mapping['queued_timestamp']
        first_queued = df.iloc[0][queued_col]
        if isinstance(first_queued, str):
            start_timestamp = int(pd.to_datetime(first_queued, utc=True).timestamp())
        elif hasattr(first_queued, 'timestamp'):
            start_timestamp = int(first_queued.timestamp())
        else:
            start_timestamp = int(first_queued)
        
        header = self.create_swf_header(start_timestamp)
        
        # Write SWF file
        with open(output_file, 'w') as f:
            f.write('\n'.join(header + updated_lines))
        
        return output_file
    
    def create_validation_test_sets(self,
                                   csv_file: str,
                                   output_dir: str) -> Tuple[str, str]:
        """
        Create validation and test sets from 2023 data
        
        Args:
            csv_file: Path to 2023 CSV file
            output_dir: Directory to save sets
            
        Returns:
            Tuple of (validation_file_path, test_file_path)
        """
        print(f"Creating validation and test sets from: {csv_file}")
        
        # Load and filter data
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} jobs from 2023")
        
        # Filter debug jobs
        df = self.filter_debug_jobs(df)
        print(f"After filtering debug jobs: {len(df)} jobs")
        
        # Filter by date (2023+)
        df = self.filter_by_date(df, start_year=2023)
        print(f"After filtering by date (2023+): {len(df)} jobs")
        
        # Filter large jobs (> 4360 nodes)
        df = self.filter_large_jobs(df)
        print(f"After filtering large jobs (> 4360 nodes): {len(df)} jobs")
        
        if len(df) == 0:
            raise ValueError("No jobs remaining after filtering")
        
        # Convert timestamp columns
        queued_col = self.column_mapping['queued_timestamp']
        if not pd.api.types.is_datetime64_any_dtype(df[queued_col]):
            df[queued_col] = pd.to_datetime(df[queued_col], utc=True)
        
        # Sort by queued timestamp
        df = df.sort_values(queued_col).reset_index(drop=True)
        
        # Split by month: January = validation, rest = test
        df['month'] = df[queued_col].dt.month
        validation_df = df[df['month'] == 1].reset_index(drop=True)
        test_df = df[df['month'] != 1].reset_index(drop=True)
        
        print(f"Validation set (January): {len(validation_df)} jobs")
        print(f"Test set (February-December): {len(test_df)} jobs")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create validation SWF file
        validation_file = os.path.join(output_dir, "validation_2023_jan.swf")
        if len(validation_df) > 0:
            self.create_single_swf(validation_df, validation_file)
            print(f"Created validation set: {validation_file}")
        else:
            print("Warning: No validation jobs found")
            validation_file = None
        
        # Create test SWF file
        test_file = os.path.join(output_dir, "test_2023_feb_dec.swf")
        if len(test_df) > 0:
            self.create_single_swf(test_df, test_file)
            print(f"Created test set: {test_file}")
        else:
            print("Warning: No test jobs found")
            test_file = None
        
        return validation_file, test_file
    
    def create_sampled_jobsets(self,
                              csv_file: str,
                              output_dir: str,
                              jobs_per_set: int = 3200,
                              num_sets: int = 9,
                              seed: int = 42) -> List[str]:
        """
        Create sampled jobsets with Poisson arrivals and random job sampling
        
        Args:
            csv_file: Path to input CSV file
            output_dir: Directory to save jobsets
            jobs_per_set: Number of jobs per jobset
            num_sets: Number of jobsets to create
            seed: Random seed for reproducibility
            
        Returns:
            List of created jobset file paths
        """
        print(f"Creating {num_sets} sampled jobsets...")
        
        # Load and filter data (same as real jobsets)
        df = pd.read_csv(csv_file)
        df = self.filter_debug_jobs(df)
        df = self.filter_by_date(df, start_year=2022)
        df = self.filter_large_jobs(df)
        
        if len(df) == 0:
            raise ValueError("No jobs remaining after filtering")
        
        # Convert timestamp columns
        queued_col = self.column_mapping['queued_timestamp']
        if not pd.api.types.is_datetime64_any_dtype(df[queued_col]):
            df[queued_col] = pd.to_datetime(df[queued_col], utc=True)
        
        df = df.sort_values(queued_col).reset_index(drop=True)
        
        # Calculate average inter-arrival time for Poisson process
        timestamps = df[queued_col].apply(lambda x: x.timestamp()).values
        inter_arrivals = np.diff(timestamps)
        mean_inter_arrival = np.mean(inter_arrivals)
        poisson_rate = 1.0 / mean_inter_arrival
        
        print(f"Average inter-arrival time: {mean_inter_arrival:.2f} seconds")
        print(f"Poisson rate: {poisson_rate:.6f} jobs/second")
        
        # Create output directory
        sampled_dir = os.path.join(output_dir, 'sampled')
        os.makedirs(sampled_dir, exist_ok=True)
        
        created_files = []
        random.seed(seed)
        np.random.seed(seed)
        
        # Get the earliest timestamp as starting point
        start_timestamp = timestamps[0]
        
        for i in range(num_sets):
            print(f"Creating sampled jobset {i+1}/{num_sets}")
            
            # Randomly sample jobs from the trace
            sampled_indices = random.sample(range(len(df)), min(jobs_per_set, len(df)))
            sampled_df = df.iloc[sampled_indices].copy()
            
            # Generate new arrival times using Poisson process
            poisson_times = []
            current_time = start_timestamp
            for _ in range(len(sampled_df)):
                # Draw inter-arrival time from exponential distribution
                inter_arrival = np.random.exponential(mean_inter_arrival)
                current_time += inter_arrival
                poisson_times.append(current_time)
            
            # Update the queued timestamps with Poisson arrivals
            sampled_df = sampled_df.reset_index(drop=True)
            for j, new_time in enumerate(poisson_times):
                sampled_df.iloc[j, sampled_df.columns.get_loc(queued_col)] = pd.Timestamp(new_time, unit='s', tz='UTC')
            
            # Sort by new arrival times
            sampled_df = sampled_df.sort_values(queued_col).reset_index(drop=True)
            
            # Add carbon consideration index
            carbon_indices = self.generate_carbon_consideration_index(len(sampled_df))
            
            # Convert to SWF format
            swf_lines = self.csv_to_swf_format(sampled_df)
            
            # Update carbon consideration index in SWF lines
            updated_lines = []
            for j, line in enumerate(swf_lines):
                parts = line.split()
                parts[-1] = f"{carbon_indices[j]:.3f}"
                updated_lines.append(' '.join(parts))
            
            # Create header
            first_queued = sampled_df.iloc[0][queued_col]
            header_timestamp = int(first_queued.timestamp())
            max_nodes = sampled_df[self.column_mapping['nodes_requested']].max()
            max_procs = sampled_df[self.column_mapping['cores_requested']].max()
            
            header = self.create_swf_header(header_timestamp)
            
            # Write jobset file
            output_file = os.path.join(sampled_dir, f"sampled_poisson_{i+1}.swf")
            with open(output_file, 'w') as f:
                f.write('\n'.join(header + updated_lines))
            
            created_files.append(output_file)
            print(f"Created {output_file} with {len(updated_lines)} jobs")
        
        return created_files
    
    def create_synthetic_jobsets(self,
                                csv_file: str,
                                output_dir: str,
                                jobs_per_set: int = 3200,
                                num_sets: int = 82,
                                seed: int = 42) -> List[str]:
        """
        Create synthetic jobsets by sampling from empirical distributions
        
        Args:
            csv_file: Path to input CSV file
            output_dir: Directory to save jobsets
            jobs_per_set: Number of jobs per jobset
            num_sets: Number of jobsets to create
            seed: Random seed for reproducibility
            
        Returns:
            List of created jobset file paths
        """
        print(f"Creating {num_sets} synthetic jobsets...")
        
        # Load and filter data
        df = pd.read_csv(csv_file)
        df = self.filter_debug_jobs(df)
        df = self.filter_by_date(df, start_year=2022)
        df = self.filter_large_jobs(df)
        
        if len(df) == 0:
            raise ValueError("No jobs remaining after filtering")
        
        # Convert timestamp columns
        queued_col = self.column_mapping['queued_timestamp']
        if not pd.api.types.is_datetime64_any_dtype(df[queued_col]):
            df[queued_col] = pd.to_datetime(df[queued_col], utc=True)
        
        df = df.sort_values(queued_col).reset_index(drop=True)
        
        # Build empirical distributions
        print("Building empirical distributions...")
        
        # 1. Inter-arrival time distribution
        timestamps = df[queued_col].apply(lambda x: x.timestamp()).values
        inter_arrivals = np.diff(timestamps)
        mean_inter_arrival = np.mean(inter_arrivals)
        
        # 2. Hour-of-day arrival pattern
        df['hour'] = df[queued_col].dt.hour
        hourly_counts = df['hour'].value_counts().sort_index()
        hourly_probs = hourly_counts / hourly_counts.sum()
        
        # 3. Day-of-week arrival pattern
        df['dayofweek'] = df[queued_col].dt.dayofweek
        daily_counts = df['dayofweek'].value_counts().sort_index()
        daily_probs = daily_counts / daily_counts.sum()
        
        # 4. Job size (nodes requested) distribution - nodes and processors are the same
        nodes_hist = Counter(df[self.column_mapping['nodes_requested']])
        nodes_values = list(nodes_hist.keys())
        nodes_weights = list(nodes_hist.values())
        
        # 5. Runtime distribution
        runtime_hist = Counter(df[self.column_mapping['runtime_seconds']])
        runtime_values = list(runtime_hist.keys())
        runtime_weights = list(runtime_hist.values())
        
        # 6. Walltime distribution
        walltime_hist = Counter(df[self.column_mapping['walltime_seconds']])
        walltime_values = list(walltime_hist.keys())
        walltime_weights = list(walltime_hist.values())
        
        print(f"Built distributions from {len(df)} jobs")
        print(f"Nodes range: {min(nodes_values)} - {max(nodes_values)}")
        print(f"Runtime range: {min(runtime_values)} - {max(runtime_values)} seconds")
        
        # Create output directory
        synthetic_dir = os.path.join(output_dir, 'synthetic')
        os.makedirs(synthetic_dir, exist_ok=True)
        
        created_files = []
        random.seed(seed)
        np.random.seed(seed)
        
        # Starting timestamp
        start_timestamp = timestamps[0]
        
        for i in range(num_sets):
            print(f"Creating synthetic jobset {i+1}/{num_sets}")
            
            synthetic_jobs = []
            current_time = start_timestamp
            
            for j in range(jobs_per_set):
                # Generate arrival time with temporal patterns
                # Sample hour and day of week based on empirical distributions
                hour = np.random.choice(hourly_probs.index, p=hourly_probs.values)
                dayofweek = np.random.choice(daily_probs.index, p=daily_probs.values)
                
                # Add some randomness to inter-arrival time
                base_inter_arrival = np.random.exponential(mean_inter_arrival)
                # Modify based on hour-of-day (peak hours get more jobs)
                hour_factor = hourly_probs[hour] * 24  # normalize to average of 1
                inter_arrival = base_inter_arrival / max(hour_factor, 0.1)  # avoid division by very small numbers
                
                current_time += inter_arrival
                
                # Sample job characteristics from empirical distributions
                nodes_req = np.random.choice(nodes_values, p=np.array(nodes_weights)/sum(nodes_weights))
                runtime = np.random.choice(runtime_values, p=np.array(runtime_weights)/sum(runtime_weights))
                walltime = np.random.choice(walltime_values, p=np.array(walltime_weights)/sum(walltime_weights))

                # Ensure runtime is positive and less than or equal to walltime
                if walltime <= 0:
                    walltime = max(runtime, 300) # Ensure a minimum walltime
                runtime = max(1, min(runtime, walltime))
                
                # Generate synthetic job data
                job_id = i * jobs_per_set + j + 1000000  # unique IDs
                submit_time = int(current_time)
                wait_time = np.random.randint(0, 3600)  # 0-1 hour wait time
                start_time = submit_time + wait_time
                
                # SWF format fields
                # For synthetic jobs, assume used processors equals requested processors
                # Nodes and processors are treated as the same thing
                req_proc = int(nodes_req)
                used_proc = req_proc
                
                used_avg_cpu = -1
                used_mem = -1
                req_mem = -1
                status = 1  # completed successfully
                user_id = random.randint(1, 10000)
                group_id = random.randint(1, 1000)
                num_exe = -1
                num_queue = -1
                num_part = -1
                num_pre = -1
                think_time = -1
                carbon_idx = 0  # will be set later
                
                swf_line = f"{job_id} {submit_time} {wait_time} {runtime} {used_proc} {used_avg_cpu} {used_mem} {req_proc} {walltime} {req_mem} {status} {user_id} {group_id} {num_exe} {num_queue} {num_part} {num_pre} {think_time} {carbon_idx}"
                synthetic_jobs.append(swf_line)
            
            # Add carbon consideration indices
            carbon_indices = self.generate_carbon_consideration_index(len(synthetic_jobs))
            
            # Update carbon consideration index in SWF lines
            updated_lines = []
            for j, line in enumerate(synthetic_jobs):
                parts = line.split()
                parts[-1] = f"{carbon_indices[j]:.3f}"
                updated_lines.append(' '.join(parts))
            
            # Create header
            header_timestamp = int(start_timestamp)
            
            header = self.create_swf_header(header_timestamp)
            
            # Write jobset file
            output_file = os.path.join(synthetic_dir, f"synthetic_{i+1}.swf")
            with open(output_file, 'w') as f:
                f.write('\n'.join(header + updated_lines))
            
            created_files.append(output_file)
            print(f"Created {output_file} with {len(updated_lines)} jobs")
        
        return created_files 