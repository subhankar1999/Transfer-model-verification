#!/usr/bin/env python3


import os
import csv
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List


class TransferExecutionLogger:
    """Logs execution metrics of the transfer abstract network program."""
    
    def __init__(self, csv_file_path: str = "transfer_execution_log.csv"):
        """
        Initialize the logger.
        
        Args:
            csv_file_path: Path to the CSV file where logs will be stored
        """
        self.csv_file_path = csv_file_path
        self.execution_id = None
        self.start_time = None
        self.execution_data = {}
        self._initialize_csv()
    
    def _initialize_csv(self) -> None:
        """Initialize CSV file with headers if it doesn't exist."""
        # Define all CSV columns
        self.fieldnames = [
            'execution_id',
            'timestamp',
            'original_network_file',
            'property_id',
            'transfer_network_name',
            'transfer network_creation_time',
            'original_network_verification_time',
            'step1_cegar_abstraction_time',
            'original_network_verification_result',
            'step1_cegar_refine_steps',
            'step1_cegar_total_ar_time',
            # 'step2_cegar_last_query_time',
            'step1_cegar_ar_times',
            'step1_cegar_ar_sizes',
            'step1_cegar_refine_sequence_times',
            'abstract_reuse_time',
            'transfer_abstract_verification_time',
            'step3_refinement_steps',
            'step3_query_times',
            'step3_refine_times',
            'transfer_network_verification_time(from scratch)',
            'step4_cegar_abstraction_time',
            'transfer_network_verification_result',
            'step4_cegar_refine_steps',
            'step4_cegar_total_ar_time',
            # 'step4_cegar_last_query_time',
            'step4_cegar_ar_times',
            'step4_cegar_ar_sizes',
            'step4_cegar_refine_sequence_times',
            'total_execution_time',
            'status',
            'notes'
        ]
        
        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(self.csv_file_path):
            # Create directory if needed (but only if there's a directory component)
            dir_path = os.path.dirname(self.csv_file_path)
            if dir_path:  # Only create if directory path is not empty
                os.makedirs(dir_path, exist_ok=True)
            
            with open(self.csv_file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            print(f" Created new CSV log file: {self.csv_file_path}")
        else:
            print(f" Using existing CSV log file: {self.csv_file_path}")
    
    def start_execution(self, original_nnet_file: str, property_id: str) -> str:
        """
        Start a new execution session.
        
        Args:
            original_nnet_file: Path/name of the original network file
            property_id: Property ID for verification
            
        Returns:
            Unique execution ID for this session
        """
        self.start_time = time.time()
        self.execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:20]}"
        
        # Extract just the filename if full path is provided
        if os.path.sep in original_nnet_file:
            filename = os.path.basename(original_nnet_file)
        else:
            filename = original_nnet_file
        
        self.execution_data = {
            'execution_id': self.execution_id,
            'timestamp': datetime.now().isoformat(),
            'original_network_file': filename,
            'property_id': property_id,
            'status': 'in_progress',
            'notes': 'Execution started'
        }
        
        print(f"\n{'='*70}")
        print(f"Logging session started: {self.execution_id}")
        print(f"  Original network: {filename}")
        print(f"  Property ID: {property_id}")
        print(f"{'='*70}")
        
        return self.execution_id
    
    def log_step1(self, transferred_network_name: str, step1_time: float) -> None:
        """
        Log Step 1 metrics.
        
        Args:
            transferred_network_name: Name of the transferred network
            step1_time: Time taken for Step 1
        """
        self.execution_data['transfer_network_name'] = transferred_network_name
        if step1_time == "N/A":
            self.execution_data['transfer network_creation_time'] = "N/A"
        else:
            self.execution_data['transfer network_creation_time'] = round(step1_time, 4)
            # print(f" Logged Step 1: {step1_time:.4f}s")
    
    def log_step2(self, step2_time: float, cegar_results: Optional[Dict[str, Any]] = None) -> None:
        """
        Log Step 2 metrics including CEGAR experiment results.
        
        Args:
            step2_time: Time taken for Step 2
            cegar_results: Dictionary containing CEGAR experiment results from one_experiment()
                          with keys like 'abstraction_time', 'query_result', etc.
        """
        if step2_time == "N/A":
            self.execution_data['original_network_verification_time'] = "N/A"
        else:
            self.execution_data['original_network_verification_time'] = round(step2_time, 4)
        
        # Parse CEGAR results if provided
        if cegar_results:
            # Convert list of tuples to dictionary if needed
            if isinstance(cegar_results, list):
                cegar_dict = dict(cegar_results)
            else:
                cegar_dict = cegar_results
            
            # Extract and store CEGAR metrics
            self.execution_data['step1_cegar_abstraction_time'] = \
                round(cegar_dict.get('abstraction_time', 0), 4)
            self.execution_data['original_network_verification_result'] = \
                cegar_dict.get('query_result', 'N/A')
            self.execution_data['step1_cegar_refine_steps'] = \
                cegar_dict.get('num_of_refine_steps', 0)
            self.execution_data['step1_cegar_total_ar_time'] = \
                round(cegar_dict.get('total_ar_query_time', 0), 4)
            # self.execution_data['step2_cegar_last_query_time'] = \
            #     round(cegar_dict.get('last_query_time', 0), 4)
            
            # Store JSON arrays
            ar_times = cegar_dict.get('ar_times', '[]')
            ar_sizes = cegar_dict.get('ar_sizes', '[]')
            refine_times = cegar_dict.get('refine_sequence_times', '[]')
            
            # Handle if they're already JSON strings
            if isinstance(ar_times, str):
                self.execution_data['step1_cegar_ar_times'] = ar_times
            else:
                self.execution_data['step1_cegar_ar_times'] = json.dumps(ar_times)
            
            if isinstance(ar_sizes, str):
                self.execution_data['step1_cegar_ar_sizes'] = ar_sizes
            else:
                self.execution_data['step1_cegar_ar_sizes'] = json.dumps(ar_sizes)
            
            if isinstance(refine_times, str):
                self.execution_data['step1_cegar_refine_sequence_times'] = refine_times
            else:
                self.execution_data['step1_cegar_refine_sequence_times'] = json.dumps(refine_times)
            
            print(f" Logged Step 1: {step2_time:.4f}s (CEGAR: {cegar_dict.get('query_result', 'N/A')})")
        else:
            print(f" Logged Step 1: {step2_time:.4f}s (no CEGAR results)")

    
    def log_step5(self, step5_time: float, cegar_results: Optional[Dict[str, Any]] = None) -> None:
        """
        Log Step 2 metrics including CEGAR experiment results.
        
        Args:
            step2_time: Time taken for Step 2
            cegar_results: Dictionary containing CEGAR experiment results from one_experiment()
                          with keys like 'abstraction_time', 'query_result', etc.
        """
        if step5_time == "N/A":
            self.execution_data['transfer_network_verification_time(from scratch)'] = "N/A"
        else:
            self.execution_data['transfer_network_verification_time(from scratch)'] = round(step5_time, 4)
        
        # Parse CEGAR results if provided
        if cegar_results:
            # Convert list of tuples to dictionary if needed
            if isinstance(cegar_results, list):
                cegar_dict = dict(cegar_results)
            else:
                cegar_dict = cegar_results
            
            # Extract and store CEGAR metrics
            self.execution_data['step4_cegar_abstraction_time'] = \
                round(cegar_dict.get('abstraction_time', 0), 4)
            self.execution_data['transfer_network_verification_result'] = \
                cegar_dict.get('query_result', 'N/A')
            self.execution_data['step4_cegar_refine_steps'] = \
                cegar_dict.get('num_of_refine_steps', 0)
            self.execution_data['step4_cegar_total_ar_time'] = \
                round(cegar_dict.get('total_ar_query_time', 0), 4)
            # self.execution_data['step4_cegar_last_query_time'] = \
            #     round(cegar_dict.get('last_query_time', 0), 4)
            # 'step4_result',
            # Store JSON arrays
            ar_times = cegar_dict.get('ar_times', '[]')
            ar_sizes = cegar_dict.get('ar_sizes', '[]')
            refine_times = cegar_dict.get('refine_sequence_times', '[]')
            
            # Handle if they're already JSON strings
            if isinstance(ar_times, str):
                self.execution_data['step4_cegar_ar_times'] = ar_times
            else:
                self.execution_data['step4_cegar_ar_times'] = json.dumps(ar_times)
            
            if isinstance(ar_sizes, str):
                self.execution_data['step4_cegar_ar_sizes'] = ar_sizes
            else:
                self.execution_data['step4_cegar_ar_sizes'] = json.dumps(ar_sizes)
            
            if isinstance(refine_times, str):
                self.execution_data['step4_cegar_refine_sequence_times'] = refine_times
            else:
                self.execution_data['step4_cegar_refine_sequence_times'] = json.dumps(refine_times)
            
            print(f" Logged Step 4: {step5_time:.4f}s (CEGAR: {cegar_dict.get('query_result', 'N/A')})")
        else:
            print(f" Logged Step 4: {step5_time:.4f}s (no CEGAR results)")
    
    def log_step3(self, step3_time: float) -> None:
        """
        Log Step 3 metrics.
        
        Args:
            step3_time: Time taken for Step 3
        """
        if step3_time == "N/A":
            self.execution_data['abstract_reuse_time'] = "N/A"
        else:
            self.execution_data['abstract_reuse_time'] = round(step3_time, 4)
        print(f" Logged Step 'abstract reuse': {step3_time:.4f}s")
    
    def log_step4(self, step4_time: float, verification_result: Optional[Dict[str, Any]] = None) -> None:
        """
        Log Step 4 metrics including verification results and refinement count.
        
        Args:
            step4_time: Time taken for Step 4
            verification_result: Dictionary containing verification results with keys like
                                'result', 'num_refine_steps', 'query_times', 'refine_times', etc.
        """
        if step4_time == "N/A":
            self.execution_data['transfer_abstract_verification_time'] = "N/A"
        self.execution_data['transfer_abstract_verification_time'] = round(step4_time, 4)
        
        if verification_result:
            # self.execution_data['step4_result'] = verification_result.get('result', 'N/A')
            self.execution_data['step3_refinement_steps'] = verification_result.get('num_refine_steps', 0)
            
            # Store timing arrays as JSON
            query_times = verification_result.get('query_times', [])
            refine_times = verification_result.get('refine_times', [])
            
            if isinstance(query_times, str):
                self.execution_data['step3_query_times'] = query_times
            else:
                self.execution_data['step3_query_times'] = json.dumps(query_times)
            
            if isinstance(refine_times, str):
                self.execution_data['step3_refine_times'] = refine_times
            else:
                self.execution_data['step3_refine_times'] = json.dumps(refine_times)
            
            print(f" Logged Step 3: {step4_time:.4f}s (Result: {verification_result.get('result', 'N/A')}, " 
                  f"Refinements: {verification_result.get('num_refine_steps', 0)})")
        else:
            print(f" Logged Step 3: {step4_time:.4f}s (no verification results)")

    
    def finish_execution(self, status: str = 'completed', notes: str = '') -> None:
        """
        Finish the execution session and write to CSV.
        
        Args:
            status: Status of execution ('completed', 'error', etc.)
            notes: Any additional notes
        """
        if self.start_time is None:
            print(" Execution was not started")
            return
        
        total_time = time.time() - self.start_time
        self.execution_data['total_execution_time'] = round(total_time, 4)
        self.execution_data['status'] = status
        self.execution_data['notes'] = notes
        
        # Write to CSV
        self._write_to_csv()
        
        print(f"\n{'='*70}")
        print(f"Execution completed: {self.execution_id}")
        print(f"  Total time: {total_time:.4f}s")
        print(f"  Status: {status}")
        print(f"  Log saved to: {self.csv_file_path}")
        print(f"{'='*70}\n")
    
    def _write_to_csv(self) -> None:
        """Write the current execution data to the CSV file."""
        try:
            # Initialize all fields with empty strings/None
            row = {field: '' for field in self.fieldnames}
            row.update(self.execution_data)
            
            # Append row to CSV
            with open(self.csv_file_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(row)
            
            print(f" Execution data written to CSV: {self.csv_file_path}")
        except Exception as e:
            print(f" Error writing to CSV: {str(e)}")
    
    def log_error(self, error_message: str) -> None:
        """
        Log an error and finish execution.
        
        Args:
            error_message: The error message
        """
        self.execution_data['notes'] = f"Error: {error_message}"
        self.finish_execution(status='error', notes=error_message)


def get_logger(csv_file_path: str = "transfer_execution_log.csv") -> TransferExecutionLogger:
    """
    Get a logger instance.
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        TransferExecutionLogger instance
    """
    return TransferExecutionLogger(csv_file_path)
