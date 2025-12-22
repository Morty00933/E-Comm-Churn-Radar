"""Optimized batch prediction with async processing and chunking."""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, AsyncIterator

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BatchJob:
    """Batch prediction job."""
    job_id: str
    status: str = "pending"  # pending, running, completed, failed
    total_records: int = 0
    processed_records: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None
    output_path: Optional[str] = None
    
    @property
    def progress(self) -> float:
        if self.total_records == 0:
            return 0.0
        return self.processed_records / self.total_records
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "progress": f"{self.progress:.1%}",
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "output_path": self.output_path,
        }


class BatchPredictor:
    """Async batch prediction processor.
    
    Features:
    - Chunked processing for large files
    - Progress tracking
    - Async execution
    - Memory-efficient streaming
    """
    
    def __init__(
        self,
        model_loader: Callable,
        feature_columns: List[str],
        chunk_size: int = 10000,
        output_dir: str = "data/predictions",
        max_concurrent_chunks: int = 4,
    ):
        self.model_loader = model_loader
        self.feature_columns = feature_columns
        self.chunk_size = chunk_size
        self.output_dir = Path(output_dir)
        self.max_concurrent_chunks = max_concurrent_chunks
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._jobs: Dict[str, BatchJob] = {}
        self._model = None
    
    @property
    def model(self):
        """Lazy model loading."""
        if self._model is None:
            self._model = self.model_loader()
        return self._model
    
    def create_job(self, input_path: str) -> BatchJob:
        """Create a new batch job."""
        import uuid
        
        job_id = str(uuid.uuid4())[:8]
        
        # Count records
        total_records = sum(1 for _ in open(input_path)) - 1  # Exclude header
        
        job = BatchJob(
            job_id=job_id,
            total_records=total_records,
        )
        
        self._jobs[job_id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get job status."""
        return self._jobs.get(job_id)
    
    async def _process_chunk(
        self,
        chunk: pd.DataFrame,
        chunk_idx: int,
    ) -> pd.DataFrame:
        """Process a single chunk."""
        # Prepare features
        X = chunk[self.feature_columns].fillna(0)
        
        # Predict
        probas = self.model.predict_proba(X)[:, 1]
        labels = (probas >= 0.5).astype(int)
        
        # Add predictions to chunk
        chunk = chunk.copy()
        chunk["churn_proba"] = probas
        chunk["churn_label"] = labels
        chunk["predicted_at"] = datetime.utcnow().isoformat()
        
        logger.debug(f"Processed chunk {chunk_idx}: {len(chunk)} records")
        
        return chunk
    
    async def _read_chunks(self, input_path: str) -> AsyncIterator[pd.DataFrame]:
        """Read file in chunks asynchronously."""
        for chunk in pd.read_csv(input_path, chunksize=self.chunk_size):
            yield chunk
    
    async def process_file(
        self,
        input_path: str,
        job: BatchJob,
        progress_callback: Optional[Callable[[BatchJob], None]] = None,
    ) -> str:
        """Process entire file with progress tracking."""
        job.status = "running"
        output_path = self.output_dir / f"predictions_{job.job_id}.csv"
        
        try:
            # Process chunks
            first_chunk = True
            chunk_idx = 0
            
            for chunk in pd.read_csv(input_path, chunksize=self.chunk_size):
                # Process chunk
                result = await self._process_chunk(chunk, chunk_idx)
                
                # Write to output
                result.to_csv(
                    output_path,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                    index=False,
                )
                first_chunk = False
                
                # Update progress
                job.processed_records += len(chunk)
                chunk_idx += 1
                
                if progress_callback:
                    progress_callback(job)
                
                # Yield control to event loop
                await asyncio.sleep(0)
            
            job.status = "completed"
            job.completed_at = datetime.utcnow().isoformat()
            job.output_path = str(output_path)
            
            logger.info(f"Batch job {job.job_id} completed: {job.processed_records} records")
            
            return str(output_path)
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"Batch job {job.job_id} failed: {e}")
            raise
    
    async def run_job(
        self,
        input_path: str,
        progress_callback: Optional[Callable[[BatchJob], None]] = None,
    ) -> BatchJob:
        """Create and run a batch job."""
        job = self.create_job(input_path)
        
        try:
            await self.process_file(input_path, job, progress_callback)
        except Exception:
            pass  # Error already logged in process_file
        
        return job
    
    def run_job_sync(
        self,
        input_path: str,
        progress_callback: Optional[Callable[[BatchJob], None]] = None,
    ) -> BatchJob:
        """Synchronous wrapper for run_job."""
        return asyncio.run(self.run_job(input_path, progress_callback))


class StreamingPredictor:
    """Memory-efficient streaming predictor for very large files."""
    
    def __init__(
        self,
        model_loader: Callable,
        feature_columns: List[str],
        batch_size: int = 1000,
    ):
        self.model_loader = model_loader
        self.feature_columns = feature_columns
        self.batch_size = batch_size
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            self._model = self.model_loader()
        return self._model
    
    def predict_stream(
        self,
        input_path: str,
        output_path: str,
    ) -> Dict[str, Any]:
        """Stream predictions to output file."""
        stats = {
            "total_records": 0,
            "churn_count": 0,
            "avg_proba": 0.0,
        }
        
        probas_sum = 0.0
        first_chunk = True
        
        for chunk in pd.read_csv(input_path, chunksize=self.batch_size):
            X = chunk[self.feature_columns].fillna(0)
            
            probas = self.model.predict_proba(X)[:, 1]
            labels = (probas >= 0.5).astype(int)
            
            chunk["churn_proba"] = probas
            chunk["churn_label"] = labels
            
            chunk.to_csv(
                output_path,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )
            first_chunk = False
            
            stats["total_records"] += len(chunk)
            stats["churn_count"] += labels.sum()
            probas_sum += probas.sum()
        
        if stats["total_records"] > 0:
            stats["avg_proba"] = probas_sum / stats["total_records"]
            stats["churn_rate"] = stats["churn_count"] / stats["total_records"]
        
        return stats


# Convenience function
def batch_predict(
    input_path: str,
    output_path: Optional[str] = None,
    chunk_size: int = 10000,
) -> Dict[str, Any]:
    """Run batch prediction on a file.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV (optional)
        chunk_size: Records per chunk
    
    Returns:
        Dict with stats and output path
    """
    from src.api.predictor import load_model, get_feature_columns
    
    if output_path is None:
        output_path = str(Path(input_path).with_suffix(".predictions.csv"))
    
    predictor = StreamingPredictor(
        model_loader=load_model,
        feature_columns=get_feature_columns(),
        batch_size=chunk_size,
    )
    
    stats = predictor.predict_stream(input_path, output_path)
    stats["output_path"] = output_path
    
    return stats


# CLI entry point
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch prediction")
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument("-o", "--output", help="Output CSV file")
    parser.add_argument("--chunk-size", type=int, default=10000)
    
    args = parser.parse_args()
    
    print(f"Processing {args.input}...")
    
    result = batch_predict(
        args.input,
        args.output,
        args.chunk_size,
    )
    
    print(f"\nCompleted!")
    print(f"  Records: {result['total_records']}")
    print(f"  Churn rate: {result.get('churn_rate', 0):.1%}")
    print(f"  Output: {result['output_path']}")


if __name__ == "__main__":
    main()
