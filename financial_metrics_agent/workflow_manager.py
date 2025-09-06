#!/usr/bin/env python3
"""
Workflow Manager for Financial Analysis

Coordinates the 8-step comprehensive workflow for Hong Kong financial analysis.
Manages step execution, error handling, and workflow state.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class WorkflowStep(Enum):
    """Enumeration of workflow steps."""
    TICKER_VALIDATION = "ticker_validation"
    DATABASE_CHECK = "database_check"
    WEB_SCRAPING = "web_scraping"
    PDF_VERIFICATION = "pdf_verification"
    PDF_DOWNLOAD = "pdf_download"
    DOCUMENT_CHUNKING = "document_chunking"
    EMBEDDING_GENERATION = "embedding_generation"
    REPORT_GENERATION = "report_generation"

@dataclass
class WorkflowStepResult:
    """Result of a workflow step execution."""
    step: WorkflowStep
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: Optional[datetime] = None

@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    max_retries: int = 3
    timeout_seconds: int = 300
    skip_on_error: bool = False
    parallel_execution: bool = False

class WorkflowState:
    """Manages the state of workflow execution."""
    
    def __init__(self, ticker: str, config: WorkflowConfig):
        self.ticker = ticker
        self.config = config
        self.start_time = time.time()
        self.steps: Dict[WorkflowStep, WorkflowStepResult] = {}
        self.current_step: Optional[WorkflowStep] = None
        self.completed_steps: List[WorkflowStep] = []
        self.failed_steps: List[WorkflowStep] = []
        self.skipped_steps: List[WorkflowStep] = []
        self.metadata: Dict[str, Any] = {}
    
    def start_step(self, step: WorkflowStep):
        """Mark the start of a workflow step."""
        self.current_step = step
        logger.info(f"🔄 Starting {step.value} for {self.ticker}")
    
    def complete_step(self, step: WorkflowStep, result: WorkflowStepResult):
        """Mark the completion of a workflow step."""
        self.steps[step] = result
        
        if result.success:
            self.completed_steps.append(step)
            logger.info(f"✅ Completed {step.value} for {self.ticker} in {result.execution_time:.2f}s")
        else:
            self.failed_steps.append(step)
            logger.error(f"❌ Failed {step.value} for {self.ticker}: {result.error}")
        
        self.current_step = None
    
    def skip_step(self, step: WorkflowStep, reason: str):
        """Mark a workflow step as skipped."""
        self.skipped_steps.append(step)
        logger.info(f"⏭️ Skipped {step.value} for {self.ticker}: {reason}")
    
    def get_total_time(self) -> float:
        """Get total workflow execution time."""
        return time.time() - self.start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get workflow execution summary."""
        return {
            "ticker": self.ticker,
            "total_time": self.get_total_time(),
            "completed_steps": len(self.completed_steps),
            "failed_steps": len(self.failed_steps),
            "skipped_steps": len(self.skipped_steps),
            "success_rate": len(self.completed_steps) / len(WorkflowStep) * 100,
            "steps": {step.value: asdict(result) for step, result in self.steps.items()},
            "metadata": self.metadata
        }

class FinancialWorkflowManager:
    """
    Manages the comprehensive 8-step financial analysis workflow.
    
    Coordinates step execution, handles errors, and maintains workflow state.
    """
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        """
        Initialize the workflow manager.
        
        Args:
            config: Workflow configuration options
        """
        self.config = config or WorkflowConfig()
        self.active_workflows: Dict[str, WorkflowState] = {}
        self.completed_workflows: List[WorkflowState] = []
        
        logger.info("✅ Financial workflow manager initialized")
    
    def create_workflow(self, ticker: str) -> WorkflowState:
        """
        Create a new workflow for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            WorkflowState instance
        """
        workflow = WorkflowState(ticker, self.config)
        self.active_workflows[ticker] = workflow
        
        logger.info(f"🚀 Created workflow for {ticker}")
        return workflow
    
    async def execute_step(self, workflow: WorkflowState, step: WorkflowStep, 
                          step_function, *args, **kwargs) -> WorkflowStepResult:
        """
        Execute a single workflow step with error handling and timing.
        
        Args:
            workflow: Workflow state
            step: Step to execute
            step_function: Function to execute for this step
            *args, **kwargs: Arguments for the step function
            
        Returns:
            WorkflowStepResult
        """
        workflow.start_step(step)
        start_time = time.time()
        
        try:
            # Execute the step function
            result_data = await step_function(*args, **kwargs)
            
            execution_time = time.time() - start_time
            result = WorkflowStepResult(
                step=step,
                success=True,
                data=result_data,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = WorkflowStepResult(
                step=step,
                success=False,
                error=str(e),
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
            logger.error(f"❌ Step {step.value} failed for {workflow.ticker}: {e}")
        
        workflow.complete_step(step, result)
        return result
    
    def should_skip_step(self, workflow: WorkflowState, step: WorkflowStep, 
                        dependencies: Optional[List[WorkflowStep]] = None) -> Optional[str]:
        """
        Determine if a step should be skipped based on dependencies and configuration.
        
        Args:
            workflow: Workflow state
            step: Step to check
            dependencies: Required previous steps
            
        Returns:
            Skip reason if step should be skipped, None otherwise
        """
        # Check if previous steps failed and skip_on_error is enabled
        if self.config.skip_on_error and workflow.failed_steps:
            return f"Previous steps failed: {[s.value for s in workflow.failed_steps]}"
        
        # Check dependencies
        if dependencies:
            missing_deps = [dep for dep in dependencies if dep not in workflow.completed_steps]
            if missing_deps:
                return f"Missing dependencies: {[d.value for d in missing_deps]}"
        
        return None
    
    def get_workflow_progress(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get progress information for a workflow.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Progress information or None if workflow not found
        """
        workflow = self.active_workflows.get(ticker)
        if not workflow:
            return None
        
        total_steps = len(WorkflowStep)
        completed = len(workflow.completed_steps)
        failed = len(workflow.failed_steps)
        skipped = len(workflow.skipped_steps)
        
        return {
            "ticker": ticker,
            "current_step": workflow.current_step.value if workflow.current_step else None,
            "progress_percent": (completed / total_steps) * 100,
            "completed_steps": completed,
            "failed_steps": failed,
            "skipped_steps": skipped,
            "total_steps": total_steps,
            "elapsed_time": workflow.get_total_time(),
            "status": "running" if workflow.current_step else "idle"
        }
    
    def complete_workflow(self, ticker: str) -> Optional[WorkflowState]:
        """
        Mark a workflow as completed and move it to completed list.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Completed workflow state or None if not found
        """
        workflow = self.active_workflows.pop(ticker, None)
        if workflow:
            self.completed_workflows.append(workflow)
            logger.info(f"✅ Completed workflow for {ticker} in {workflow.get_total_time():.2f}s")
        
        return workflow
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get overall workflow statistics."""
        total_workflows = len(self.completed_workflows)
        
        if total_workflows == 0:
            return {
                "total_workflows": 0,
                "average_execution_time": 0,
                "success_rate": 0,
                "step_statistics": {}
            }
        
        total_time = sum(w.get_total_time() for w in self.completed_workflows)
        successful_workflows = sum(1 for w in self.completed_workflows if not w.failed_steps)
        
        # Calculate step statistics
        step_stats = {}
        for step in WorkflowStep:
            step_results = [w.steps.get(step) for w in self.completed_workflows if step in w.steps]
            successful_steps = sum(1 for r in step_results if r and r.success)
            avg_time = sum(r.execution_time for r in step_results if r) / len(step_results) if step_results else 0
            
            step_stats[step.value] = {
                "total_executions": len(step_results),
                "successful_executions": successful_steps,
                "success_rate": (successful_steps / len(step_results) * 100) if step_results else 0,
                "average_execution_time": avg_time
            }
        
        return {
            "total_workflows": total_workflows,
            "successful_workflows": successful_workflows,
            "success_rate": (successful_workflows / total_workflows * 100),
            "average_execution_time": total_time / total_workflows,
            "active_workflows": len(self.active_workflows),
            "step_statistics": step_stats
        }
    
    def cleanup_old_workflows(self, max_completed: int = 100):
        """Clean up old completed workflows to prevent memory buildup."""
        if len(self.completed_workflows) > max_completed:
            removed = len(self.completed_workflows) - max_completed
            self.completed_workflows = self.completed_workflows[-max_completed:]
            logger.info(f"🧹 Cleaned up {removed} old workflow records")
