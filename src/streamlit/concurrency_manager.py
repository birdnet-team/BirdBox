"""
Concurrency control for Streamlit app.

This module provides:
- Global semaphore to limit concurrent detections
- Waiting pool for users who can't acquire a slot immediately
"""

import threading
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrency control."""
    max_concurrent_detections: int = config.MAX_CONCURRENT_DETECTIONS


class ConcurrencyManager:
    """
    Manages concurrent detections to prevent server overload.
    
    Uses semaphore-based concurrency control to limit simultaneous detections.
    
    Thread-safe implementation using locks and semaphores.
    """
    
    def __init__(self, config: ConcurrencyConfig = None):
        """
        Initialize concurrency manager.
        
        Args:
            config: ConcurrencyConfig instance, or None for defaults
        """
        self.config = config or ConcurrencyConfig()
        
        # Semaphore to limit concurrent detections
        self._detection_semaphore = threading.Semaphore(self.config.max_concurrent_detections)
        
        # Lock for thread-safe access to shared data
        self._lock = threading.Lock()
        
        # Track users waiting in pool (set for O(1) lookups)
        self._waiting_pool: set = set()  # Set of session_ids waiting for a slot
        
        # Track active detections (session_ids currently processing)
        self._active_detections: set = set()
    
    def can_start_detection(self, session_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a session can start a new detection.
        
        This method does NOT acquire the semaphore - it only checks availability.
        The semaphore should be acquired via start_detection().
        
        Args:
            session_id: Unique session identifier (e.g., Streamlit session ID)
            
        Returns:
            Tuple of (can_start: bool, reason: Optional[str])
            If can_start is False, reason explains why (already active, no slots available, etc.)
        """
        # Acquire lock for all checks
        with self._lock:
            # Check if already in waiting pool
            if session_id in self._waiting_pool:
                return False, "Server is busy. Try again later."
            
            # Check if already processing
            if session_id in self._active_detections:
                return False, "You already have a detection in progress."
            
            # Check if semaphore is available (don't acquire it yet - that happens in start_detection)
            # Use count-based check instead of trying to acquire (which might block)
            available_slots = self.config.max_concurrent_detections - len(self._active_detections)
            
            if available_slots > 0:
                # Slot available - can start (but don't acquire semaphore yet)
                return True, None
            else:
                # No slots available - add to waiting pool
                self._waiting_pool.add(session_id)
                return False, "Server is busy. Try again later."
    
    def start_detection(self, session_id: str) -> bool:
        """
        Start a detection (should be called after can_start_detection returns True).
        
        This will block until a slot is available if the session is in queue.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if detection started, False if cancelled/timed out
        """
        # Check if already active first (with lock)
        with self._lock:
            if session_id in self._active_detections:
                return True
        
        # Check if in waiting pool and remove if so
        was_in_pool = False
        with self._lock:
            if session_id in self._waiting_pool:
                was_in_pool = True
                self._waiting_pool.remove(session_id)
        
        # Acquire semaphore
        # If user was in pool, they should wait briefly for a slot to become available
        # Otherwise, try non-blocking first
        if was_in_pool:
            # User was in pool - use a short timeout to wait for slot
            timeout = 0.1  # 100ms wait for users in pool
            acquired = self._detection_semaphore.acquire(timeout=timeout)
        else:
            # Not in pool - try non-blocking first
            acquired = self._detection_semaphore.acquire(blocking=False)
            
            if not acquired:
                # Not immediately available - try with short timeout
                timeout = 0.05  # 50ms max wait
                acquired = self._detection_semaphore.acquire(timeout=timeout)
        
        if not acquired:
            # Still not available - add back to pool if we removed them
            if was_in_pool:
                # Add back to waiting pool
                with self._lock:
                    if session_id not in self._waiting_pool:
                        self._waiting_pool.add(session_id)
            return False
        
        # Mark as active
        with self._lock:
            self._active_detections.add(session_id)
        
        return True
    
    def remove_from_waiting_pool(self, session_id: str):
        """
        Remove a user from the waiting pool without finishing a detection.
        Use this when user cancels or removes their file.
        
        Args:
            session_id: Unique session identifier
        """
        with self._lock:
            if session_id in self._waiting_pool:
                self._waiting_pool.remove(session_id)
    
    def finish_detection(self, session_id: str):
        """
        Mark a detection as finished and release resources.
        Also removes user from waiting pool if they're in it.
        
        Args:
            session_id: Unique session identifier
        """
        with self._lock:
            was_active = session_id in self._active_detections
            if was_active:
                self._active_detections.remove(session_id)
            
            # Remove from waiting pool if somehow still there
            was_in_pool = session_id in self._waiting_pool
            if was_in_pool:
                self._waiting_pool.remove(session_id)
        
        # Release semaphore only if we were actually active
        if was_active:
            self._detection_semaphore.release()
    
    def get_status(self, session_id: str) -> Dict:
        """
        Get current status for a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Dictionary with status information
        """
        with self._lock:
            is_active = session_id in self._active_detections
            is_waiting = session_id in self._waiting_pool
            
            status = {
                'is_active': is_active,
                'is_waiting': is_waiting,
                'active_detections': len(self._active_detections),
                'waiting_pool_size': len(self._waiting_pool),
                'max_concurrent': self.config.max_concurrent_detections,
            }
            
            return status


# Global concurrency manager instance (shared across all sessions)
_global_concurrency_manager: Optional[ConcurrencyManager] = None
_concurrency_manager_lock = threading.Lock()


def get_concurrency_manager(config: ConcurrencyConfig = None) -> ConcurrencyManager:
    """
    Get or create the global concurrency manager instance.
    
    Args:
        config: Optional configuration. Only used on first call.
        
    Returns:
        Global ConcurrencyManager instance
    """
    global _global_concurrency_manager
    
    with _concurrency_manager_lock:
        if _global_concurrency_manager is None:
            _global_concurrency_manager = ConcurrencyManager(config)
        return _global_concurrency_manager


def reset_concurrency_manager():
    """Reset the global concurrency manager (useful for testing)."""
    global _global_concurrency_manager
    
    with _concurrency_manager_lock:
        _global_concurrency_manager = None

