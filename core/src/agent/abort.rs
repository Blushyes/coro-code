//! Abort (cancellation) controller for AgentCore

#[derive(Clone)]
pub struct AbortController {
    tx: tokio::sync::watch::Sender<bool>,
}

#[derive(Clone)]
pub struct AbortRegistration {
    rx: tokio::sync::watch::Receiver<bool>,
}

impl AbortController {
    /// Create a new controller and its registration
    pub fn new() -> (Self, AbortRegistration) {
        let (tx, rx) = tokio::sync::watch::channel(false);
        (Self { tx: tx.clone() }, AbortRegistration { rx })
    }

    /// Subscribe to this controller to obtain a registration
    pub fn subscribe(&self) -> AbortRegistration {
        AbortRegistration {
            rx: self.tx.subscribe(),
        }
    }

    /// Trigger cancellation (idempotent)
    pub fn cancel(&self) {
        let result = self.tx.send(true);
        // We can't use output handler here as AbortController doesn't have access to it
        // This is fine as this is the lowest level and should work silently
        let _ = result; // Suppress unused variable warning
    }
}

impl AbortRegistration {
    /// Check current cancellation state without blocking
    pub fn is_cancelled(&self) -> bool {
        *self.rx.borrow()
    }

    /// Wait until cancellation is triggered (returns immediately if already cancelled)
    pub async fn cancelled(&mut self) {
        if !*self.rx.borrow() {
            let _ = self.rx.changed().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::{sleep, timeout};

    #[tokio::test]
    async fn test_abort_controller_basic() {
        let (controller, mut registration) = AbortController::new();

        // Initially not cancelled
        assert!(!registration.is_cancelled());

        // Cancel the controller
        controller.cancel();

        // Should be cancelled now
        assert!(registration.is_cancelled());

        // cancelled() should return immediately
        let result = timeout(Duration::from_millis(100), registration.cancelled()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_abort_controller_multiple_subscribers() {
        let (controller, mut reg1) = AbortController::new();
        let mut reg2 = controller.subscribe();

        // Both should not be cancelled initially
        assert!(!reg1.is_cancelled());
        assert!(!reg2.is_cancelled());

        // Cancel the controller
        controller.cancel();

        // Both should be cancelled
        assert!(reg1.is_cancelled());
        assert!(reg2.is_cancelled());

        // Both should return immediately from cancelled()
        let result1 = timeout(Duration::from_millis(100), reg1.cancelled()).await;
        let result2 = timeout(Duration::from_millis(100), reg2.cancelled()).await;
        assert!(result1.is_ok());
        assert!(result2.is_ok());
    }

    #[tokio::test]
    async fn test_abort_controller_task_interruption() {
        // Case 1: No cancel → should complete other branch quickly
        let (_controller, mut registration) = AbortController::new();
        let result = tokio::select! {
            _ = registration.cancelled() => "interrupted",
            _ = sleep(Duration::from_millis(20)) => "completed",
        };
        assert_eq!(result, "completed");

        // Case 2: Cancel after short delay → should interrupt
        let (controller2, mut registration2) = AbortController::new();
        let controller2_clone = controller2.clone();
        tokio::spawn(async move {
            sleep(Duration::from_millis(30)).await;
            controller2_clone.cancel();
        });
        let result2 = tokio::select! {
            _ = registration2.cancelled() => "interrupted",
            _ = sleep(Duration::from_secs(2)) => "completed",
        };
        assert_eq!(result2, "interrupted");
    }

    #[tokio::test]
    async fn test_abort_controller_idempotent_cancel() {
        let (controller, mut registration) = AbortController::new();

        // Cancel multiple times
        controller.cancel();
        controller.cancel();
        controller.cancel();

        // Should still work correctly
        assert!(registration.is_cancelled());

        let result = timeout(Duration::from_millis(100), registration.cancelled()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_abort_controller_clone() {
        let (controller, mut registration) = AbortController::new();
        let controller_clone = controller.clone();

        // Clone should work the same way
        controller_clone.cancel();

        assert!(registration.is_cancelled());

        let result = timeout(Duration::from_millis(100), registration.cancelled()).await;
        assert!(result.is_ok());
    }
}
