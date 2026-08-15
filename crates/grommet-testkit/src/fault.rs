//! Deterministic fault injection for the adapters a processor depends on.
//!
//! A [`FaultPlan`] decides, at each labelled point in your adapters, whether
//! this call fails. The labels are yours: any small `Copy + Eq + Debug` type
//! works, typically an enum naming each place an external dependency can fail.
//!
//! The plan is shared by cheap clones and is deliberately `!Send`, matching the
//! shard model where adapters live on one core.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::fmt;
use std::rc::Rc;

/// A label for a place where a fault can be injected.
pub trait FaultPoint: Copy + Eq + fmt::Debug + 'static {}

impl<T: Copy + Eq + fmt::Debug + 'static> FaultPoint for T {}

enum Mode<P> {
    Off,
    Ordered(VecDeque<P>),
    Countdown(i64),
    Bytes { bytes: Vec<u8>, position: usize },
}

struct State<P> {
    mode: Mode<P>,
    enabled: Option<Vec<P>>,
    fired: Vec<P>,
    checks: usize,
}

pub struct FaultPlan<P: FaultPoint>(Rc<RefCell<State<P>>>);

impl<P: FaultPoint> Clone for FaultPlan<P> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<P: FaultPoint> Default for FaultPlan<P> {
    fn default() -> Self {
        Self::off()
    }
}

impl<P: FaultPoint> FaultPlan<P> {
    /// Never inject. Use this for the adapters a test is not exercising.
    pub fn off() -> Self {
        Self::with_mode(Mode::Off)
    }

    /// Fail exactly these points, in this order, one occurrence each.
    pub fn ordered(points: impl IntoIterator<Item = P>) -> Self {
        Self::with_mode(Mode::Ordered(points.into_iter().collect()))
    }

    /// Fail the `n`-th eligible operation and no other, where zero means the
    /// very next one.
    ///
    /// Sweeping `n = 0, 1, 2, …` until no fault fires visits every single
    /// failure position a deterministic workload can reach. See
    /// [`crate::conformance::single_fault_campaign`].
    pub fn countdown(n: i64) -> Self {
        Self::with_mode(Mode::Countdown(n))
    }

    /// Let a coverage-guided fuzzer steer which operations fail. Roughly one
    /// byte in eight injects, which keeps plenty of happy paths in the mix.
    pub fn from_fuzz_bytes(bytes: &[u8]) -> Self {
        Self::with_mode(Mode::Bytes { bytes: bytes.to_vec(), position: 0 })
    }

    /// Restrict injection to these points, leaving every other point healthy.
    pub fn only(self, points: &[P]) -> Self {
        self.0.borrow_mut().enabled = Some(points.to_vec());
        self
    }

    /// Ask whether this call should fail. Call it at each labelled point in
    /// your adapter, and fail that call when it returns true.
    pub fn fires(&self, point: P) -> bool {
        let mut state = self.0.borrow_mut();
        if state.enabled.as_ref().is_some_and(|enabled| !enabled.contains(&point)) {
            return false;
        }
        state.checks += 1;
        let fire = match &mut state.mode {
            Mode::Off => false,
            Mode::Ordered(points) if points.front() == Some(&point) => {
                points.pop_front();
                true
            }
            Mode::Ordered(_) => false,
            Mode::Countdown(remaining) if *remaining == 0 => {
                *remaining = -1;
                true
            }
            Mode::Countdown(remaining) if *remaining > 0 => {
                *remaining -= 1;
                false
            }
            Mode::Countdown(_) => false,
            Mode::Bytes { bytes, position } => {
                let byte = bytes.get(*position).copied();
                *position += 1;
                byte.is_some_and(|value| value.is_multiple_of(8))
            }
        };
        if fire {
            state.fired.push(point);
        }
        fire
    }

    /// Whether the plan has nothing left to inject. A test that expected a
    /// specific failure should assert this, so a plan that silently never fired
    /// cannot pass as a successful recovery.
    pub fn is_exhausted(&self) -> bool {
        let state = self.0.borrow();
        match &state.mode {
            Mode::Off => true,
            Mode::Ordered(points) => points.is_empty(),
            Mode::Countdown(_) => !state.fired.is_empty(),
            Mode::Bytes { bytes, position } => *position >= bytes.len(),
        }
    }

    pub fn did_fire(&self) -> bool {
        !self.0.borrow().fired.is_empty()
    }

    pub fn fired(&self) -> Vec<P> {
        self.0.borrow().fired.clone()
    }

    /// How many eligible points have been consulted. Useful for asserting a
    /// workload actually reaches the adapters you meant to test.
    pub fn checks(&self) -> usize {
        self.0.borrow().checks
    }

    fn with_mode(mode: Mode<P>) -> Self {
        Self(Rc::new(RefCell::new(State { mode, enabled: None, fired: Vec::new(), checks: 0 })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Point {
        Load,
        Commit,
        Cache,
    }

    #[test]
    fn an_off_plan_never_fires_and_is_immediately_exhausted() {
        let plan = FaultPlan::<Point>::off();
        assert!(!plan.fires(Point::Load));
        assert!(plan.is_exhausted() && !plan.did_fire());
        assert_eq!(plan.checks(), 1);
    }

    #[test]
    fn an_ordered_plan_fires_each_point_once_in_sequence() {
        let plan = FaultPlan::ordered([Point::Load, Point::Commit]);
        assert!(!plan.fires(Point::Commit), "out of sequence points are healthy");
        assert!(plan.fires(Point::Load));
        assert!(!plan.fires(Point::Load), "each listed occurrence fires once");
        assert!(plan.fires(Point::Commit));
        assert!(plan.is_exhausted());
        assert_eq!(plan.fired(), vec![Point::Load, Point::Commit]);
    }

    #[test]
    fn a_countdown_plan_fires_exactly_the_nth_eligible_operation() {
        let plan = FaultPlan::countdown(2);
        assert!(!plan.fires(Point::Load));
        assert!(!plan.fires(Point::Load));
        assert!(plan.fires(Point::Commit), "the third eligible call fails");
        assert!(!plan.fires(Point::Commit), "and the plan is then inert");
        assert!(plan.did_fire() && plan.is_exhausted());
    }

    #[test]
    fn restricting_points_leaves_every_other_adapter_healthy() {
        let plan = FaultPlan::countdown(0).only(&[Point::Commit]);
        assert!(!plan.fires(Point::Load), "an excluded point is not even counted");
        assert!(!plan.fires(Point::Cache));
        assert_eq!(plan.checks(), 0);
        assert!(plan.fires(Point::Commit));
    }

    #[test]
    fn fuzz_bytes_steer_injection_and_exhaust_with_the_input() {
        let plan = FaultPlan::<Point>::from_fuzz_bytes(&[1, 8, 3]);
        assert!(!plan.fires(Point::Load));
        assert!(plan.fires(Point::Commit), "a byte divisible by eight injects");
        assert!(!plan.fires(Point::Cache));
        assert!(plan.is_exhausted(), "the input is consumed");
        assert!(!plan.fires(Point::Load), "and injection stops with it");
    }

    #[test]
    fn clones_share_one_plan() {
        let plan = FaultPlan::ordered([Point::Load]);
        let handle = plan.clone();
        assert!(handle.fires(Point::Load));
        assert!(plan.is_exhausted(), "a clone's firing is visible through the original");
    }
}
