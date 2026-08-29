//! The one `UnsafeCell` shape, so that loom can check both users of it.
//!
//! `loom::cell::UnsafeCell` and `std::cell::UnsafeCell` do not have the same
//! API: loom hands out access through a closure so that it can bracket the
//! access, notice two overlapping ones and fail the model. `std` hands out a
//! raw pointer and checks nothing. Presenting loom's shape on both means the
//! code that uses a cell is written once and is the code loom checks, rather
//! than a `cfg`-selected variant of it.
//!
//! Deliberately the smallest surface that serves its users: one constructor
//! and one accessor. A shared read would need `with`, and nothing here does.
//! Every access is a move in or a move out, under a lock the caller holds.
#![allow(unsafe_code)]

/// A cell whose accesses loom can see.
#[derive(Debug)]
pub(crate) struct UnsafeCell<T>(Inner<T>);

#[cfg(not(loom))]
type Inner<T> = std::cell::UnsafeCell<T>;
#[cfg(loom)]
type Inner<T> = loom::cell::UnsafeCell<T>;

impl<T> UnsafeCell<T> {
    pub(crate) fn new(data: T) -> Self {
        Self(Inner::new(data))
    }

    /// Access the contents exclusively.
    ///
    /// # Safety
    ///
    /// The caller must hold whatever lock its module uses to serialize access
    /// to this cell, and no other access may overlap this one. Each caller
    /// states which lock that is at the point of the call.
    ///
    /// Under `--cfg loom` this is additionally checked at runtime: two
    /// overlapping accesses fail the model rather than silently being
    /// undefined behaviour, which is the reason this shape exists.
    #[inline]
    pub(crate) unsafe fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
        #[cfg(not(loom))]
        {
            f(self.0.get())
        }
        #[cfg(loom)]
        {
            self.0.with_mut(f)
        }
    }
}
