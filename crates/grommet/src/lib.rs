//! A hardware-aware work scheduler: key-affine dispatch and CPU-bound offload,
//! hosted on the async runtime you already use.
//!
//! Work carries an affine key. Every item for one key is handled by one shard,
//! in submission order, one at a time, so the state behind that key needs no
//! locking and no atomics: while an item is being processed it holds the only
//! copy. Shards are pinned to cores chosen from the machine's own topology.
//!
//! Grommet does not drive futures. Polling IO, waking a socket and owning the
//! thread belong to a host runtime, selected by a Cargo feature: `driver-tokio`
//! gives each shard thread its own current-thread tokio runtime. A processor
//! keeps using the database and HTTP clients it already has, and grommet adds
//! affinity, fairness and offload above them.
//!
//! Within a shard, keys are dispatched round-robin from per-class ready rings,
//! which bounds starvation strictly rather than statistically: a key at
//! position `k` runs within `k` dispatches, no matter how much work a busier
//! key has queued. Each class has its own in-flight budget, so saturating one
//! class: CPU-bound work, a slow dependency: cannot starve another.
//!
//! # What you provide
//!
//! - [`Work`]: an item, its affine [`ShardKey`], its class, and optionally how
//!   long it stays worth doing.
//! - [`Processor`]: what to do with an item, given the key's resident state.
//! - Optionally an [`Offload`] pool for CPU-bound work, so a long computation
//!   never stalls the shard core that submitted it.
//!
//! ```no_run
//! use std::convert::Infallible;
//! use std::time::Duration;
//! use grommet::{Call, ClassId, Disposition, IO, Processor, Scheduler, Work};
//!
//! struct Job { account: u64, amount: i64, attempt: u128 }
//!
//! impl Work for Job {
//!     type Key = u64;
//!     type Id = u128;
//!     fn key(&self) -> u64 { self.account }
//!     fn class(&self) -> ClassId { IO }
//!     fn request_id(&self) -> Option<u128> { Some(self.attempt) }
//!     fn time_to_live(&self) -> Option<Duration> { Some(Duration::from_millis(50)) }
//! }
//!
//! #[derive(Clone)]
//! struct Ledger;
//!
//! impl Processor for Ledger {
//!     // Wrapping in `Call` attaches a reply channel to each item.
//!     type Work = Call<Job, i64>;
//!     type State = i64;
//!     type Error = Infallible;
//!
//!     async fn process(
//!         &self,
//!         _key: u64,
//!         balance: Option<i64>,
//!         call: Call<Job, i64>,
//!     ) -> Result<Disposition<i64>, Infallible> {
//!         let (job, responder) = call.into_parts();
//!         let balance = balance.unwrap_or(0) + job.amount;
//!         responder.send(balance);
//!         Ok(Disposition::Keep(balance))
//!     }
//! }
//!
//! # async fn run() {
//! // The clock defaults to `SystemClock` and the class count to the IO +
//! // COMPUTE split, so the common case names neither.
//! let runtime = Scheduler::<Ledger>::builder(4, [2048, 64])
//!     .spawn(|_shard| Ledger)
//!     .expect("start shards");
//!
//! let balance = runtime.router().call(Job { account: 7, amount: 100, attempt: 1 }).await;
//! # }
//! ```
//!
//! # Replies are opt-in
//!
//! Submission itself is one-way: it reports whether work was accepted, not what
//! it produced. Wrapping work in a [`Call`] as above adds a reply channel and
//! gives you [`Router::call`], which is what most request/response services
//! want.
//!
//! It is a wrapper rather than a built-in because a reply channel costs a heap
//! allocation and two atomics per item, and plenty of workloads have no caller
//! to answer: ingestion and feed handling, processors that reply by writing to
//! their own socket, and anything that wants to answer a batch of items with a
//! single syscall. Those keep the steady state allocation-free by submitting
//! plain [`Work`].
//!
//! # `!Send` on purpose
//!
//! Work is `Send`, because it crosses once from the submitter to its shard.
//! Nothing after that is: per-key state, processor futures and anything held
//! across an await stay on one core. That is what makes `Rc` and `Cell` correct
//! here, and it is also a real constraint: code written against `Send` futures
//! and work stealing will not fit. If you want that, use an ordinary
//! multi-threaded executor; this crate is deliberately the other thing.

#![deny(unsafe_code)]

pub mod clock;
pub(crate) mod doorbell;
pub(crate) mod driver;
pub mod error;
pub mod key;
pub mod mailbox;
pub mod metrics;
pub(crate) mod outstanding;

pub mod offload;
pub mod processor;
pub mod respond;
pub mod router;
pub mod scheduler;
pub mod shard;
pub mod topology;
pub(crate) mod waiters;
pub mod work;

pub use clock::{Clock, ManualClock, SystemClock};
pub use error::{Fallout, ProcessError};
pub use key::{RequestId, ShardKey, mix};
pub use mailbox::{Inbox, Mailbox, channel};
pub use offload::{InlineOffload, Offload, OffloadError};
pub use processor::{KeyOf, PanicPolicy, Processor};
pub use respond::{Answer, Call, CallError, Cancelled, Responder};
pub use router::{BatchError, Router, SubmitError};
pub use scheduler::{BuildError, Builder, Scheduler, ShardContext};
pub use shard::ShardConfig;
pub use topology::{PinPolicy, Plan, ShardPlacement, TopologyReport, Workload};
pub use work::{CLASSES, COMPUTE, Envelope, IO, Work};

pub use grommet_core::{ClassId, Config as DispatchConfig, Disposition, Snapshot};
