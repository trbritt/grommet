//! Per-key FIFOs as intrusive lists over one shard-wide slab.
//!
//! Every queued item lives in a single arena; a key owns nothing but a head
//! index, a tail index and a length. This deliberately assumes nothing about
//! how depth is distributed across keys. A shard holding a hundred thousand
//! idle keys pays twelve bytes each and no allocation, and a key holding a
//! hundred thousand items pays the same twelve bytes and no allocation, so a
//! workload where a few keys carry most of the traffic costs no more than a
//! uniform one.
//!
//! The slab only ever grows to the peak number of simultaneously queued items,
//! which admission already bounds by `max_pending`. Reserving that much up
//! front makes the steady state allocation-free.
//!
//! # Safety invariant
//!
//! This is the only module in the crate that indexes without bounds checks, on
//! the strength of one invariant:
//!
//! > Every non-`NIL` index reachable from a [`List`] or from the free list was
//! > returned by [`Slab::alloc`] for this same slab, and is therefore less than
//! > `nodes.len()`.
//!
//! It holds because `alloc` returns either an index it took off the free list
//! (which was itself allocated earlier) or `nodes.len()` immediately before
//! pushing, and because `nodes` is never shortened — there is no `remove`,
//! `truncate`, `clear` or `shrink` anywhere in this file, so an index that was
//! valid once stays valid for the life of the slab. Freed nodes are recycled
//! through the free list rather than removed.
//!
//! A [`List`] is meaningful only against the slab that allocated it. Both types
//! are crate-private and the scheduler owns exactly one slab alongside the
//! slots holding its lists, so a list can never reach a foreign slab.
//!
//! Every unchecked access is paired with a `debug_assert!`, so ordinary test,
//! simulation and Miri runs check what release builds assume.
#![allow(unsafe_code)]

const NIL: u32 = u32::MAX;

struct Node<T> {
    item: Option<T>,
    /// Next queued item for the owning key, or the next free node when this
    /// node is on the free list.
    next: u32,
}

/// One key's FIFO. Cheap to copy and independent of the item type, so the
/// per-key slot stays small no matter how large a work item is.
#[derive(Clone, Copy, Debug)]
pub(crate) struct List {
    head: u32,
    tail: u32,
    len: u32,
}

impl Default for List {
    fn default() -> Self {
        Self { head: NIL, tail: NIL, len: 0 }
    }
}

impl List {
    pub(crate) fn len(&self) -> usize {
        self.len as usize
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// The shard-wide arena backing every key's FIFO.
pub(crate) struct Slab<T> {
    nodes: Vec<Node<T>>,
    free: u32,
}

impl<T> Slab<T> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self { nodes: Vec::with_capacity(capacity), free: NIL }
    }

    /// Nodes currently held by the arena, freed ones included. The arena never
    /// shrinks, so this is the high-water mark of simultaneously queued items.
    pub(crate) fn capacity(&self) -> usize {
        self.nodes.len()
    }

    /// # Safety
    ///
    /// `index` must satisfy the module's safety invariant.
    #[inline]
    unsafe fn node(&self, index: u32) -> &Node<T> {
        debug_assert!((index as usize) < self.nodes.len(), "slab read out of range: {index}");
        // SAFETY: the caller guarantees `index` was allocated from this slab,
        // and the node vector never shrinks.
        unsafe { self.nodes.get_unchecked(index as usize) }
    }

    /// # Safety
    ///
    /// `index` must satisfy the module's safety invariant.
    #[inline]
    unsafe fn node_mut(&mut self, index: u32) -> &mut Node<T> {
        debug_assert!((index as usize) < self.nodes.len(), "slab write out of range: {index}");
        // SAFETY: as above; `&mut self` makes this the only live reference.
        unsafe { self.nodes.get_unchecked_mut(index as usize) }
    }

    fn alloc(&mut self, item: T) -> u32 {
        let index = self.free;
        if index != NIL {
            // SAFETY: the free list only ever holds previously allocated
            // indices, so `index` is in range.
            let node = unsafe { self.node_mut(index) };
            let next_free = node.next;
            node.item = Some(item);
            node.next = NIL;
            self.free = next_free;
            return index;
        }
        let index = self.nodes.len() as u32;
        assert_ne!(index, NIL, "queue slab exceeded {NIL} live items");
        self.nodes.push(Node { item: Some(item), next: NIL });
        index
    }

    pub(crate) fn push_back(&mut self, list: &mut List, item: T) {
        let index = self.alloc(item);
        if list.tail == NIL {
            list.head = index;
        } else {
            // SAFETY: a non-NIL tail was allocated from this slab.
            unsafe { self.node_mut(list.tail) }.next = index;
        }
        list.tail = index;
        list.len += 1;
    }

    pub(crate) fn pop_front(&mut self, list: &mut List) -> Option<T> {
        let index = list.head;
        if index == NIL {
            return None;
        }
        let free = self.free;
        // SAFETY: a non-NIL head was allocated from this slab.
        let node = unsafe { self.node_mut(index) };
        let item = node.item.take();
        let next = node.next;
        node.next = free;
        self.free = index;
        list.head = next;
        if next == NIL {
            list.tail = NIL;
        }
        list.len -= 1;
        debug_assert!(item.is_some(), "a linked node always holds an item");
        item
    }

    pub(crate) fn front(&self, list: &List) -> Option<&T> {
        if list.head == NIL {
            return None;
        }
        // SAFETY: a non-NIL head was allocated from this slab.
        unsafe { self.node(list.head) }.item.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[test]
    fn each_list_keeps_its_own_fifo_order_while_sharing_one_slab() {
        let mut slab = Slab::with_capacity(0);
        let mut hot = List::default();
        let mut cold = List::default();

        // Interleave a deep queue with a shallow one: the deep key must not
        // disturb the shallow key's ordering or vice versa.
        for item in 0..6 {
            slab.push_back(&mut hot, item);
            if item == 3 {
                slab.push_back(&mut cold, 100);
            }
        }
        assert_eq!((hot.len(), cold.len()), (6, 1));
        assert_eq!(slab.front(&hot), Some(&0));
        assert_eq!(slab.front(&cold), Some(&100));

        let mut drained = Vec::new();
        while let Some(item) = slab.pop_front(&mut hot) {
            drained.push(item);
        }
        assert_eq!(drained, vec![0, 1, 2, 3, 4, 5]);
        assert!(hot.is_empty() && slab.front(&hot).is_none());
        assert_eq!(slab.pop_front(&mut cold), Some(100));
        assert_eq!(slab.pop_front(&mut cold), None);
    }

    #[test]
    fn freed_nodes_are_reused_so_the_slab_tracks_peak_depth_not_throughput() {
        let mut slab = Slab::with_capacity(0);
        let mut list = List::default();
        for item in 0..4 {
            slab.push_back(&mut list, item);
        }
        assert_eq!(slab.capacity(), 4);

        for _ in 0..4 {
            slab.pop_front(&mut list);
        }
        // Ten thousand more items through the same key must not grow the arena
        // beyond the peak depth that was actually live at once.
        for round in 0..10_000 {
            slab.push_back(&mut list, round);
            assert_eq!(slab.pop_front(&mut list), Some(round));
        }
        assert_eq!(slab.capacity(), 4);
    }

    #[test]
    fn refilling_an_emptied_list_relinks_both_ends() {
        let mut slab = Slab::with_capacity(2);
        let mut list = List::default();
        slab.push_back(&mut list, 1);
        assert_eq!(slab.pop_front(&mut list), Some(1));
        assert!(list.is_empty());

        slab.push_back(&mut list, 2);
        slab.push_back(&mut list, 3);
        assert_eq!(list.len(), 2);
        assert_eq!(slab.pop_front(&mut list), Some(2));
        assert_eq!(slab.pop_front(&mut list), Some(3));
        assert_eq!(slab.pop_front(&mut list), None);
    }

    /// The unchecked indexing above is only as good as the invariant that every
    /// reachable index was allocated from this slab. This drives many
    /// interleaved lists against independent reference queues, so a linking
    /// mistake surfaces as a wrong value here and as undefined behaviour under
    /// Miri, which runs this same test.
    #[test]
    fn randomized_interleaving_matches_independent_reference_queues() {
        const LISTS: usize = 8;
        let steps = if cfg!(miri) { 2_000 } else { 200_000 };

        let mut slab = Slab::with_capacity(0);
        let mut lists = [List::default(); LISTS];
        let mut model: [VecDeque<u64>; LISTS] = std::array::from_fn(|_| VecDeque::new());
        let mut live = 0usize;
        let mut peak = 0usize;
        let mut state = 0x2545_f491_4f6c_dd1du64;

        for step in 0..steps {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let which = (state >> 3) as usize % LISTS;
            // Biased towards pushing so queues actually get deep, which is the
            // case a per-key inline container would have handled worst.
            if state & 0b11 != 0 {
                slab.push_back(&mut lists[which], step);
                model[which].push_back(step);
                live += 1;
                peak = peak.max(live);
            } else {
                let popped = slab.pop_front(&mut lists[which]);
                assert_eq!(popped, model[which].pop_front(), "divergence at step {step}");
                live -= usize::from(popped.is_some());
            }
            assert_eq!(slab.front(&lists[which]), model[which].front());
            assert_eq!(lists[which].len(), model[which].len());
        }

        for (list, reference) in lists.iter_mut().zip(&mut model) {
            while let Some(expected) = reference.pop_front() {
                assert_eq!(slab.pop_front(list), Some(expected));
            }
            assert_eq!(slab.pop_front(list), None);
        }
        assert_eq!(slab.capacity(), peak, "the arena must not exceed peak live depth");
    }
}
