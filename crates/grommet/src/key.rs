//! Affine keys and the placement hash.

use std::hash::Hash;

/// The identity that work is made affine to: an account, an instrument, a
/// session, a partition. All work for one key is processed by one shard, in
/// submission order, one item at a time.
pub trait ShardKey: Copy + Eq + Hash + Send + 'static {
    /// A value derived from this key's identity. The router applies its own
    /// avalanche step afterwards, so returning the raw identity is both correct
    /// and the cheapest option — there is no need to hash here.
    fn shard_hash(&self) -> u64;
}

/// An idempotency key: the caller-stable identity of one logical operation,
/// which a retry reuses.
///
/// Any exact, hashable identity works — a `u64` sequence number, a ULID or UUID
/// as `u128`, a `(session, seq)` pair. It is deliberately not narrowed to one
/// representation, and deliberately not hashed down to a fixed width, because a
/// collision here would suppress work that was never a duplicate.
///
/// The runtime uses this only to coalesce duplicates that are *concurrently*
/// live for the same key. Durable deduplication — answering a retry that
/// arrives after the original completed, with the original's outcome — needs
/// the authoritative store and belongs in your processor.
pub trait RequestId: Clone + Eq + Hash + Send + 'static {}

impl<T: Clone + Eq + Hash + Send + 'static> RequestId for T {}

/// The MurmurHash3 64-bit finalizer (`fmix64`), applied by the router so that
/// structured or sequential keys still spread evenly across shards.
#[inline]
pub const fn mix(mut hash: u64) -> u64 {
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^= hash >> 33;
    hash
}

macro_rules! integer_keys {
    ($($ty:ty),* $(,)?) => {
        $(
            impl ShardKey for $ty {
                #[inline]
                fn shard_hash(&self) -> u64 {
                    *self as u64
                }
            }
        )*
    };
}

integer_keys!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

impl ShardKey for u128 {
    #[inline]
    fn shard_hash(&self) -> u64 {
        (*self as u64) ^ ((*self >> 64) as u64)
    }
}

impl ShardKey for i128 {
    #[inline]
    fn shard_hash(&self) -> u64 {
        (*self as u128).shard_hash()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn mixing_spreads_sequential_keys_across_buckets() {
        const BUCKETS: u64 = 16;
        let mut counts = HashMap::new();
        for key in 0..16_000u64 {
            *counts.entry(mix(key.shard_hash()) % BUCKETS).or_insert(0usize) += 1;
        }
        assert_eq!(counts.len(), BUCKETS as usize, "every bucket must be reachable");
        // A dense sequential range is the worst case for an unmixed identity
        // hash; after mixing no bucket should be far off its fair share.
        let fair = 16_000 / BUCKETS as usize;
        for count in counts.values() {
            assert!(*count > fair / 2 && *count < fair * 2, "bucket skew: {count} vs {fair}");
        }
    }

    #[test]
    fn integer_and_wide_keys_carry_their_identity() {
        assert_eq!(7u64.shard_hash(), 7);
        assert_eq!(7usize.shard_hash(), 7);
        assert_eq!((-1i32).shard_hash(), u64::from(u32::MAX) | 0xffff_ffff_0000_0000);
        assert_eq!(0u128.shard_hash(), 0);
        assert_eq!((u128::from(u64::MAX) + 1).shard_hash(), 1, "the high half folds in");
        assert_eq!(5i128.shard_hash(), 5);
    }

    #[test]
    fn mixing_is_deterministic_and_injective_on_a_sample() {
        let mut seen = HashMap::new();
        for key in 0..10_000u64 {
            assert_eq!(mix(key), mix(key));
            assert!(seen.insert(mix(key), key).is_none(), "collision at {key}");
        }
    }

    /// Known answers for `fmix64`, cross-checked against an independent
    /// implementation of the same finalizer.
    ///
    /// Distribution and injectivity are both too weak to pin this function: a
    /// mixer with a shift or an operator wrong still spreads a sequential range
    /// evenly and still avoids collisions on a small sample. Only exact outputs
    /// catch that, and the placement of every key depends on getting it right.
    #[test]
    fn the_mixer_reproduces_fmix64_exactly() {
        assert_eq!(mix(0), 0, "the finalizer fixes zero");
        assert_eq!(mix(1), 0xb456_bcfc_34c2_cb2c);
        assert_eq!(mix(2), 0x3abf_2a20_6506_83e7);
        assert_eq!(mix(42), 0x8108_7960_8e42_59cc);
        assert_eq!(mix(1000), 0xaf00_2c11_4878_da41);
        assert_eq!(mix(u64::MAX), 0x64b5_720b_4b82_5f21);
    }

    #[test]
    fn a_wide_key_folds_its_halves_together_rather_than_merging_their_bits() {
        // Low and high halves that share a set bit: xor drops it, or keeps it.
        // Without a case like this, folding with `|` looks the same as `^`.
        let key = (1u128 << 64) | 3;
        assert_eq!(key.shard_hash(), 2, "the halves are combined with xor");
    }
}
