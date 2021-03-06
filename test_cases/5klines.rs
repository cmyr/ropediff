// Copyright 2016 The xi-editor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! An engine for handling edits (possibly from async sources) and undo. It
//! conceptually represents the current text and all edit history for that
//! text.
//!
//! This module actually implements a mini Conflict-free Replicated Data Type
//! under `Engine::edit_rev`, which is considerably simpler than the usual
//! CRDT implementation techniques, because all operations are serialized in
//! this central engine. It provides the ability to apply edits that depend on
//! a previously committed version of the text rather than the current text,
//! which is sufficient for asynchronous plugins that can only have one
//! pending edit in flight each.
//!
//! There is also a full CRDT merge operation implemented under
//! `Engine::merge`, which is more powerful but considerably more complex.
//! It enables support for full asynchronous and even peer-to-peer editing.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::collections::hash_map::DefaultHasher;
use std;

use rope::{Rope, RopeInfo};
use multiset::{Subset, CountMatcher};
use interval::Interval;
use delta::{Delta, InsertDelta};

/// Represents the current state of a document and all of its history
#[derive(Serialize, Deserialize, Debug)]
pub struct Engine {
    /// The session ID used to create new `RevId`s for edits made on this device
    #[serde(default = "default_session", skip_serializing)]
    session: SessionId,
    /// The incrementing revision number counter for this session used for `RevId`s
    #[serde(default = "initial_revision_counter", skip_serializing)]
    rev_id_counter: u32,
    /// The current contents of the document as would be displayed on screen
    text: Rope,
    /// Storage for all the characters that have been deleted  but could
    /// return if a delete is un-done or an insert is re- done.
    tombstones: Rope,
    /// Imagine a "union string" that contained all the characters ever
    /// inserted, including the ones that were later deleted, in the locations
    /// they would be if they hadn't been deleted.
    ///
    /// This is a `Subset` of the "union string" representing the characters
    /// that are currently deleted, and thus in `tombstones` rather than
    /// `text`. The count of a character in `deletes_from_union` represents
    /// how many times it has been deleted, so if a character is deleted twice
    /// concurrently it will have count `2` so that undoing one delete but not
    /// the other doesn't make it re-appear.
    ///
    /// You could construct the "union string" from `text`, `tombstones` and
    /// `deletes_from_union` by splicing a segment of `tombstones` into `text`
    /// wherever there's a non-zero-count segment in `deletes_from_union`.
    deletes_from_union: Subset,
    // TODO: switch to a persistent Set representation to avoid O(n) copying
    undone_groups: BTreeSet<usize>,  // set of undo_group id's
    /// The revision history of the document
    revs: Vec<Revision>,
}

// The advantage of using a session ID over random numbers is that it can be
// easily delta-compressed later.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct RevId {
    // 96 bits has a 10^(-12) chance of collision with 400 million sessions and 10^(-6) with 100 billion.
    // `session1==session2==0` is reserved for initialization which is the same on all sessions.
    // A colliding session will break merge invariants and the document will start crashing Xi.
    session1: u64,
    // if this was a tuple field instead of two fields, alignment padding would add 8 more bytes.
    session2: u32,
    // There will probably never be a document with more than 4 billion edits
    // in a single session.
    num: u32,
}

#[derive(Serialize, Deserialize, Debug)]
struct Revision {
    /// This uniquely represents the identity of this revision and it stays
    /// the same even if it is rebased or merged between devices.
    rev_id: RevId,
    /// The largest undo group number of any edit in the history up to this
    /// point. Used to optimize undo to not look further back.
    max_undo_so_far: usize,
    edit: Contents,
}

/// Valid within a session. If there's a collision the most recent matching
/// Revision will be used, which means only the (small) set of concurrent edits
/// could trigger incorrect behavior if they collide, so u64 is safe.
pub type RevToken = u64;

/// the session ID component of a `RevId`
pub type SessionId = (u64, u32);

#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
struct FullPriority {
    priority: usize,
    session_id: SessionId,
}

use self::Contents::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
enum Contents {
    Edit {
        /// Used to order concurrent inserts, for example auto-indentation
        /// should go before typed text.
        priority: usize,
        /// Groups related edits together so that they are undone and re-done
        /// together. For example, an auto-indent insertion would be un-done
        /// along with the newline that triggered it.
        undo_group: usize,
        /// The subset of the characters of the union string from after this
        /// revision that were added by this revision.
        inserts: Subset,
        /// The subset of the characters of the union string from after this
        /// revision that were deleted by this revision.
        deletes: Subset,
    },
    Undo {
        /// The set of groups toggled between undone and done.
        /// Just the `symmetric_difference` (XOR) of the two sets.
        toggled_groups: BTreeSet<usize>,  // set of undo_group id's
        /// Used to store a reversible difference between the old
        /// and new deletes_from_union
        deletes_bitxor: Subset,
    }
}

/// for single user cases, used by serde and ::empty
fn default_session() -> (u64,u32) {
    (1, 0)
}

/// Revision 0 is always an Undo of the empty set of groups
fn initial_revision_counter() -> u32 {
    1
}

impl RevId {
    /// Returns a u64 that will be equal for equivalent revision IDs and
    /// should be as unlikely to collide as two random u64s.
    pub fn token(&self) -> RevToken {
        use std::hash::{Hash, Hasher};
        // Rust is unlikely to break the property that this hash is strongly collision-resistant
        // and it only needs to be consistent over one execution.
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    pub fn session_id(&self) -> SessionId {
        (self.session1, self.session2)
    }
}

impl Engine {
    /// Create a new Engine with a single edit that inserts `initial_contents`
    /// if it is non-empty. It needs to be a separate commit rather than just
    /// part of the initial contents since any two `Engine`s need a common
    /// ancestor in order to be mergeable.
    pub fn new(initial_contents: Rope) -> Engine {
        let mut engine = Engine::empty();
        if initial_contents.len() > 0 {
            let first_rev = engine.get_head_rev_id().token();
            let delta = Delta::simple_edit(Interval::new_closed_closed(0,0), initial_contents, 0);
            engine.edit_rev(0, 0, first_rev, delta);
        }
        engine
    }

    pub fn empty() -> Engine {
        let deletes_from_union = Subset::new(0);
        let rev = Revision {
            rev_id: RevId { session1: 0, session2: 0, num: 0 },
            edit: Undo { toggled_groups: BTreeSet::new(), deletes_bitxor: deletes_from_union.clone() },
            max_undo_so_far: 0,
        };
        Engine {
            session: default_session(),
            rev_id_counter: 1,
            text: Rope::default(),
            tombstones: Rope::default(),
            deletes_from_union,
            undone_groups: BTreeSet::new(),
            revs: vec![rev],
        }
    }

    fn next_rev_id(&self) -> RevId {
        RevId { session1: self.session.0, session2: self.session.1, num: self.rev_id_counter }
    }

    fn find_rev(&self, rev_id: RevId) -> Option<usize> {
        self.revs.iter().enumerate().rev()
            .find(|&(_, ref rev)| rev.rev_id == rev_id)
            .map(|(i, _)| i)
    }

    fn find_rev_token(&self, rev_token: RevToken) -> Option<usize> {
        self.revs.iter().enumerate().rev()
            .find(|&(_, ref rev)| rev.rev_id.token() == rev_token)
            .map(|(i, _)| i)
    }


    // TODO: does Cow really help much here? It certainly won't after making Subsets a rope.
    /// Find what the `deletes_from_union` field in Engine would have been at the time
    /// of a certain `rev_index`. In other words, the deletes from the union string at that time.
    fn deletes_from_union_for_index(&self, rev_index: usize) -> Cow<Subset> {
        self.deletes_from_union_before_index(rev_index + 1, true)
    }

    /// Garbage collection means undo can sometimes need to replay the very first
    /// revision, and so needs a way to get the deletion set before then.
    fn deletes_from_union_before_index(&self, rev_index: usize, invert_undos: bool) -> Cow<Subset> {
        let mut deletes_from_union = Cow::Borrowed(&self.deletes_from_union);
        let mut undone_groups = Cow::Borrowed(&self.undone_groups);

        // invert the changes to deletes_from_union starting in the present and working backwards
        for rev in self.revs[rev_index..].iter().rev() {
            deletes_from_union = match rev.edit {
                Edit { ref inserts, ref deletes, ref undo_group, .. } => {
                    if undone_groups.contains(undo_group) {
                        // no need to un-delete undone inserts since we'll just shrink them out
                        Cow::Owned(deletes_from_union.transform_shrink(inserts))
                    } else {
                        let un_deleted = deletes_from_union.subtract(deletes);
                        Cow::Owned(un_deleted.transform_shrink(inserts))
                    }
                }
                Undo { ref toggled_groups, ref deletes_bitxor } => {
                    if invert_undos {
                        let new_undone = undone_groups.symmetric_difference(toggled_groups).cloned().collect();
                        undone_groups = Cow::Owned(new_undone);
                        Cow::Owned(deletes_from_union.bitxor(deletes_bitxor))
                    } else {
                        deletes_from_union
                    }
                }
            }
        }
        deletes_from_union
    }

    /// Get the contents of the document at a given revision number
    fn rev_content_for_index(&self, rev_index: usize) -> Rope {
        let old_deletes_from_union = self.deletes_from_cur_union_for_index(rev_index);
        let delta = Delta::synthesize(&self.tombstones,
            &self.deletes_from_union, &old_deletes_from_union);
        delta.apply(&self.text)
    }

    /// Get the Subset to delete from the current union string in order to obtain a revision's content
    fn deletes_from_cur_union_for_index(&self, rev_index: usize) -> Cow<Subset> {
        let mut deletes_from_union = self.deletes_from_union_for_index(rev_index);
        for rev in &self.revs[rev_index + 1..] {
            if let Edit { ref inserts, .. } = rev.edit {
                if !inserts.is_empty() {
                    deletes_from_union = Cow::Owned(deletes_from_union.transform_union(inserts));
                }
            }
        }
        deletes_from_union
    }

    /// Returns the largest undo group ID used so far
    pub fn max_undo_group_id(&self) -> usize {
        self.revs.last().unwrap().max_undo_so_far
    }

    /// Get revision id of head revision.
    pub fn get_head_rev_id(&self) -> RevId {
        self.revs.last().unwrap().rev_id
    }

    /// Get text of head revision.
    pub fn get_head(&self) -> &Rope {
        &self.text
    }

    /// Get text of a given revision, if it can be found.
    pub fn get_rev(&self, rev: RevToken) -> Option<Rope> {
        self.find_rev_token(rev).map(|rev_index| self.rev_content_for_index(rev_index))
    }

    /// A delta that, when applied to `base_rev`, results in the current head. Panics
    /// if there is not at least one edit.
    pub fn delta_rev_head(&self, base_rev: RevToken) -> Delta<RopeInfo> {
        let ix = self.find_rev_token(base_rev).expect("base revision not found");
        let prev_from_union = self.deletes_from_cur_union_for_index(ix);
        // TODO: this does 2 calls to Delta::synthesize and 1 to apply, this probably could be better.
        let old_tombstones = shuffle_tombstones(&self.text, &self.tombstones, &self.deletes_from_union, &prev_from_union);
        Delta::synthesize(&old_tombstones, &prev_from_union, &self.deletes_from_union)
    }

    // TODO: don't construct transform if subsets are empty
    // TODO: maybe switch to using a revision index for `base_rev` once we disable GC
    /// Returns a tuple of a new `Revision` representing the edit based on the
    /// current head, a new text `Rope`, a new tombstones `Rope` and a new `deletes_from_union`.
    fn mk_new_rev(&self, new_priority: usize, undo_group: usize,
            base_rev: RevToken, delta: Delta<RopeInfo>) -> (Revision, Rope, Rope, Subset) {
        let ix = self.find_rev_token(base_rev).expect("base revision not found");
        let (ins_delta, deletes) = delta.factor();

        // rebase delta to be on the base_rev union instead of the text
        let deletes_at_rev = self.deletes_from_union_for_index(ix);
        let mut union_ins_delta = ins_delta.transform_expand(&deletes_at_rev, true);
        let mut new_deletes = deletes.transform_expand(&deletes_at_rev);

        // rebase the delta to be on the head union instead of the base_rev union
        let new_full_priority = FullPriority { priority: new_priority, session_id: self.session };
        for r in &self.revs[ix + 1..] {
            if let Edit { priority, ref inserts, .. } = r.edit {
                if !inserts.is_empty() {
                    let full_priority = FullPriority { priority, session_id: r.rev_id.session_id() };
                    let after = new_full_priority >= full_priority;  // should never be ==
                    union_ins_delta = union_ins_delta.transform_expand(inserts, after);
                    new_deletes = new_deletes.transform_expand(inserts);
                }
            }
        }

        // rebase the deletion to be after the inserts instead of directly on the head union
        let new_inserts = union_ins_delta.inserted_subset();
        if !new_inserts.is_empty() {
            new_deletes = new_deletes.transform_expand(&new_inserts);
        }

        // rebase insertions on text and apply
        let cur_deletes_from_union = &self.deletes_from_union;
        let text_ins_delta = union_ins_delta.transform_shrink(cur_deletes_from_union);
        let text_with_inserts = text_ins_delta.apply(&self.text);
        let rebased_deletes_from_union = cur_deletes_from_union.transform_expand(&new_inserts);

        // is the new edit in an undo group that was already undone due to concurrency?
        let undone = self.undone_groups.contains(&undo_group);
        let new_deletes_from_union = {
            let to_delete = if undone { &new_inserts } else { &new_deletes };
            rebased_deletes_from_union.union(to_delete)
        };

        // move deleted or undone-inserted things from text to tombstones
        let (new_text, new_tombstones) = shuffle(&text_with_inserts, &self.tombstones,
            &rebased_deletes_from_union, &new_deletes_from_union);

        let head_rev = &self.revs.last().unwrap();
        (Revision {
            rev_id: self.next_rev_id(),
            max_undo_so_far: std::cmp::max(undo_group, head_rev.max_undo_so_far),
            edit: Edit {
                priority: new_priority,
                undo_group,
                inserts: new_inserts,
                deletes: new_deletes,
            }
        }, new_text, new_tombstones, new_deletes_from_union)
    }

    // TODO: have `base_rev` be an index so that it can be used maximally efficiently with the
    // head revision, a token or a revision ID. Efficiency loss of token is negligible but unfortunate.
    pub fn edit_rev(&mut self, priority: usize, undo_group: usize,
            base_rev: RevToken, delta: Delta<RopeInfo>) {
        let (new_rev, new_text, new_tombstones, new_deletes_from_union) =
            self.mk_new_rev(priority, undo_group, base_rev, delta);
        self.rev_id_counter += 1;
        self.revs.push(new_rev);
        self.text = new_text;
        self.tombstones = new_tombstones;
        self.deletes_from_union = new_deletes_from_union;
    }

    // since undo and gc replay history with transforms, we need an empty set
    // of the union string length *before* the first revision.
    fn empty_subset_before_first_rev(&self) -> Subset {
        let first_rev = &self.revs.first().unwrap();
        // it will be immediately transform_expanded by inserts if it is an Edit, so length must be before
        let len = match first_rev.edit {
            Edit { ref inserts, .. } => inserts.count(CountMatcher::Zero),
            Undo { ref deletes_bitxor, .. } => deletes_bitxor.count(CountMatcher::All),
        };
        Subset::new(len)
    }

    /// Find the first revision that could be affected by toggling a set of undo groups
    fn find_first_undo_candidate_index(&self, toggled_groups: &BTreeSet<usize>) -> usize {
        // find the lowest toggled undo group number
        if let Some(lowest_group) = toggled_groups.iter().cloned().next() {
            for (i,rev) in self.revs.iter().enumerate().rev() {
                if rev.max_undo_so_far < lowest_group {
                    return i + 1; // +1 since we know the one we just found doesn't have it
                }
            }
            return 0;
        } else { // no toggled groups, return past end
            return self.revs.len();
        }
    }

    // This computes undo all the way from the beginning. An optimization would be to not
    // recompute the prefix up to where the history diverges, but it's not clear that's
    // even worth the code complexity.
    fn compute_undo(&self, groups: &BTreeSet<usize>) -> (Revision, Subset) {
        let toggled_groups = self.undone_groups.symmetric_difference(&groups).cloned().collect();
        let first_candidate = self.find_first_undo_candidate_index(&toggled_groups);
        // the `false` below: don't invert undos since our first_candidate is based on the current undo set, not past
        let mut deletes_from_union = self.deletes_from_union_before_index(first_candidate, false).into_owned();

        for rev in &self.revs[first_candidate..] {
            if let Edit { ref undo_group, ref inserts, ref deletes, .. } = rev.edit {
                if groups.contains(undo_group) {
                    if !inserts.is_empty() {
                        deletes_from_union = deletes_from_union.transform_union(inserts);
                    }
                } else {
                    if !inserts.is_empty() {
                        deletes_from_union = deletes_from_union.transform_expand(inserts);
                    }
                    if !deletes.is_empty() {
                        deletes_from_union = deletes_from_union.union(deletes);
                    }
                }
            }
        }

        let deletes_bitxor = self.deletes_from_union.bitxor(&deletes_from_union);
        let max_undo_so_far = self.revs.last().unwrap().max_undo_so_far;
        (Revision {
            rev_id: self.next_rev_id(),
            max_undo_so_far,
            edit: Undo { toggled_groups, deletes_bitxor }
        }, deletes_from_union)
    }

    // TODO: maybe refactor this API to take a toggle set
    pub fn undo(&mut self, groups: BTreeSet<usize>) {
        let (new_rev, new_deletes_from_union) = self.compute_undo(&groups);

        let (new_text, new_tombstones) =
            shuffle(&self.text, &self.tombstones, &self.deletes_from_union, &new_deletes_from_union);

        self.text = new_text;
        self.tombstones = new_tombstones;
        self.deletes_from_union = new_deletes_from_union;
        self.undone_groups = groups;
        self.revs.push(new_rev);
        self.rev_id_counter += 1;
    }

    pub fn is_equivalent_revision(&self, base_rev: RevId, other_rev: RevId) -> bool {
        let base_subset = self.find_rev(base_rev).map(|rev_index| self.deletes_from_cur_union_for_index(rev_index));
        let other_subset = self.find_rev(other_rev).map(|rev_index| self.deletes_from_cur_union_for_index(rev_index));

        base_subset.is_some() && base_subset == other_subset
    }

    // Note: this function would need some work to handle retaining arbitrary revisions,
    // partly because the reachability calculation would become more complicated (a
    // revision might hold content from an undo group that would otherwise be gc'ed),
    // and partly because you need to retain more undo history, to supply input to the
    // reachability calculation.
    //
    // Thus, it's easiest to defer gc to when all plugins quiesce, but it's certainly
    // possible to fix it so that's not necessary.
    pub fn gc(&mut self, gc_groups: &BTreeSet<usize>) {
        let mut gc_dels = self.empty_subset_before_first_rev();
        // TODO: want to let caller retain more rev_id's.
        let mut retain_revs = BTreeSet::new();
        if let Some(last) = self.revs.last() {
            retain_revs.insert(last.rev_id);
        }
        {
            for rev in &self.revs {
                if let Edit { ref undo_group, ref inserts, ref deletes, .. } = rev.edit {
                    if !retain_revs.contains(&rev.rev_id) && gc_groups.contains(undo_group) {
                        if self.undone_groups.contains(undo_group) {
                            if !inserts.is_empty() {
                                gc_dels = gc_dels.transform_union(inserts);
                            }
                        } else {
                            if !inserts.is_empty() {
                                gc_dels = gc_dels.transform_expand(inserts);
                            }
                            if !deletes.is_empty() {
                                gc_dels = gc_dels.union(deletes);
                            }
                        }
                    } else if !inserts.is_empty() {
                        gc_dels = gc_dels.transform_expand(inserts);
                    }
                }
            }
        }
        if !gc_dels.is_empty() {
            let not_in_tombstones = self.deletes_from_union.complement();
            let dels_from_tombstones = gc_dels.transform_shrink(&not_in_tombstones);
            self.tombstones = dels_from_tombstones.delete_from(&self.tombstones);
            self.deletes_from_union = self.deletes_from_union.transform_shrink(&gc_dels);
        }
        let old_revs = std::mem::replace(&mut self.revs, Vec::new());
        for rev in old_revs.into_iter().rev() {
            match rev.edit {
                Edit { priority, undo_group, inserts, deletes } => {
                    let new_gc_dels = if inserts.is_empty() {
                        None
                    } else {
                        Some(gc_dels.transform_shrink(&inserts))
                    };
                    if retain_revs.contains(&rev.rev_id) || !gc_groups.contains(&undo_group) {
                        let (inserts, deletes) = if gc_dels.is_empty() {
                            (inserts, deletes)
                        } else {
                            (inserts.transform_shrink(&gc_dels),
                                deletes.transform_shrink(&gc_dels))
                        };
                        self.revs.push(Revision {
                            rev_id: rev.rev_id,
                            max_undo_so_far: rev.max_undo_so_far,
                            edit: Edit {
                                priority,
                                undo_group,
                                inserts,
                                deletes,
                            }
                        });
                    }
                    if let Some(new_gc_dels) = new_gc_dels {
                        gc_dels = new_gc_dels;
                    }
                }
                Undo { toggled_groups, deletes_bitxor } => {
                    // We're super-aggressive about dropping these; after gc, the history
                    // of which undos were used to compute deletes_from_union in edits may be lost.
                    if retain_revs.contains(&rev.rev_id) {
                        let new_deletes_bitxor = if gc_dels.is_empty() {
                            deletes_bitxor
                        } else {
                            deletes_bitxor.transform_shrink(&gc_dels)
                        };
                        self.revs.push(Revision {
                            rev_id: rev.rev_id,
                            max_undo_so_far: rev.max_undo_so_far,
                            edit: Undo {
                                toggled_groups: &toggled_groups - gc_groups,
                                deletes_bitxor: new_deletes_bitxor,
                            }
                        })
                    }
                }
            }
        }
        self.revs.reverse();
    }

    /// Merge the new content from another Engine into this one with a CRDT merge
    pub fn merge(&mut self, other: &Engine) {
        let (mut new_revs, text, tombstones, deletes_from_union) = {
            let base_index = find_base_index(&self.revs, &other.revs);
            let a_to_merge = &self.revs[base_index..];
            let b_to_merge = &other.revs[base_index..];

            let common = find_common(a_to_merge, b_to_merge);

            let a_new = rearrange(a_to_merge, &common, self.deletes_from_union.len());
            let b_new = rearrange(b_to_merge, &common, other.deletes_from_union.len());

            let b_deltas = compute_deltas(&b_new, &other.text, &other.tombstones, &other.deletes_from_union);
            let expand_by = compute_transforms(a_new);

            let max_undo = self.max_undo_group_id();
            rebase(expand_by, b_deltas, self.text.clone(), self.tombstones.clone(), self.deletes_from_union.clone(), max_undo)
        };

        self.text = text;
        self.tombstones = tombstones;
        self.deletes_from_union = deletes_from_union;
        self.revs.append(&mut new_revs);
    }

    /// When merging between multiple concurrently-editing sessions, each session should have a unique ID
    /// set with this function, which will make the revisions they create not have colliding IDs.
    /// For safety, this will panic if any revisions have already been added to the Engine.
    ///
    /// Merge may panic or return incorrect results if session IDs collide, which is why they can be
    /// 96 bits which is more than sufficient for this to never happen.
    pub fn set_session_id(&mut self, session: SessionId) {
        assert_eq!(1, self.revs.len(), "Revisions were added to an Engine before set_session_id, these may collide.");
        self.session = session;
    }
}

// ======== Generic helpers

/// Move sections from text to tombstones and out of tombstones based on a new and old set of deletions
fn shuffle_tombstones(text: &Rope, tombstones: &Rope,
        old_deletes_from_union: &Subset, new_deletes_from_union: &Subset) -> Rope {
    // Taking the complement of deletes_from_union leads to an interleaving valid for swapped text and tombstones,
    // allowing us to use the same method to insert the text into the tombstones.
    let inverse_tombstones_map = old_deletes_from_union.complement();
    let move_delta = Delta::synthesize(text, &inverse_tombstones_map, &new_deletes_from_union.complement());
    move_delta.apply(tombstones)
}

/// Move sections from text to tombstones and vice versa based on a new and old set of deletions.
/// Returns a tuple of a new text `Rope` and a new `Tombstones` rope described by `new_deletes_from_union`.
fn shuffle(text: &Rope, tombstones: &Rope,
        old_deletes_from_union: &Subset, new_deletes_from_union: &Subset) -> (Rope,Rope) {
    // Delta that deletes the right bits from the text
    let del_delta = Delta::synthesize(tombstones, old_deletes_from_union, new_deletes_from_union);
    let new_text = del_delta.apply(text);
    // println!("shuffle: old={:?} new={:?} old_text={:?} new_text={:?} old_tombstones={:?}",
    //     old_deletes_from_union, new_deletes_from_union, text, new_text, tombstones);
    (new_text, shuffle_tombstones(text,tombstones,old_deletes_from_union,new_deletes_from_union))
}

// ======== Merge helpers

/// Find an index before which everything is the same
fn find_base_index(a: &[Revision], b: &[Revision]) -> usize {
    assert!(!a.is_empty() && !b.is_empty());
    assert!(a[0].rev_id == b[0].rev_id);
    // TODO find the maximum base revision.
    // this should have the same behavior, but worse performance
    1
}

/// Find a set of revisions common to both lists
fn find_common(a: &[Revision], b: &[Revision]) -> BTreeSet<RevId> {
    // TODO make this faster somehow?
    let a_ids: BTreeSet<RevId> = a.iter().map(|r| r.rev_id).collect();
    let b_ids: BTreeSet<RevId> = b.iter().map(|r| r.rev_id).collect();
    a_ids.intersection(&b_ids).cloned().collect()
}

/// Returns the operations in `revs` that don't have their `rev_id` in
/// `base_revs`, but modified so that they are in the same order but based on
/// the `base_revs`. This allows the rest of the merge to operate on only
/// revisions not shared by both sides.
///
/// Conceptually, see the diagram below, with `.` being base revs and `n` being
/// non-base revs, `N` being transformed non-base revs, and rearranges it:
/// .n..n...nn..  -> ........NNNN -> returns vec![N,N,N,N]
fn rearrange(revs: &[Revision], base_revs: &BTreeSet<RevId>, head_len: usize) -> Vec<Revision> {
    // transform representing the characters added by common revisions after a point.
    let mut s = Subset::new(head_len);

    let mut out = Vec::with_capacity(revs.len() - base_revs.len());
    for rev in revs.iter().rev() {
        let is_base = base_revs.contains(&rev.rev_id);
        let contents = match rev.edit {
            Contents::Edit {priority, undo_group, ref inserts, ref deletes} => {
                if is_base {
                    s = inserts.transform_union(&s);
                    None
                } else {
                    // fast-forward this revision over all common ones after it
                    let transformed_inserts = inserts.transform_expand(&s);
                    let transformed_deletes = deletes.transform_expand(&s);
                    // we don't want new revisions before this to be transformed after us
                    s = s.transform_shrink(&transformed_inserts);
                    Some(Contents::Edit {
                        inserts: transformed_inserts,
                        deletes: transformed_deletes,
                        priority, undo_group,
                    })
                }
            },
            Contents::Undo { .. } => panic!("can't merge undo yet"),
        };
        if let Some(edit) = contents {
            out.push(Revision { edit, rev_id: rev.rev_id, max_undo_so_far: rev.max_undo_so_far });
        }
    }

    out.as_mut_slice().reverse();
    out
}

#[derive(Clone, Debug)]
struct DeltaOp {
    rev_id: RevId,
    priority: usize,
    undo_group: usize,
    inserts: InsertDelta<RopeInfo>,
    deletes: Subset,
}

/// Transform `revs`, which doesn't include information on the actual content of the operations,
/// into an `InsertDelta`-based representation that does by working backward from the text and tombstones.
fn compute_deltas(revs: &[Revision], text: &Rope, tombstones: &Rope, deletes_from_union: &Subset) -> Vec<DeltaOp> {
    let mut out = Vec::with_capacity(revs.len());

    let mut cur_all_inserts = Subset::new(deletes_from_union.len());
    for rev in revs.iter().rev() {
        match rev.edit {
            Contents::Edit {priority, undo_group, ref inserts, ref deletes} => {
                let older_all_inserts = inserts.transform_union(&cur_all_inserts);

                // TODO could probably be more efficient by avoiding shuffling from head every time
                let tombstones_here = shuffle_tombstones(text, tombstones, deletes_from_union, &older_all_inserts);
                let delta = Delta::synthesize(&tombstones_here, &older_all_inserts, &cur_all_inserts);
                // TODO create InsertDelta directly and more efficiently instead of factoring
                let (ins, _) = delta.factor();
                out.push(DeltaOp {
                    rev_id: rev.rev_id,
                    priority, undo_group,
                    inserts: ins,
                    deletes: deletes.clone(),
                });

                cur_all_inserts = older_all_inserts;
            },
            Contents::Undo { .. } => panic!("can't merge undo yet"),
        }
    }

    out.as_mut_slice().reverse();
    out
}

/// Computes a series of priorities and transforms for the deltas on the right
/// from the new revisions on the left.
///
/// Applies an optimization where it combines sequential revisions with the
/// same priority into one transform to decrease the number of transforms that
/// have to be considered in `rebase` substantially for normal editing
/// patterns. Any large runs of typing in the same place by the same user (e.g
/// typing a paragraph) will be combined into a single segment in a transform
/// as opposed to thousands of revisions.
fn compute_transforms(revs: Vec<Revision>) -> Vec<(FullPriority, Subset)> {
    let mut out = Vec::new();
    let mut last_priority: Option<usize> = None;
    for r in revs {
        if let Contents::Edit {priority, inserts, .. } = r.edit {
            if inserts.is_empty() {
                continue;
            }
            if Some(priority) == last_priority {
                let last: &mut (FullPriority, Subset) = out.last_mut().unwrap();
                last.1 = last.1.transform_union(&inserts);
            } else {
                last_priority = Some(priority);
                let prio = FullPriority { priority, session_id: r.rev_id.session_id() };
                out.push((prio, inserts));
            }
        }
    }
    out
}

/// Rebase `b_new` on top of `expand_by` and return revision contents that can be appended as new
/// revisions on top of the revisions represented by `expand_by`.
fn rebase(mut expand_by: Vec<(FullPriority, Subset)>, b_new: Vec<DeltaOp>, mut text: Rope, mut tombstones: Rope,
        mut deletes_from_union: Subset, mut max_undo_so_far: usize) -> (Vec<Revision>, Rope, Rope, Subset) {
    let mut out = Vec::with_capacity(b_new.len());

    let mut next_expand_by = Vec::with_capacity(expand_by.len());
    for op in b_new {
        let DeltaOp { rev_id, priority, undo_group, mut inserts, mut deletes } = op;
        let full_priority = FullPriority { priority, session_id: rev_id.session_id() };
        // expand by each in expand_by
        for &(trans_priority, ref trans_inserts) in &expand_by {
            let after = full_priority >= trans_priority;  // should never be ==
            // d-expand by other
            inserts = inserts.transform_expand(trans_inserts, after);
            // trans-expand other by expanded so they have the same context
            let inserted = inserts.inserted_subset();
            let new_trans_inserts = trans_inserts.transform_expand(&inserted);
            // The deletes are already after our inserts, but we need to include the other inserts
            deletes = deletes.transform_expand(&new_trans_inserts);
            // On the next step we want things in expand_by to have op in the context
            next_expand_by.push((trans_priority, new_trans_inserts));
        }

        let text_inserts = inserts.transform_shrink(&deletes_from_union);
        let text_with_inserts = text_inserts.apply(&text);
        let inserted = inserts.inserted_subset();

        let expanded_deletes_from_union = deletes_from_union.transform_expand(&inserted);
        let new_deletes_from_union = expanded_deletes_from_union.union(&deletes);
        let (new_text, new_tombstones) =
            shuffle(&text_with_inserts, &tombstones, &expanded_deletes_from_union, &new_deletes_from_union);

        text = new_text;
        tombstones = new_tombstones;
        deletes_from_union = new_deletes_from_union;

        max_undo_so_far = std::cmp::max(max_undo_so_far, undo_group);
        out.push(Revision {
            rev_id, max_undo_so_far,
            edit: Contents::Edit {
                priority, undo_group, deletes,
                inserts: inserted,
            }
        });

        expand_by = next_expand_by;
        next_expand_by = Vec::with_capacity(expand_by.len());
    }

    (out, text, tombstones, deletes_from_union)
}


#[cfg(test)]
mod tests {
    use engine::*;
    use rope::{Rope, RopeInfo};
    use delta::{Builder, Delta};
    use multiset::Subset;
    use interval::Interval;
    use std::collections::BTreeSet;
    use test_helpers::{parse_subset_list, parse_subset, parse_delta, debug_subsets};

    const TEST_STR: &'static str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    fn build_delta_1() -> Delta<RopeInfo> {
        let mut d_builder = Builder::new(TEST_STR.len());
        d_builder.delete(Interval::new_closed_open(10, 36));
        d_builder.replace(Interval::new_closed_open(39, 42), Rope::from("DEEF"));
        d_builder.replace(Interval::new_closed_open(54, 54), Rope::from("999"));
        d_builder.delete(Interval::new_closed_open(58, 61));
        d_builder.build()
    }

    fn build_delta_2() -> Delta<RopeInfo> {
        let mut d_builder = Builder::new(TEST_STR.len());
        d_builder.replace(Interval::new_closed_open(1, 3), Rope::from("!"));
        d_builder.delete(Interval::new_closed_open(10, 36));
        d_builder.replace(Interval::new_closed_open(42, 45), Rope::from("GI"));
        d_builder.replace(Interval::new_closed_open(54, 54), Rope::from("888"));
        d_builder.replace(Interval::new_closed_open(59, 60), Rope::from("HI"));
        d_builder.build()
    }

    #[test]
    fn edit_rev_simple() {
        let mut engine = Engine::new(Rope::from(TEST_STR));
        let first_rev = engine.get_head_rev_id().token();
        engine.edit_rev(0, 1, first_rev, build_delta_1());
        assert_eq!("0123456789abcDEEFghijklmnopqr999stuvz", String::from(engine.get_head()));
    }

    #[test]
    fn edit_rev_concurrent() {
        let mut engine = Engine::new(Rope::from(TEST_STR));
        let first_rev = engine.get_head_rev_id().token();
        engine.edit_rev(1, 1, first_rev, build_delta_1());
        engine.edit_rev(0, 2, first_rev, build_delta_2());
        assert_eq!("0!3456789abcDEEFGIjklmnopqr888999stuvHIz", String::from(engine.get_head()));
    }

    fn undo_test(before: bool, undos : BTreeSet<usize>, output: &str) {
        let mut engine = Engine::new(Rope::from(TEST_STR));
        let first_rev = engine.get_head_rev_id().token();
        if before {
            engine.undo(undos.clone());
        }
        engine.edit_rev(1, 1, first_rev, build_delta_1());
        engine.edit_rev(0, 2, first_rev, build_delta_2());
        if !before {
            engine.undo(undos);
        }
        assert_eq!(output, String::from(engine.get_head()));
    }

    #[test]
    fn edit_rev_undo() {
        undo_test(true, [1,2].iter().cloned().collect(), TEST_STR);
    }

    #[test]
    fn edit_rev_undo_2() {
        undo_test(true, [2].iter().cloned().collect(), "0123456789abcDEEFghijklmnopqr999stuvz");
    }

    #[test]
    fn edit_rev_undo_3() {
        undo_test(true, [1].iter().cloned().collect(), "0!3456789abcdefGIjklmnopqr888stuvwHIyz");
    }

    #[test]
    fn delta_rev_head() {
        let mut engine = Engine::new(Rope::from(TEST_STR));
        let first_rev = engine.get_head_rev_id().token();
        engine.edit_rev(1, 1, first_rev, build_delta_1());
        let d = engine.delta_rev_head(first_rev);
        assert_eq!(String::from(engine.get_head()), d.apply_to_string(TEST_STR));
    }

    #[test]
    fn delta_rev_head_2() {
        let mut engine = Engine::new(Rope::from(TEST_STR));
        let first_rev = engine.get_head_rev_id().token();
        engine.edit_rev(1, 1, first_rev, build_delta_1());
        engine.edit_rev(0, 2, first_rev, build_delta_2());
        let d = engine.delta_rev_head(first_rev);
        assert_eq!(String::from(engine.get_head()), d.apply_to_string(TEST_STR));
    }

    #[test]
    fn delta_rev_head_3() {
        let mut engine = Engine::new(Rope::from(TEST_STR));
        let first_rev = engine.get_head_rev_id().token();
        engine.edit_rev(1, 1, first_rev, build_delta_1());
        let after_first_edit = engine.get_head_rev_id().token();
        engine.edit_rev(0, 2, first_rev, build_delta_2());
        let d = engine.delta_rev_head(after_first_edit);
        assert_eq!(String::from(engine.get_head()), d.apply_to_string("0123456789abcDEEFghijklmnopqr999stuvz"));
    }

    #[test]
    fn undo() {
        undo_test(false, [1,2].iter().cloned().collect(), TEST_STR);
    }

    #[test]
    fn undo_2() {
        undo_test(false, [2].iter().cloned().collect(), "0123456789abcDEEFghijklmnopqr999stuvz");
    }

    #[test]
    fn undo_3() {
        undo_test(false, [1].iter().cloned().collect(), "0!3456789abcdefGIjklmnopqr888stuvwHIyz");
    }

    #[test]
    fn undo_4() {
        let mut engine = Engine::new(Rope::from(TEST_STR));
        let d1 = Delta::simple_edit(Interval::new_closed_open(0,0), Rope::from("a"), TEST_STR.len());
        let first_rev = engine.get_head_rev_id().token();
        engine.edit_rev(1, 1, first_rev, d1.clone());
        let new_head = engine.get_head_rev_id().token();
        engine.undo([1].iter().cloned().collect());
        let d2 = Delta::simple_edit(Interval::new_closed_open(0,0), Rope::from("a"), TEST_STR.len()+1);
        engine.edit_rev(1, 2, new_head, d2); // note this is based on d1 before, not the undo
        let new_head_2 = engine.get_head_rev_id().token();
        let d3 = Delta::simple_edit(Interval::new_closed_open(0,0), Rope::from("b"), TEST_STR.len()+1);
        engine.edit_rev(1, 3, new_head_2, d3);
        engine.undo([1,3].iter().cloned().collect());
        assert_eq!("a0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", String::from(engine.get_head()));
    }

    #[test]
    fn undo_5() {
        let mut engine = Engine::new(Rope::from(TEST_STR));
        let d1 = Delta::simple_edit(Interval::new_closed_open(0,10), Rope::from(""), TEST_STR.len());
        let first_rev = engine.get_head_rev_id().token();
        engine.edit_rev(1, 1, first_rev, d1.clone());
        engine.edit_rev(1, 2, first_rev, d1.clone());
        engine.undo([1].iter().cloned().collect());
        assert_eq!("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", String::from(engine.get_head()));
        engine.undo([1,2].iter().cloned().collect());
        assert_eq!(TEST_STR, String::from(engine.get_head()));
        engine.undo([].iter().cloned().collect());
        assert_eq!("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", String::from(engine.get_head()));
    }

    #[test]
    fn gc() {
        let mut engine = Engine::new(Rope::from(TEST_STR));
        let d1 = Delta::simple_edit(Interval::new_closed_open(0,0), Rope::from("c"), TEST_STR.len());
        let first_rev = engine.get_head_rev_id().token();
        engine.edit_rev(1, 1, first_rev, d1);
        let new_head = engine.get_head_rev_id().token();
        engine.undo([1].iter().cloned().collect());
        let d2 = Delta::simple_edit(Interval::new_closed_open(0,0), Rope::from("a"), TEST_STR.len()+1);
        engine.edit_rev(1, 2, new_head, d2);
        let gc : BTreeSet<usize> = [1].iter().cloned().collect();
        engine.gc(&gc);
        let d3 = Delta::simple_edit(Interval::new_closed_open(0,0), Rope::from("b"), TEST_STR.len()+1);
        let new_head_2 = engine.get_head_rev_id().token();
        engine.edit_rev(1, 3, new_head_2, d3);
        engine.undo([3].iter().cloned().collect());
        assert_eq!("a0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", String::from(engine.get_head()));
    }

    /// This case is a regression test reproducing a panic I found while using the UI.
    /// It does undos and gcs in a pattern that can actually happen when using the editor.
    fn gc_scenario(edits: usize, max_undos: usize) {
        let mut engine = Engine::new(Rope::from(""));

        // insert `edits` letter "b"s in separate undo groups
        for i in 0..edits {
            let d = Delta::simple_edit(Interval::new_closed_open(0,0), Rope::from("b"), i);
            let head = engine.get_head_rev_id().token();
            engine.edit_rev(1, i+1, head, d);
            if i >= max_undos {
                let to_gc : BTreeSet<usize> = [i-max_undos].iter().cloned().collect();
                engine.gc(&to_gc)
            }
        }

        // spam cmd+z until the available undo history is exhausted
        let mut to_undo = BTreeSet::new();
        for i in ((edits-max_undos)..edits).rev() {
            to_undo.insert(i+1);
            engine.undo(to_undo.clone());
        }

        // insert a character at the beginning
        let d1 = Delta::simple_edit(Interval::new_closed_open(0,0), Rope::from("h"), engine.get_head().len());
        let head = engine.get_head_rev_id().token();
        engine.edit_rev(1, edits+1, head, d1);

        // since character was inserted after gc, editor gcs all undone things
        engine.gc(&to_undo);

        // insert character at end, when this test was added, it panic'd here
        let chars_left = (edits-max_undos)+1;
        let d2 = Delta::simple_edit(Interval::new_closed_open(chars_left, chars_left), Rope::from("f"), engine.get_head().len());
        let head2 = engine.get_head_rev_id().token();
        engine.edit_rev(1, edits+1, head2, d2);

        let mut soln = String::from("h");
        for _ in 0..(edits-max_undos) {
            soln.push('b');
        }
        soln.push('f');
        assert_eq!(soln, String::from(engine.get_head()));
    }

    #[test]
    fn gc_2() {
        // the smallest values with which it still fails:
        gc_scenario(4,3);
    }

    #[test]
    fn gc_3() {
        // original values this test was created/found with in the UI:
        gc_scenario(35,20);
    }

    #[test]
    fn gc_4() {
        let mut engine = Engine::new(Rope::from(TEST_STR));
        let d1 = Delta::simple_edit(Interval::new_closed_open(0,10), Rope::from(""), TEST_STR.len());
        let first_rev = engine.get_head_rev_id().token();
        engine.edit_rev(1, 1, first_rev, d1.clone());
        engine.edit_rev(1, 2, first_rev, d1.clone());
        let gc : BTreeSet<usize> = [1].iter().cloned().collect();
        engine.gc(&gc);
        // shouldn't do anything since it was double-deleted and one was GC'd
        engine.undo([1,2].iter().cloned().collect());
        assert_eq!("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", String::from(engine.get_head()));
    }

    #[test]
    fn gc_5() {
        let mut engine = Engine::new(Rope::from(TEST_STR));
        let d1 = Delta::simple_edit(Interval::new_closed_open(0,10), Rope::from(""), TEST_STR.len());
        let initial_rev = engine.get_head_rev_id().token();
        engine.undo([1].iter().cloned().collect());
        engine.edit_rev(1, 1, initial_rev, d1.clone());
        engine.edit_rev(1, 2, initial_rev, d1.clone());
        let gc : BTreeSet<usize> = [1].iter().cloned().collect();
        engine.gc(&gc);
        // only one of the deletes was gc'd, the other should still be in effect
        assert_eq!("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", String::from(engine.get_head()));
        // since one of the two deletes was gc'd this should undo the one that wasn't
        engine.undo([2].iter().cloned().collect());
        assert_eq!(TEST_STR, String::from(engine.get_head()));
    }

    #[test]
    fn gc_6() {
        let mut engine = Engine::new(Rope::from(TEST_STR));
        let d1 = Delta::simple_edit(Interval::new_closed_open(0,10), Rope::from(""), TEST_STR.len());
        let initial_rev = engine.get_head_rev_id().token();
        engine.edit_rev(1, 1, initial_rev, d1.clone());
        engine.undo([1,2].iter().cloned().collect());
        engine.edit_rev(1, 2, initial_rev, d1.clone());
        let gc : BTreeSet<usize> = [1].iter().cloned().collect();
        engine.gc(&gc);
        assert_eq!(TEST_STR, String::from(engine.get_head()));
        // since one of the two deletes was gc'd this should re-do the one that wasn't
        engine.undo([].iter().cloned().collect());
        assert_eq!("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", String::from(engine.get_head()));
    }

    fn basic_rev(i: usize) -> RevId {
        RevId { session1: 1, session2: 0, num: i as u32 }
    }

    fn basic_insert_ops(inserts: Vec<Subset>, priority: usize) -> Vec<Revision> {
        inserts.into_iter().enumerate().map(|(i, inserts)| {
            let deletes = Subset::new(inserts.len());
            Revision {
                rev_id: basic_rev(i+1),
                max_undo_so_far: i+1,
                edit: Contents::Edit {
                    priority, inserts, deletes,
                    undo_group: i+1,
                }
            }
        }).collect()
    }

    #[test]
    fn rearrange_1() {
        let inserts = parse_subset_list("
        ##
        -#-
        #---
        ---#-
        -----#
        #------
        ");
        let revs = basic_insert_ops(inserts, 1);
        let base: BTreeSet<RevId> = [3,5].iter().cloned().map(basic_rev).collect();

        let rearranged = rearrange(&revs, &base, 7);
        let rearranged_inserts: Vec<Subset> = rearranged.into_iter().map(|c| {
            match c.edit {
                Contents::Edit {inserts, ..} => inserts,
                Contents::Undo { .. } => panic!(),
            }
        }).collect();

        debug_subsets(&rearranged_inserts);
        let correct = parse_subset_list("
        -##-
        --#--
        ---#--
        #------
        ");
        assert_eq!(correct, rearranged_inserts);
    }

    fn ids_to_fake_revs(ids: &[usize]) -> Vec<Revision> {
        let contents = Contents::Edit {
            priority: 0,
            undo_group: 0,
            inserts: Subset::new(0),
            deletes: Subset::new(0),
        };

        ids.iter().cloned().map(|i| {
            Revision {
                rev_id: basic_rev(i),
                max_undo_so_far: i,
                edit: contents.clone()
            }
        }).collect()
    }

    #[test]
    fn find_common_1() {
        let a: Vec<Revision> = ids_to_fake_revs(&[0,2,4,6,8,10,12]);
        let b: Vec<Revision> = ids_to_fake_revs(&[0,1,2,4,5,8,9]);
        let res = find_common(&a, &b);

        let correct: BTreeSet<RevId> = [0,2,4,8].iter().cloned().map(basic_rev).collect();
        assert_eq!(correct, res);
    }


    #[test]
    fn find_base_1() {
        let a: Vec<Revision> = ids_to_fake_revs(&[0,2,4,6,8,10,12]);
        let b: Vec<Revision> = ids_to_fake_revs(&[0,1,2,4,5,8,9]);
        let res = find_base_index(&a, &b);

        assert_eq!(1, res);
    }

    #[test]
    fn compute_deltas_1() {
        let inserts = parse_subset_list("
        -##-
        --#--
        ---#--
        #------
        ");
        let revs = basic_insert_ops(inserts, 1);

        let text = Rope::from("13456");
        let tombstones = Rope::from("27");
        let deletes_from_union = parse_subset("-#----#");
        let delta_ops = compute_deltas(&revs, &text, &tombstones, &deletes_from_union);

        println!("{:#?}", delta_ops);

        let mut r = Rope::from("27");
        for op in &delta_ops {
            r = op.inserts.apply(&r);
        }
        assert_eq!("1234567", String::from(r));
    }

    #[test]
    fn compute_transforms_1() {
        let inserts = parse_subset_list("
        -##-
        --#--
        ---#--
        #------
        ");
        let revs = basic_insert_ops(inserts, 1);

        let expand_by = compute_transforms(revs);
        assert_eq!(1, expand_by.len());
        assert_eq!(1, expand_by[0].0.priority);
        let subset_str = format!("{:#?}", expand_by[0].1);
        assert_eq!("#-####-", &subset_str);
    }

    #[test]
    fn compute_transforms_2() {
        let inserts_1 = parse_subset_list("
        -##-
        --#--
        ");
        let mut revs = basic_insert_ops(inserts_1, 1);
        let inserts_2 = parse_subset_list("
        ----
        ");
        let mut revs_2 = basic_insert_ops(inserts_2, 4);
        revs.append(&mut revs_2);
        let inserts_3 = parse_subset_list("
        ---#--
        #------
        ");
        let mut revs_3 = basic_insert_ops(inserts_3, 2);
        revs.append(&mut revs_3);

        let expand_by = compute_transforms(revs);
        assert_eq!(2, expand_by.len());
        assert_eq!(1, expand_by[0].0.priority);
        assert_eq!(2, expand_by[1].0.priority);

        let subset_str = format!("{:#?}", expand_by[0].1);
        assert_eq!("-###-", &subset_str);
        let subset_str = format!("{:#?}", expand_by[1].1);
        assert_eq!("#---#--", &subset_str);
    }

    #[test]
    fn rebase_1() {
        let inserts = parse_subset_list("
        --#-
        ----#
        ");
        let a_revs = basic_insert_ops(inserts.clone(), 1);
        let b_revs = basic_insert_ops(inserts, 2);

        let text_b = Rope::from("zpbj");
        let tombstones_b = Rope::from("a");
        let deletes_from_union_b = parse_subset("-#---");
        let b_delta_ops = compute_deltas(&b_revs, &text_b, &tombstones_b, &deletes_from_union_b);

        println!("{:#?}", b_delta_ops);

        let text_a = Rope::from("zcbd");
        let tombstones_a = Rope::from("a");
        let deletes_from_union_a = parse_subset("-#---");
        let expand_by = compute_transforms(a_revs);

        let (revs, text_2, tombstones_2, deletes_from_union_2) =
            rebase(expand_by, b_delta_ops, text_a, tombstones_a, deletes_from_union_a, 0);

        let rebased_inserts: Vec<Subset> = revs.into_iter().map(|c| {
            match c.edit {
                Contents::Edit {inserts, ..} => inserts,
                Contents::Undo { .. } => panic!(),
            }
        }).collect();

        debug_subsets(&rebased_inserts);
        let correct = parse_subset_list("
        ---#--
        ------#
        ");
        assert_eq!(correct, rebased_inserts);


        assert_eq!("zcpbdj", String::from(&text_2));
        assert_eq!("a", String::from(&tombstones_2));
        assert_eq!("-#-----", format!("{:#?}", deletes_from_union_2));
    }

    // ============== Merge script tests

    #[derive(Clone, Debug)]
    enum MergeTestOp {
        Merge(usize, usize),
        Assert(usize, String),
        AssertAll(String),
        AssertMaxUndoSoFar(usize, usize),
        Edit { ei: usize, p: usize, u: usize, d: Delta<RopeInfo> },
    }

    #[derive(Debug)]
    struct MergeTestState {
        peers: Vec<Engine>,
    }

    impl MergeTestState {
        fn new(count: usize) -> MergeTestState {
            let mut peers = Vec::with_capacity(count);
            for i in 0..count {
                let mut peer = Engine::new(Rope::from(""));
                peer.set_session_id(((i*1000) as u64, 0));
                peers.push(peer);
            }
            MergeTestState { peers }
        }

        fn run_op(&mut self, op: &MergeTestOp) {
            match *op {
                MergeTestOp::Merge(ai, bi) => {
                    let (start, end) = self.peers.split_at_mut(ai);
                    let (mut a, rest) = end.split_first_mut().unwrap();
                    let b = if bi < ai {
                        &mut start[bi]
                    } else {
                        &mut rest[bi - ai - 1]
                    };
                    a.merge(b);
                },
                MergeTestOp::Assert(ei, ref correct) => {
                    let e = &mut self.peers[ei];
                    assert_eq!(correct, &String::from(e.get_head()), "for peer {}", ei);
                },
                MergeTestOp::AssertMaxUndoSoFar(ei, correct) => {
                    let e = &mut self.peers[ei];
                    assert_eq!(correct, e.max_undo_group_id(), "for peer {}", ei);
                },
                MergeTestOp::AssertAll(ref correct) => {
                    for (ei, e) in self.peers.iter().enumerate() {
                        assert_eq!(correct, &String::from(e.get_head()), "for peer {}", ei);
                    }
                },
                MergeTestOp::Edit { ei, p, u, d: ref delta } => {
                    let mut e = &mut self.peers[ei];
                    let head = e.get_head_rev_id().token();
                    e.edit_rev(p, u, head, delta.clone());
                },
            }
        }

        fn run_script(&mut self, script: &[MergeTestOp]) {
            for (i, op) in script.iter().enumerate() {
                println!("running {:?} at index {}", op, i);
                self.run_op(op);
            }
        }
    }

    /// Like the scanned whiteboard diagram I have, but without deleting 'a'
    #[test]
    fn merge_insert_only_whiteboard() {
        use self::MergeTestOp::*;
        let script = vec![
            Edit { ei: 2, p: 1, u: 1, d: parse_delta("ab") },
            Merge(0,2), Merge(1, 2),
            Assert(0, "ab".to_owned()),
            Assert(1, "ab".to_owned()),
            Assert(2, "ab".to_owned()),
            Edit { ei: 0, p: 3, u: 1, d: parse_delta("-c-") },
            Edit { ei: 0, p: 3, u: 1, d: parse_delta("---d") },
            Assert(0, "acbd".to_owned()),
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("-p-") },
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("---j") },
            Assert(1, "apbj".to_owned()),
            Edit { ei: 2, p: 1, u: 1, d: parse_delta("z--") },
            Merge(0,2), Merge(1, 2),
            Assert(0, "zacbd".to_owned()),
            Assert(1, "zapbj".to_owned()),
            Merge(0,1),
            Assert(0, "zacpbdj".to_owned()),
        ];
        MergeTestState::new(3).run_script(&script[..]);
    }

    /// Tests that priorities are used to break ties correctly
    #[test]
    fn merge_priorities() {
        use self::MergeTestOp::*;
        let script = vec![
            Edit { ei: 2, p: 1, u: 1, d: parse_delta("ab") },
            Merge(0,2), Merge(1, 2),
            Assert(0, "ab".to_owned()),
            Assert(1, "ab".to_owned()),
            Assert(2, "ab".to_owned()),
            Edit { ei: 0, p: 3, u: 1, d: parse_delta("-c-") },
            Edit { ei: 0, p: 3, u: 1, d: parse_delta("---d") },
            Assert(0, "acbd".to_owned()),
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("-p-") },
            Assert(1, "apb".to_owned()),
            Edit { ei: 2, p: 4, u: 1, d: parse_delta("-r-") },
            Merge(0,2), Merge(1, 2),
            Assert(0, "acrbd".to_owned()),
            Assert(1, "arpb".to_owned()),
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("----j") },
            Assert(1, "arpbj".to_owned()),
            Edit { ei: 2, p: 4, u: 1, d: parse_delta("---z") },
            Merge(0,2), Merge(1, 2),
            Assert(0, "acrbdz".to_owned()),
            Assert(1, "arpbzj".to_owned()),
            Merge(0,1),
            Assert(0, "acrpbdzj".to_owned()),
        ];
        MergeTestState::new(3).run_script(&script[..]);
    }

    /// Tests that merging again when there are no new revisions does nothing
    #[test]
    fn merge_idempotent() {
        use self::MergeTestOp::*;
        let script = vec![
            Edit { ei: 2, p: 1, u: 1, d: parse_delta("ab") },
            Merge(0,2), Merge(1, 2),
            Assert(0, "ab".to_owned()),
            Assert(1, "ab".to_owned()),
            Assert(2, "ab".to_owned()),
            Edit { ei: 0, p: 3, u: 1, d: parse_delta("-c-") },
            Edit { ei: 0, p: 3, u: 1, d: parse_delta("---d") },
            Assert(0, "acbd".to_owned()),
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("-p-") },
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("---j") },
            Merge(0,1),
            Assert(0, "acpbdj".to_owned()),
            Merge(0,1), Merge(1,0), Merge(0,1), Merge(1,0),
            Assert(0, "acpbdj".to_owned()),
            Assert(1, "acpbdj".to_owned()),
        ];
        MergeTestState::new(3).run_script(&script[..]);
    }

    #[test]
    fn merge_associative() {
        use self::MergeTestOp::*;
        let script = vec![
            Edit { ei: 2, p: 1, u: 1, d: parse_delta("ab") },
            Merge(0,2), Merge(1, 2),
            Edit { ei: 0, p: 3, u: 1, d: parse_delta("-c-") },
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("-p-") },
            Edit { ei: 2, p: 2, u: 1, d: parse_delta("z--") },
            // copy the current state
            Merge(3, 0), Merge(4, 1), Merge(5, 2),
            // Do the merge one direction
            Merge(1,2),
            Merge(0,1),
            Assert(0, "zacpb".to_owned()),
            // Do it the other way on the copy
            Merge(4,3),
            Merge(5,4),
            Assert(5, "zacpb".to_owned()),
            // Go crazy
            Merge(0,5), Merge(2,5), Merge(4,5), Merge(1,4),
            Merge(3,1), Merge(5,3),
            AssertAll("zacpb".to_owned()),
        ];
        MergeTestState::new(6).run_script(&script[..]);
    }

    #[test]
    fn merge_simple_delete_1() {
        use self::MergeTestOp::*;
        let script = vec![
            Edit { ei: 0, p: 1, u: 1, d: parse_delta("abc") },
            Merge(1,0),
            Assert(0, "abc".to_owned()),
            Assert(1, "abc".to_owned()),
            Edit { ei: 0, p: 1, u: 1, d: parse_delta("!-d-") },
            Assert(0, "bdc".to_owned()),
            Edit { ei: 1, p: 3, u: 1, d: parse_delta("--efg!") },
            Assert(1, "abefg".to_owned()),
            Merge(1,0),
            Assert(1, "bdefg".to_owned()),
        ];
        MergeTestState::new(2).run_script(&script[..]);
    }

    #[test]
    fn merge_simple_delete_2() {
        use self::MergeTestOp::*;
        let script = vec![
            Edit { ei: 0, p: 1, u: 1, d: parse_delta("ab") },
            Merge(1,0),
            Assert(0, "ab".to_owned()),
            Assert(1, "ab".to_owned()),
            Edit { ei: 0, p: 1, u: 1, d: parse_delta("!-") },
            Assert(0, "b".to_owned()),
            Edit { ei: 1, p: 3, u: 1, d: parse_delta("-c-") },
            Assert(1, "acb".to_owned()),
            Merge(1,0),
            Assert(1, "cb".to_owned()),
        ];
        MergeTestState::new(2).run_script(&script[..]);
    }

    /// I have a scanned whiteboard diagram of doing this merge by hand, good for reference
    #[test]
    fn merge_whiteboard() {
        use self::MergeTestOp::*;
        let script = vec![
            Edit { ei: 2, p: 1, u: 1, d: parse_delta("ab") },
            Merge(0,2), Merge(1, 2), Merge(3, 2),
            Assert(0, "ab".to_owned()),
            Assert(1, "ab".to_owned()),
            Assert(2, "ab".to_owned()),
            Assert(3, "ab".to_owned()),
            Edit { ei: 2, p: 1, u: 1, d: parse_delta("!-") },
            Assert(2, "b".to_owned()),
            Edit { ei: 0, p: 3, u: 1, d: parse_delta("-c-") },
            Edit { ei: 0, p: 3, u: 1, d: parse_delta("---d") },
            Assert(0, "acbd".to_owned()),
            Merge(0,2),
            Assert(0, "cbd".to_owned()),
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("-p-") },
            Merge(1,2),
            Assert(1, "pb".to_owned()),
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("--j") },
            Assert(1, "pbj".to_owned()),
            // to replicate whiteboard, z must be before a tombstone
            // which we can do with another peer that inserts before a and merges.
            Edit { ei: 3, p: 7, u: 1, d: parse_delta("z--") },
            Merge(2,3),
            Merge(0,2), Merge(1, 2),
            Assert(0, "zcbd".to_owned()),
            Assert(1, "zpbj".to_owned()),
            Merge(0,1), // the merge from the whiteboard scan
            Assert(0, "zcpbdj".to_owned()),
        ];
        MergeTestState::new(4).run_script(&script[..]);
    }

    #[test]
    fn merge_max_undo_so_far() {
        use self::MergeTestOp::*;
        let script = vec![
            Edit { ei: 0, p: 1, u: 1, d: parse_delta("ab") },
            Merge(1,0), Merge(2,0),
            AssertMaxUndoSoFar(1,1),
            Edit { ei: 0, p: 1, u: 2, d: parse_delta("!-") },
            Edit { ei: 1, p: 3, u: 3, d: parse_delta("-!") },
            Merge(1,0),
            AssertMaxUndoSoFar(1,3),
            AssertMaxUndoSoFar(0,2),
            Merge(0,1),
            AssertMaxUndoSoFar(0,3),
            Edit { ei: 2, p: 1, u: 1, d: parse_delta("!!") },
            Merge(1,2),
            AssertMaxUndoSoFar(1,3),
        ];
        MergeTestState::new(3).run_script(&script[..]);
    }

    /// This is a regression test to ensure that session IDs are used to break
    /// ties in edit priorities. Otherwise the results may be inconsistent.
    #[test]
    fn merge_session_priorities() {
        use self::MergeTestOp::*;
        let script = vec![
            Edit { ei: 0, p: 1, u: 1, d: parse_delta("ac") },
            Merge(1,0),
            Merge(2,0),
            AssertAll("ac".to_owned()),
            Edit { ei: 0, p: 1, u: 1, d: parse_delta("-d-") },
            Assert(0, "adc".to_owned()),
            Edit { ei: 1, p: 1, u: 1, d: parse_delta("-f-") },
            Merge(2,1),
            Assert(1, "afc".to_owned()),
            Assert(2, "afc".to_owned()),
            Merge(2,0),
            Merge(0,1),
            // These two will be different without using session IDs
            Assert(2, "adfc".to_owned()),
            Assert(0, "adfc".to_owned()),
        ];
        MergeTestState::new(3).run_script(&script[..]);
    }
}
// Copyright 2018 The xi-editor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![cfg_attr(feature = "benchmarks", feature(test))]
#![cfg_attr(feature = "collections_range", feature(collections_range))]

#![cfg_attr(feature = "cargo-clippy", allow(
    identity_op,
    new_without_default_derive,
))]

#[macro_use]
extern crate lazy_static;
extern crate time;

#[macro_use]
extern crate serde_derive;

extern crate serde;

#[macro_use]
extern crate log;

extern crate libc;

#[cfg(feature = "benchmarks")]
extern crate test;

#[cfg(feature = "json_payload")]
#[macro_use]
extern crate serde_json;

mod fixed_lifo_deque;
mod sys_pid;
mod sys_tid;

use std::borrow::Cow;
use std::collections::HashMap;
use std::cmp;
use std::fmt;
use std::mem::size_of;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Mutex;
use fixed_lifo_deque::FixedLifoDeque;

pub type StrCow = Cow<'static, str>;

#[derive(Clone, Debug)]
pub enum CategoriesT {
    StaticArray(&'static[&'static str]),
    DynamicArray(Vec<String>),
}

trait StringArrayEq<Rhs: ?Sized = Self> {
    fn arr_eq(&self, other: &Rhs) -> bool;
}

impl StringArrayEq<[&'static str]> for Vec<String> {
    fn arr_eq(&self, other: &[&'static str]) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for i in 0..self.len() {
            if self[i] != other[i] {
                return false;
            }
        }
        true
    }
}

impl StringArrayEq<Vec<String>> for &'static [&'static str] {
    fn arr_eq(&self, other: &Vec<String>) -> bool {
        if self.len() != other.len() {
            return false;
        }
        for i in 0..self.len() {
            if self[i] != other[i] {
                return false;
            }
        }
        true
    }
}

impl PartialEq for CategoriesT {
    fn eq(&self, other: &CategoriesT) -> bool {
        match *self {
            CategoriesT::StaticArray(ref self_arr) => {
                match *other {
                    CategoriesT::StaticArray(ref other_arr) => self_arr.eq(other_arr),
                    CategoriesT::DynamicArray(ref other_arr) => self_arr.arr_eq(other_arr),
                }
            },
            CategoriesT::DynamicArray(ref self_arr) => {
                match *other {
                    CategoriesT::StaticArray(ref other_arr) => self_arr.arr_eq(other_arr),
                    CategoriesT::DynamicArray(ref other_arr) => self_arr.eq(other_arr),
                }
            }
        }
    }
}

impl Eq for CategoriesT {}

impl serde::Serialize for CategoriesT {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer
    {
        self.join(",").serialize(serializer)
    }
}


impl<'de> serde::Deserialize<'de> for CategoriesT {
    fn deserialize<D>(deserializer: D)
        -> Result<CategoriesT, D::Error>
        where D: serde::Deserializer<'de>
    {
        use serde::de::Visitor;
        struct CategoriesTVisitor;

        impl<'de> Visitor<'de> for CategoriesTVisitor {
            type Value = CategoriesT;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("comma-separated strings")
            }

            fn visit_str<E>(self, v: &str) -> Result<CategoriesT, E>
                where E: serde::de::Error
            {
                let categories = v.split(",").map(|s| s.to_string()).collect();
                Ok(CategoriesT::DynamicArray(categories))
            }
        }

        deserializer.deserialize_str(CategoriesTVisitor)
    }
}

impl CategoriesT {
    pub fn join(&self, sep: &str) -> String {
        match *self {
            CategoriesT::StaticArray(ref arr) => arr.join(sep),
            CategoriesT::DynamicArray(ref vec) => vec.join(sep),
        }
    }
}

macro_rules! categories_from_constant_array {
    ($num_args: expr) => {
        impl From<&'static[&'static str; $num_args]> for CategoriesT {
            fn from(c: &'static[&'static str; $num_args]) -> CategoriesT {
                CategoriesT::StaticArray(c)
            }
        }
    }
}

categories_from_constant_array!(0);
categories_from_constant_array!(1);
categories_from_constant_array!(2);
categories_from_constant_array!(3);
categories_from_constant_array!(4);
categories_from_constant_array!(5);
categories_from_constant_array!(6);
categories_from_constant_array!(7);
categories_from_constant_array!(8);
categories_from_constant_array!(9);
categories_from_constant_array!(10);

impl From<Vec<String>> for CategoriesT {
    fn from(c: Vec<String>) -> CategoriesT {
        CategoriesT::DynamicArray(c)
    }
}

#[cfg(all(not(feature = "dict_payload"), not(feature = "json_payload")))]
pub type TracePayloadT = StrCow;

#[cfg(feature = "json_payload")]
pub type TracePayloadT = serde_json::Value;

#[cfg(feature = "dict_payload")]
pub type TracePayloadT = std::collections::HashMap<StrCow, StrCow>;

/// How tracing should be configured.
#[derive(Copy, Clone)]
pub struct Config {
    sample_limit_count: usize
}

impl Config {
    /// The maximum number of bytes the tracing data should take up.  This limit
    /// won't be exceeded by the underlying storage itself (i.e. rounds down).
    pub fn with_limit_bytes(size: usize) -> Self {
        Self::with_limit_count(size / size_of::<Sample>())
    }

    /// The maximum number of entries the tracing data should allow.  Total
    /// storage allocated will be limit * size_of<Sample>
    pub fn with_limit_count(limit: usize) -> Self {
        Self {
            sample_limit_count: limit
        }
    }

    /// The default amount of storage to allocate for tracing.  Currently 1 MB.
    pub fn default() -> Self {
        // 1 MB
        Self::with_limit_bytes(1 * 1024 * 1024)
    }

    /// The maximum amount of space the tracing data will take up.  This does
    /// not account for any overhead of storing the data itself (i.e. pointer to
    /// the heap, counters, etc); just the data itself.
    pub fn max_size_in_bytes(&self) -> usize {
        self.sample_limit_count * size_of::<Sample>()
    }

    /// The maximum number of samples that should be stored.
    pub fn max_samples(&self) -> usize {
        self.sample_limit_count
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SampleEventType {
    DurationBegin,
    DurationEnd,
    CompleteDuration,
    Instant,
    AsyncStart,
    AsyncInstant,
    AsyncEnd,
    FlowStart,
    FlowInstant,
    FlowEnd,
    ObjectCreated,
    ObjectSnapshot,
    ObjectDestroyed,
    Metadata
}

impl SampleEventType {
    // TODO(vlovich): Replace all of this with serde flatten + rename once
    // https://github.com/serde-rs/serde/issues/1189 is fixed.
    #[inline]
    fn into_chrome_id(&self) -> char {
        match *self {
            SampleEventType::DurationBegin => 'B',
            SampleEventType::DurationEnd => 'E',
            SampleEventType::CompleteDuration => 'X',
            SampleEventType::Instant => 'i',
            SampleEventType::AsyncStart => 'b',
            SampleEventType::AsyncInstant => 'n',
            SampleEventType::AsyncEnd => 'e',
            SampleEventType::FlowStart => 's',
            SampleEventType::FlowInstant => 't',
            SampleEventType::FlowEnd => 'f',
            SampleEventType::ObjectCreated => 'N',
            SampleEventType::ObjectSnapshot => 'O',
            SampleEventType::ObjectDestroyed => 'D',
            SampleEventType::Metadata => 'M'
        }
    }

    #[inline]
    fn from_chrome_id(symbol: char) -> Self {
        match symbol {
            'B' => SampleEventType::DurationBegin,
            'E' => SampleEventType::DurationEnd,
            'X' => SampleEventType::CompleteDuration,
            'i' => SampleEventType::Instant,
            'b' => SampleEventType::AsyncStart,
            'n' => SampleEventType::AsyncInstant,
            'e' => SampleEventType::AsyncEnd,
            's' => SampleEventType::FlowStart,
            't' => SampleEventType::FlowInstant,
            'f' => SampleEventType::FlowEnd,
            'N' => SampleEventType::ObjectCreated,
            'O' => SampleEventType::ObjectSnapshot,
            'D' => SampleEventType::ObjectDestroyed,
            'M' => SampleEventType::Metadata,
            _ => panic!("Unexpected chrome sample type '{}'", symbol)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum MetadataType {
    ProcessName { name: String },
    #[allow(dead_code)]
    ProcessLabels { labels: String },
    #[allow(dead_code)]
    ProcessSortIndex { sort_index: i32 },
    ThreadName { name: String },
    #[allow(dead_code)]
    ThreadSortIndex { sort_index: i32 },
}

impl MetadataType {
    fn sample_name(&self) -> &'static str {
        match *self {
            MetadataType::ProcessName {..} => "process_name",
            MetadataType::ProcessLabels {..} => "process_labels",
            MetadataType::ProcessSortIndex {..} => "process_sort_index",
            MetadataType::ThreadName {..} => "thread_name",
            MetadataType::ThreadSortIndex {..} => "thread_sort_index",
        }
    }

    fn consume(self) -> (Option<String>, Option<i32>) {
        match self {
            MetadataType::ProcessName {name} => (Some(name), None),
            MetadataType::ThreadName {name} => (Some(name), None),
            MetadataType::ProcessSortIndex {sort_index} => (None, Some(sort_index)),
            MetadataType::ThreadSortIndex {sort_index} => (None, Some(sort_index)),
            MetadataType::ProcessLabels {..} => (None, None)
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SampleArgs {
    /// An arbitrary payload to associate with the sample.  The type is
    /// controlled by features (default string).
    #[serde(rename = "xi_payload")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<TracePayloadT>,

    /// The name to associate with the pid/tid.  Whether it's associated with
    /// the pid or the tid depends on the name of the event
    /// via process_name/thread_name respectively.
    #[serde(rename = "name")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata_name: Option<StrCow>,

    /// Sorting priority between processes/threads in the view.
    #[serde(rename = "sort_index")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata_sort_index: Option<i32>,
}

#[inline]
fn ns_to_us(ns: u64) -> u64 {
    ns / 1000
}

fn serialize_event_type<S>(ph: &SampleEventType, s: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
    s.serialize_char(ph.into_chrome_id())
}

fn deserialize_event_type<'de, D>(d: D) -> Result<SampleEventType, D::Error>
    where D: serde::Deserializer<'de> {
    serde::Deserialize::deserialize(d).map(|ph : char| SampleEventType::from_chrome_id(ph))
}

/// Stores the relevant data about a sample for later serialization.
/// The payload associated with any sample is by default a string but may be
/// configured via the `dict_payload` or `json_payload` features (there is an
/// associated performance hit across the board for turning it on).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Sample {
    /// The name of the event to be shown.
    pub name: StrCow,
    /// List of categories the event applies to.
    #[serde(rename = "cat")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub categories: Option<CategoriesT>,
    /// When was the sample started.
    #[serde(rename = "ts")]
    pub timestamp_us: u64,
    /// What kind of sample this is.
    #[serde(rename = "ph")]
    #[serde(serialize_with = "serialize_event_type")]
    #[serde(deserialize_with = "deserialize_event_type")]
    pub event_type: SampleEventType,
    #[serde(rename = "dur")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_us: Option<u64>,
    /// The process the sample was captured in.
    pub pid: u64,
    /// The thread the sample was captured on.  Omitted for Metadata events that
    /// want to set the process name (if provided then sets the thread name).
    pub tid: u64,
    #[serde(skip_serializing)]
    pub thread_name: Option<StrCow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<SampleArgs>
}

fn to_cow_str<S>(s: S) -> StrCow where S: Into<StrCow> {
    s.into()
}

impl Sample {

    fn thread_name() -> Option<StrCow> {
        let thread = std::thread::current();
        thread.name().map(|ref s| to_cow_str(s.to_string()))
    }

    /// Constructs a Begin or End sample.  Should not be used directly.  Instead
    /// should be constructed via SampleGuard.
    pub fn new_duration_marker<S, C>(name: S,
                                     categories: C,
                                     payload: Option<TracePayloadT>,
                                     event_type: SampleEventType)
        -> Self
        where S: Into<StrCow>, C: Into<CategoriesT>
    {
        Self {
            name: name.into(),
            categories: Some(categories.into()),
            timestamp_us: ns_to_us(time::precise_time_ns()),
            event_type: event_type,
            duration_us: None,
            tid: sys_tid::current_tid().unwrap(),
            thread_name: Sample::thread_name(),
            pid: sys_pid::current_pid(),
            args: Some(SampleArgs {
                payload: payload,
                metadata_name: None,
                metadata_sort_index: None,
            }),
        }
    }

    /// Constructs a Duration sample.  For use via xi_trace::closure.
    pub fn new_duration<S, C>(name: S,
                              categories: C,
                              payload: Option<TracePayloadT>,
                              start_ns: u64,
                              duration_ns: u64) -> Self
        where S: Into<StrCow>, C: Into<CategoriesT>
    {
        Self {
            name: name.into(),
            categories: Some(categories.into()),
            timestamp_us: ns_to_us(start_ns),
            event_type: SampleEventType::CompleteDuration,
            duration_us: Some(ns_to_us(duration_ns)),
            tid: sys_tid::current_tid().unwrap(),
            thread_name: Sample::thread_name(),
            pid: sys_pid::current_pid(),
            args: Some(SampleArgs {
                payload: payload,
                metadata_name: None,
                metadata_sort_index: None,
            }),
        }
    }

    /// Constructs an instantaneous sample.
    pub fn new_instant<S, C>(name: S, categories: C,
                          payload: Option<TracePayloadT>) -> Self
        where S: Into<StrCow>, C: Into<CategoriesT>
    {
        Self {
            name: name.into(),
            categories: Some(categories.into()),
            timestamp_us: ns_to_us(time::precise_time_ns()),
            event_type: SampleEventType::Instant,
            duration_us: None,
            tid: sys_tid::current_tid().unwrap(),
            thread_name: Sample::thread_name(),
            pid: sys_pid::current_pid(),
            args: Some(SampleArgs {
                payload: payload,
                metadata_name: None,
                metadata_sort_index: None,
            }),
        }
    }

    fn new_metadata(timestamp_ns: u64, meta: MetadataType, tid: u64) -> Self {
        let sample_name = to_cow_str(meta.sample_name());
        let (metadata_name, sort_index) = meta.consume();

        Self {
            name: sample_name,
            categories: None,
            timestamp_us: ns_to_us(timestamp_ns),
            event_type: SampleEventType::Metadata,
            duration_us: None,
            tid: tid,
            thread_name: None,
            pid: sys_pid::current_pid(),
            args: Some(SampleArgs {
                payload: None,
                metadata_name: metadata_name.map(|s| Cow::Owned(s)),
                metadata_sort_index: sort_index,
            }),
        }
    }
}

impl PartialEq for Sample {
    fn eq(&self, other: &Sample) -> bool {
        self.timestamp_us == other.timestamp_us &&
            self.name == other.name &&
            self.categories == other.categories &&
            self.pid == other.pid &&
            self.tid == other.tid &&
            self.event_type == other.event_type &&
            self.args == other.args
    }
}

impl Eq for Sample {}

impl PartialOrd for Sample {
    fn partial_cmp(&self, other: &Sample) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Sample {
    fn cmp(&self, other: &Sample) -> cmp::Ordering {
        self.timestamp_us.cmp(&other.timestamp_us)
    }
}

impl Hash for Sample {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.pid, self.timestamp_us).hash(state);
    }
}

#[must_use]
pub struct SampleGuard<'a> {
    sample: Option<Sample>,
    trace: Option<&'a Trace>,
}

impl<'a> SampleGuard<'a> {
    #[inline]
    pub fn new_disabled() -> Self {
        Self {
            sample: None,
            trace: None,
        }
    }

    #[inline]
    fn new<S, C>(trace: &'a Trace, name: S, categories: C, payload: Option<TracePayloadT>)
        -> Self
        where S: Into<StrCow>, C: Into<CategoriesT>
    {
        // TODO(vlovich): optimize this path to use the Complete event type
        // rather than emitting an explicit start/stop to reduce the size of
        // the generated JSON.
        let guard = Self {
            sample: Some(Sample::new_duration_marker(
                name, categories, payload, SampleEventType::DurationBegin)),
            trace: Some(&trace),
        };
        trace.record(guard.sample.as_ref().unwrap().clone());
        guard
    }
}

impl<'a> Drop for SampleGuard<'a> {
    fn drop(&mut self) {
        if let Some(ref mut trace) = self.trace {
            let mut sample = self.sample.take().unwrap();
            sample.timestamp_us = ns_to_us(time::precise_time_ns());
            sample.event_type = SampleEventType::DurationEnd;
            trace.record(sample);
        }
    }
}

/// Returns the file name of the EXE if possible, otherwise the full path, or
/// None if an irrecoverable error occured.
fn exe_name() -> Option<String> {
    match std::env::current_exe() {
        Ok(exe_name) => {
            match exe_name.clone().file_name() {
                Some(filename) => {
                    filename.to_str().map(|s| s.to_string())
                },
                None => {
                    let full_path = exe_name.into_os_string();
                    let full_path_str = full_path.into_string();
                    match full_path_str {
                        Ok(s) => Some(s),
                        Err(e) => {
                            warn!("Failed to get string representation: {:?}", e);
                            None
                        },
                    }
                }
            }
        },
        Err(ref e) => {
            warn!("Failed to get path to current exe: {:?}", e);
            None
        },
    }
}

/// Stores the tracing data.
pub struct Trace {
    enabled: AtomicBool,
    samples: Mutex<FixedLifoDeque<Sample>>,
}

impl Trace {
    pub fn disabled() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            samples: Mutex::new(FixedLifoDeque::new())
        }
    }

    pub fn enabled(config: Config) -> Self {
        Self {
            enabled: AtomicBool::new(true),
            samples: Mutex::new(FixedLifoDeque::with_limit(config.max_samples())),
        }
    }

    pub fn disable(&self) {
        let mut all_samples = self.samples.lock().unwrap();
        all_samples.reset_limit(0);
        self.enabled.store(false, AtomicOrdering::Relaxed);
    }

    #[inline]
    pub fn enable(&self) {
        self.enable_config(Config::default());
    }

    pub fn enable_config(&self, config: Config) {
        let mut all_samples = self.samples.lock().unwrap();
        all_samples.reset_limit(config.max_samples());
        self.enabled.store(true, AtomicOrdering::Relaxed);
    }

    /// Generally racy since the underlying storage might be mutated in a separate thread.
    /// Exposed for unit tests.
    pub fn get_samples_count(&self) -> usize {
        self.samples.lock().unwrap().len()
    }

    /// Exposed for unit tests only.
    pub fn get_samples_limit(&self) -> usize {
        self.samples.lock().unwrap().limit()
    }

    #[inline]
    pub(crate) fn record(&self, sample: Sample) {
        let mut all_samples = self.samples.lock().unwrap();
        all_samples.push_back(sample);
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(AtomicOrdering::Relaxed)
    }

    pub fn instant<S, C>(&self, name: S, categories: C)
        where S: Into<StrCow>, C: Into<CategoriesT>
    {
        if self.is_enabled() {
            self.record(Sample::new_instant(name, categories, None));
        }
    }

    pub fn instant_payload<S, C, P>(&self, name: S, categories: C, payload: P)
        where S: Into<StrCow>, C:Into<CategoriesT>, P: Into<TracePayloadT>
    {
        if self.is_enabled() {
            self.record(Sample::new_instant(name, categories, Some(payload.into())));
        }
    }

    pub fn block<S, C>(&self, name: S, categories: C) -> SampleGuard
        where S: Into<StrCow>, C: Into<CategoriesT>
    {
        if !self.is_enabled() {
            SampleGuard::new_disabled()
        } else {
            SampleGuard::new(&self, name, categories, None)
        }
    }

    pub fn block_payload<S, C, P>(&self, name: S, categories: C, payload: P)
        -> SampleGuard
        where S: Into<StrCow>, C: Into<CategoriesT>, P: Into<TracePayloadT>
    {
        if !self.is_enabled() {
            SampleGuard::new_disabled()
        } else {
            SampleGuard::new(&self, name, categories, Some(payload.into()))
        }
    }

    pub fn closure<S, C, F, R>(&self, name: S, categories: C, closure: F) -> R
        where S: Into<StrCow>, C: Into<CategoriesT>, F: FnOnce() -> R
    {
        // TODO: simplify this through the use of scopeguard crate
        let start = time::precise_time_ns();
        let result = closure();
        let end = time::precise_time_ns();
        if self.is_enabled() {
            self.record(Sample::new_duration(
                name, categories, None, start, end - start));
        }
        result
    }

    pub fn closure_payload<S, C, P, F, R>(&self, name: S, categories: C,
                                          closure: F, payload: P)
        -> R
        where S: Into<StrCow>, C: Into<CategoriesT>, P: Into<TracePayloadT>,
              F: FnOnce() -> R
    {
        // TODO: simplify this through the use of scopeguard crate
        let start = time::precise_time_ns();
        let result = closure();
        let end = time::precise_time_ns();
        if self.is_enabled() {
            self.record(Sample::new_duration(
                name, categories, Some(payload.into()), start, end - start));
        }
        result
    }

    pub fn samples_cloned_unsorted<'a>(&'a self) -> Vec<Sample> {
        let all_samples = self.samples.lock().unwrap();
        if all_samples.is_empty() {
            return Vec::with_capacity(0);
        }

        let mut as_vec = Vec::with_capacity(all_samples.len() + 10);
        let first_sample_timestamp = all_samples.front()
            .map_or(0, |ref s| s.timestamp_us);
        let tid = all_samples.front()
            .map_or_else(|| sys_tid::current_tid().unwrap(), |ref s| s.tid);

        if let Some(exe_name) = exe_name() {
            as_vec.push(Sample::new_metadata(
                first_sample_timestamp,
                MetadataType::ProcessName {name: exe_name},
                tid));
        }

        let mut thread_names: HashMap<u64, StrCow> = HashMap::new();

        for sample in all_samples.iter() {
            if let Some(ref thread_name) = sample.thread_name {
                let previous_name = thread_names.insert(sample.tid, thread_name.clone());
                if previous_name.is_none() || previous_name.unwrap() != *thread_name {
                    as_vec.push(Sample::new_metadata(
                        first_sample_timestamp,
                        MetadataType::ThreadName { name: thread_name.to_string() },
                        sample.tid));
                }
            }
        }

        as_vec.extend(all_samples.iter().cloned());
        as_vec
    }

    #[inline]
    pub fn samples_cloned_sorted(&self) -> Vec<Sample> {
        let mut samples = self.samples_cloned_unsorted();
        samples.sort_unstable();
        samples
    }
}

lazy_static! { static ref TRACE : Trace = Trace::disabled(); }

/// Enable tracing with the default configuration.  See Config::default.
/// Tracing is disabled initially on program launch.
#[inline]
pub fn enable_tracing() {
    TRACE.enable();
}

/// Enable tracing with a specific configuration. Tracing is disabled initially
/// on program launch.
#[inline]
pub fn enable_tracing_with_config(config: Config) {
    TRACE.enable_config(config);
}

/// Disable tracing.  This clears all trace data (& frees the memory).
#[inline]
pub fn disable_tracing() {
    TRACE.disable();
}

/// Is tracing enabled.  Technically doesn't guarantee any samples will be
/// stored as tracing could still be enabled but set with a limit of 0.
#[inline]
pub fn is_enabled() -> bool {
    TRACE.is_enabled()
}

/// Create an instantaneous sample without any payload.  This is the lowest
/// overhead tracing routine available.
///
/// # Performance
/// The `dict_payload` or `json_payload` feature makes this ~1.3-~1.5x slower.
/// See `trace_payload` for a more complete discussion.
///
/// # Arguments
///
/// * `name` - A string that provides some meaningful name to this sample.
/// Usage of static strings is encouraged for best performance to avoid copies.
/// However, anything that can be converted into a Cow string can be passed as
/// an argument.
///
/// * `categories` - A static array of static strings that tags the samples in
/// some way.
///
/// # Examples
///
/// ```
/// xi_trace::trace("something happened", &["rpc", "response"]);
/// ```
#[inline]
pub fn trace<S, C>(name: S, categories: C)
    where S: Into<StrCow>, C: Into<CategoriesT>
{
    TRACE.instant(name, categories);
}


/// Create an instantaneous sample with a payload.  The type the payload
/// conforms to is currently determined by the feature this library is compiled
/// with.  By default, the type is string-like just like name.  If compiled with
/// `dict_payload` then a Rust HashMap is expected while the `json_payload`
/// feature makes the payload a `serde_json::Value` (additionally the library
/// acquires a dependency on the `serde_json` crate.
///
/// # Performance
/// A static string has the lowest overhead as no copies are necessary, roughly
/// equivalent performance to a regular trace.  A string that needs to be copied
/// first can make it ~1.7x slower than a regular trace.
///
/// When compiling with `dict_payload` or `json_payload`, this is ~2.1x slower
/// than a string that needs to be copied (or ~4.5x slower than a static string)
///
/// # Arguments
///
/// * `name` - A string that provides some meaningful name to this sample.
/// Usage of static strings is encouraged for best performance to avoid copies.
/// However, anything that can be converted into a Cow string can be passed as
/// an argument.
///
/// * `categories` - A static array of static strings that tags the samples in
/// some way.
///
/// # Examples
///
/// ```
/// xi_trace::trace_payload("something happened", &["rpc", "response"], "a note about this");
/// ```
///
/// With `json_payload` feature:
///
/// ```rust,ignore
/// xi_trace::trace_payload("my event", &["rpc", "response"], json!({"key": "value"}));
/// ```
#[inline]
pub fn trace_payload<S, C, P>(name: S, categories: C, payload: P)
    where S: Into<StrCow>, C: Into<CategoriesT>, P: Into<TracePayloadT>
{
    TRACE.instant_payload(name, categories, payload);
}

/// Creates a duration sample.  The sample is finalized (end_ns set) when the
/// returned value is dropped.  `trace_closure` may be prettier to read.
///
/// # Performance
/// See `trace_payload` for a more complete discussion.
///
/// # Arguments
///
/// * `name` - A string that provides some meaningful name to this sample.
/// Usage of static strings is encouraged for best performance to avoid copies.
/// However, anything that can be converted into a Cow string can be passed as
/// an argument.
///
/// * `categories` - A static array of static strings that tags the samples in
/// some way.
///
/// # Returns
/// A guard that when dropped will update the Sample with the timestamp & then
/// record it.
///
/// # Examples
///
/// ```
/// fn something_expensive() {
/// }
///
/// fn something_else_expensive() {
/// }
///
/// let trace_guard = xi_trace::trace_block("something_expensive", &["rpc", "request"]);
/// something_expensive();
/// std::mem::drop(trace_guard); // finalize explicitly if
///
/// {
///     let _guard = xi_trace::trace_block("something_else_expensive", &["rpc", "response"]);
///     something_else_expensive();
/// }
/// ```
#[inline]
pub fn trace_block<'a, S, C>(name: S, categories: C) -> SampleGuard<'a>
    where S: Into<StrCow>, C: Into<CategoriesT>
{
    TRACE.block(name, categories)
}


/// See `trace_block` for how the block works and `trace_payload` for a
/// discussion on payload.
#[inline]
pub fn trace_block_payload<'a, S, C, P>(name: S, categories: C, payload: P)
    -> SampleGuard<'a>
    where S: Into<StrCow>, C: Into<CategoriesT>, P: Into<TracePayloadT>
{
    TRACE.block_payload(name, categories, payload)
}

/// Creates a duration sample that measures how long the closure took to execute.
///
/// # Performance
/// See `trace_payload` for a more complete discussion.
///
/// # Arguments
///
/// * `name` - A string that provides some meaningful name to this sample.
/// Usage of static strings is encouraged for best performance to avoid copies.
/// However, anything that can be converted into a Cow string can be passed as
/// an argument.
///
/// * `categories` - A static array of static strings that tags the samples in
/// some way.
///
/// # Returns
/// The result of the closure.
///
/// # Examples
///
/// ```
/// fn something_expensive() -> u32 {
///     0
/// }
///
/// fn something_else_expensive(value: u32) {
/// }
///
/// let result = xi_trace::trace_closure("something_expensive", &["rpc", "request"], || {
///     something_expensive()
/// });
/// xi_trace::trace_closure("something_else_expensive", &["rpc", "response"], || {
///     something_else_expensive(result);
/// });
/// ```
#[inline]
pub fn trace_closure<S, C, F, R>(name: S, categories: C, closure: F) -> R
    where S: Into<StrCow>, C: Into<CategoriesT>, F: FnOnce() -> R
{
    TRACE.closure(name, categories, closure)
}

/// See `trace_closure` for how the closure works and `trace_payload` for a
/// discussion on payload.
#[inline]
pub fn trace_closure_payload<S, C, P, F, R>(name: S, categories: C,
                                            closure: F, payload: P) -> R
    where S: Into<StrCow>, C: Into<CategoriesT>, P: Into<TracePayloadT>,
          F: FnOnce() -> R
{
    TRACE.closure_payload(name, categories, closure, payload)
}

#[inline]
pub fn samples_len() -> usize {
    TRACE.get_samples_count()
}

/// Returns all the samples collected so far.  There is no guarantee that the
/// samples are ordered chronologically for several reasons:
///
/// 1. Samples that span sections of code may be inserted on end instead of
/// beginning.
/// 2. Performance optimizations might have per-thread buffers.  Keeping all
/// that sorted would be prohibitively expensive.
/// 3. You may not care about them always being sorted if you're merging samples
/// from multiple distributed sources (i.e. you want to sort the merged result
/// rather than just this processe's samples).
#[inline]
pub fn samples_cloned_unsorted() -> Vec<Sample> {
    TRACE.samples_cloned_unsorted()
}

/// Returns all the samples collected so far ordered chronologically by
/// creation.  Roughly corresponds to start_ns but instead there's a
/// monotonically increasing single global integer (when tracing) per creation
/// of Sample that determines order.
#[inline]
pub fn samples_cloned_sorted() -> Vec<Sample> {
    TRACE.samples_cloned_sorted()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "benchmarks")]
    use test::Bencher;
    #[cfg(feature = "benchmarks")]
    use test::black_box;

    #[cfg(all(not(feature = "dict_payload"), not(feature = "json_payload")))]
    fn to_payload(value: &'static str) -> &'static str {
        value
    }

    #[cfg(feature = "dict_payload")]
    fn to_payload(value: &'static str) -> TracePayloadT {
        let mut d = TracePayloadT::with_capacity(1);
        d.insert(StrCow::from("test"), StrCow::from(value));
        d
    }

    #[cfg(feature = "json_payload")]
    fn to_payload(value: &'static str) -> TracePayloadT {
        json!({"test": value})
    }

    #[test]
    fn test_samples_pulse() {
        let trace = Trace::enabled(Config::with_limit_count(10));
        for _i in 0..50 {
            trace.instant("test_samples_pulse", &["test"]);
        }
    }

    #[test]
    fn test_samples_block() {
        let trace = Trace::enabled(Config::with_limit_count(10));
        for _i in 0..50 {
            let _ = trace.block("test_samples_block", &["test"]);
        }
    }

    #[test]
    fn test_samples_closure() {
        let trace = Trace::enabled(Config::with_limit_count(10));
        for _i in 0..50 {
            trace.closure("test_samples_closure", &["test"], || {});
        }
    }

    #[test]
    fn test_disable_drops_all_samples() {
        let trace = Trace::enabled(Config::with_limit_count(10));
        assert_eq!(trace.is_enabled(), true);
        trace.instant("1", &["test"]);
        trace.instant("2", &["test"]);
        trace.instant("3", &["test"]);
        trace.instant("4", &["test"]);
        trace.instant("5", &["test"]);
        assert_eq!(trace.get_samples_count(), 5);
        // 1 for exe name & 1 for the thread name
        assert_eq!(trace.samples_cloned_unsorted().len(), 7);
        trace.disable();
        assert_eq!(trace.get_samples_count(), 0);
    }

    #[test]
    fn test_get_samples() {
        let trace = Trace::enabled(Config::with_limit_count(20));
        assert_eq!(trace.samples_cloned_unsorted().len(), 0);

        assert_eq!(trace.is_enabled(), true);
        assert_eq!(trace.get_samples_limit(), 20);
        assert_eq!(trace.samples_cloned_unsorted().len(), 0);

        trace.closure_payload("x", &["test"], || (),
                              to_payload("test_get_samples"));
        assert_eq!(trace.get_samples_count(), 1);
        // +2 for exe & thread name.
        assert_eq!(trace.samples_cloned_unsorted().len(), 3);

        trace.closure_payload("y", &["test"], || {},
                              to_payload("test_get_samples"));
        assert_eq!(trace.samples_cloned_unsorted().len(), 4);

        trace.closure_payload("z", &["test"], || {},
                              to_payload("test_get_samples"));

        let snapshot = trace.samples_cloned_unsorted();
        assert_eq!(snapshot.len(), 5);

        assert_eq!(snapshot[0].name, "process_name");
        assert_eq!(snapshot[0].args.as_ref().unwrap().metadata_name.as_ref().is_some(), true);
        assert_eq!(snapshot[1].name, "thread_name");
        assert_eq!(snapshot[1].args.as_ref().unwrap().metadata_name.as_ref().is_some(), true);
        assert_eq!(snapshot[2].name, "x");
        assert_eq!(snapshot[3].name, "y");
        assert_eq!(snapshot[4].name, "z");
    }

    #[test]
    fn test_trace_disabled() {
        let trace = Trace::disabled();
        assert_eq!(trace.get_samples_limit(), 0);
        assert_eq!(trace.get_samples_count(), 0);

        {
            trace.instant("something", &[]);
            let _x = trace.block("something", &[]);
            trace.closure("something", &[], || ());
        }

        assert_eq!(trace.get_samples_count(), 0);
    }

    #[test]
    fn test_get_samples_nested_trace() {
        let trace = Trace::enabled(Config::with_limit_count(11));
        assert_eq!(trace.is_enabled(), true);
        assert_eq!(trace.get_samples_limit(), 11);

        // current recording mechanism should see:
        // a, b, y, z, c, x
        // even though the actual sampling order (from timestamp of
        // creation) is:
        // x, a, y, b, z, c
        // This might be an over-specified test as it will
        // probably change as the recording internals change.
        trace.closure_payload("x", &["test"], || {
            trace.instant_payload("a", &["test"], to_payload("test_get_samples_nested_trace"));
            trace.closure_payload("y", &["test"], || {
                trace.instant_payload("b", &["test"], to_payload("test_get_samples_nested_trace"));
            }, to_payload("test_get_samples_nested_trace"));
            trace.block_payload("z", &["test"], to_payload("test_get_samples_nested_trace"));
            trace.instant_payload("c", &["test"], to_payload("test_get_samples_nested_trace"));
        }, to_payload("test_get_samples_nested_trace"));

        let snapshot = trace.samples_cloned_unsorted();
        // +2 for exe & thread name
        assert_eq!(snapshot.len(), 9);

        assert_eq!(snapshot[0].name, "process_name");
        assert_eq!(snapshot[0].args.as_ref().unwrap().metadata_name.as_ref().is_some(), true);
        assert_eq!(snapshot[1].name, "thread_name");
        assert_eq!(snapshot[1].args.as_ref().unwrap().metadata_name.as_ref().is_some(), true);
        assert_eq!(snapshot[2].name, "a");
        assert_eq!(snapshot[3].name, "b");
        assert_eq!(snapshot[4].name, "y");
        assert_eq!(snapshot[5].name, "z");
        assert_eq!(snapshot[6].name, "z");
        assert_eq!(snapshot[7].name, "c");
        assert_eq!(snapshot[8].name, "x");
    }

    #[test]
    fn test_get_sorted_samples() {
        let trace = Trace::enabled(Config::with_limit_count(10));

        // current recording mechanism should see:
        // a, b, y, z, c, x
        // even though the actual sampling order (from timestamp of
        // creation) is:
        // x, a, y, b, z, c
        // This might be an over-specified test as it will
        // probably change as the recording internals change.

        // NOTE: 1 us sleeps are inserted as the first line of a closure to
        // ensure that when the samples are sorted by time they come out in a
        // stable order since the resolution of timestamps is 1us.
        // NOTE 2: from_micros is currently in unstable so using new
        trace.closure_payload("x", &["test"], || {
            std::thread::sleep(std::time::Duration::new(0, 1000));
            trace.instant_payload("a", &["test"], to_payload("test_get_sorted_samples"));
            trace.closure_payload("y", &["test"], || {
                std::thread::sleep(std::time::Duration::new(0, 1000));
                trace.instant_payload("b", &["test"], to_payload("test_get_sorted_samples"));
            }, to_payload("test_get_sorted_samples"));
            trace.block_payload("z", &["test"], to_payload("test_get_sorted_samples"));
            trace.instant("c", &["test"]);
        }, to_payload("test_get_sorted_samples"));

        let snapshot = trace.samples_cloned_sorted();
        // +2 for exe & thread name.
        assert_eq!(snapshot.len(), 9);

        assert_eq!(snapshot[0].name, "process_name");
        assert_eq!(snapshot[0].args.as_ref().unwrap().metadata_name.as_ref().is_some(), true);
        assert_eq!(snapshot[1].name, "thread_name");
        assert_eq!(snapshot[1].args.as_ref().unwrap().metadata_name.as_ref().is_some(), true);
        assert_eq!(snapshot[2].name, "x");
        assert_eq!(snapshot[3].name, "a");
        assert_eq!(snapshot[4].name, "y");
        assert_eq!(snapshot[5].name, "b");
        assert_eq!(snapshot[6].name, "z");
        assert_eq!(snapshot[7].name, "z");
        assert_eq!(snapshot[8].name, "c");
    }

    #[test]
    fn test_cross_process_samples() {
        let mut samples = vec![
            Sample::new_instant("local pid", &[], None),
            Sample::new_instant("remote pid", &[], None)];
        samples[0].pid = 1;
        samples[0].timestamp_us = 10;

        samples[1].pid = 2;
        samples[1].timestamp_us = 5;

        samples.sort();

        assert_eq!(samples[0].name, "remote pid");
        assert_eq!(samples[1].name, "local pid");
    }

    #[cfg(feature = "benchmarks")]
    #[bench]
    fn bench_trace_instant_disabled(b: &mut Bencher) {
        let trace = Trace::disabled();

        b.iter(|| black_box(trace.instant("nothing", &["benchmark"])));
    }

    #[cfg(feature = "benchmarks")]
    #[bench]
    fn bench_trace_instant(b: &mut Bencher) {
        let trace = Trace::enabled(Config::default());
        b.iter(|| black_box(trace.instant("something", &["benchmark"])));
    }

    #[cfg(feature = "benchmarks")]
    #[bench]
    fn bench_trace_instant_with_payload(b: &mut Bencher) {
        let trace = Trace::enabled(Config::default());
        b.iter(|| black_box(trace.instant_payload(
            "something", &["benchmark"],
            to_payload("some description of the trace"))));
    }

    #[cfg(feature = "benchmarks")]
    #[bench]
    fn bench_trace_block_disabled(b: &mut Bencher) {
        let trace = Trace::disabled();
        b.iter(|| black_box(trace.block("something", &["benchmark"])));
    }

    #[cfg(feature = "benchmarks")]
    #[bench]
    fn bench_trace_block(b: &mut Bencher) {
        let trace = Trace::enabled(Config::default());
        b.iter(|| black_box(trace.block("something", &["benchmark"])));
    }


    #[cfg(feature = "benchmarks")]
    #[bench]
    fn bench_trace_block_payload(b: &mut Bencher) {
        let trace = Trace::enabled(Config::default());
        b.iter(|| {
            black_box(trace.block_payload(
                    "something", &["benchmark"],
                    to_payload(("some payload for the block"))));
        });
    }

    #[cfg(feature = "benchmarks")]
    #[bench]
    fn bench_trace_closure_disabled(b: &mut Bencher) {
        let trace = Trace::disabled();

        b.iter(|| black_box(trace.closure("something", &["benchmark"], || {})));
    }

    #[cfg(feature = "benchmarks")]
    #[bench]
    fn bench_trace_closure(b: &mut Bencher) {
        let trace = Trace::enabled(Config::default());
        b.iter(|| black_box(trace.closure("something", &["benchmark"], || {})));
    }

    #[cfg(feature = "benchmarks")]
    #[bench]
    fn bench_trace_closure_payload(b: &mut Bencher) {
        let trace = Trace::enabled(Config::default());
        b.iter(|| black_box(trace.closure_payload(
                    "something", &["benchmark"], || {},
                    to_payload(("some description of the closure")))));
    }

    // this is the cost contributed by the timestamp to trace()
    #[cfg(feature = "benchmarks")]
    #[bench]
    fn bench_single_timestamp(b: &mut Bencher) {
        b.iter(|| black_box(time::precise_time_ns()));
    }

    // this is the cost contributed by the timestamp to
    // trace_block()/trace_closure
    #[cfg(feature = "benchmarks")]
    #[bench]
    fn bench_two_timestamps(b: &mut Bencher) {
        b.iter(|| {
            black_box(time::precise_time_ns());
            black_box(time::precise_time_ns());
        });
    }

    #[cfg(feature = "benchmarks")]
    #[bench]
    fn bench_get_tid(b: &mut Bencher) {
        b.iter(|| black_box(sys_tid::current_tid()));
    }

    #[cfg(feature = "benchmarks")]
    #[bench]
    fn bench_get_pid(b: &mut Bencher) {
        b.iter(|| sys_pid::current_pid());
    }
}
// Copyright 2016 The xi-editor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::cmp::{min,max};
use std::cell::RefCell;
use std::ops::Range;

use serde_json::Value;

use xi_rope::rope::{Rope, LinesMetric, RopeInfo};
use xi_rope::delta::Delta;
use xi_rope::tree::Cursor;
use xi_rope::breaks::{Breaks, BreaksInfo, BreaksMetric, BreaksBaseMetric};
use xi_rope::interval::Interval;
use xi_rope::spans::Spans;
use xi_trace::trace_block;
use client::Client;
use edit_types::ViewEvent;
use line_cache_shadow::{self, LineCacheShadow, RenderPlan, RenderTactic};
use movement::{Movement, region_movement, selection_movement};
use rpc::{GestureType, MouseAction, SelectionModifier};
use styles::{Style, ThemeStyleMap};
use selection::{Affinity, Selection, SelRegion};
use tabs::{ViewId, BufferId};
use width_cache::WidthCache;
use word_boundaries::WordCursor;
use find::Find;
use linewrap;
use internal::find::FindStatus;

type StyleMap = RefCell<ThemeStyleMap>;


/// A flag used to indicate when legacy actions should modify selections
const FLAG_SELECT: u64 = 2;

pub struct View {
    view_id: ViewId,
    buffer_id: BufferId,

    /// Tracks whether this view has been scheduled to render.
    /// We attempt to reduce duplicate renders by setting a small timeout
    /// after an edit is applied, to allow batching with any plugin updates.
    pending_render: bool,
    size: Size,
    /// The selection state for this view. Invariant: non-empty.
    selection: Selection,

    drag_state: Option<DragState>,

    /// vertical scroll position
    first_line: usize,
    /// height of visible portion
    height: usize,
    breaks: Option<Breaks>,
    wrap_col: WrapWidth,

    /// Front end's line cache state for this view. See the `LineCacheShadow`
    /// description for the invariant.
    lc_shadow: LineCacheShadow,

    /// New offset to be scrolled into position after an edit.
    scroll_to: Option<usize>,

    /// The state for finding text for this view.
    /// Each instance represents a separate search query.
    find: Vec<Find>,

    /// Tracks whether there has been changes in find results or find parameters.
    /// This is used to determined whether FindStatus should be sent to the frontend.
    find_changed: FindStatusChange,

    /// Tracks whether find highlights should be rendered.
    /// Highlights are only rendered when search dialog is open.
    highlight_find: bool,

    /// The state for replacing matches for this view.
    replace: Option<Replace>,

    /// Tracks whether the replacement string or replace parameters changed.
    replace_changed: bool,
}

/// Indicates what changed in the find state.
#[derive(PartialEq, Debug)]
enum FindStatusChange {
    /// None of the find parameters or number of matches changed.
    None,

    /// Find parameters and number of matches changed.
    All,

    /// Only number of matches changed
    Matches
}

/// Contains replacement string and replace options.
#[derive(Debug, Default, PartialEq, Serialize, Deserialize, Clone)]
pub struct Replace {
    /// Replacement string.
    pub chars: String,
    pub preserve_case: bool
}

/// A size, in pixel units (not display pixels).
#[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Size {
    pub width: f64,
    pub height: f64,
}

/// The visual width of the buffer for the purpose of word wrapping.
enum WrapWidth {
    /// No wrapping in effect.
    None,

    /// Width in bytes (utf-8 code units).
    ///
    /// Only works well for ASCII, will probably not be maintained long-term.
    Bytes(usize),

    /// Width in px units, requiring measurement by the front-end.
    Width(f64),
}

/// State required to resolve a drag gesture into a selection.
struct DragState {
    /// All the selection regions other than the one being dragged.
    base_sel: Selection,

    /// Offset of the point where the drag started.
    offset: usize,

    /// Start of the region selected when drag was started (region is
    /// assumed to be forward).
    min: usize,

    /// End of the region selected when drag was started.
    max: usize,
}

impl View {
    pub fn new(view_id: ViewId, buffer_id: BufferId) -> View {
        View {
            view_id: view_id,
            buffer_id: buffer_id,
            pending_render: false,
            selection: SelRegion::caret(0).into(),
            scroll_to: Some(0),
            size: Size::default(),
            drag_state: None,
            first_line: 0,
            height: 10,
            breaks: None,
            wrap_col: WrapWidth::None,
            lc_shadow: LineCacheShadow::default(),
            find: Vec::new(),
            find_changed: FindStatusChange::None,
            highlight_find: false,
            replace: None,
            replace_changed: false,
        }
    }

    pub(crate) fn get_buffer_id(&self) -> BufferId {
        self.buffer_id
    }

    pub(crate) fn get_view_id(&self) -> ViewId {
        self.view_id
    }

    pub(crate) fn get_replace(&self) -> Option<Replace> {
        self.replace.clone()
    }

    pub(crate) fn set_has_pending_render(&mut self, pending: bool) {
        self.pending_render = pending
    }

    pub(crate) fn has_pending_render(&self) -> bool {
        self.pending_render
    }

    pub(crate) fn do_edit(&mut self, text: &Rope, cmd: ViewEvent) {
        use self::ViewEvent::*;
        match cmd {
            Move(movement) => self.do_move(text, movement, false),
            ModifySelection(movement) => self.do_move(text, movement, true),
            SelectAll => self.select_all(text),
            Scroll(range) => self.set_scroll(range.first, range.last),
            AddSelectionAbove =>
                self.add_selection_by_movement(text, Movement::Up),
            AddSelectionBelow =>
                self.add_selection_by_movement(text, Movement::Down),
            Gesture { line, col, ty } =>
                self.do_gesture(text, line, col, ty),
            GotoLine { line } => self.goto_line(text, line),
            Find { chars, case_sensitive, regex, whole_words } =>
                self.do_find(text, chars, case_sensitive, regex, whole_words),
            FindNext { wrap_around, allow_same, modify_selection } =>
                self.do_find_next(text, false, wrap_around, allow_same, &modify_selection),
            FindPrevious { wrap_around, allow_same, modify_selection } =>
                self.do_find_next(text, true, wrap_around, allow_same, &modify_selection),
            FindAll => self.do_find_all(text),
            Click(MouseAction { line, column, flags, click_count }) => {
                // Deprecated (kept for client compatibility):
                // should be removed in favor of do_gesture
                warn!("Usage of click is deprecated; use do_gesture");
                if (flags & FLAG_SELECT) != 0 {
                    self.do_gesture(text, line, column, GestureType::RangeSelect)
                } else if click_count == Some(2) {
                    self.do_gesture(text, line, column, GestureType::WordSelect)
                } else if click_count == Some(3) {
                    self.do_gesture(text, line, column, GestureType::LineSelect)
                } else {
                    self.do_gesture(text, line, column, GestureType::PointSelect)
                }
            }
            Drag(MouseAction { line, column, .. }) =>
                self.do_drag(text, line, column, Affinity::default()),
            Cancel => self.do_cancel(text),
            HighlightFind { visible } => {
                self.highlight_find = visible;
                self.find_changed = FindStatusChange::All;
                self.set_dirty(text);
            },
            SelectionForFind { case_sensitive } =>
                self.do_selection_for_find(text, case_sensitive),
            Replace { chars, preserve_case } =>
                self.do_set_replace(chars, preserve_case),
            SelectionForReplace => self.do_selection_for_replace(text),
            SelectionIntoLines => self.do_split_selection_into_lines(text),
        }
    }

    fn do_gesture(&mut self, text: &Rope, line: u64, col: u64, ty: GestureType) {
        let line = line as usize;
        let col = col as usize;
        let offset = self.line_col_to_offset(text, line, col);
        match ty {
            GestureType::PointSelect => {
                self.set_selection(text, SelRegion::caret(offset));
                self.start_drag(offset, offset, offset);
            },
            GestureType::RangeSelect => self.select_range(text, offset),
            GestureType::ToggleSel => self.toggle_sel(text, offset),
            GestureType::LineSelect =>
                self.select_line(text, offset, line, false),
            GestureType::WordSelect =>
                self.select_word(text, offset, false),
            GestureType::MultiLineSelect =>
                self.select_line(text, offset, line, true),
            GestureType::MultiWordSelect =>
                self.select_word(text, offset, true)
        }
    }

    fn do_cancel(&mut self, text: &Rope) {
        // if we have active find highlights, we don't collapse selections
        if self.find.is_empty() {
            self.collapse_selections(text);
        } else {
            self.unset_find();
        }
    }

    pub(crate) fn unset_find(&mut self) {
        for mut find in self.find.iter_mut() {
            find.unset();
        }
        self.find.clear();
    }

    fn goto_line(&mut self, text: &Rope, line: u64) {
        let offset = self.line_col_to_offset(text, line as usize, 0);
        self.set_selection(text, SelRegion::caret(offset));
    }

    pub fn set_size(&mut self, size: Size) {
        self.size = size;
    }

    pub fn set_scroll(&mut self, first: i64, last: i64) {
        let first = max(first, 0) as usize;
        let last = max(last, 0) as usize;
        self.first_line = first;
        self.height = last - first;
    }

    pub fn scroll_height(&self) -> usize {
        self.height
    }

    fn scroll_to_cursor(&mut self, text: &Rope) {
        let end = self.sel_regions().last().unwrap().end;
        let line = self.line_of_offset(text, end);
        if line < self.first_line {
            self.first_line = line;
        } else if self.first_line + self.height <= line {
            self.first_line = line - (self.height - 1);
        }
        // We somewhat arbitrarily choose the last region for setting the old-style
        // selection state, and for scrolling it into view if needed. This choice can
        // likely be improved.
        self.scroll_to = Some(end);
    }

    /// Toggles a caret at the given offset.
    pub fn toggle_sel(&mut self, text: &Rope, offset: usize) {
        // We could probably reduce the cloning of selections by being clever.
        let mut selection = self.selection.clone();
        if !selection.regions_in_range(offset, offset).is_empty() {
            selection.delete_range(offset, offset, true);
            if !selection.is_empty() {
                self.drag_state = None;
                self.set_selection_raw(text, selection);
                return;
            }
        }
        self.drag_state = Some(DragState {
            base_sel: selection.clone(),
            offset,
            min: offset,
            max: offset,
        });
        let region = SelRegion::caret(offset);
        selection.add_region(region);
        self.set_selection_raw(text, selection);
    }

    /// Move the selection by the given movement. Return value is the offset of
    /// a point that should be scrolled into view.
    ///
    /// If `modify` is `true`, the selections are modified, otherwise the results
    /// of individual region movements become carets.
    pub fn do_move(&mut self, text: &Rope, movement: Movement, modify: bool) {
        self.drag_state = None;
        let new_sel = selection_movement(movement, &self.selection,
                                         self, text, modify);
        self.set_selection(text, new_sel);
    }

    /// Set the selection to a new value.
    pub fn set_selection<S: Into<Selection>>(&mut self, text: &Rope, sel: S) {
        self.set_selection_raw(text, sel.into());
        self.scroll_to_cursor(text);
    }

    /// Sets the selection to a new value, without invalidating.
    fn set_selection_for_edit(&mut self, text: &Rope, sel: Selection) {
        self.selection = sel;
        self.scroll_to_cursor(text);
    }

    /// Sets the selection to a new value, invalidating the line cache as needed.
    /// This function does not perform any scrolling.
    fn set_selection_raw(&mut self, text: &Rope, sel: Selection) {
        self.invalidate_selection(text);
        self.selection = sel;
        self.invalidate_selection(text);
    }

    /// Invalidate the current selection. Note that we could be even more
    /// fine-grained in the case of multiple cursors, but we also want this
    /// method to be fast even when the selection is large.
    fn invalidate_selection(&mut self, text: &Rope) {
        // TODO: refine for upstream (caret appears on prev line)
        let first_line = self.line_of_offset(text, self.selection.first().unwrap().min());
        let last_line = self.line_of_offset(text, self.selection.last().unwrap().max()) + 1;
        let all_caret = self.selection.iter().all(|region| region.is_caret());
        let invalid = if all_caret {
            line_cache_shadow::CURSOR_VALID
        } else {
            line_cache_shadow::CURSOR_VALID | line_cache_shadow::STYLES_VALID
        };
        self.lc_shadow.partial_invalidate(first_line, last_line, invalid);
    }

    fn add_selection_by_movement(&mut self, text: &Rope, movement: Movement) {
        let mut sel = Selection::new();
        for &region in self.sel_regions() {
            sel.add_region(region);
            let new_region = region_movement(movement, region, self,
                                             &text, false);
            sel.add_region(new_region);
        }
        self.set_selection(text, sel);
    }

    // TODO: insert from keyboard or input method shouldn't break undo group,
    /// Invalidates the styles of the given range (start and end are offsets within
    /// the text).
    pub fn invalidate_styles(&mut self, text: &Rope, start: usize, end: usize) {
        let first_line = self.line_of_offset(text, start);
        let (mut last_line, last_col) = self.offset_to_line_col(text, end);
        last_line += if last_col > 0 { 1 } else { 0 };
        self.lc_shadow.partial_invalidate(first_line, last_line, line_cache_shadow::STYLES_VALID);
    }

    /// Select entire buffer.
    ///
    /// Note: unlike movement based selection, this does not scroll.
    pub fn select_all(&mut self, text: &Rope) {
        let selection = SelRegion::new(0, text.len()).into();
        self.set_selection_raw(text, selection);
    }

    /// Selects a specific range (eg. when the user performs SHIFT + click).
    pub fn select_range(&mut self, text: &Rope, offset: usize) {
        if !self.is_point_in_selection(offset) {
            let sel = {
                let (last, rest) = self.sel_regions().split_last().unwrap();
                let mut sel = Selection::new();
                for &region in rest {
                    sel.add_region(region);
                }
                // TODO: small nit, merged region should be backward if end < start.
                // This could be done by explicitly overriding, or by tweaking the
                // merge logic.
                sel.add_region(SelRegion::new(last.start, offset));
                sel
            };
            self.set_selection(text, sel);
            self.start_drag(offset, offset, offset);
        }
    }

    /// Selects the given region and supports multi selection.
    fn select_region(&mut self, text: &Rope, offset: usize, region: SelRegion, multi_select: bool) {
        let mut selection = match multi_select {
            true => self.selection.clone(),
            false => Selection::new(),
        };

        selection.add_region(region);
        self.set_selection(text, selection);

        self.start_drag(offset, region.start, region.end);
    }

    /// Selects an entire word and supports multi selection.
    pub fn select_word(&mut self, text: &Rope, offset: usize, multi_select: bool) {
        let (start, end) = {
            let mut word_cursor = WordCursor::new(text, offset);
            word_cursor.select_word()
        };

        self.select_region(text, offset, SelRegion::new(start, end), multi_select);
    }

    /// Selects an entire line and supports multi selection.
    pub fn select_line(&mut self, text: &Rope, offset: usize, line: usize, multi_select: bool) {
        let start = self.line_col_to_offset(text, line, 0);
        let end = self.line_col_to_offset(text, line + 1, 0);

        self.select_region(text, offset, SelRegion::new(start, end), multi_select);
    }

    /// Splits current selections into lines.
    fn do_split_selection_into_lines(&mut self, text: &Rope) {
        let mut selection = Selection::new();

        for region in self.selection.iter() {
            if region.is_caret() {
                selection.add_region(SelRegion::caret(region.max()));
            } else {
                let mut cursor = Cursor::new(&text, region.min());

                while cursor.pos() < region.max() {
                    let sel_start = cursor.pos();
                    let end_of_line = match cursor.next::<LinesMetric>() {
                        Some(end) if end >= region.max() => max(0, region.max() - 1),
                        Some(end) => max(0, end - 1),
                        None if cursor.pos() == text.len() => cursor.pos(),
                        _ => break
                    };

                    selection.add_region(SelRegion::new(sel_start, end_of_line));
                }
            }
        }

        self.set_selection_raw(text, selection);
    }

    /// Starts a drag operation.
    pub fn start_drag(&mut self, offset: usize, min: usize, max: usize) {
        let base_sel = Selection::new();
        self.drag_state = Some(DragState { base_sel, offset, min, max });
    }

    /// Does a drag gesture, setting the selection from a combination of the drag
    /// state and new offset.
    fn do_drag(&mut self, text: &Rope, line: u64, col: u64, affinity: Affinity) {
        let offset = self.line_col_to_offset(text, line as usize, col as usize);
        let new_sel = self.drag_state.as_ref().map(|drag_state| {
            let mut sel = drag_state.base_sel.clone();
            // TODO: on double or triple click, quantize offset to requested granularity.
            let (start, end) = if offset < drag_state.offset {
                (drag_state.max, min(offset, drag_state.min))
            } else {
                (drag_state.min, max(offset, drag_state.max))
            };
            let horiz = None;
            sel.add_region(
                SelRegion::new(start, end)
                    .with_horiz(horiz)
                    .with_affinity(affinity)
            );
            sel
        });

        if let Some(sel) = new_sel {
            self.set_selection(text, sel);
        }
    }

    /// Returns the regions of the current selection.
    pub fn sel_regions(&self) -> &[SelRegion] {
        &self.selection
    }

    /// Collapse all selections in this view into a single caret
    pub fn collapse_selections(&mut self, text: &Rope) {
        let mut sel = self.selection.clone();
        sel.collapse();
        self.set_selection(text, sel);
    }

    /// Determines whether the offset is in any selection (counting carets and
    /// selection edges).
    pub fn is_point_in_selection(&self, offset: usize) -> bool {
        !self.selection.regions_in_range(offset, offset).is_empty()
    }

    // Render a single line, and advance cursors to next line.
    fn render_line(&self, client: &Client, styles: &StyleMap,
                   text: &Rope, start_of_line: &mut Cursor<RopeInfo>,
                   soft_breaks: Option<&mut Cursor<BreaksInfo>>,
                   style_spans: &Spans<Style>, line_num: usize) -> Value
    {
        let start_pos = start_of_line.pos();
        let pos = soft_breaks.map_or(start_of_line.next::<LinesMetric>(), |bc| {
            let pos = bc.next::<BreaksMetric>();
            // if using breaks update cursor
            if let Some(pos) = pos { start_of_line.set(pos) }
            pos
        }).unwrap_or(text.len());

        let l_str = text.slice_to_string(start_pos, pos);
        let mut cursors = Vec::new();
        let mut selections = Vec::new();
        for region in self.selection.regions_in_range(start_pos, pos) {
            // cursor
            let c = region.end;
            if (c > start_pos && c < pos) ||
                (!region.is_upstream() && c == start_pos) ||
                (region.is_upstream() && c == pos) ||
                (c == pos && c == text.len() && self.line_of_offset(text, c) == line_num)
            {
                cursors.push(c - start_pos);
            }

            // selection with interior
            let sel_start_ix = clamp(region.min(), start_pos, pos) - start_pos;
            let sel_end_ix = clamp(region.max(), start_pos, pos) - start_pos;
            if sel_end_ix > sel_start_ix {
                selections.push((sel_start_ix, sel_end_ix));
            }
        }

        let mut hls = Vec::new();

        if self.highlight_find {
            for find in self.find.iter() {
                for region in find.occurrences().regions_in_range(start_pos, pos) {
                    let sel_start_ix = clamp(region.min(), start_pos, pos) - start_pos;
                    let sel_end_ix = clamp(region.max(), start_pos, pos) - start_pos;
                    if sel_end_ix > sel_start_ix {
                        hls.push((sel_start_ix, sel_end_ix));
                    }
                }
            }
        }

        let styles = self.render_styles(client, styles, start_pos, pos,
                                        &selections, &hls, style_spans);

        let mut result = json!({
            "text": &l_str,
            "styles": styles,
        });

        if !cursors.is_empty() {
            result["cursor"] = json!(cursors);
        }
        result
    }

    pub fn render_styles(&self, client: &Client, styles: &StyleMap,
                         start: usize, end: usize, sel: &[(usize, usize)],
                         hls: &[(usize, usize)],
                         style_spans: &Spans<Style>) -> Vec<isize>
    {
        let mut rendered_styles = Vec::new();
        let style_spans = style_spans.subseq(Interval::new_closed_open(start, end));

        let mut ix = 0;
        // we add the special find highlights (1) and selection (0) styles first.
        // We add selection after find because we want it to be preferred if the
        // same span exists in both sets (as when there is an active selection)
        for &(sel_start, sel_end) in hls {
            rendered_styles.push((sel_start as isize) - ix);
            rendered_styles.push(sel_end as isize - sel_start as isize);
            rendered_styles.push(1);
            ix = sel_end as isize;
        }
        for &(sel_start, sel_end) in sel {
            rendered_styles.push((sel_start as isize) - ix);
            rendered_styles.push(sel_end as isize - sel_start as isize);
            rendered_styles.push(0);
            ix = sel_end as isize;
        }
        for (iv, style) in style_spans.iter() {
            let style_id = self.get_or_def_style_id(client, styles, &style);
            rendered_styles.push((iv.start() as isize) - ix);
            rendered_styles.push(iv.end() as isize - iv.start() as isize);
            rendered_styles.push(style_id as isize);
            ix = iv.end() as isize;
        }
        rendered_styles
    }

    fn get_or_def_style_id(&self, client: &Client, style_map: &StyleMap,
                           style: &Style) -> usize {
        let mut style_map = style_map.borrow_mut();
        if let Some(ix) = style_map.lookup(style) {
            return ix;
        }
        let ix = style_map.add(style);
        let style = style_map.merge_with_default(style);
        client.def_style(&style.to_json(ix));
        ix
    }

    fn build_update_op(&self, op: &str, lines: Option<Vec<Value>>, n: usize) -> Value {
        let mut update = json!({
            "op": op,
            "n": n,
        });

        if let Some(lines) = lines {
            update["lines"] = json!(lines);
        }

        update
    }

    fn send_update_for_plan(&mut self, text: &Rope, client: &Client,
                            styles: &StyleMap, style_spans: &Spans<Style>,
                            plan: &RenderPlan, pristine: bool)
    {
        if !self.lc_shadow.needs_render(plan) { return; }

        // send updated find status only if there have been changes
        if self.find_changed != FindStatusChange::None {
            let matches_only = self.find_changed == FindStatusChange::Matches;
            client.find_status(self.view_id, &json!(self.find_status(matches_only)));
        }

        // send updated replace status if changed
        if self.replace_changed {
            if let Some(replace) = self.get_replace() {
                client.replace_status(self.view_id, &json!(replace))
            }
        }

        let mut b = line_cache_shadow::Builder::new();
        let mut ops = Vec::new();
        let mut line_num = 0;  // tracks old line cache

        for seg in self.lc_shadow.iter_with_plan(plan) {
            match seg.tactic {
                RenderTactic::Discard => {
                    ops.push(self.build_update_op("invalidate", None, seg.n));
                    b.add_span(seg.n, 0, 0);
                }
                RenderTactic::Preserve => {
                    // TODO: in the case where it's ALL_VALID & !CURSOR_VALID, and cursors
                    // are empty, could send update removing the cursor.
                    if seg.validity == line_cache_shadow::ALL_VALID {
                        let n_skip = seg.their_line_num - line_num;
                        if n_skip > 0 {
                            ops.push(self.build_update_op("skip", None, n_skip));
                        }
                        ops.push(self.build_update_op("copy", None, seg.n));
                        b.add_span(seg.n, seg.our_line_num, line_cache_shadow::ALL_VALID);
                        line_num = seg.their_line_num + seg.n;
                    } else {
                        ops.push(self.build_update_op("invalidate", None, seg.n));
                        b.add_span(seg.n, 0, 0);
                    }
                }
                RenderTactic::Render => {
                    // TODO: update (rather than re-render) in cases of text valid
                    if seg.validity == line_cache_shadow::ALL_VALID {
                        let n_skip = seg.their_line_num - line_num;
                        if n_skip > 0 {
                            ops.push(self.build_update_op("skip", None, n_skip));
                        }
                        ops.push(self.build_update_op("copy", None, seg.n));
                        b.add_span(seg.n, seg.our_line_num, line_cache_shadow::ALL_VALID);
                        line_num = seg.their_line_num + seg.n;
                    } else {
                        let start_line = seg.our_line_num;
                        let end_line = start_line + seg.n;

                        let offset = self.offset_of_line(text, start_line);
                        let mut line_cursor = Cursor::new(text, offset);
                        let mut soft_breaks = self.breaks.as_ref().map(|breaks|
                            Cursor::new(breaks, offset));
                        let mut rendered_lines = Vec::new();
                        for line_num in start_line..end_line {
                            let line = self.render_line(client, styles, text,
                                                        &mut line_cursor,
                                                        soft_breaks.as_mut(),
                                                        style_spans, line_num);
                            rendered_lines.push(line);
                        }
                        ops.push(self.build_update_op("ins", Some(rendered_lines), seg.n));
                        b.add_span(seg.n, seg.our_line_num, line_cache_shadow::ALL_VALID);
                    }
                }
            }
        }
        let params = json!({
            "ops": ops,
            "pristine": pristine,
        });

        client.update_view(self.view_id, &params);
        self.lc_shadow = b.build();
        for find in &mut self.find {
            find.set_hls_dirty(false)
        }
    }

    /// Determines the current number of find results and search parameters to send them to
    /// the frontend.
    pub fn find_status(&mut self, matches_only: bool) -> Vec<FindStatus> {
        self.find_changed = FindStatusChange::None;

        self.find.iter().map(|find| {
            find.find_status(matches_only)
        }).collect::<Vec<FindStatus>>()
    }

    /// Update front-end with any changes to view since the last time sent.
    /// The `pristine` argument indicates whether or not the buffer has
    /// unsaved changes.
    pub fn render_if_dirty(&mut self, text: &Rope, client: &Client,
                           styles: &StyleMap, style_spans: &Spans<Style>,
                           pristine: bool)
    {
        let height = self.line_of_offset(text, text.len()) + 1;
        let plan = RenderPlan::create(height, self.first_line, self.height);
        self.send_update_for_plan(text, client, styles,
                                  style_spans, &plan, pristine);
        if let Some(new_scroll_pos) = self.scroll_to.take() {
            let (line, col) = self.offset_to_line_col(text, new_scroll_pos);
            client.scroll_to(self.view_id, line, col);
        }
    }

    // Send the requested lines even if they're outside the current scroll region.
    pub fn request_lines(&mut self, text: &Rope, client: &Client,
                         styles: &StyleMap, style_spans: &Spans<Style>,
                         first_line: usize, last_line: usize, pristine: bool) {
        let height = self.line_of_offset(text, text.len()) + 1;
        let mut plan = RenderPlan::create(height, self.first_line, self.height);
        plan.request_lines(first_line, last_line);
        self.send_update_for_plan(text, client, styles,
                                  style_spans, &plan, pristine);
    }

    /// Invalidates front-end's entire line cache, forcing a full render at the next
    /// update cycle. This should be a last resort, updates should generally cause
    /// finer grain invalidation.
    pub fn set_dirty(&mut self, text: &Rope) {
        let height = self.line_of_offset(text, text.len()) + 1;
        let mut b = line_cache_shadow::Builder::new();
        b.add_span(height, 0, 0);
        b.set_dirty(true);
        self.lc_shadow = b.build();
    }

    // How should we count "column"? Valid choices include:
    // * Unicode codepoints
    // * grapheme clusters
    // * Unicode width (so CJK counts as 2)
    // * Actual measurement in text layout
    // * Code units in some encoding
    //
    // Of course, all these are identical for ASCII. For now we use UTF-8 code units
    // for simplicity.

    pub(crate) fn offset_to_line_col(&self, text: &Rope, offset: usize) -> (usize, usize) {
        let line = self.line_of_offset(text, offset);
        (line, offset - self.offset_of_line(text, line))
    }

    pub(crate) fn line_col_to_offset(&self, text: &Rope, line: usize, col: usize) -> usize {
        let mut offset = self.offset_of_line(text, line).saturating_add(col);
        if offset >= text.len() {
            offset = text.len();
            if self.line_of_offset(text, offset) <= line {
                return offset;
            }
        } else {
            // Snap to grapheme cluster boundary
            offset = text.prev_grapheme_offset(offset + 1).unwrap();
        }

        // clamp to end of line
        let next_line_offset = self.offset_of_line(text, line + 1);
        if offset >= next_line_offset {
            if let Some(prev) = text.prev_grapheme_offset(next_line_offset) {
                offset = prev;
            }
        }
        offset
    }

    // use own breaks if present, or text if not (no line wrapping)

    /// Returns the visible line number containing the given offset.
    pub fn line_of_offset(&self, text: &Rope, offset: usize) -> usize {
        match self.breaks {
            Some(ref breaks) => {
                breaks.convert_metrics::<BreaksBaseMetric, BreaksMetric>(offset)
            }
            None => text.line_of_offset(offset)
        }
    }

    /// Returns the byte offset corresponding to the line `line`.
    pub fn offset_of_line(&self, text: &Rope, line: usize) -> usize {
        match self.breaks {
            Some(ref breaks) => {
                breaks.convert_metrics::<BreaksMetric, BreaksBaseMetric>(line)
            }
            None => {
                // sanitize input
                let line = line.min(text.measure::<LinesMetric>() + 1);
                text.offset_of_line(line)
            }
        }
    }

    pub(crate) fn rewrap(&mut self, text: &Rope, wrap_col: usize) {
        if wrap_col > 0 {
            self.breaks = Some(linewrap::linewrap(text, wrap_col));
            self.wrap_col = WrapWidth::Bytes(wrap_col);
        } else {
            self.breaks = None
        }
    }

    /// Generate line breaks based on width measurement. Currently batch-mode,
    /// and currently in a debugging state.
    pub(crate) fn wrap_width(&mut self, text: &Rope, width_cache: &mut WidthCache,
                             client: &Client, style_spans: &Spans<Style>)
    {
        let _t = trace_block("View::wrap_width", &["core"]);
        self.breaks = Some(linewrap::linewrap_width(text, width_cache,
                                                    style_spans, client,
                                                    self.size.width));
        self.wrap_col = WrapWidth::Width(self.size.width);
    }

    /// Updates the view after the text has been modified by the given `delta`.
    /// This method is responsible for updating the cursors, and also for
    /// recomputing line wraps.
    pub fn after_edit(&mut self, text: &Rope, last_text: &Rope,
                      delta: &Delta<RopeInfo>, client: &Client,
                      width_cache: &mut WidthCache, keep_selections: bool)
    {
        let (iv, new_len) = delta.summary();
        if let Some(breaks) = self.breaks.as_mut() {
            match self.wrap_col {
                WrapWidth::None => (),
                WrapWidth::Bytes(col) => linewrap::rewrap(breaks, text, iv,
                                                          new_len, col),
                WrapWidth::Width(px) =>
                    linewrap::rewrap_width(breaks, text, width_cache,
                                           client, iv, new_len, px),
            }
        }
        if self.breaks.is_some() {
            // TODO: finer grain invalidation for the line wrapping, needs info
            // about what wrapped.
            self.set_dirty(text);
        } else {
            let start = self.line_of_offset(last_text, iv.start());
            let end = self.line_of_offset(last_text, iv.end()) + 1;
            let new_end = self.line_of_offset(text, iv.start() + new_len) + 1;
            self.lc_shadow.edit(start, end, new_end - start);
        }
        // Any edit cancels a drag. This is good behavior for edits initiated through
        // the front-end, but perhaps not for async edits.
        self.drag_state = None;

        // update only find highlights affected by change
        for find in &mut self.find {
            find.update_highlights(text, delta);
        }

        self.find_changed = FindStatusChange::Matches;

        // Note: for committing plugin edits, we probably want to know the priority
        // of the delta so we can set the cursor before or after the edit, as needed.
        let new_sel = self.selection.apply_delta(delta, true, keep_selections);
        self.set_selection_for_edit(text, new_sel);
    }

    fn do_selection_for_find(&mut self, text: &Rope, case_sensitive: bool) {
        // set last selection or word under current cursor as search query
        let search_query = match self.selection.last() {
            Some(region) => {
                if !region.is_caret() {
                    text.slice_to_string(region.min(), region.max())
                } else {
                    let (start, end) = {
                        let mut word_cursor = WordCursor::new(text, region.max());
                        word_cursor.select_word()
                    };
                    text.slice_to_string(start, end)
                }
            },
            _ => return
        };

        self.find_changed = FindStatusChange::All;
        self.set_dirty(text);

        // todo: this will be changed once multiple queries are supported
        // todo: for now only a single search query is supported however in the future
        // todo: the correct Find instance needs to be updated with the new parameters
        if self.find.is_empty() {
            self.find.push(Find::new());
        }

        self.find.first_mut().unwrap().do_find(text, search_query, case_sensitive, false, true);
    }

    pub fn do_find(&mut self, text: &Rope, chars: String, case_sensitive: bool, is_regex: bool,
                   whole_words: bool) {
        self.set_dirty(text);
        self.find_changed = FindStatusChange::Matches;

        // todo: this will be changed once multiple queries are supported
        // todo: for now only a single search query is supported however in the future
        // todo: the correct Find instance needs to be updated with the new parameters
        if self.find.is_empty() {
            self.find.push(Find::new());
        }

        self.find.first_mut().unwrap().do_find(text, chars, case_sensitive, is_regex, whole_words);
    }

    /// Selects the next find match.
    pub fn do_find_next(&mut self, text: &Rope, reverse: bool, wrap: bool, allow_same: bool,
                     modify_selection: &SelectionModifier) {
        self.select_next_occurrence(text, reverse, false, allow_same, modify_selection);
        if self.scroll_to.is_none() && wrap {
            self.select_next_occurrence(text, reverse, true, allow_same, modify_selection);
        }
    }

    /// Selects all find matches.
    pub fn do_find_all(&mut self, text: &Rope) {
        let mut selection = Selection::new();
        for find in self.find.iter() {
            for &occurrence in find.occurrences().iter() {
                selection.add_region(occurrence);
            }
        }

        if !selection.is_empty() { // todo: invalidate so that nothing selected accidentally replaced
            self.set_selection(text, selection);
        }
    }

    /// Select the next occurrence relative to the last cursor. `reverse` determines whether the
    /// next occurrence before (`true`) or after (`false`) the last cursor is selected. `wrapped`
    /// indicates a search for the next occurrence past the end of the file.
    pub fn select_next_occurrence(&mut self, text: &Rope, reverse: bool, wrapped: bool,
                                  _allow_same: bool, modify_selection: &SelectionModifier) {
        // multiple queries; select closest occurrence
        let closest_occurrence = self.find.iter().flat_map(|x|
            x.next_occurrence(text, reverse, wrapped, &self.selection)
        ).min_by_key(|x| {
            match reverse {
                true => x.end,
                false => x.start
            }
        });

        if let Some(occ) = closest_occurrence {
            match modify_selection {
                SelectionModifier::Set => self.set_selection(text, occ),
                SelectionModifier::Add => {
                    let mut selection = self.selection.clone();
                    selection.add_region(occ);
                    self.set_selection(text, selection);
                },
                SelectionModifier::AddRemovingCurrent => {
                    let mut selection = self.selection.clone();

                    if let Some(last_selection) = self.selection.last() {
                        if !last_selection.is_caret() {
                            selection.delete_range(last_selection.min(), last_selection.max(), false);
                        }
                    }

                    selection.add_region(occ);
                    self.set_selection(text, selection);
                }
                _ => { }
            }
        }
    }

    fn do_set_replace(&mut self, chars: String, preserve_case: bool) {
        self.replace = Some(Replace { chars, preserve_case });
        self.replace_changed = true;
    }

    fn do_selection_for_replace(&mut self, text: &Rope) {
        // set last selection or word under current cursor as replacement string
        let replacement = match self.selection.last() {
            Some(region) => {
                if !region.is_caret() {
                    text.slice_to_string(region.min(), region.max())
                } else {
                    let (start, end) = {
                        let mut word_cursor = WordCursor::new(text, region.max());
                        word_cursor.select_word()
                    };
                    text.slice_to_string(start, end)
                }
            },
            _ => return
        };

        self.set_dirty(text);
        self.do_set_replace(replacement, false);
    }

    /// Get the line range of a selected region.
    pub fn get_line_range(&self, text: &Rope, region: &SelRegion) -> Range<usize> {
        let (first_line, _) = self.offset_to_line_col(text, region.min());
        let (mut last_line, last_col) = self.offset_to_line_col(text, region.max());
        if last_col == 0 && last_line > first_line {
            last_line -= 1;
        }

        first_line..(last_line + 1)
    }

    pub fn get_caret_offset(&self) -> Option<usize> {
        match self.selection.len() {
            1 if self.selection[0].is_caret() => {
                let offset = self.selection[0].start;
                Some(offset)
            }
            _ => None
        }
    }
}

// utility function to clamp a value within the given range
fn clamp(x: usize, min: usize, max: usize) -> usize {
    if x < min {
        min
    } else if x < max {
        x
    } else {
        max
    }
}
// Copyright 2016 The xi-editor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! A rope data structure with a line count metric and (soon) other useful
//! info.

use std::cmp::{min,max};
use std::borrow::Cow;
use std::str::FromStr;
use std::string::ParseError;
use std::fmt;
use std::str;
use std::ops::Add;

use tree::{Leaf, Node, NodeInfo, Metric, TreeBuilder, Cursor};
use delta::{Delta, DeltaElement};
use interval::Interval;

use bytecount;
use memchr::{memrchr, memchr};
use serde::ser::{Serialize, Serializer, SerializeStruct, SerializeTupleVariant};
use serde::de::{Deserialize, Deserializer};

use unicode_segmentation::GraphemeCursor;
use unicode_segmentation::GraphemeIncomplete;

const MIN_LEAF: usize = 511;
const MAX_LEAF: usize = 1024;

/// A rope data structure.
///
/// A [rope](https://en.wikipedia.org/wiki/Rope_(data_structure)) is a data structure
/// for strings, specialized for incremental editing operations. Most operations
/// (such as insert, delete, substring) are O(log n). This module provides an immutable
/// (also known as [persistent](https://en.wikipedia.org/wiki/Persistent_data_structure))
/// version of Ropes, and if there are many copies of similar strings, the common parts
/// are shared.
///
/// Internally, the implementation uses reference counting (not thread safe, though
/// it would be easy enough to modify to use `Arc` instead of `Rc` if that were
/// required). Mutations are generally copy-on-write, though in-place edits are
/// supported as an optimization when only one reference exists, making the
/// implementation as efficient as a mutable version.
///
/// Also note: in addition to the `From` traits described below, this module
/// implements `From<Rope> for String` and `From<&Rope> for String`, for easy
/// conversions in both directions.
///
/// # Examples
///
/// Create a `Rope` from a `String`:
///
/// ```rust
/// # use xi_rope::Rope;
/// let a = Rope::from("hello ");
/// let b = Rope::from("world");
/// assert_eq!("hello world", String::from(a.clone() + b.clone()));
/// assert!("hello world" == String::from(a + b));
/// ```
///
/// Get a slice of a `Rope`:
///
/// ```rust
/// # use xi_rope::Rope;
/// let a = Rope::from("hello world");
/// let b = a.slice(1, 9);
/// assert_eq!("ello wor", String::from(&b));
/// let c = b.slice(1, 7);
/// assert_eq!("llo wo", String::from(c));
/// ```
///
/// Replace part of a `Rope`:
///
/// ```rust
/// # use xi_rope::Rope;
/// let mut a = Rope::from("hello world");
/// a.edit_str(1, 9, "era");
/// assert_eq!("herald", String::from(a));
/// ```
pub type Rope = Node<RopeInfo>;

/// Represents a transform from one rope to another.
pub type RopeDelta = Delta<RopeInfo>;

/// An element in a `RopeDelta`.
pub type RopeDeltaElement = DeltaElement<RopeInfo>;

impl Leaf for String {
    fn len(&self) -> usize {
        self.len()
    }

    fn is_ok_child(&self) -> bool {
        self.len() >= MIN_LEAF
    }

    fn push_maybe_split(&mut self, other: &String, iv: Interval) -> Option<String> {
        //println!("push_maybe_split [{}] [{}] {:?}", self, other, iv);
        let (start, end) = iv.start_end();
        self.push_str(&other[start..end]);
        if self.len() <= MAX_LEAF {
            None
        } else {
            let splitpoint = find_leaf_split_for_merge(self);
            let right_str = self[splitpoint..].to_owned();
            self.truncate(splitpoint);
            self.shrink_to_fit();
            Some(right_str)
        }
    }
}

#[derive(Clone, Copy)]
pub struct RopeInfo {
    lines: usize,
    utf16_size: usize,
}

impl NodeInfo for RopeInfo {
    type L = String;

    fn accumulate(&mut self, other: &Self) {
        self.lines += other.lines;
        self.utf16_size += other.utf16_size;
    }

    fn compute_info(s: &String) -> Self {
        RopeInfo {
            lines: count_newlines(s),
            utf16_size: count_utf16_code_units(s),
        }
    }

    fn identity() -> Self {
        RopeInfo {
            lines: 0,
            utf16_size: 0,
        }
    }
}

//TODO: document metrics, based on https://github.com/google/xi-editor/issues/456
//See ../docs/MetricsAndBoundaries.md for more information.
#[derive(Clone, Copy)]
pub struct BaseMetric(());

/// Measured unit is utf8 code unit.
/// Base unit is utf8 code unit.
/// Boundary is atomic and determined by codepoint boundary.
/// Atomicity is implicit, putting the offset
/// between two utf8 code units that form a code point is considered invalid.
/// For example, take a string that starts with a 0xC2 byte.
/// Then offset=1 is invalid.
impl Metric<RopeInfo> for BaseMetric {
    fn measure(_: &RopeInfo, len: usize) -> usize {
        len
    }

    fn to_base_units(_: &String, in_measured_units: usize) -> usize {
        in_measured_units
    }

    fn from_base_units(_: &String, in_base_units: usize) -> usize {
        in_base_units
    }

    fn is_boundary(s: &String, offset: usize) -> bool {
        s.is_char_boundary(offset)
    }

    fn prev(s: &String, offset: usize) -> Option<usize> {
        if offset == 0 {
            // I think it's a precondition that this will never be called
            // with offset == 0, but be defensive.
            None
        } else {
            let mut len = 1;
            while !s.is_char_boundary(offset - len) {
                len += 1;
            }
            Some(offset - len)
        }
    }

    fn next(s: &String, offset: usize) -> Option<usize> {
        if offset == s.len() {
            // I think it's a precondition that this will never be called
            // with offset == s.len(), but be defensive.
            None
        } else {
            let b = s.as_bytes()[offset];
            Some(offset + len_utf8_from_first_byte(b))
        }
    }

    fn can_fragment() -> bool {
        false
    }
}

/// Given the inital byte of a UTF-8 codepoint, returns the number of
/// bytes required to represent the codepoint.
/// RFC reference : https://tools.ietf.org/html/rfc3629#section-4
pub fn len_utf8_from_first_byte(b: u8) -> usize {
    match b {
        b if b < 0x80 => 1,
        b if b < 0xe0 => 2,
        b if b < 0xf0 => 3,
        _ => 4
    }
}

#[derive(Clone, Copy)]
pub struct LinesMetric(usize);  // number of lines

/// Measured unit is newline amount.
/// Base unit is utf8 code unit.
/// Boundary is trailing and determined by a newline char.
impl Metric<RopeInfo> for LinesMetric {
    fn measure(info: &RopeInfo, _: usize) -> usize {
        info.lines
    }

    fn is_boundary(s: &String, offset: usize) -> bool {
        if offset == 0 {
            // shouldn't be called with this, but be defensive
            false
        } else {
            s.as_bytes()[offset - 1] == b'\n'
        }
    }

    fn to_base_units(s: &String, in_measured_units: usize) -> usize {
        let mut offset = 0;
        for _ in 0..in_measured_units {
            match memchr(b'\n', &s.as_bytes()[offset..]) {
                Some(pos) => offset += pos + 1,
                _ => panic!("to_base_units called with arg too large")
            }
        }
        offset
    }

    fn from_base_units(s: &String, in_base_units: usize) -> usize {
        count_newlines(&s[..in_base_units])
    }

    fn prev(s: &String, offset: usize) -> Option<usize> {
        memrchr(b'\n', &s.as_bytes()[..offset])
            .map(|pos| pos + 1)
    }

    fn next(s: &String, offset: usize) -> Option<usize> {
        memchr(b'\n', &s.as_bytes()[offset..])
            .map(|pos| offset + pos + 1)
    }

    fn can_fragment() -> bool { true }
}

#[derive(Clone, Copy)]
pub struct Utf16CodeUnitsMetric(usize);

impl Metric<RopeInfo> for Utf16CodeUnitsMetric {
    fn measure(info: &RopeInfo, _: usize) -> usize {
        info.utf16_size
    }

    fn is_boundary(s: &String, offset: usize) -> bool {
        s.is_char_boundary(offset)
    }

    fn to_base_units(s: &String, in_measured_units: usize) -> usize {
        let mut cur_len_utf16 = 0;
        let mut cur_len_utf8 = 0;
        for u in s.chars() {
            if cur_len_utf16 >= in_measured_units {
                break;
            }
            cur_len_utf16 += u.len_utf16();
            cur_len_utf8 += u.len_utf8();
        }
        cur_len_utf8
    }

    fn from_base_units(s: &String, in_base_units: usize) -> usize {
        count_utf16_code_units(&s[..in_base_units])
    }

    fn prev(s: &String, offset: usize) -> Option<usize> {
        if offset == 0 {
            // I think it's a precondition that this will never be called
            // with offset == 0, but be defensive.
            None
        } else {
            let mut len = 1;
            while !s.is_char_boundary(offset - len) {
                len += 1;
            }
            Some(offset - len)
        }
    }

    fn next(s: &String, offset: usize) -> Option<usize> {
        if offset == s.len() {
            // I think it's a precondition that this will never be called
            // with offset == s.len(), but be defensive.
            None
        } else {
            let b = s.as_bytes()[offset];
            Some(offset + len_utf8_from_first_byte(b))
        }
    }

    fn can_fragment() -> bool { false }
}

// Low level functions

pub fn count_newlines(s: &str) -> usize {
    bytecount::count(s.as_bytes(), b'\n')
}

fn count_utf16_code_units(s: &str) -> usize {
    let mut utf16_count = 0;
    for &b in s.as_bytes() {
        if (b as i8) >= -0x40 {
            utf16_count += 1;
        }
        if b >= 0xf0 {
            utf16_count += 1;
        }
    }
    utf16_count
}

fn find_leaf_split_for_bulk(s: &str) -> usize {
    find_leaf_split(s, MIN_LEAF)
}

fn find_leaf_split_for_merge(s: &str) -> usize {
    find_leaf_split(s, max(MIN_LEAF, s.len() - MAX_LEAF))
}

// Try to split at newline boundary (leaning left), if not, then split at codepoint
fn find_leaf_split(s: &str, minsplit: usize) -> usize {
    let mut splitpoint = min(MAX_LEAF, s.len() - MIN_LEAF);
    match memrchr(b'\n', &s.as_bytes()[minsplit - 1..splitpoint]) {
        Some(pos) => minsplit + pos,
        None => {
            while !s.is_char_boundary(splitpoint) {
                splitpoint -= 1;
            }
            splitpoint
        }
    }
}

// Additional APIs custom to strings

impl FromStr for Rope {
    type Err = ParseError;
    fn from_str(s: &str) -> Result<Rope, Self::Err> {
        let mut b = TreeBuilder::new();
        b.push_str(s);
        Ok(b.build())
    }
}

impl Serialize for Rope {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        serializer.serialize_str(&String::from(self))
    }
}

impl<'de> Deserialize<'de> for Rope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Rope::from(s))
    }
}

impl Serialize for DeltaElement<RopeInfo> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        match *self {
            DeltaElement::Copy(ref start, ref end) => {
                let mut el = serializer.serialize_tuple_variant("DeltaElement",
                                                                0, "copy", 2)?;
                el.serialize_field(start)?;
                el.serialize_field(end)?;
                el.end()
            }
            DeltaElement::Insert(ref node) =>
                serializer.serialize_newtype_variant("DeltaElement", 1,
                                                     "insert", node)
        }
    }
}

impl Serialize for Delta<RopeInfo> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        let mut delta = serializer.serialize_struct("Delta", 2)?;
        delta.serialize_field("els", &self.els)?;
        delta.serialize_field("base_len", &self.base_len)?;
        delta.end()
    }
}

impl<'de> Deserialize<'de> for Delta<RopeInfo> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>,
    {
        // NOTE: we derive to an interim representation and then convert
        // that into our actual target.
        #[derive(Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        enum RopeDeltaElement_ {
            Copy(usize, usize),
            Insert(String),
        }

        #[derive(Serialize, Deserialize)]
        struct RopeDelta_ {
            els: Vec<RopeDeltaElement_>,
            base_len: usize
        }

        impl From<RopeDeltaElement_> for DeltaElement<RopeInfo> {
            fn from(elem: RopeDeltaElement_) -> DeltaElement<RopeInfo> {
                match elem {
                    RopeDeltaElement_::Copy(start, end) =>
                        DeltaElement::Copy(start, end),
                    RopeDeltaElement_::Insert(s) =>
                        DeltaElement::Insert(Rope::from(s)),
                }
            }
        }

        impl From<RopeDelta_> for Delta<RopeInfo> {
            fn from(mut delta: RopeDelta_) -> Delta<RopeInfo> {
                Delta {
                    els: delta.els.drain(..)
                        .map(DeltaElement::from).collect(),
                    base_len: delta.base_len
                }
            }
        }
        let d = RopeDelta_::deserialize(deserializer)?;
        Ok(Delta::from(d))
    }
}

impl Rope {
    /// Edit the string, replacing the byte range [`start`..`end`] with `new`.
    ///
    /// Note: `edit` and `edit_str` may be merged, using traits.
    ///
    /// Time complexity: O(log n)
    pub fn edit_str(&mut self, start: usize, end: usize, new: &str) {
        let mut b = TreeBuilder::new();
        // TODO: may make this method take the iv directly
        let edit_iv = Interval::new_closed_open(start, end);
        let self_iv = Interval::new_closed_closed(0, self.len());
        self.push_subseq(&mut b, self_iv.prefix(edit_iv));
        b.push_str(new);
        self.push_subseq(&mut b, self_iv.suffix(edit_iv));
        *self = b.build();
    }

    /// Returns a slice of the string from the byte range [`start`..`end`).
    pub fn slice(&self, start: usize, end: usize) -> Rope {
        let iv = Interval::new_closed_open(start, end);
        self.subseq(iv)
    }

    // encourage callers to use Cursor instead?

    /// Determine whether `offset` lies on a codepoint boundary.
    pub fn is_codepoint_boundary(&self, offset: usize) -> bool {
        let mut cursor = Cursor::new(self, offset);
        cursor.is_boundary::<BaseMetric>()
    }

    /// Return the offset of the codepoint before `offset`.
    pub fn prev_codepoint_offset(&self, offset: usize) -> Option<usize> {
        let mut cursor = Cursor::new(self, offset);
        cursor.prev::<BaseMetric>()
    }

    /// Return the offset of the codepoint after `offset`.
    pub fn next_codepoint_offset(&self, offset: usize) -> Option<usize> {
        let mut cursor = Cursor::new(self, offset);
        cursor.next::<BaseMetric>()
    }

    pub fn prev_grapheme_offset(&self, offset: usize) -> Option<usize> {
        let mut cursor = Cursor::new(self, offset);
        cursor.prev_grapheme()
    }

    pub fn next_grapheme_offset(&self, offset: usize) -> Option<usize> {
        let mut cursor = Cursor::new(self, offset);
        cursor.next_grapheme()
    }

    /// Return the line number corresponding to the byte index `offset`.
    ///
    /// The line number is 0-based, thus this is equivalent to the count of newlines
    /// in the slice up to `offset`.
    ///
    /// Time complexity: O(log n)
    ///
    /// # Panics
    ///
    /// This function will panic if `offset > self.len()`. Callers are expected to
    /// validate their input.
    pub fn line_of_offset(&self, offset: usize) -> usize {
        self.convert_metrics::<BaseMetric, LinesMetric>(offset)
    }

    /// Return the byte offset corresponding to the line number `line`.
    /// If `line` is equal to one plus the current number of lines,
    /// this returns the offset of the end of the rope. Arguments higher
    /// than this will panic.
    ///
    /// The line number is 0-based.
    ///
    /// Time complexity: O(log n)
    ///
    /// # Panics
    ///
    /// This function will panic if `line > self.measure::<LinesMetric>() + 1`.
    /// Callers are expected to validate their input.
    pub fn offset_of_line(&self, line: usize) -> usize {
        let max_line = self.measure::<LinesMetric>() + 1;
        if line > max_line {
            panic!("line number {} beyond last line {}", line, max_line);
        } else if line == max_line {
            return self.len();
        }
        self.convert_metrics::<LinesMetric, BaseMetric>(line)
    }

    /// Returns an iterator over chunks of the rope.
    ///
    /// Each chunk is a `&str` slice borrowed from the rope's storage. The size
    /// of the chunks is indeterminate but for large strings will generally be
    /// in the range of 511-1024 bytes.
    ///
    /// The empty string will yield a single empty slice. In all other cases, the
    /// slices will be nonempty.
    ///
    /// Time complexity: technically O(n log n), but the constant factor is so
    /// tiny it is effectively O(n). This iterator does not allocate.
    pub fn iter_chunks(&self, start: usize, end: usize) -> ChunkIter {
        ChunkIter {
            cursor: Cursor::new(self, start),
            end,
        }
    }

    //TODO: implement iter_chunks using ranges and delete this
    pub fn iter_chunks_all(&self) -> ChunkIter {
        self.iter_chunks(0, self.len())
    }

    /// An iterator over the raw lines. The lines, except the last, include the
    /// terminating newline.
    ///
    /// The return type is a `Cow<str>`, and in most cases the lines are slices
    /// borrowed from the rope.
    pub fn lines_raw(&self, start: usize, end: usize) -> LinesRaw {
        LinesRaw {
            inner: self.iter_chunks(start, end),
            fragment: ""
        }
    }

    //TODO: implement lines_raw using ranges and delete this
    pub fn lines_raw_all(&self) -> LinesRaw {
        self.lines_raw(0, self.len())
    }

    /// An iterator over the lines of a rope.
    ///
    /// Lines are ended with either Unix (`\n`) or MS-DOS (`\r\n`) style line endings.
    /// The line ending is stripped from the resulting string. The final line ending
    /// is optional.
    ///
    /// The return type is a `Cow<str>`, and in most cases the lines are slices borrowed
    /// from the rope.
    ///
    /// The semantics are intended to match `str::lines()`.
    pub fn lines(&self, start: usize, end: usize) -> Lines {
        Lines {
            inner: self.lines_raw(start, end)
        }
    }

    // TODO: replace this with a version of `lines` that accepts a range
    pub fn lines_all(&self) -> Lines {
        self.lines(0, self.len())
    }

    // callers should be encouraged to use cursor instead
    pub fn byte_at(&self, offset: usize) -> u8 {
        let cursor = Cursor::new(self, offset);
        let (leaf, pos) = cursor.get_leaf().unwrap();
        leaf.as_bytes()[pos]
    }

    // TODO: this should be a Cow
    // TODO: a case can be made to hang this on Cursor instead
    pub fn slice_to_string(&self, start: usize, end: usize) -> String {
        let mut result = String::new();
        for chunk in self.iter_chunks(start, end) {
            result.push_str(chunk);
        }
        result
    }
}

// should make this generic, but most leaf types aren't going to be sliceable
pub struct ChunkIter<'a> {
    cursor: Cursor<'a, RopeInfo>,
    end: usize,
}

impl<'a> Iterator for ChunkIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.cursor.pos() >= self.end {
            return None;
        }
        let (leaf, start_pos) = self.cursor.get_leaf().unwrap();
        // if self.end is inside this chunk, verify that it is a codepoint boundary
        //let len = if self.end - self.cursor.pos() < leaf.len() - start_pos {
            //let prev_pos = self.cursor.pos();
            //self.cursor.set(self.end);
            //if self.cursor.is_boundary::<BaseMetric>() {
                //self.end - prev_pos
            //} else {
                //// if we aren't on a boundary we can't be at the end of the chunk
                //self.cursor.next::<BaseMetric>().unwrap() - prev_pos
            //}
        //} else {
            //leaf.len() - start_pos
        //};

        let len = min(self.end - self.cursor.pos(), leaf.len() - start_pos);
        let mut end = start_pos + len;
        while !leaf.is_char_boundary(end) {
            end += 1;
        }
        assert!(end <= leaf.len());
        assert!(leaf.is_char_boundary(end));
        self.cursor.next_leaf();
        Some(&leaf[start_pos .. end])
    }
}

pub struct ByteIter<'a> {
    inner: ChunkIter<'a>,
    current: Option<&'a str>,
    idx: usize,
}

impl<'a> Iterator for ByteIter<'a> {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.current.is_none() {
            let next_chunk = self.inner.next();
            if next_chunk.is_none() { return None; }
            self.current = next_chunk;
            self.idx = 0;
        }

        assert!(self.current.is_some());

        if self.idx <= self.current.unwrap().len() {
            let b = self.current.unwrap().as_bytes()[self.idx];
            self.idx += 1;
            Some(b)
        } else {
            self.current = None;
            // recurse at most once; this doesn't feel great but I wrote myself into a
            // corner and I'm lazy -cmyr
            self.next()
        }
    }
}

impl TreeBuilder<RopeInfo> {
    pub fn push_str(&mut self, mut s: &str) {
        if s.len() <= MAX_LEAF {
            if !s.is_empty() {
                self.push_leaf(s.to_owned());
            }
            return;
        }
        while !s.is_empty() {
            let splitpoint = if s.len() > MAX_LEAF {
                find_leaf_split_for_bulk(s)
            } else {
                s.len()
            };
            self.push_leaf(s[..splitpoint].to_owned());
            s = &s[splitpoint..];
        }
    }
}

impl<T: AsRef<str>> From<T> for Rope {
    fn from(s: T) -> Rope {
        Rope::from_str(s.as_ref()).unwrap()
    }
}

impl From<Rope> for String {
    // maybe explore grabbing leaf? would require api in tree
    fn from(r: Rope) -> String {
        String::from(&r)
    }
}

impl<'a> From<&'a Rope> for String {
    fn from(r: &Rope) -> String {
        r.slice_to_string(0, r.len())
    }
}

impl fmt::Debug for Rope {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if f.alternate() {
            write!(f, "{}", String::from(self))
        } else {
            write!(f, "Rope({:?})", String::from(self))
        }
    }
}

impl Add<Rope> for Rope {
    type Output = Rope;
    fn add(self, rhs: Rope) -> Rope {
        let mut b = TreeBuilder::new();
        b.push(self);
        b.push(rhs);
        b.build()
    }
}

 //additional cursor features

impl<'a> Cursor<'a, RopeInfo> {
    /// Get previous codepoint before cursor position, and advance cursor backwards.
    pub fn prev_codepoint(&mut self) -> Option<char> {
        self.prev::<BaseMetric>();
        if let Some((l, offset)) = self.get_leaf() {
            l[offset..].chars().next()
        } else {
            None
        }
    }

    /// Get next codepoint after cursor position, and advance cursor.
    pub fn next_codepoint(&mut self) -> Option<char> {
        if let Some((l, offset)) = self.get_leaf() {
            self.next::<BaseMetric>();
            l[offset..].chars().next()
        } else {
            None
        }
    }

    pub fn next_grapheme(&mut self) -> Option<usize> {
        let (mut l, mut offset) = self.get_leaf()?;
        let mut pos = self.pos();
        while offset < l.len() && !l.is_char_boundary(offset) {
            pos -= 1;
            offset -= 1;
        }
        let mut leaf_offset = pos - offset;
        let mut c = GraphemeCursor::new(pos, self.total_len(), true);
        let mut next_boundary = c.next_boundary(&l, leaf_offset);
        while let Err(incomp) = next_boundary {
            if let GraphemeIncomplete::PreContext(_) = incomp {
                let (pl, poffset) = self.prev_leaf()?;
                c.provide_context(&pl, self.pos() - poffset);
            } else if incomp == GraphemeIncomplete::NextChunk {
                self.set(pos);
                let (nl, noffset) = self.next_leaf()?;
                l = nl;
                leaf_offset = self.pos() - noffset;
                pos = leaf_offset + nl.len();
            } else {
                return None;
            }
            next_boundary = c.next_boundary(&l, leaf_offset);
        }
        next_boundary.unwrap_or(None)
    }

    pub fn prev_grapheme(&mut self) -> Option<usize> {
        let (mut l, mut offset) = self.get_leaf()?;
        let mut pos = self.pos();
        while offset < l.len() && !l.is_char_boundary(offset) {
            pos += 1;
            offset += 1;
        }
        let mut leaf_offset = pos - offset;
        let mut c = GraphemeCursor::new(pos, l.len() + leaf_offset, true);
        let mut prev_boundary = c.prev_boundary(&l, leaf_offset);
        while let Err(incomp) = prev_boundary {
            if let GraphemeIncomplete::PreContext(_) = incomp {
                let (pl, poffset) = self.prev_leaf()?;
                c.provide_context(&pl, self.pos() - poffset);
            } else if incomp == GraphemeIncomplete::PrevChunk {
                self.set(pos);
                let (pl, poffset) = self.prev_leaf()?;
                l = pl;
                leaf_offset = self.pos() - poffset;
                pos = leaf_offset + pl.len();
            } else {
                return None;
            }
            prev_boundary = c.prev_boundary(&l, leaf_offset);
        }
        prev_boundary.unwrap_or(None)
    }

    /// Returns the largest chunk of valid utf-8 with length <= chunk_size.
    /// If the cursor is at the end of the leaf, advances to the next leaf.
    /// If the cursor is at the end of the rope, returns the empty string.
    pub fn next_utf8_chunk_in_leaf(&mut self, chunk_size: usize) -> &'a str {
        let (leaf, offset) = match self.get_leaf() {
            Some((l, off)) => (l, off),
            None => return "",
        };

        let mut end = leaf.len().min(offset.saturating_add(chunk_size));
        while !leaf.is_char_boundary(end) {
            end -= 1;
        }
        let new_pos = self.pos() + (end - offset);
        self.set(new_pos);
        &leaf[offset..end]
    }
}

// line iterators

pub struct LinesRaw<'a> {
    inner: ChunkIter<'a>,
    fragment: &'a str
}

fn cow_append<'a>(a: Cow<'a, str>, b: &'a str) -> Cow<'a, str> {
    if a.is_empty() {
        Cow::from(b)
    } else {
        Cow::from(a.into_owned() + b)
    }
}

impl<'a> Iterator for LinesRaw<'a> {
    type Item = Cow<'a, str>;

    fn next(&mut self) -> Option<Cow<'a, str>> {
        let mut result = Cow::from("");
        loop {
            if self.fragment.is_empty() {
                match self.inner.next() {
                    Some(chunk) => self.fragment = chunk,
                    None => return if result.is_empty() { None } else { Some(result) }
                }
                if self.fragment.is_empty() {
                    // can only happen on empty input
                    return None;
                }
            }
            match memchr(b'\n', self.fragment.as_bytes()) {
                Some(i) => {
                    result = cow_append(result, &self.fragment[.. i + 1]);
                    self.fragment = &self.fragment[i + 1 ..];
                    return Some(result);
                },
                None => {
                    result = cow_append(result, self.fragment);
                    self.fragment = "";
                }
            }
        }
    }
}

pub struct Lines<'a> {
    inner: LinesRaw<'a>
}

impl<'a> Iterator for Lines<'a> {
    type Item = Cow<'a, str>;

    fn next(&mut self) -> Option<Cow<'a, str>> {
        match self.inner.next() {
            Some(Cow::Borrowed(mut s)) => {
                if s.ends_with('\n') {
                    s = &s[..s.len() - 1];
                    if s.ends_with('\r') {
                        s = &s[..s.len() - 1];
                    }
                }
                Some(Cow::from(s))
            },
            Some(Cow::Owned(mut s)) => {
                if s.ends_with('\n') {
                    let _ = s.pop();
                    if s.ends_with('\r') {
                        let _ = s.pop();
                    }
                }
                Some(Cow::from(s))
            }
            None => None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_test::{Token, assert_tokens};

    #[test]
    fn replace_small() {
        let mut a = Rope::from("hello world");
        a.edit_str(1, 9, "era");
        assert_eq!("herald", String::from(a));
    }

    #[test]
    fn lines_raw_small() {
        let a = Rope::from("a\nb\nc");
        assert_eq!(vec!["a\n", "b\n", "c"], a.lines_raw_all().collect::<Vec<_>>());

        let a = Rope::from("a\nb\n");
        assert_eq!(vec!["a\n", "b\n"], a.lines_raw_all().collect::<Vec<_>>());

        let a = Rope::from("\n");
        assert_eq!(vec!["\n"], a.lines_raw_all().collect::<Vec<_>>());

        let a = Rope::from("");
        assert_eq!(0, a.lines_raw_all().count());
    }

    #[test]
    fn lines_small() {
        let a = Rope::from("a\nb\nc");
        assert_eq!(vec!["a", "b", "c"], a.lines_all().collect::<Vec<_>>());
        assert_eq!(String::from(&a).lines().collect::<Vec<_>>(),
        a.lines_all().collect::<Vec<_>>());

        let a = Rope::from("a\nb\n");
        assert_eq!(vec!["a", "b"], a.lines_all().collect::<Vec<_>>());
        assert_eq!(String::from(&a).lines().collect::<Vec<_>>(),
        a.lines_all().collect::<Vec<_>>());

        let a = Rope::from("\n");
        assert_eq!(vec![""], a.lines_all().collect::<Vec<_>>());
        assert_eq!(String::from(&a).lines().collect::<Vec<_>>(),
        a.lines_all().collect::<Vec<_>>());

        let a = Rope::from("");
        assert_eq!(0, a.lines_all().count());
        assert_eq!(String::from(&a).lines().collect::<Vec<_>>(),
        a.lines_all().collect::<Vec<_>>());

        let a = Rope::from("a\r\nb\r\nc");
        assert_eq!(vec!["a", "b", "c"], a.lines_all().collect::<Vec<_>>());
        assert_eq!(String::from(&a).lines().collect::<Vec<_>>(),
        a.lines_all().collect::<Vec<_>>());

        let a = Rope::from("a\rb\rc");
        assert_eq!(vec!["a\rb\rc"], a.lines_all().collect::<Vec<_>>());
        assert_eq!(String::from(&a).lines().collect::<Vec<_>>(),
               a.lines_all().collect::<Vec<_>>());
    }

    #[test]
    fn lines_med() {
        let mut a = String::new();
        let mut b = String::new();
        let line_len = MAX_LEAF + MIN_LEAF - 1;
        for _ in 0..line_len {
            a.push('a');
            b.push('b');
        }
        a.push('\n');
        b.push('\n');
        let r = Rope::from(&a[..MAX_LEAF]);
        let r = r + Rope::from(String::from(&a[MAX_LEAF..]) + &b[..MIN_LEAF]);
        let r = r + Rope::from(&b[MIN_LEAF..]);
        //println!("{:?}", r.iter_chunks().collect::<Vec<_>>());

        assert_eq!(vec![a.as_str(), b.as_str()], r.lines_raw_all().collect::<Vec<_>>());
        assert_eq!(vec![&a[..line_len], &b[..line_len]], r.lines_all().collect::<Vec<_>>());
        assert_eq!(String::from(&r).lines().collect::<Vec<_>>(),
                   r.lines_all().collect::<Vec<_>>());

        // additional tests for line indexing
        assert_eq!(a.len(), r.offset_of_line(1));
        assert_eq!(r.len(), r.offset_of_line(2));
        assert_eq!(0, r.line_of_offset(a.len() - 1));
        assert_eq!(1, r.line_of_offset(a.len()));
        assert_eq!(1, r.line_of_offset(r.len() - 1));
        assert_eq!(2, r.line_of_offset(r.len()));
    }

    #[test]
    fn append_large() {
        let mut a = Rope::from("");
        let mut b = String::new();
        for i in 0..5_000 {
            let c = i.to_string() + "\n";
            b.push_str(&c);
            a = a + Rope::from(&c);
        }
        assert_eq!(b, String::from(a));
    }

    #[test]
    fn prev_codepoint_offset_small() {
        let a = Rope::from("a\u{00A1}\u{4E00}\u{1F4A9}");
        assert_eq!(Some(6), a.prev_codepoint_offset(10));
        assert_eq!(Some(3), a.prev_codepoint_offset(6));
        assert_eq!(Some(1), a.prev_codepoint_offset(3));
        assert_eq!(Some(0), a.prev_codepoint_offset(1));
        assert_eq!(None, a.prev_codepoint_offset(0));
        let b = a.slice(1, 10);
        assert_eq!(Some(5), b.prev_codepoint_offset(9));
        assert_eq!(Some(2), b.prev_codepoint_offset(5));
        assert_eq!(Some(0), b.prev_codepoint_offset(2));
        assert_eq!(None, b.prev_codepoint_offset(0));
    }

    #[test]
    fn next_codepoint_offset_small() {
        let a = Rope::from("a\u{00A1}\u{4E00}\u{1F4A9}");
        assert_eq!(Some(10), a.next_codepoint_offset(6));
        assert_eq!(Some(6), a.next_codepoint_offset(3));
        assert_eq!(Some(3), a.next_codepoint_offset(1));
        assert_eq!(Some(1), a.next_codepoint_offset(0));
        assert_eq!(None, a.next_codepoint_offset(10));
        let b = a.slice(1, 10);
        assert_eq!(Some(9), b.next_codepoint_offset(5));
        assert_eq!(Some(5), b.next_codepoint_offset(2));
        assert_eq!(Some(2), b.next_codepoint_offset(0));
        assert_eq!(None, b.next_codepoint_offset(9));
    }

    #[test]
    fn prev_grapheme_offset() {
        // A with ring, hangul, regional indicator "US"
        let a = Rope::from("A\u{030a}\u{110b}\u{1161}\u{1f1fa}\u{1f1f8}");
        assert_eq!(Some(9), a.prev_grapheme_offset(17));
        assert_eq!(Some(3), a.prev_grapheme_offset(9));
        assert_eq!(Some(0), a.prev_grapheme_offset(3));
        assert_eq!(None, a.prev_grapheme_offset(0));
    }

    #[test]
    fn next_grapheme_offset() {
        // A with ring, hangul, regional indicator "US"
        let a = Rope::from("A\u{030a}\u{110b}\u{1161}\u{1f1fa}\u{1f1f8}");
        assert_eq!(Some(3), a.next_grapheme_offset(0));
        assert_eq!(Some(9), a.next_grapheme_offset(3));
        assert_eq!(Some(17), a.next_grapheme_offset(9));
        assert_eq!(None, a.next_grapheme_offset(17));
    }

    #[test]
    fn next_grapheme_offset_with_ris_of_leaf_boundaries() {
        let s1 = "\u{1f1fa}\u{1f1f8}".repeat(100);
        let a = Rope::concat(
            Rope::from(s1.clone()),
            Rope::concat(
                Rope::from(String::from(s1.clone()) + "\u{1f1fa}"),
                Rope::from(s1.clone()),
            ),
        );
        for i in 1..(s1.len() * 3) {
            assert_eq!(Some((i - 1) / 8 * 8), a.prev_grapheme_offset(i));
            assert_eq!(Some(i / 8 * 8 + 8), a.next_grapheme_offset(i));
        }
        for i in (s1.len() * 3 + 1)..(s1.len() * 3 + 4) {
            assert_eq!(Some(s1.len() * 3), a.prev_grapheme_offset(i));
            assert_eq!(Some(s1.len() * 3 + 4), a.next_grapheme_offset(i));
        }
        assert_eq!(None, a.prev_grapheme_offset(0));
        assert_eq!(Some(8), a.next_grapheme_offset(0));
        assert_eq!(Some(s1.len() * 3), a.prev_grapheme_offset(s1.len() * 3 + 4));
        assert_eq!(None, a.next_grapheme_offset(s1.len() * 3 + 4));
    }

    #[test]
    fn test_ser_de() {
        let rope = Rope::from("a\u{00A1}\u{4E00}\u{1F4A9}");
        assert_tokens(&rope, &[
            Token::Str("a\u{00A1}\u{4E00}\u{1F4A9}"),
        ]);
        assert_tokens(&rope, &[
            Token::String("a\u{00A1}\u{4E00}\u{1F4A9}"),
        ]);
        assert_tokens(&rope, &[
            Token::BorrowedStr("a\u{00A1}\u{4E00}\u{1F4A9}"),
        ]);
    }

    #[test]
    fn line_of_offset_small() {
        let a = Rope::from("a\nb\nc");
        assert_eq!(0, a.line_of_offset(0));
        assert_eq!(0, a.line_of_offset(1));
        assert_eq!(1, a.line_of_offset(2));
        assert_eq!(1, a.line_of_offset(3));
        assert_eq!(2, a.line_of_offset(4));
        assert_eq!(2, a.line_of_offset(5));
        let b = a.slice(2, 4);
        assert_eq!(0, b.line_of_offset(0));
        assert_eq!(0, b.line_of_offset(1));
        assert_eq!(1, b.line_of_offset(2));
    }

    #[test]
    fn offset_of_line_small() {
        let a = Rope::from("a\nb\nc");
        assert_eq!(0, a.offset_of_line(0));
        assert_eq!(2, a.offset_of_line(1));
        assert_eq!(4, a.offset_of_line(2));
        assert_eq!(5, a.offset_of_line(3));
        let b = a.slice(2, 4);
        assert_eq!(0, b.offset_of_line(0));
        assert_eq!(2, b.offset_of_line(1));
    }

    #[test]
    fn eq_small() {
        let a = Rope::from("a");
        let a2 = Rope::from("a");
        let b = Rope::from("b");
        let empty = Rope::from("");
        assert!(a == a2);
        assert!(a != b);
        assert!(a != empty);
        assert!(empty == empty);
        assert!(a.slice(0, 0) == empty);
    }

    #[test]
    fn eq_med() {
        let mut a = String::new();
        let mut b = String::new();
        let line_len = MAX_LEAF + MIN_LEAF - 1;
        for _ in 0..line_len {
            a.push('a');
            b.push('b');
        }
        a.push('\n');
        b.push('\n');
        let r = Rope::from(&a[..MAX_LEAF]);
        let r = r + Rope::from(String::from(&a[MAX_LEAF..]) + &b[..MIN_LEAF]);
        let r = r + Rope::from(&b[MIN_LEAF..]);

        let a_rope = Rope::from(&a);
        let b_rope = Rope::from(&b);
        assert!(r != a_rope);
        assert!(r.clone().slice(0, a.len()) == a_rope);
        assert!(r.clone().slice(a.len(), r.len()) == b_rope);
        assert!(r == a_rope.clone() + b_rope.clone());
        assert!(r != b_rope + a_rope);
    }

    #[test]
    fn line_offsets() {
        let rope = Rope::from("hi\ni'm\nfour\nlines");
        assert_eq!(rope.offset_of_line(0), 0);
        assert_eq!(rope.offset_of_line(1), 3);
        assert_eq!(rope.line_of_offset(0), 0);
        assert_eq!(rope.line_of_offset(3), 1);
        // interior of first line should be first line
        assert_eq!(rope.line_of_offset(1), 0);
        // interior of last line should be last line
        assert_eq!(rope.line_of_offset(15), 3);
        assert_eq!(rope.offset_of_line(4), rope.len());
    }

    #[test]
    #[should_panic]
    fn line_of_offset_panic() {
        let rope = Rope::from("hi\ni'm\nfour\nlines");
        rope.line_of_offset(20);
    }

    #[test]
    #[should_panic]
    fn offset_of_line_panic() {
        let rope = Rope::from("hi\ni'm\nfour\nlines");
        rope.offset_of_line(5);
    }

    #[test]
    fn utf16_code_units_metric() {
        let rope = Rope::from("hi\ni'm\nfour\nlines");
        let utf16_units = rope.measure::<Utf16CodeUnitsMetric>();
        assert_eq!(utf16_units, 17);

        // position after 'f' in four
        let utf8_offset = 9;
        let utf16_units = rope.convert_metrics::<BaseMetric, Utf16CodeUnitsMetric>(utf8_offset);
        assert_eq!(utf16_units, 9);

        let utf8_offset = rope.convert_metrics::<Utf16CodeUnitsMetric, BaseMetric>(utf16_units);
        assert_eq!(utf8_offset, 9);

        let rope_with_emoji = Rope::from("hi\ni'm\n😀 four\nlines");
        let utf16_units = rope_with_emoji.measure::<Utf16CodeUnitsMetric>();
        
        assert_eq!(utf16_units, 20);

        // position after 'f' in four
        let utf8_offset = 13;
        let utf16_units = rope_with_emoji.convert_metrics::<BaseMetric, Utf16CodeUnitsMetric>(utf8_offset);
        assert_eq!(utf16_units, 11);

        let utf8_offset = rope_with_emoji.convert_metrics::<Utf16CodeUnitsMetric, BaseMetric>(utf16_units);
        assert_eq!(utf8_offset, 13);

        //for next line
        let utf8_offset = 19;
        let utf16_units = rope_with_emoji.convert_metrics::<BaseMetric, Utf16CodeUnitsMetric>(utf8_offset);
        assert_eq!(utf16_units, 17);

        let utf8_offset = rope_with_emoji.convert_metrics::<Utf16CodeUnitsMetric, BaseMetric>(utf16_units);
        assert_eq!(utf8_offset, 19);
    }
}
