use std::collections::HashMap;
use std::ops::Range;

use suffix::SuffixTable;

use xi_rope::tree::{Cursor, Node, NodeInfo};
use xi_rope::delta::{Delta, DeltaElement};
use xi_rope::rope::{BaseMetric, Rope, RopeInfo, RopeDelta};
use xi_rope::interval::Interval;
use xi_rope::compare::{ne_idx, ne_idx_rev, RopeScanner};
use xi_rope::diff::{Diff as XiDiff, LineHashDiff};

use memchr::memchr;

pub trait Diff<N: NodeInfo> {
    fn compute_delta(base: &Node<N>, target: &Node<N>, min_size: usize) -> Delta<N>;
}

pub trait MinimalDiff {
    /// Compute delta for the smallest different interior region.
    fn compute_delta_body(base: &Rope, target: &Rope, min_size: usize,
                          offset: usize, builder: &mut DeltaOps);
}


impl Diff<RopeInfo> for LineHashDiff {
    fn compute_delta(base: &Rope, target: &Rope, _min_size: usize) -> RopeDelta {
        <LineHashDiff as XiDiff<RopeInfo>>::compute_delta(base, target)
    }
}
/// Shared between various diff implementations
pub fn find_diff_start(base: &Rope, target: &Rope) -> (usize, usize) {
    let mut base_cursor = Cursor::new(base, 0);
    let mut target_cursor = Cursor::new(target, 0);
    loop {
        let b_next = base_cursor.next_codepoint();
        if b_next.is_none() || b_next != target_cursor.next_codepoint() {
            base_cursor.prev_codepoint();
            break;
        }
    }

    let diff_start = base_cursor.pos();

    base_cursor.set(base.len());
    target_cursor.set(target.len());
    loop {
        let b_prev = base_cursor.prev_codepoint();
        if b_prev.is_none() || b_prev != target_cursor.prev_codepoint() {
            base_cursor.next_codepoint();
            break;
        }
    }
    (diff_start, base.len() - base_cursor.pos())
}

impl<T> Diff<RopeInfo> for T where T: MinimalDiff {
    fn compute_delta(base: &Rope, target: &Rope, min_size: usize) -> RopeDelta {
        let mut builder = DeltaOps::default();
        let (diff_start, diff_end) = find_diff_start(base, target);
        //let mut scanner = RopeScanner::new(base, target);
        //assert_eq!((diff_start, diff_end), scanner.find_min_diff_range());
        if diff_start > 0 {
            builder.copy(0, diff_start);
        }
        if diff_start == base.len() && target.len() == base.len() {
            return builder.to_delta(base, target);
        }

        let base_iv = Interval::new(diff_start, base.len() - diff_end);
        let base_d = base.subseq(base_iv);
        let targ_iv = Interval::new(diff_start, target.len() - diff_end);
        let targ_d = target.subseq(targ_iv);

        T::compute_delta_body(&base_d, &targ_d, min_size, diff_start, &mut builder);
        if diff_end > 0 {
            builder.copy(base.len() - diff_end, base.len());
        }
        builder.to_delta(base, target)
    }
}

/// An efficient approximately minimal diff, not intended to be human readable.
/// Based on the approach outlined in File System Support for Delta
/// Compression (http://www.xmailserver.org/xdfs.pdf)
pub struct SmallSlowDiff;

impl Diff<RopeInfo> for SmallSlowDiff {
    fn compute_delta(base: &Rope, target: &Rope, min_size: usize) -> RopeDelta {
        let mut targ_idx = 0;
        let mut builder = DeltaOps::default();
        let mut cursor = Cursor::new(target, 0);

        while targ_idx < target.len() {
            match longest_common(base, target, targ_idx, min_size) {
                Some((start, end)) => {
                    builder.copy(start, end);
                    targ_idx += end - start;
                }
                None => {
                    cursor.set(targ_idx + min_size);
                    if !cursor.is_boundary::<BaseMetric>() {
                        cursor.prev::<BaseMetric>();
                    }
                    builder.insert(targ_idx, cursor.pos());
                    targ_idx = cursor.pos();
                }
            }
        }
        //eprintln!("{:?}", &builder.ops);
        builder.to_delta(base, target)
    }
}

/// Returns the longest common subsequence starting from `targ_idx` of `target`.
/// Returns `None` if no subsequence longer than `min_length` can be found.
fn longest_common(base: &Rope, target: &Rope, targ_idx: usize, min_length: usize)
    -> Option<(usize, usize)>
{
    let mut best = (0, 0);
    let mut base_cursor = Cursor::new(base, 0);
    let mut target_cursor = Cursor::new(target, targ_idx);

    loop {
        let start_pos = base_cursor.pos();
        let mut end_pos = start_pos;
        target_cursor.set(targ_idx);

        loop {
            let next_base = base_cursor.next_codepoint();
            let next_target = target_cursor.next_codepoint();
            if next_base.is_none() || next_base != next_target { break; }
            end_pos = base_cursor.pos();
        }

        if end_pos - start_pos > best.1 - best.0 {
            best = (start_pos, end_pos);
        }

        base_cursor.set(start_pos);
        if base_cursor.next_codepoint().is_none() { break; }
    }

    if best.1 - best.0 >= min_length {
        Some(best)
    } else {
        None
    }
}

pub struct SmallDiff;

impl Diff<RopeInfo> for SmallDiff {
    fn compute_delta(base: &Rope, target: &Rope, min_size: usize) -> RopeDelta {
        let mut builder = DeltaOps::default();
        let mut processed_idx = 0;
        let matcher = Matcher::new(base, min_size);

        for chunk in target.iter_chunks(..) {
            let mut chunk_idx = 0;
            while chunk_idx < chunk.len() {
                match matcher.find_match(chunk, chunk_idx) {
                    Some((start, end)) => {
                        builder.copy(start, end);
                        chunk_idx += end - start;
                    }
                    None => {
                        let cur_idx = processed_idx + chunk_idx;
                        let mut insert_end = chunk.len().min(chunk_idx + min_size);
                        while !chunk.is_char_boundary(insert_end) {
                            insert_end += 1;
                        }
                        builder.insert(cur_idx, processed_idx + insert_end);
                        chunk_idx += insert_end - chunk_idx;
                    }
                }
            }
            processed_idx += chunk.len();
        }
        builder.to_delta(base, target)
    }
}

pub struct SmallTricksyDiff;

impl MinimalDiff for SmallTricksyDiff {
    fn compute_delta_body(base: &Rope, target: &Rope, min_size: usize,
                     offset: usize, builder: &mut DeltaOps) {
        let mut processed_idx = 0;
        let matcher = Matcher::new(&base, min_size);

        for chunk in target.iter_chunks(..) {
            let mut chunk_idx = 0;
            while chunk_idx < chunk.len() {
                match matcher.find_match(chunk, chunk_idx) {
                    Some((start, end)) => {
                        assert!(end > start, "{} > {}", end, start);
                        builder.copy(offset + start, offset + end);
                        chunk_idx += end - start;
                    }
                    None => {
                        let cur_idx = processed_idx + chunk_idx;
                        let mut insert_end = chunk.len().min(chunk_idx + min_size);
                        while !chunk.is_char_boundary(insert_end) {
                            insert_end += 1;
                        }
                        builder.insert(offset + cur_idx, offset + processed_idx + insert_end);
                        chunk_idx += insert_end - chunk_idx;
                    }
                }
            }
            processed_idx += chunk.len();
        }
        //eprintln!("ops: {:?}", &builder.ops);
    }
}



pub struct Matcher<'a> {
    inner: HashMap<&'a [u8], usize>,
    min_size: usize,
    rope: &'a Rope,
}

impl<'a> Matcher<'a> {
    pub fn new(rope: &'a Rope, min_size: usize) -> Self {
        let mut inner = HashMap::with_capacity(rope.len());
        let mut idx = 0;
        for chunk in rope.iter_chunks(..) {
            let mut chunk_idx = 0;
            for slice in chunk.as_bytes().chunks(min_size) {
                if slice.len() != min_size { continue; }
                inner.insert(slice, idx + chunk_idx);
                chunk_idx += slice.len();
            }
            idx += chunk.len();
        }

        Matcher { inner, min_size, rope }
    }

    pub fn find_match(&self, target: &str, idx: usize) -> Option<(usize, usize)> {
        let mut end = idx + self.min_size;
        if end >= target.len() { return None; }
        while !target.is_char_boundary(end) {
            end += 1;
        }
        let start_idx = self.inner.get(&target.as_bytes()[idx..end])?;
        Some(self.max_match(target, idx, *start_idx))
    }

    fn max_match(&self, target: &str, targ_start: usize, self_idx: usize) -> (usize, usize) {
        let base_cursor = Cursor::new(self.rope, self_idx);

        let (base, base_start) = base_cursor.get_leaf().unwrap();

        let mut base_end = base_start;
        let mut targ_end = targ_start;

        while base_end < base.len() && targ_end < target.len() {
            if base.as_bytes()[base_end] != target.as_bytes()[targ_end] {
                break;
            }
            base_end += 1;
            targ_end += 1;
        }
        (self_idx, self_idx + (base_end - base_start))
    }
}


pub struct FastHashDiff;

pub struct FastHashDiffSimpler;

impl Diff<RopeInfo> for FastHashDiffSimpler {
    fn compute_delta(base: &Rope, target: &Rope, min_size: usize) -> RopeDelta {
        let mut builder = DeltaOps::default();
        FastHashDiff::compute_delta_body(base, target, min_size, 0, &mut builder);
        builder.to_delta(base, target)
    }
}

pub fn make_line_hashes<'a>(base: &'a str, min_size: usize) -> HashMap<&'a str, usize> {
    let mut offset = 0;
    let mut line_hashes = HashMap::with_capacity(base.len() / 60);
    let iter = LineScanner { inner: base, idx: 0 };
    for line in iter {
        let non_ws = non_ws_offset(&line);
        if line.len() - non_ws >= min_size {
            line_hashes.insert(&line[non_ws..], offset + non_ws);
        }
        offset += line.len();
    }
    line_hashes
}

impl MinimalDiff for FastHashDiff {
    fn compute_delta_body(base: &Rope, target: &Rope,
                          min_size: usize, start_offset: usize, builder: &mut DeltaOps) {
        let mut copies = Vec::new();
        let base_string = String::from(base);
        let line_hashes = make_line_hashes(&base_string, min_size);

        let mut offset = 0;
        let mut prev_targ_end = 0;

        for line in target.lines_raw(..) {
            let non_ws = non_ws_offset(&line);
            if offset + non_ws < prev_targ_end {
                // no-op, but we don't break because we still want to increment offset
            } else if line.len() - non_ws >= min_size {
                if let Some(base_off) = line_hashes.get(&line[non_ws..]) {
                    let targ_off = offset + non_ws;
                    let (left_dist, right_dist) = expand_match(base, target, *base_off,
                                                             targ_off, prev_targ_end);
                    //let (left_dist2, right_dist2) = fast_expand_match(base, target, *base_off,
                                                             //targ_off, prev_match_base_end,
                                                             //prev_targ_end);
                    //assert_eq!((left_dist, right_dist), (left_dist2, right_dist2),
                               //"base_off {} targ_off {} prev_base {} prev_targ {}",
                               //*base_off, targ_off, prev_match_base_end, prev_targ_end);
                    let (targ_start, targ_end) = (targ_off - left_dist, targ_off + right_dist);
                    let (base_start, base_end) = (base_off - left_dist, base_off + right_dist);

                    copies.push(((targ_start, targ_end), (base_start, base_end)));
                    prev_targ_end = targ_end;
                }
            }
            offset += line.len();
        }
        // make the delta ops:

        let mut targ_pos = 0;
        for ((targ_start, targ_end), (base_start, base_end)) in copies {
            if targ_start > targ_pos {
                builder.insert(start_offset + targ_pos, start_offset + targ_start);
            }
            builder.copy(start_offset + base_start, start_offset + base_end);
            targ_pos = targ_end;
        }

        if targ_pos != target.len() {
            builder.insert(start_offset + targ_pos, start_offset + target.len());
        }
    }
}

/// Given two ropes and the offsets of two equal bytes, finds the largest
/// identical substring shared between the two ropes which contains the offset.
///
/// The return value is a pair of offsets, each of which represents an absolute
/// distance. That is to say, the position of the start and end boundaries
/// relative to the input offset.
fn expand_match(base: &Rope, target: &Rope, base_off: usize, targ_off: usize,
                prev_match_targ_end: usize) -> (usize, usize) {
    let mut base_curs = Cursor::new(base, base_off);
    let mut targ_curs = Cursor::new(target, targ_off);
    let mut end_pos = base_off;
    let mut start_pos = base_off;

    // find the end
    loop {
        let next_base = base_curs.next_codepoint();
        let next_target = targ_curs.next_codepoint();
        if next_base.is_none() || next_base != next_target { break; }
        end_pos = base_curs.pos();
    }

    // find the start
    base_curs.set(base_off);
    targ_curs.set(targ_off);

    //TODO: instead of passing in two prev_end things we should just pass in max_prev_move
    //or something
    loop {
        let next_base = base_curs.prev_codepoint();
        let next_target = targ_curs.prev_codepoint();

        if next_base.is_none()
        || next_base != next_target
        || targ_curs.pos() < prev_match_targ_end { break; }
        start_pos = base_curs.pos();
    }

    (base_off - start_pos, end_pos - base_off)
}

pub fn fast_expand_match(base: &Rope, target: &Rope, base_off: usize, targ_off: usize,
                prev_match_targ_end: usize) -> (usize, usize) {

    let mut scanner = RopeScanner::new(base, target);
    debug_assert!(targ_off >= prev_match_targ_end, "{} >= {}", targ_off, prev_match_targ_end);
    let max_left = targ_off - prev_match_targ_end;
    let start = scanner.find_ne_char_back(base_off, targ_off, max_left);
    debug_assert!(start <= max_left, "{} <= {}", start, max_left);
    let end = scanner.find_ne_char(base_off, targ_off, None);
    (start.min(max_left), end)
}

struct LineScanner<'a> {
    inner: &'a str,
    idx: usize,
}

impl<'a> Iterator for LineScanner<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<&'a str> {
        let idx = memchr(b'\n', &self.inner.as_bytes()[self.idx..])?;
        let result = &self.inner[self.idx..self.idx + idx+1];
        self.idx += idx + 1;
        Some(result)
    }
}

pub struct MockParallelHashDiff;

impl Diff<RopeInfo> for MockParallelHashDiff {
    fn compute_delta(base: &Rope, target: &Rope, min_size: usize) -> RopeDelta {
        let mut builder = DeltaOps::default();
        let base_str = String::from(base);
        let line_hashes = make_line_hashes(&base_str, min_size);
        let chunk_results = target.iter_chunks(..)
            .scan(0, |offset, chunk| {
                let result = (*offset, chunk);
                *offset += chunk.len();
                Some(result)
            })
            .flat_map(|(offset, chunk)|
                      ops_for_chunk(chunk, offset, base, &line_hashes, min_size))
            .collect::<Vec<_>>();

        let mut targ_pos = 0;
        //chunk_results.sort_unstable_by(|one, two| one.target.start.cmp(&two.target.start));
        for copy in chunk_results {
            if copy.target.start > targ_pos {
                builder.insert(targ_pos, copy.target.start);
            }
            if copy.target.start < targ_pos {
                // if two concurrent batches have matched overlapping regions of the target,
                // adjust the second op to not overlap
                let delta = targ_pos - copy.target.start;
                builder.copy(copy.base.start + delta, copy.base.end);
            } else {
                builder.copy(copy.base.start, copy.base.end);
            }
            targ_pos = copy.target.end;
        }

        if targ_pos != target.len() {
            builder.insert(targ_pos, target.len());
        }
        //eprintln!("ops: {:?}", &builder.ops);

        builder.to_delta(base, target)
    }
}

pub struct ParallelHashDiff;
use crossbeam::queue::MsQueue;
use crossbeam::thread;

impl Diff<RopeInfo> for ParallelHashDiff {
    fn compute_delta(base: &Rope, target: &Rope, min_size: usize) -> RopeDelta {
        let mut builder = DeltaOps::default();
        let base_str = String::from(base);
        let line_hashes = make_line_hashes(&base_str, min_size);
        let  queue = MsQueue::new();
        let finished = MsQueue::new();

        let result: Vec<CopyOp> = thread::scope(|scope| {

        let mut chunk_off = 0;
        let mut chunk_count = 0;
        for chunk in target.iter_chunks(..) {
            queue.push((chunk_off, chunk));
            chunk_off += chunk.len();
            chunk_count += 1;
        }

        //TODO: how many threads do we want to start?
        //let mut v = Vec::new();
        for _ in 0..4 {
            let _ = scope.spawn(|| {
                //let mut v = Vec::new();
                while let Some((offset, chunk)) = queue.try_pop() {
                    let mut ops = ops_for_chunk(chunk, offset, base, &line_hashes, min_size);
                    finished.push(ops);
                    //v.append(&mut ops);
                }
            });
        }
        let mut result = Vec::with_capacity(chunk_count * 5);
        for _ in 0..chunk_count {
            let mut chunk_result = finished.pop();
            result.append(&mut chunk_result);
        }
        result.sort_unstable_by(|one, two| one.target.start.cmp(&two.target.start));
        result

        });
        let mut targ_pos = 0;
        for copy in result {
            if copy.target.start > targ_pos {
                builder.insert(targ_pos, copy.target.start);
            }
            if copy.target.start <= targ_pos {
                // if two concurrent batches have matched overlapping regions of the target,
                // adjust the second op to not overlap
                let delta = targ_pos - copy.target.start;
                builder.copy(copy.base.start + delta, copy.base.end);
            } else {
                builder.copy(copy.base.start, copy.base.end);
            }
            targ_pos = copy.target.end;
        }

        if targ_pos != target.len() {
            builder.insert(targ_pos, target.len());
        }
        //eprintln!("ops: {:?}", &builder.ops);

        builder.to_delta(base, target)
    }
}

#[derive(Debug, Clone)]
struct CopyOp {
    base: Range<usize>,
    target: Range<usize>,
}

fn ops_for_chunk(chunk: &str, offset_of_chunk_in_target: usize, base: &Rope,
                 matches: &HashMap<&str, usize>, min_size: usize) -> Vec<CopyOp> {

    let mut copies = Vec::new();
    let lines = LineScanner { inner: chunk, idx: 0 };
    let mut line_offset = 0;
    let mut cursor = Cursor::new(base, 0);
    let mut prev_match_end = 0;

    for line in lines {
        let non_ws = non_ws_offset(&line);
        if line.len() - non_ws < min_size
        // if we've already copied this line as part of the previous op, skip
        || line_offset + non_ws < prev_match_end {
            line_offset += line.len();
            continue;
        }
        if let Some(offset_in_base) = matches.get(&line[non_ws..]) {
            let offset_in_chunk = line_offset + non_ws;
            cursor.set(*offset_in_base);
            let start = expand_match_left(&chunk[..offset_in_chunk], &mut cursor);
            assert!(start <= offset_in_chunk, "{} <= {}", start, offset_in_chunk);
            cursor.set(*offset_in_base);
            let mut end = expand_match_right(&chunk[offset_in_chunk..], &mut cursor);

            let base_start = offset_in_base - start;
            let base_end = offset_in_base + end;
            let target_start = offset_of_chunk_in_target + offset_in_chunk - start;
            let target_end = offset_of_chunk_in_target + offset_in_chunk + end;

            let op = CopyOp {
                base: base_start..base_end,
                target: target_start..target_end,
            };
            copies.push(op);
            prev_match_end = offset_in_chunk + end;
        }
        line_offset += line.len();
    }
    copies
}

fn expand_match_right(chunk: &str, base: &mut Cursor<RopeInfo>) -> usize {
    let max_size = chunk.len().min(base.total_len() - base.pos());
    let (leaf, offset_in_leaf) = base.get_leaf().unwrap();
    let leaf = &leaf[offset_in_leaf..];

    if let Some(idx) = ne_idx(chunk.as_bytes(), leaf.as_bytes()) {
        return idx;
    }
    // no hit, and at end of chunk: the whole chunk matches
    if leaf.len() >= chunk.len() { return chunk.len(); }
    //let remainder = chunk.len() - leaf.len();
    let scanned = leaf.len();

    // expand at most into one neighbouring leaf
    base.next_leaf().and_then(|(leaf, _)| {
        let chunk = &chunk[chunk.len() - scanned..];
        ne_idx(chunk.as_bytes(), leaf.as_bytes())
            .map(|mut idx| {
                while idx > 0 && !chunk.is_char_boundary(idx) {
                    idx -= 1;
                }
                idx + scanned
            })
    })
    .unwrap_or(max_size)
}

fn expand_match_left(chunk: &str, base: &mut Cursor<RopeInfo>) -> usize {
    let max_size = chunk.len().min(base.pos());
    assert!(max_size <= chunk.len());
    let (leaf, offset_in_leaf) = base.get_leaf().unwrap();
    let leaf = &leaf[..offset_in_leaf];

    if let Some(idx) = ne_idx_rev(chunk.as_bytes(), leaf.as_bytes()) {
        return idx;
    }

    // no hit, and at end of chunk: the whole chunk matches
    if chunk.len() <= leaf.len() { return chunk.len(); }
    // expand at most into one neighbouring leaf
    let scanned = leaf.len();

    base.prev_leaf().and_then(|(leaf, _)| {
        let chunk = &chunk[..chunk.len() - scanned];
        ne_idx_rev(chunk.as_bytes(), leaf.as_bytes())
            .map(|mut idx| {
                while idx > 1 && !chunk.is_char_boundary(chunk.len() - idx) {
                    idx -= 1;
                }
                idx + scanned
            })
    })
    .unwrap_or(max_size)
}

#[inline]
pub fn non_ws_offset(s: &str) -> usize {
    s.as_bytes()
        .iter()
        .take_while(|b| **b == b' ' || **b == b'\t')
        .count()
}

pub struct SuffixDiff;

impl SuffixDiff {
    fn match_for_cursor(cursor: &mut Cursor<RopeInfo>, suffix: &SuffixTable,
                        chunk_size: usize) -> Option<(usize, usize)> {

        let (leaf, offset_in_leaf) = cursor.get_leaf().unwrap();
        let leaf_head_end = (offset_in_leaf + chunk_size).min(leaf.len());
        let leaf_head = &leaf[offset_in_leaf..leaf_head_end];
        debug_assert!(!leaf_head.is_empty());
        let new_curs_pos = cursor.pos() + leaf_head.len();
        cursor.set(new_curs_pos);

        let match_positions = suffix.positions(&leaf_head);
        if match_positions.is_empty() { return None; }
        // end of rope: no need to expand the match, just return the last
        if cursor.pos() >= cursor.total_len() || leaf_head_end == leaf.len() {
            let offset = *match_positions.last().unwrap() as usize;
            return Some((offset, offset + leaf_head.len()));
        }

        let leaf_tail = &leaf.as_bytes()[leaf_head_end..];

        // find which chunk is the matchiest:
        let (pos, match_len) = match_positions.iter()
            .map(|match_pos| {
                //FIXME: this could panic on input where the first different byte
                //is not a codepoint boundary. maybe just compare char_indicies?
                let match_pos = *match_pos as usize;
                let mut off = 0;
                loop {
                    let suf_pos = match_pos + leaf_head.len() + off;
                    if suf_pos >= suffix.text().len()
                    || off >= leaf_tail.len()
                    || suffix.text().as_bytes()[suf_pos] != leaf_tail[off] {
                        break;
                    }
                    off += 1;
                }
                 off + leaf_head.len()
            })
            .enumerate()
            .max_by(|a, b| a.1.cmp(&b.1))?;

        let match_start = match_positions[pos] as usize;
        let final_curs_pos =cursor.pos() + (match_len - leaf_head.len());
        cursor.set(final_curs_pos);
        Some((match_start, match_start + match_len))
    }
}

pub struct SuffixDiffOpt;

impl Diff<RopeInfo> for SuffixDiff {
    fn compute_delta(base: &Rope, target: &Rope, min_size: usize) -> RopeDelta {
        let mut builder = DeltaOps::default();
        SuffixDiffOpt::compute_delta_body(base, target, min_size, 0, &mut builder);
        builder.to_delta(base, target)
    }
}

impl MinimalDiff for SuffixDiffOpt {
    fn compute_delta_body(base: &Rope, target: &Rope,
                          min_size: usize, offset: usize, builder: &mut DeltaOps) {
        let suffix = SuffixTable::new(base.slice_to_cow(..));
        let mut cursor = Cursor::new(target, 0);
        let mut prev_cursor_pos;

        while cursor.pos() < target.len() {
            prev_cursor_pos = cursor.pos();
            match SuffixDiff::match_for_cursor(&mut cursor, &suffix, min_size) {
                Some((start, end)) => builder.copy(offset+ start, offset+ end),
                None => builder.insert(offset + prev_cursor_pos, offset + cursor.pos()),
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeltaOps {
    ops: Vec<InterimOp>,
}

impl DeltaOps {
    fn copy(&mut self, start: usize, end: usize) {
        if let Some(InterimOp::Copy(_, e)) = self.ops.last_mut() {
            if e == &start {
                *e = end;
                return;
            }
        }
        self.ops.push(InterimOp::Copy(start, end));
    }

    fn insert(&mut self, start: usize, end: usize) {
        if let Some(InterimOp::Insert(_, e)) = self.ops.last_mut() {
            if e == &start {
                *e = end;
                return;
            }
        }
        self.ops.push(InterimOp::Insert(start, end));
    }

    fn to_delta<N: NodeInfo>(self, base: &Node<N>, target: &Node<N>) -> Delta<N> {
        let els: Vec<DeltaElement<_>> = self.ops.into_iter().map(|op| match op {
            InterimOp::Copy(s, e) => DeltaElement::Copy(s, e),
            InterimOp::Insert(s, e) => {
                let iv = Interval::new(s, e);
                DeltaElement::Insert(target.subseq(iv))
            }
        }).collect();
        Delta { els, base_len: base.len() }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum InterimOp {
    /// A range in the base document.
    Copy(usize, usize),
    /// A range in the target document.
    Insert(usize, usize),
}

/// Represents copying `len` bytes from base to target.
#[derive(Debug, Clone, Copy)]
struct DiffOp {
    target_idx: usize,
    base_idx: usize,
    len: usize,
}

#[derive(Debug, Clone, Default)]
pub struct DiffBuilder {
    ops: Vec<DiffOp>,
}

impl DiffBuilder {
    pub fn copy(&mut self, base: usize, target: usize, len: usize) {
        let op = DiffOp {target_idx: target, base_idx: base, len };
        if let Some(prev) = self.ops.last_mut() {
            let prev_end = prev.target_idx + prev.len;
            let base_end = prev.base_idx + prev.len;
            assert!(prev_end <= target, "{} <= {} prev {:?}, op {:?}",
                    prev_end, target, prev, op);
            if prev_end == target && base_end == base {
                prev.len += len;
                return;
            }
        }
        self.ops.push(DiffOp { target_idx: target, base_idx: base, len: len })
    }

    pub fn to_delta(self, base: &Rope, target: &Rope) -> RopeDelta {
        let mut els = Vec::with_capacity(self.ops.len() * 2);
        let mut targ_pos = 0;
        for DiffOp { base_idx, target_idx, len } in self.ops {
            if target_idx > targ_pos {
                let iv = Interval::new(targ_pos, target_idx);
                els.push(DeltaElement::Insert(target.subseq(iv)));
            }
            els.push(DeltaElement::Copy(base_idx, base_idx + len));
            targ_pos = target_idx + len;
        }

        if targ_pos < target.len() {
            let iv = Interval::new(targ_pos, target.len());
            els.push(DeltaElement::Insert(target.subseq(iv)));
        }

        Delta { els, base_len: base.len() }
    }

    pub fn to_nondecreasing_delta(self, base: &Rope, target: &Rope) -> RopeDelta {
        if self.ops.is_empty() { return self.to_delta(base, target); }

        // indicies in self.ops
        let mut used_regions = vec![0];
        let mut prev_chain = vec![0; self.ops.len()];

        for i in 1..self.ops.len() {
            let DiffOp { target_idx, len, .. } = self.ops[*used_regions.last().unwrap()];
            if target_idx < self.ops[0].target_idx {
                prev_chain[i] = *used_regions.last().unwrap();
                used_regions.push(i);
            }

        }

        unimplemented!()
    }
}

pub struct FinalDiff;

impl Diff<RopeInfo> for FinalDiff {
    fn compute_delta(base: &Rope, target: &Rope, min_size: usize) -> RopeDelta {
        let mut builder = DiffBuilder::default();

        // before doing anything, scan top down and bottom up for like-ness.
        let mut scanner = RopeScanner::new(base, target);
        let (start_offset, diff_end) = scanner.find_min_diff_range();
        let target_end = target.len() - diff_end;

        if start_offset > 0 {
            builder.copy(0, 0, start_offset);
        }

        // if our preliminary scan finds no differences we're done
        if start_offset == base.len() && target.len() == base.len() {
            return builder.to_delta(base, target);
        }

        //TODO: because of how `lines_raw` returns Cows, we can't easily build
        //the lookup table without allocating. The eventual solution would be to have
        //a custom iter on the rope that returns suitable chunks.
        let base_string = String::from(base);
        let line_hashes = make_line_hashes(&base_string, min_size);

        let mut offset = start_offset;

        // When we find a matching region, we extend it forwards and backwards.
        // we keep track of how far forward we extend it each time, to avoid
        // having a subsequent scan extend backwards over the same region.
        let mut prev_targ_end = start_offset;
        let mut prev_base_end = 0;

        for line in target.lines_raw(start_offset..target_end) {
            let non_ws = non_ws_offset(&line);
            if offset + non_ws < prev_targ_end {
                // no-op, but we don't break because we still want to bump offset
            } else if line.len() - non_ws >= min_size {
                if let Some(base_off) = line_hashes.get(&line[non_ws..]) {
                    let targ_off = offset + non_ws;
                    let (left_dist, mut right_dist) = fast_expand_match(base, target,
                                                                        *base_off,
                                                                        targ_off,
                                                                        prev_targ_end);
                    if targ_off + right_dist > target_end {
                        // don't let last match expand past target_end
                        right_dist = target_end - targ_off;
                    }
                    let targ_start = targ_off - left_dist;
                    let base_start = base_off - left_dist;
                    let len = left_dist + right_dist;

                    // other parts of the code (Delta::factor) require that delta ops
                    // be in non-decreasing order, so we only actually copy a region
                    // when this is true. This algorithm was initially designed without
                    // this constraint; a better design would prioritize early matches,
                    // and more efficiently avoid searching in disallowed regions.
                    if base_start >= prev_base_end {
                        builder.copy(base_start, targ_start, len);
                        prev_targ_end = targ_start + len;
                        prev_base_end = base_start + len;
                    }
                }
            }
            offset += line.len();
        }

        if diff_end > 0 {
            builder.copy(base.len() - diff_end, target.len() - diff_end, diff_end);
        }

        builder.to_delta(base, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter;
    static SMALL_ONE: &str = "This adds FixedSizeAdler32, that has a size set at construction, and keeps bytes in a cyclic buffer of that size to be removed when it fills up.

Current logic (and implementing Write) might be too much, since bytes will probably always be fed one by one anyway. Otherwise a faster way of removing a sequence might be needed (one by one is inefficient).";

    static SMALL_TWO: &str = "This adds some function, I guess?, that has a size set at construction, and keeps bytes in a cyclic buffer of that size to be ground up and injested when it fills up.

Currently my sense of smell (and the pain of implementing Write) might be too much, since bytes will probably always be fed one by one anyway. Otherwise crying might be needed (one by one is inefficient).";


    #[test]
    fn simple_lcs() {
        let one = "a long time ago".into();
        let two = "a few long times tables".into();
        let result = longest_common(&one, &two, 0, 16);
        assert!(result.is_none());

        let result = longest_common(&one, &two, 5, 4);
        assert_eq!(result, Some((1, 11)));
    }

    #[test]
    fn first_diff_byte() {
        let one: String = iter::repeat('a').take(100)
            .chain(iter::once('b'))
            .chain(iter::repeat('c').take(5))
            .chain(iter::repeat('a').take(20))
            .collect();
        let two: String = iter::repeat('a').take(200).collect();

        let one = one.into();
        let two = two.into();
        let result = find_diff_start(&one, &two);
        let result_inv = find_diff_start(&two, &one);
        assert_eq!(result, result_inv);
        assert_eq!(result, (100, 20));
    }

    #[test]
    fn diff_range_two() {
        let one = "this is one string";
        let two = "this is two string";
        let result = find_diff_start(&one.into(), &two.into());
        assert_eq!(&one[result.0..one.len() - result.1], "one");
    }


    #[test]
    fn diff_smoke_test() {
        let one = SMALL_ONE.into();
        let two = SMALL_TWO.into();

        let delta = SmallSlowDiff::compute_delta(&one, &two, 16);
        println!("delta: {:?}", &delta);
        //assert!(false);

        let result = delta.apply(&one);
        assert_eq!(result, two);

        let delta = SmallDiff::compute_delta(&one, &two, 16);
        println!("delta: {:?}", &delta);
        //assert!(false);

        let result = delta.apply(&one);
        assert_eq!(result, two);
    }

    #[test]
    fn test_larger_diff() {
        let one = INTERVAL_STR.into();
        let two = BREAKS_STR.into();

        let delta = SmallDiff::compute_delta(&one, &two, 16);
        let result = delta.apply(&one);
        assert_eq!(String::from(result), String::from(two));
    }

    #[test]
    fn test_suffix_diff() {
        let one = SMALL_ONE.into();
        let two = SMALL_TWO.into();

        let delta = SuffixDiff::compute_delta(&one, &two, 8);
        println!("{:?}", &delta);
        let result = delta.apply(&one);
        assert_eq!(String::from(result), String::from(two));
    }

    #[test]
    fn test_larger_diff_tricksy() {
        let one = INTERVAL_STR.into();
        let two = BREAKS_STR.into();

        let delta = SmallTricksyDiff::compute_delta(&one, &two, 16);
        let result = delta.apply(&one);
        assert_eq!(String::from(result), String::from(two));
    }

    #[test]
    fn test_suffix_diff_tricksy() {
        let one = SMALL_ONE.into();
        let two = SMALL_TWO.into();

        let delta = SmallTricksyDiff::compute_delta(&one, &two, 16);
        println!("{:?}", &delta);
        let result = delta.apply(&one);
        assert_eq!(String::from(result), String::from(two));
    }


    #[test]
    fn cursor_chunk_iter() {
        let rope = Rope::from(INTERVAL_STR);
        for chunk_size in &[3, 16, 33, 1, 42, 420, 100000000] {
            let mut cursor = Cursor::new(&rope, 0);
            let mut chunks = Vec::with_capacity(rope.len() / chunk_size);
            while cursor.pos() < rope.len() {
                chunks.push(cursor.next_utf8_chunk_in_leaf(*chunk_size));
            }
            println!("chunk_size == {}", chunk_size);
            assert_eq!(INTERVAL_STR, chunks.into_iter().collect::<String>().as_str());
        }
    }

    #[test]
    fn suffix_smoke_test() {
        let one = Rope::from("this is my thistly test string, isn't mycium stringly?");
        let two = Rope::from("my this is surely test stuff, thistles and gore");
        let suffix = SuffixTable::new(String::from(one));
        let mut cursor = Cursor::new(&two, 0);
        let slice = cursor.next_utf8_chunk_in_leaf(4);
        assert_eq!(slice, "my t");
        let slice = cursor.next_utf8_chunk_in_leaf(8);
        assert_eq!(slice, "his is s");
        let slice = cursor.next_utf8_chunk_in_leaf(usize::max_value());
        assert_eq!(slice, "urely test stuff, thistles and gore");
        cursor.set(0);

        assert_eq!(SuffixDiff::match_for_cursor(&mut cursor, &suffix, 4), Some((8, 15)));
        cursor.set(3);
        assert_eq!(SuffixDiff::match_for_cursor(&mut cursor, &suffix, 4), Some((0, 8)));
        cursor.set(11);
        assert_eq!(SuffixDiff::match_for_cursor(&mut cursor, &suffix, 4), None);
        cursor.set(30);
        assert_eq!(SuffixDiff::match_for_cursor(&mut cursor, &suffix, 4), Some((11, 17)));

        let three = Rope::from("this is not my thistly test string");
        let mut cursor = Cursor::new(&three, 0);
        assert_eq!(SuffixDiff::match_for_cursor(&mut cursor, &suffix, 4), Some((0, 8)));
        //let iv = Interval::new(0, 8);
        //els.push(DeltaElement::Insert(one.subseq(iv)));
    }

    //#[test]
    //fn making_hashes() {
        //let s = include_str!("../Cargo.toml");
        //let hashes = make_line_hashes(&s);
        //assert_eq!(hashes.get("[package]\n"), Some(0));
    //}

    fn make_lines(n: usize) -> String {
        let mut s = String::with_capacity(n * 81);
        let line: String = iter::repeat('a').take(79).chain(iter::once('\n')).collect();
        for _ in 0..n {
            s.push_str(&line);
        }
        s
    }

    #[test]
    fn match_right_simple() {
        let rope =     Rope::from("aaaaaaaaaaaaaaaa");
        let chunk1 = String::from("aaaaaaaaaaaaaaaa");
        let chunk2 = String::from("baaaaaaaaaaaaaaa");
        let chunk3 = String::from("abaaaaaaaaaaaaaa");
        let chunk4 = String::from("aaaaaabaaaaaaaaa");
        let mut cursor = Cursor::new(&rope, 0);
        assert_eq!(expand_match_right(&chunk1, &mut cursor), 16);
        assert_eq!(expand_match_right(&chunk2, &mut cursor), 0);
        assert_eq!(expand_match_right(&chunk3, &mut cursor), 1);
        assert_eq!(expand_match_right(&chunk4, &mut cursor), 6);
    }

    #[test]
    fn match_right() {
        let rope = Rope::from(make_lines(500));

        let mut chunk = make_lines(5);
        assert_eq!(chunk.len(), 400);
        chunk.push_str("bbb\n");

        let mut cursor = Cursor::new(&rope, 0);
        assert_eq!(expand_match_right(&chunk, &mut cursor), 400);
        assert_eq!(expand_match_right(&chunk, &mut cursor), 400);
        assert_eq!(expand_match_right(&chunk, &mut cursor), 400);
        assert_eq!(expand_match_right(&chunk, &mut cursor), 400);
        assert_eq!(expand_match_right(&chunk, &mut cursor), 400);

    }

    #[test]
    fn match_left_simple() {
        let rope =     Rope::from("aaaaaaaaaaaaaaaa");
        let chunk1 = String::from("aaaaaaaaaaaaaaaa");
        let chunk2 = String::from("aaaaaaaaaaaaaaba");
        let chunk3 = String::from("aaaaaaaaaaaaaaab");
        let chunk4 = String::from("aaaaaabaaaaaaaaa");
        let mut cursor = Cursor::new(&rope, rope.len());
        assert!(cursor.get_leaf().is_some());
        assert_eq!(expand_match_left(&chunk1, &mut cursor), 16);
        cursor.set(rope.len());
        assert_eq!(expand_match_left(&chunk2, &mut cursor), 1);
        cursor.set(rope.len());
        assert_eq!(expand_match_left(&chunk3, &mut cursor), 0);
        cursor.set(rope.len());
        assert_eq!(expand_match_left(&chunk4, &mut cursor), 9);
    }

    #[test]
    fn match_left_ne_lens() {
        let rope =     Rope::from("aaaaaaaaaaaaaaaa");
        let chunk1 = String::from("aaaaaaaaaaaaa");
        let chunk2 = String::from("aaaaaaaaaaaaab");
        let mut cursor = Cursor::new(&rope, rope.len());
        assert_eq!(expand_match_left(&chunk1, &mut cursor), 13);
        cursor.set(rope.len());
        assert_eq!(expand_match_left(&chunk2, &mut cursor), 0);

    }

    #[test]
    fn match_left() {
        let rope = Rope::from(make_lines(10));
        let mut chunk = String::from("bbb");
        chunk.push_str(&make_lines(5));
        //let exp = chunk.len();
        //chunk.push_str("bbb\n");

        let mut cursor = Cursor::new(&rope, rope.len());
        assert_eq!(expand_match_left(&chunk, &mut cursor), 400);
        assert_eq!(expand_match_left(&chunk, &mut cursor), 400);
        assert_eq!(expand_match_left(&chunk, &mut cursor), 400);
        assert_eq!(expand_match_left(&chunk, &mut cursor), 400);
        assert_eq!(expand_match_left(&chunk, &mut cursor), 400);

    }

    static INTERVAL_STR: &str = include_str!("../src/interval.rs");
    static BREAKS_STR: &str = include_str!("../src/breaks.rs");
}
