use std::collections::HashMap;
use xi_rope::rope::{Rope, RopeInfo, RopeDelta, LinesMetric};
//use xi_rope::interval::Interval;
//use xi_rope::delta::{Delta, DeltaElement};
use xi_rope::compare::RopeScanner;

use diff_play::*;

pub struct LineHashLisDiff;

const MIN_SIZE: usize = 32;

impl Diff<RopeInfo> for LineHashLisDiff {
    fn compute_delta(base: &Rope, target: &Rope, _min_size: usize) -> RopeDelta {
        LineHashLisDiff::compute_delta_impl(base, target)
    }
}

//impl Diff<RopeInfo> for LineHashLisDiff {
impl LineHashLisDiff {
    fn compute_delta_impl(base: &Rope, target: &Rope) -> RopeDelta {
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
        let line_hashes = make_line_hashes(&base_string, MIN_SIZE);

        //let mut offset = start_offset;

        let line_count = target.measure::<LinesMetric>() + 1;
        let mut matches = Vec::with_capacity(line_count);

        let mut targ_line_offset = 0;
        let mut prev_base = 0;

        let mut needs_subseq = false;
        for line in target.lines_raw(start_offset..target_end) {
            let non_ws = non_ws_offset(&line);
            if line.len() - non_ws >= MIN_SIZE {
                if let Some(base_off) = line_hashes.get(&line[non_ws..]) {
                    let targ_off = targ_line_offset + non_ws;
                    matches.push((start_offset + targ_off, *base_off));
                    if *base_off < prev_base {
                        needs_subseq = true;
                    }
                    prev_base = *base_off;
                }
            }
            targ_line_offset += line.len();
        }

        // we now have an ordered list of matches and their positions.
        // to ensure that our delta only copies non-decreasing base regions,
        // we take the longest increasing subsequence.
        // TODO: a possible optimization here would be to expand matches
        // to adjacent lines first? this would be at best a small win though..

        let longest_subseq = if needs_subseq {
            longest_increasing_region_set(&matches)
        } else {
            matches
        };

        // for each matching region, we extend it forwards and backwards.
        // we keep track of how far forward we extend it each time, to avoid
        // having a subsequent scan extend backwards over the same region.
        let mut prev_targ_end = start_offset;

        for (targ_off, base_off) in longest_subseq {
            if targ_off <= prev_targ_end { continue; }
            let (left_dist, mut right_dist) = fast_expand_match(base, target, base_off,
                                                                targ_off, prev_targ_end);
            if targ_off + right_dist > target_end {
                // don't let last match expand past target_end
                right_dist = target_end - targ_off;
            }

            let targ_start = targ_off - left_dist;
            let base_start = base_off - left_dist;
            let len = left_dist + right_dist;
            prev_targ_end = targ_start + len;

            builder.copy(base_start, targ_start, len);
        }

        if diff_end > 0 {
            builder.copy(base.len() - diff_end, target.len() - diff_end, diff_end);
        }

        builder.to_delta(base, target)
    }
}

/// Finds the longest increasing subset of copyable regions. This is essentially
/// the longest increasing subsequence problem. This implementation is adapted
/// from https://codereview.stackexchange.com/questions/187337/longest-increasing-subsequence-algorithm
fn longest_increasing_region_set(items: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut result = vec![0];
    let mut prev_chain = vec![0; items.len()];

    for i in 1..items.len() {
        // If the next item is greater than the last item of the current longest subsequence,
        // push its index at the end of the result and continue.
        let last_idx = *result.last().unwrap();
        if items[last_idx].1 < items[i].1 {
            prev_chain[i] = last_idx;
            result.push(i);
            continue;
        }

        let next_idx = match result.binary_search_by(|&j| items[j].1.cmp(&items[i].1)) {
            Ok(idx) => continue, // we ignore duplicates
            Err(idx) => idx,
        };

        if &items[i].1 < &items[result[next_idx]].1 {
            if next_idx > 0 {
                prev_chain[i] = result[next_idx - 1];
            }
            result[next_idx] = i;
        }
    }

    // walk backwards from the last item in result to construct the final sequence
    let mut u = result.len();
    let mut v = *result.last().unwrap();
    while u != 0 {
        u -= 1;
        result[u] = v;
        v = prev_chain[v];
    }
    result.iter().map(|i| items[*i]).collect()
}
