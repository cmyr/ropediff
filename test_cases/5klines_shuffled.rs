        let first_line = self.line_of_offset(text, self.selection.first().unwrap().min());
        if let Some(lowest_group) = toggled_groups.iter().cloned().next() {
            Merge(4,3),
            Token::BorrowedStr("a\u{00A1}\u{4E00}\u{1F4A9}"),
                metadata_name: metadata_name.map(|s| Cow::Owned(s)),
}
/// State required to resolve a drag gesture into a selection.

                metadata_sort_index: sort_index,
                    let mut e = &mut self.peers[ei];
        Self {

        // todo: the correct Find instance needs to be updated with the new parameters
        assert_eq!(0, a.offset_of_line(0));
/// # Performance
        }
                    } else {
                    let _ = s.pop();
    fragment: &'a str
                for region in find.occurrences().regions_in_range(start_pos, pos) {
        // that into our actual target.
    fn to_base_units(_: &String, in_measured_units: usize) -> usize {

        let mut update = json!({

use unicode_segmentation::GraphemeCursor;
    // since undo and gc replay history with transforms, we need an empty set

        // send updated find status only if there have been changes
        engine.undo([].iter().cloned().collect());
    }
/// Usage of static strings is encouraged for best performance to avoid copies.
        let a = Rope::from("a\u{00A1}\u{4E00}\u{1F4A9}");

            MetadataType::ThreadName {..} => "thread_name",
        let contents = match rev.edit {
    Edit {
                //self.end - prev_pos
///

        assert_eq!(None, b.next_codepoint_offset(9));
                RenderTactic::Preserve => {
                         start: usize, end: usize, sel: &[(usize, usize)],
                self.do_find_next(text, false, wrap_around, allow_same, &modify_selection),
        a.lines_all().collect::<Vec<_>>());
        let mut all_samples = self.samples.lock().unwrap();

        b.iter(|| black_box(trace.closure("something", &["benchmark"], || {})));
                        let offset = self.offset_of_line(text, start_line);
            trace.instant("something", &[]);
                                        &selections, &hls, style_spans);
    fn eq(&self, other: &Sample) -> bool {
                            Cursor::new(breaks, offset));
extern crate serde;
//     http://www.apache.org/licenses/LICENSE-2.0
        self.tombstones = new_tombstones;
        // TODO: this does 2 calls to Delta::synthesize and 1 to apply, this probably could be better.
    /// This uniquely represents the identity of this revision and it stays
                MergeTestOp::Edit { ei, p, u, d: ref delta } => {
        let a_revs = basic_insert_ops(inserts.clone(), 1);
                        } else {
    fn is_ok_child(&self) -> bool {
        for i in 1..(s1.len() * 3) {

    ($num_args: expr) => {
        let a = Rope::from("a");
        r.slice_to_string(0, r.len())
    }
/// xi_trace::trace_payload("something happened", &["rpc", "response"], "a note about this");
        if offset >= text.len() {
        samples.sort_unstable();
        self.do_set_replace(replacement, false);
}

//! It enables support for full asynchronous and even peer-to-peer editing.
            _ => None
            SelectionForFind { case_sensitive } =>
        }
        let offset = self.line_col_to_offset(text, line as usize, 0);

                self.add_selection_by_movement(text, Movement::Down),
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        }

}

                }
    #[serde(skip_serializing_if = "Option::is_none")]
    }
#[derive(Clone, Copy)]
            b.push('b');
                let mut sel = Selection::new();

            Assert(1, "cb".to_owned()),

    fn accumulate(&mut self, other: &Self) {
        let ix = self.find_rev_token(base_rev).expect("base revision not found");
    pub fn edit_rev(&mut self, priority: usize, undo_group: usize,
    }
    priority: usize,
        prev_boundary.unwrap_or(None)
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        // send updated replace status if changed
///
//! pending edit in flight each.
        self.pending_render
            base_sel: selection.clone(),
    fn next(&mut self) -> Option<Cow<'a, str>> {

        let cursor = Cursor::new(self, offset);

    }
    }
    let mut out = Vec::with_capacity(revs.len() - base_revs.len());

        assert_eq!(vec!["a", "b", "c"], a.lines_all().collect::<Vec<_>>());
    pub fn offset_of_line(&self, line: usize) -> usize {
                    let e = &mut self.peers[ei];
/// Represents the current state of a document and all of its history
        ];
        assert_eq!(vec!["\n"], a.lines_raw_all().collect::<Vec<_>>());
            name: name.into(),
                });

            SampleGuard::new_disabled()
                warn!("Usage of click is deprecated; use do_gesture");
/// creation.  Roughly corresponds to start_ns but instead there's a
        let mut engine = Engine::new(Rope::from(TEST_STR));
        tombstones = new_tombstones;
        where D: Deserializer<'de>,
                el.end()
            return false;
        use self::MergeTestOp::*;
            min: offset,

        engine.edit_rev(1, 2, new_head, d2);
    // * Unicode width (so CJK counts as 2)

        let (iv, new_len) = delta.summary();
}
}
        assert_eq!(snapshot[6].name, "z");
        if offset == s.len() {


    fn to_base_units(s: &String, in_measured_units: usize) -> usize {
                        }

        s.is_char_boundary(offset)
            categories: None,
        }
    deletes: Subset,
            }
pub struct SampleGuard<'a> {
extern crate test;
        let new_inserts = union_ins_delta.inserted_subset();

            self.push_leaf(s[..splitpoint].to_owned());
}
    #[cfg(feature = "benchmarks")]
            pid: sys_pid::current_pid(),
/// ```rust
        next_expand_by = Vec::with_capacity(expand_by.len());
                        ops.push(self.build_update_op("invalidate", None, seg.n));
        engine.gc(&to_undo);
}
        // the smallest values with which it still fails:
                        if self.undone_groups.contains(undo_group) {
    }
            Drag(MouseAction { line, column, .. }) =>

        revs.append(&mut revs_2);
    pub fn collapse_selections(&mut self, text: &Rope) {
        assert_eq!(0, b.line_of_offset(0));
        assert_eq!(TEST_STR, String::from(engine.get_head()));
// Try to split at newline boundary (leaning left), if not, then split at codepoint
                        s = &s[..s.len() - 1];
                (region.is_upstream() && c == pos) ||

//! info.
pub struct View {
    }
    fn lines_med() {

        a.push('\n');
        // ensure that when the samples are sorted by time they come out in a
        }
    /// in the slice up to `offset`.
        /// revision that were deleted by this revision.
        let first_candidate = self.find_first_undo_candidate_index(&toggled_groups);
        for i in ((edits-max_undos)..edits).rev() {
/// * `categories` - A static array of static strings that tags the samples in
        for (iv, style) in style_spans.iter() {
/// * `name` - A string that provides some meaningful name to this sample.
            let base_index = find_base_index(&self.revs, &other.revs);

        for rev in old_revs.into_iter().rev() {

                },

    }
/// A flag used to indicate when legacy actions should modify selections
/// some way.
    }
    fn bench_trace_instant_disabled(b: &mut Bencher) {
            self.wrap_col = WrapWidth::Bytes(wrap_col);
        b if b < 0x80 => 1,
    #[test]
    }

        }
use styles::{Style, ThemeStyleMap};
        } else if self.first_line + self.height <= line {
            }
            // with offset == 0, but be defensive.
        let delta_ops = compute_deltas(&revs, &text, &tombstones, &deletes_from_union);
pub struct SampleArgs {
        Assert(usize, String),
            }
        let script = vec![
    utf16_size: usize,


        Self::with_limit_bytes(1 * 1024 * 1024)
    #[bench]
        trace.instant("1", &["test"]);
            black_box(time::precise_time_ns());
                      delta: &Delta<RopeInfo>, client: &Client,
        delta.apply(&self.text)
    }
///
            Edit { ei: 2, p: 2, u: 1, d: parse_delta("z--") },
        }
        revs.append(&mut revs_3);
        assert_eq!(correct, res);
    pub(crate) fn get_buffer_id(&self) -> BufferId {
        for _i in 0..50 {

            Assert(2, "adfc".to_owned()),
            Merge(0,1),

    }
            Assert(1, "acpbdj".to_owned()),
            self.select_next_occurrence(text, reverse, true, allow_same, modify_selection);
    fn offset_of_line_panic() {
        assert_eq!(utf16_units, 17);
        assert_eq!(trace.samples_cloned_unsorted().len(), 0);
    fn next(&mut self) -> Option<&'a str> {
            let b_to_merge = &other.revs[base_index..];
    /// Find parameters and number of matches changed.
        assert_eq!(snapshot[0].name, "process_name");
    max: usize,
        if (b as i8) >= -0x40 {
            replace: None,
            Merge(1,0),
            if i >= max_undos {
            Edit { ei: 0, p: 3, u: 1, d: parse_delta("-c-") },
            AssertMaxUndoSoFar(0,2),
        assert_eq!(utf16_units, 9);
}
        let gc : BTreeSet<usize> = [1].iter().cloned().collect();
            }
            .map(|(i, _)| i)
        self.invalidate_selection(text);

        let search_query = match self.selection.last() {
#[inline]
/// # Performance
        engine.edit_rev(1, 3, new_head_2, d3);
    ns / 1000
            MetadataType::ProcessName {name} => (Some(name), None),
    ///
/// from multiple distributed sources (i.e. you want to sort the merged result
    }
                if !inserts.is_empty() {
        max_undo_so_far = std::cmp::max(max_undo_so_far, undo_group);
                           style: &Style) -> usize {
            if let GraphemeIncomplete::PreContext(_) = incomp {
}
                    let sel_start_ix = clamp(region.min(), start_pos, pos) - start_pos;
        json!({"test": value})
    fn edit_rev_undo() {

        let revs = basic_insert_ops(inserts, 1);
        let text_inserts = inserts.transform_shrink(&deletes_from_union);
        Cow::from(a.into_owned() + b)
        -##-
#[cfg(feature = "dict_payload")]
        let (leaf, start_pos) = self.cursor.get_leaf().unwrap();
/// assert_eq!("hello world", String::from(a.clone() + b.clone()));
        self.set_selection_for_edit(text, new_sel);
    #[test]
/// Also note: in addition to the `From` traits described below, this module
    // use own breaks if present, or text if not (no line wrapping)
            let _x = trace.block("something", &[]);
            event_type: SampleEventType::Instant,
/// Returns a tuple of a new text `Rope` and a new `Tombstones` rope described by `new_deletes_from_union`.
            }
        let pos = soft_breaks.map_or(start_of_line.next::<LinesMetric>(), |bc| {
        }

            Merge(2,0),
                if let Edit { ref undo_group, ref inserts, ref deletes, .. } = rev.edit {
            }
        Some(&leaf[start_pos .. end])
        let mut engine = Engine::new(Rope::from(TEST_STR));
        self.revs.iter().enumerate().rev()
                    if seg.validity == line_cache_shadow::ALL_VALID {
    #[bench]
                    text.slice_to_string(region.min(), region.max())
            replace_changed: false,

    }
        engine.gc(&gc);
        }
        }
                Edit { priority, undo_group, inserts, deletes } => {
}
                        gc_dels = new_gc_dels;
        }
            AssertAll("zacpb".to_owned()),
    }
                Contents::Undo { .. } => panic!(),

/// Usage of static strings is encouraged for best performance to avoid copies.

    #[test]
            result.push_str(chunk);
        let trace = Trace::enabled(Config::default());
    find: Vec<Find>,
                    let new_gc_dels = if inserts.is_empty() {
        let new_head_2 = engine.get_head_rev_id().token();
        deletes_from_union = new_deletes_from_union;
    } else {
        }
///
                }
    pub fn gc(&mut self, gc_groups: &BTreeSet<usize>) {
        let a = Rope::from("");
            Contents::Edit {priority, undo_group, ref inserts, ref deletes} => {

        let closest_occurrence = self.find.iter().flat_map(|x|
        assert_eq!(snapshot[3].name, "y");
/// Get a slice of a `Rope`:
        engine.undo([1,2].iter().cloned().collect());
            MetadataType::ProcessSortIndex {sort_index} => (None, Some(sort_index)),
/// ```

    fn from_base_units(s: &String, in_base_units: usize) -> usize {
                breaks.convert_metrics::<BreaksBaseMetric, BreaksMetric>(offset)

    }
        as_vec.extend(all_samples.iter().cloned());
    let mut out = Vec::with_capacity(revs.len());
        self.set_selection_raw(text, selection);
        assert_eq!(1, r.line_of_offset(r.len() - 1));
            {
            Assert(1, "ab".to_owned()),
    }
        engine.edit_rev(0, 2, first_rev, build_delta_2());

    }
    }
                Some(pos) => offset += pos + 1,
            name: name.into(),
        // A with ring, hangul, regional indicator "US"
            Contents::Edit {priority, undo_group, ref inserts, ref deletes} => {

            }
///     something_else_expensive(result);

    TRACE.enable_config(config);
    }
        where S: Serializer
    pub fn request_lines(&mut self, text: &Rope, client: &Client,

    fn find_common_1() {
            self.find.push(Find::new());
pub type TracePayloadT = StrCow;
            soln.push('b');
        let mut sel = self.selection.clone();
        }
        let mut ix = 0;
        let r = r + Rope::from(&b[MIN_LEAF..]);
    /// concurrently it will have count `2` so that undoing one delete but not
    scroll_to: Option<usize>,
        where S: Into<StrCow>, C: Into<CategoriesT>
        let expand_by = compute_transforms(a_revs);
        assert_eq!(String::from(&a).lines().collect::<Vec<_>>(),
        &leaf[offset..end]
    fn bench_trace_instant(b: &mut Bencher) {
            categories: Some(categories.into()),
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
    pub fn session_id(&self) -> SessionId {
/// }
            _ => panic!("Unexpected chrome sample type '{}'", symbol)
                   whole_words: bool) {
        let b_revs = basic_insert_ops(inserts, 2);
        assert_eq!(snapshot[1].args.as_ref().unwrap().metadata_name.as_ref().is_some(), true);
        let mut as_vec = Vec::with_capacity(all_samples.len() + 10);
            MergeTestState { peers }
        let utf16_units = rope_with_emoji.convert_metrics::<BaseMetric, Utf16CodeUnitsMetric>(utf8_offset);
        // update only find highlights affected by change
        for _ in 0..in_measured_units {
    use test::black_box;
        // the front-end, but perhaps not for async edits.
            match reverse {
                   style_spans: &Spans<Style>, line_num: usize) -> Value
                self.set(pos);
        for find in &mut self.find {
use std::fmt;
        base_subset.is_some() && base_subset == other_subset
            let b = self.current.unwrap().as_bytes()[self.idx];
            let a_new = rearrange(a_to_merge, &common, self.deletes_from_union.len());
    fn render_line(&self, client: &Client, styles: &StyleMap,
        assert_eq!(trace.samples_cloned_unsorted().len(), 7);
        cursor.is_boundary::<BaseMetric>()
            deletes = deletes.transform_expand(&new_trans_inserts);
    #[test]
    let mut out = Vec::new();
                // TODO create InsertDelta directly and more efficiently instead of factoring
    fn is_boundary(s: &String, offset: usize) -> bool {
                sel.add_region(SelRegion::new(last.start, offset));
    pub metadata_sort_index: Option<i32>,
                                                        &mut line_cursor,


            if (c > start_pos && c < pos) ||
    /// The line number is 0-based.
    highlight_find: bool,
                self.start_drag(offset, offset, offset);
        }


            Edit { ei: 2, p: 1, u: 1, d: parse_delta("ab") },
    #[cfg(feature = "dict_payload")]

            MetadataType::ProcessLabels {..} => "process_labels",
    }
        assert_eq!(snapshot[0].args.as_ref().unwrap().metadata_name.as_ref().is_some(), true);
            prev_boundary = c.prev_boundary(&l, leaf_offset);




        /// along with the newline that triggered it.
    /// update cycle. This should be a last resort, updates should generally cause
                        let mut line_cursor = Cursor::new(text, offset);
    }
        undo_test(false, [2].iter().cloned().collect(), "0123456789abcDEEFghijklmnopqr999stuvz");
    pub args: Option<SampleArgs>

            Edit { ei: 0, p: 3, u: 1, d: parse_delta("-c-") },

        x
        ");
            Merge(0,2), Merge(1, 2),

        self.send_update_for_plan(text, client, styles,
                                     event_type: SampleEventType)
            categories: Some(categories.into()),
            let splitpoint = if s.len() > MAX_LEAF {
                    let mut selection = self.selection.clone();
        let mut result = Cow::from("");
use std::collections::BTreeSet;

        engine.gc(&gc);
// Licensed under the Apache License, Version 2.0 (the "License");
    #[test]
/// fn something_else_expensive(value: u32) {
    fn compute_deltas_1() {
        } else {
    }
                        };
                }
        if b >= 0xf0 {
        where S: Into<StrCow>, C: Into<CategoriesT>

    #[serde(skip_serializing_if = "Option::is_none")]
        let mut engine = Engine::new(Rope::from(TEST_STR));

    Metadata
    deletes_from_union: Subset,
/// * `categories` - A static array of static strings that tags the samples in


            if next_chunk.is_none() { return None; }
/// Revision will be used, which means only the (small) set of concurrent edits
            Assert(1, "ab".to_owned()),
        b.iter(|| black_box(trace.closure_payload(
        // probably change as the recording internals change.
///
}
                payload: payload,
        }
    fn measure(info: &RopeInfo, _: usize) -> usize {
fn find_common(a: &[Revision], b: &[Revision]) -> BTreeSet<RevId> {
        let (new_text, new_tombstones) =
/// # use xi_rope::Rope;
        let b = a.slice(1, 10);
pub struct Sample {

    ///
///     let _guard = xi_trace::trace_block("something_else_expensive", &["rpc", "response"]);
        soln.push('f');
        self.select_region(text, offset, SelRegion::new(start, end), multi_select);
    find_leaf_split(s, max(MIN_LEAF, s.len() - MAX_LEAF))
    #[bench]
// line iterators

/// Rebase `b_new` on top of `expand_by` and return revision contents that can be appended as new

    // trace_block()/trace_closure
/// `base_revs`, but modified so that they are in the same order but based on
/// revisions not shared by both sides.

        a.lines_all().collect::<Vec<_>>());
        self.replace = Some(Replace { chars, preserve_case });

    match std::env::current_exe() {
            Revision {
        let mut delta = serializer.serialize_struct("Delta", 2)?;
                                             &text, false);
    /// finer grain invalidation.
        match *self {
        } else {
/// Applies an optimization where it combines sequential revisions with the
        assert!(self.current.is_some());
    fn merge_session_priorities() {
        self.pending_render = pending

        if let Some(edit) = contents {

        assert_tokens(&rope, &[
            }

        let a = Rope::from("A\u{030a}\u{110b}\u{1161}\u{1f1fa}\u{1f1f8}");
        assert_eq!("0!3456789abcDEEFGIjklmnopqr888999stuvHIz", String::from(engine.get_head()));
    where D: serde::Deserializer<'de> {
    /// Front end's line cache state for this view. See the `LineCacheShadow`

                }
    }
pub fn len_utf8_from_first_byte(b: u8) -> usize {

        // current recording mechanism should see:
            buffer_id: buffer_id,
    pub event_type: SampleEventType,
        }

        }

    #[cfg(feature = "benchmarks")]
        let mut cur_len_utf8 = 0;
    #[test]

    /// The incrementing revision number counter for this session used for `RevId`s
        // todo: for now only a single search query is supported however in the future
                self.set_selection(text, SelRegion::caret(offset));
            self.first_line = line;
            utf16_count += 1;
            // TODO: on double or triple click, quantize offset to requested granularity.
    pub width: f64,

        println!("{:#?}", b_delta_ops);
                        word_cursor.select_word()
    {
        let mut c = GraphemeCursor::new(pos, l.len() + leaf_offset, true);

            Merge(3,1), Merge(5,3),
impl Rope {
    /// return if a delete is un-done or an insert is re- done.
    fn build_delta_1() -> Delta<RopeInfo> {
    }
                        assert_eq!(correct, &String::from(e.get_head()), "for peer {}", ei);
                    assert_eq!(correct, e.max_undo_group_id(), "for peer {}", ei);
            Merge(1,0),
    fn from(r: Rope) -> String {
        self.rev_id_counter += 1;
                    let full_path_str = full_path.into_string();
    fn eq_small() {
        }
                out.push((prio, inserts));
        Ok(exe_name) => {
        assert_eq!(a.len(), r.offset_of_line(1));
        });
        self.convert_metrics::<BaseMetric, LinesMetric>(offset)
        ];
                    .with_horiz(horiz)


        // x, a, y, b, z, c
    Width(f64),
        }
        trace.instant("4", &["test"]);
        // position after 'f' in four
            GestureType::ToggleSel => self.toggle_sel(text, offset),
            if self[i] != other[i] {


    // in a single session.
            shuffle(&self.text, &self.tombstones, &self.deletes_from_union, &new_deletes_from_union);
        assert_eq!(snapshot.len(), 9);
                                                    self.size.width));
use delta::{Delta, InsertDelta};
        true
            //}
        self.subseq(iv)
        // rebase the delta to be on the head union instead of the base_rev union

    pub fn set_scroll(&mut self, first: i64, last: i64) {
        /// and new deletes_from_union

        assert_eq!(1, a.line_of_offset(2));

                    result = cow_append(result, self.fragment);
// distributed under the License is distributed on an "AS IS" BASIS,
    #[allow(dead_code)]
}
        ];
        let after_first_edit = engine.get_head_rev_id().token();
    pub fn samples_cloned_unsorted<'a>(&'a self) -> Vec<Sample> {
    /// This is used to determined whether FindStatus should be sent to the frontend.
        //let len = if self.end - self.cursor.pos() < leaf.len() - start_pos {
    fn test_samples_closure() {
    }
// You may obtain a copy of the License at
            Click(MouseAction { line, column, flags, click_count }) => {
    size: Size,
                if self.fragment.is_empty() {
        assert!(end <= leaf.len());
            if self.fragment.is_empty() {
    /// slices will be nonempty.
    ObjectDestroyed,

            duration_us: None,
    }
        }
            offset = text.len();
    1
        ");
                             client: &Client, style_spans: &Spans<Style>)
        self.revs.push(new_rev);
    for r in revs {
    fn can_fragment() -> bool { false }
    /// Constructs an instantaneous sample.
            trace.instant_payload("a", &["test"], to_payload("test_get_samples_nested_trace"));
    fn undo_test(before: bool, undos : BTreeSet<usize>, output: &str) {
        // probably change as the recording internals change.
                        word_cursor.select_word()

    #[test]
}
    fn new_metadata(timestamp_ns: u64, meta: MetadataType, tid: u64) -> Self {
    TRACE.is_enabled()
        String::from(&r)
        rendered_styles
    }
// Licensed under the Apache License, Version 2.0 (the "License");
        if let Some(breaks) = self.breaks.as_mut() {
        engine.edit_rev(1, 1, first_rev, d1.clone());
use word_boundaries::WordCursor;

        assert_eq!(1, b.line_of_offset(2));
    fn delta_rev_head_3() {
                        };
        let first_rev = engine.get_head_rev_id().token();
// See the License for the specific language governing permissions and
            let mut word_cursor = WordCursor::new(text, offset);
        let text_with_inserts = text_inserts.apply(&text);
            // with offset == 0, but be defensive.
            } else {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
///
        let utf16_units = rope.measure::<Utf16CodeUnitsMetric>();
        self.set_selection(text, new_sel);
            offset = text.prev_grapheme_offset(offset + 1).unwrap();
            name: name.into(),
#[inline]
    fn bench_get_tid(b: &mut Bencher) {
/// A [rope](https://en.wikipedia.org/wiki/Rope_(data_structure)) is a data structure
                if !region.is_caret() {
        inserts: Subset,

        trace.instant("5", &["test"]);

    }
    let mut cur_all_inserts = Subset::new(deletes_from_union.len());
    /// description for the invariant.
    pub fn next_codepoint(&mut self) -> Option<char> {
    pub fn max_size_in_bytes(&self) -> usize {
impl CategoriesT {
    {
                    }


        let start = self.line_col_to_offset(text, line, 0);
            SampleEventType::AsyncInstant => 'n',
        let mut engine = Engine::new(Rope::from(TEST_STR));
    }
            self.set_selection(text, selection);
                    new_deletes = new_deletes.transform_expand(inserts);
            Assert(0, "acrbdz".to_owned()),
            None
        } else {
/// implements `From<Rope> for String` and `From<&Rope> for String`, for easy
        if last_col == 0 && last_line > first_line {

                    self.fragment = "";
        let head2 = engine.get_head_rev_id().token();
pub struct Config {
            rebase(expand_by, b_deltas, self.text.clone(), self.tombstones.clone(), self.deletes_from_union.clone(), max_undo)
            Assert(0, "zacpbdj".to_owned()),
            if region.is_caret() {

        let correct = parse_subset_list("
            sel
    view_id: ViewId,
            rev_id, max_undo_so_far,
                    }
                                                                0, "copy", 2)?;
/// on program launch.
        fn run_op(&mut self, op: &MergeTestOp) {
                },
        let script = vec![
}
            },
pub struct ByteIter<'a> {
}


    type L = String;

        let mut revs = basic_insert_ops(inserts_1, 1);
    /// the same even if it is rebased or merged between devices.
        all_samples.push_back(sample);
        while !leaf.is_char_boundary(end) {
}
    match b {
/// an argument.
    }

//     http://www.apache.org/licenses/LICENSE-2.0
        },
            self.set_selection(text, sel);
    /// controlled by features (default string).
        self.replace.clone()
                           styles: &StyleMap, style_spans: &Spans<Style>,
            Assert(2, "ab".to_owned()),
                leaf_offset = self.pos() - noffset;
        for rev in &self.revs[first_candidate..] {

use std::hash::{Hash, Hasher};
        }).collect();
/// Then offset=1 is invalid.

    fn bench_single_timestamp(b: &mut Bencher) {
    }
            // These two will be different without using session IDs
    #[test]
        let first_rev = engine.get_head_rev_id().token();
use std::cell::RefCell;

    }
        }

        toggled_groups: BTreeSet<usize>,  // set of undo_group id's
            Edit { ei: 2, p: 1, u: 1, d: parse_delta("z--") },

                                                        soft_breaks.as_mut(),
    /// Storage for all the characters that have been deleted  but could
            GestureType::MultiWordSelect =>
fn shuffle_tombstones(text: &Rope, tombstones: &Rope,
    fn line_of_offset_small() {
        self.view_id
        b.push('\n');
                metadata_name: None,
                if s.ends_with('\n') {
            for &occurrence in find.occurrences().iter() {
        if before {
fn find_leaf_split_for_merge(s: &str) -> usize {
        cur_len_utf8
        } else {
///
        }
        assert_eq!("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", String::from(engine.get_head()));
        let mut engine = Engine::new(Rope::from(TEST_STR));

            GestureType::RangeSelect => self.select_range(text, offset),
    fn bench_trace_block(b: &mut Bencher) {
impl<'de> Deserialize<'de> for Delta<RopeInfo> {
        let a = Rope::from("a\rb\rc");
            engine.edit_rev(1, i+1, head, d);
/// of Sample that determines order.
            Assert(0, "zacbd".to_owned()),
            //leaf.len() - start_pos
    }
            // if using breaks update cursor
            //} else {
        let d1 = Delta::simple_edit(Interval::new_closed_open(0,10), Rope::from(""), TEST_STR.len());
    /// Returns an iterator over chunks of the rope.
    fn set_selection_for_edit(&mut self, text: &Rope, sel: Selection) {
            Merge(0,2), Merge(1, 2),
    /// Tracks whether this view has been scheduled to render.
            }
    pub fn line_of_offset(&self, offset: usize) -> usize {
}
            self.record(Sample::new_duration(
#[inline]
    }

#[derive(Clone, Copy)]
    pub fn do_find_next(&mut self, text: &Rope, reverse: bool, wrap: bool, allow_same: bool,
//! This module actually implements a mini Conflict-free Replicated Data Type

        let mut a = String::new();
                }
        let script = vec![
            engine.undo(to_undo.clone());
                trace.instant_payload("b", &["test"], to_payload("test_get_sorted_samples"));

    fn next(&mut self) -> Option<u8> {

        self.timestamp_us == other.timestamp_us &&
            Assert(0, "acrpbdzj".to_owned()),
}
    /// vertical scroll position
                              to_payload("test_get_samples"));
/// How tracing should be configured.
        assert_eq!(vec!["a", "b"], a.lines_all().collect::<Vec<_>>());

    }
/// let trace_guard = xi_trace::trace_block("something_expensive", &["rpc", "request"]);
        // even though the actual sampling order (from timestamp of
use std::borrow::Cow;
                        gc_dels = gc_dels.transform_expand(inserts);
#[derive(Clone, Debug)]
/// Atomicity is implicit, putting the offset
            tid: sys_tid::current_tid().unwrap(),
    #[test]
    }
    type Item = Cow<'a, str>;
    undo_group: usize,
    // `session1==session2==0` is reserved for initialization which is the same on all sessions.
/// Replace part of a `Rope`:
    }
impl Serialize for DeltaElement<RopeInfo> {
                }

        for region in self.selection.regions_in_range(start_pos, pos) {
    /// Returns the byte offset corresponding to the line `line`.
impl StringArrayEq<Vec<String>> for &'static [&'static str] {
    }
        let inserts_2 = parse_subset_list("
const MAX_LEAF: usize = 1024;
                Some(offset)
            self.tombstones = dels_from_tombstones.delete_from(&self.tombstones);
        let (new_text, new_tombstones) = shuffle(&text_with_inserts, &self.tombstones,
    fn goto_line(&mut self, text: &Rope, line: u64) {
        !self.selection.regions_in_range(offset, offset).is_empty()
/// fn something_expensive() -> u32 {
        }).collect::<Vec<FindStatus>>()
        }
        engine.edit_rev(0, 2, first_rev, build_delta_2());
    }

        assert!(r.clone().slice(0, a.len()) == a_rope);
        ];
    #[serde(skip_serializing_if = "Option::is_none")]
    }
        }

    //TODO: implement iter_chunks using ranges and delete this
        let inserted = inserts.inserted_subset();
            self.collapse_selections(text);
///
        // multiple queries; select closest occurrence
}
        for _i in 0..50 {
        // of the delta so we can set the cursor before or after the edit, as needed.
}
        let rope = Rope::from("hi\ni'm\nfour\nlines");
}
        self.size = size;
    #[bench]
            self.first_line = line - (self.height - 1);
            Assert(0, "acbd".to_owned()),
            'b' => SampleEventType::AsyncStart,

        use self::MergeTestOp::*;
        }
    /// Each instance represents a separate search query.
            let (line, col) = self.offset_to_line_col(text, new_scroll_pos);
        }
        d

    fn merge_priorities() {
        assert_eq!(utf16_units, 17);
        if self.is_enabled() {
        use self::MergeTestOp::*;
        }
            offset += 1;
            }),
        }

        let mut deletes_from_union = Cow::Borrowed(&self.deletes_from_union);
    // this should have the same behavior, but worse performance
        assert_eq!(snapshot[3].name, "a");
    }
    /// Create a new Engine with a single edit that inserts `initial_contents`
    #[bench]
/// }
    }
                    CategoriesT::StaticArray(ref other_arr) => self_arr.arr_eq(other_arr),

    // Send the requested lines even if they're outside the current scroll region.
use std::ops::Add;
    #[test]
                    let (mut a, rest) = end.split_first_mut().unwrap();
        // interior of first line should be first line
        }
    }
        delta.serialize_field("base_len", &self.base_len)?;

        let (start, end) = iv.start_end();
    pub fn edit_str(&mut self, start: usize, end: usize, new: &str) {
        // +2 for exe & thread name.
    /// height of visible portion
        self.session = session;
        #[serde(rename_all = "snake_case")]

        let trace = Trace::enabled(Config::default());
    /// Width in px units, requiring measurement by the front-end.
            'M' => SampleEventType::Metadata,
                if rev.max_undo_so_far < lowest_group {
                                         self, text, modify);
        let a = Rope::from("A\u{030a}\u{110b}\u{1161}\u{1f1fa}\u{1f1f8}");
/// # Examples

        self.lc_shadow.partial_invalidate(first_line, last_line, line_cache_shadow::STYLES_VALID);
            a.push('a');
        let rearranged = rearrange(&revs, &base, 7);
                }
                    if undone_groups.contains(undo_group) {
            highlight_find: false,

            }
        if let Contents::Edit {priority, inserts, .. } = r.edit {
    {
                match self.inner.next() {
            pid: sys_pid::current_pid(),
            fn visit_str<E>(self, v: &str) -> Result<CategoriesT, E>
    {

    fn deserialize<D>(deserializer: D)
        use self::MergeTestOp::*;
        }
    }
}
            let a_to_merge = &self.revs[base_index..];
            let style_id = self.get_or_def_style_id(client, styles, &style);
                }

                        Cow::Owned(un_deleted.transform_shrink(inserts))
        let start_pos = start_of_line.pos();
    #[test]
    pub fn new(initial_contents: Rope) -> Engine {
}
mod tests {
        let inserts = parse_subset_list("
            rev_id: self.next_rev_id(),
    }

            shuffle(&text_with_inserts, &tombstones, &expanded_deletes_from_union, &new_deletes_from_union);
            Assert(1, "bdefg".to_owned()),
    /// Find what the `deletes_from_union` field in Engine would have been at the time
extern crate serde_json;
        let mut engine = Engine::new(Rope::from(TEST_STR));
        assert_eq!("herald", String::from(a));
    fn empty_subset_before_first_rev(&self) -> Subset {
        let all_caret = self.selection.iter().all(|region| region.is_caret());
    let mut splitpoint = min(MAX_LEAF, s.len() - MIN_LEAF);
                Contents::Edit {inserts, ..} => inserts,
                // TODO: small nit, merged region should be backward if end < start.
            trace.instant("test_samples_pulse", &["test"]);
    pub(crate) fn wrap_width(&mut self, text: &Rope, width_cache: &mut WidthCache,
}
        self.send_update_for_plan(text, client, styles,
                println!("running {:?} at index {}", op, i);
        assert_eq!(Some(0), a.prev_codepoint_offset(1));
        RopeInfo {
        let self_iv = Interval::new_closed_closed(0, self.len());
            find.set_hls_dirty(false)
            Merge(1,0),
    replace_changed: bool,
                        } else {
    }
                        // no need to un-delete undone inserts since we'll just shrink them out

}
        samples
use xi_rope::breaks::{Breaks, BreaksInfo, BreaksMetric, BreaksBaseMetric};
        deletes_bitxor: Subset,
#[inline]
        }
    }
        // the generated JSON.
    /// of a certain `rev_index`. In other words, the deletes from the union string at that time.
}
        self.set_dirty(text);
        assert_eq!(1, r.line_of_offset(a.len()));
    }
        assert_eq!("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", String::from(engine.get_head()));
/// Move sections from text to tombstones and out of tombstones based on a new and old set of deletions
    pub fn lines(&self, start: usize, end: usize) -> Lines {
            Contents::Undo { .. } => panic!("can't merge undo yet"),
    }
    /// Selects an entire line and supports multi selection.
            drag_state: None,
    trace: Option<&'a Trace>,

        }
///
        /// The set of groups toggled between undone and done.
    pub fn select_line(&mut self, text: &Rope, offset: usize, line: usize, multi_select: bool) {
        assert_eq!(r.len(), r.offset_of_line(2));
//
            Assert(0, "bdc".to_owned()),

    ///

            r = op.inserts.apply(&r);
                    }
        // todo: for now only a single search query is supported however in the future
        assert_eq!(utf8_offset, 13);
#[cfg(test)]

    /// Merge may panic or return incorrect results if session IDs collide, which is why they can be
        /// The subset of the characters of the union string from after this

        let mut hls = Vec::new();
            Some(offset - len)
        // a, b, y, z, c, x
        /// revision that were added by this revision.
        assert_eq!(vec!["a\rb\rc"], a.lines_all().collect::<Vec<_>>());
    pub fn find_status(&mut self, matches_only: bool) -> Vec<FindStatus> {
                self.do_find(text, chars, case_sensitive, regex, whole_words),
pub struct LinesRaw<'a> {
        a.edit_str(1, 9, "era");
// you may not use this file except in compliance with the License.

        assert_eq!(rope.offset_of_line(4), rope.len());
        self.scroll_to = Some(end);
            SampleEventType::Metadata => 'M'
    {
                pos = leaf_offset + pl.len();
        peers: Vec<Engine>,
        where D: serde::Deserializer<'de>
    let mut next_expand_by = Vec::with_capacity(expand_by.len());
#[derive(Serialize, Deserialize, Debug)]
                              categories: C,
                Rope::from(s1.clone()),
fn find_leaf_split_for_bulk(s: &str) -> usize {
        // We could probably reduce the cloning of selections by being clever.
        engine.undo([2].iter().cloned().collect());
// easily delta-compressed later.
        let ix = self.find_rev_token(base_rev).expect("base revision not found");

    }
#![cfg_attr(feature = "collections_range", feature(collections_range))]
    fn gc_2() {
            Contents::Undo { .. } => panic!("can't merge undo yet"),
                let (pl, poffset) = self.prev_leaf()?;
impl fmt::Debug for Rope {
    pub fn offset_of_line(&self, text: &Rope, line: usize) -> usize {
        let mut cursor = Cursor::new(self, offset);
    }
/// fn something_expensive() {
        }).collect();
        let l_str = text.slice_to_string(start_pos, pos);
        };
                }
    /// A delta that, when applied to `base_rev`, results in the current head. Panics
    fn compute_info(s: &String) -> Self {
                // should be removed in favor of do_gesture
    session: SessionId,
            MetadataType::ThreadSortIndex {sort_index} => (None, Some(sort_index)),
        d_builder.replace(Interval::new_closed_open(54, 54), Rope::from("999"));

//
// Unless required by applicable law or agreed to in writing, software
        engine.edit_rev(1, edits+1, head2, d2);
        let mut selection = self.selection.clone();
    /// End of the region selected when drag was started.
                self.do_drag(text, line, column, Affinity::default()),

        // current recording mechanism should see:
                    if invert_undos {
        let first_rev = engine.get_head_rev_id().token();
        assert_eq!(Some(3), a.next_codepoint_offset(1));
        let mut union_ins_delta = ins_delta.transform_expand(&deletes_at_rev, true);

                    // fast-forward this revision over all common ones after it
            if self[i] != other[i] {


        for find in self.find.iter() {
        let mut engine = Engine::new(Rope::from(TEST_STR));
    fn find_first_undo_candidate_index(&self, toggled_groups: &BTreeSet<usize>) -> usize {
            _ => return
            rev_id_counter: 1,
        d_builder.delete(Interval::new_closed_open(10, 36));
}
                                toggled_groups: &toggled_groups - gc_groups,
                    selection.add_region(occ);
                            plan: &RenderPlan, pristine: bool)
    fn edit_rev_simple() {
        self.deletes_from_union = deletes_from_union;
fn deserialize_event_type<'de, D>(d: D) -> Result<SampleEventType, D::Error>
        let s1 = "\u{1f1fa}\u{1f1f8}".repeat(100);
#[must_use]
    fn partial_cmp(&self, other: &Sample) -> Option<cmp::Ordering> {

    rev_id: RevId,
/// Given the inital byte of a UTF-8 codepoint, returns the number of
            ModifySelection(movement) => self.do_move(text, movement, true),
            panic!("line number {} beyond last line {}", line, max_line);
    }

        let old_tombstones = shuffle_tombstones(&self.text, &self.tombstones, &self.deletes_from_union, &prev_from_union);
    fn bench_trace_block_payload(b: &mut Bencher) {
    }
        (self.session1, self.session2)
    {
impl<'a> Iterator for LinesRaw<'a> {
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("----j") },
    /// unsaved changes.
        use serde::de::Visitor;
        let mut r = Rope::from("27");
        let mut to_undo = BTreeSet::new();
    }
    }
            l[offset..].chars().next()
                splitpoint -= 1;
            if let Some(ref thread_name) = sample.thread_name {

        }
    #[test]



/// # use xi_rope::Rope;
        Rope::from_str(s.as_ref()).unwrap()

        // shouldn't do anything since it was double-deleted and one was GC'd
            "n": n,
    ///
pub fn trace_block_payload<'a, S, C, P>(name: S, categories: C, payload: P)

    sample_limit_count: usize
    pub fn select_range(&mut self, text: &Rope, offset: usize) {
/// Create an instantaneous sample without any payload.  This is the lowest
            Edit { ei: 0, p: 3, u: 1, d: parse_delta("---d") },
                            }
            trace.closure("something", &[], || ());
            return Vec::with_capacity(0);
        --#--
            last_line -= 1;
/// required). Mutations are generally copy-on-write, though in-place edits are
//TODO: document metrics, based on https://github.com/google/xi-editor/issues/456
        for &region in self.sel_regions() {
    end: usize,
use tabs::{ViewId, BufferId};
    {
        let r = r + Rope::from(String::from(&a[MAX_LEAF..]) + &b[..MIN_LEAF]);
    pub fn enabled(config: Config) -> Self {
                c.provide_context(&pl, self.pos() - poffset);
}
    }
        Self {
        } else { // no toggled groups, return past end
    fn prev_grapheme_offset() {
    {
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    pub fn get_rev(&self, rev: RevToken) -> Option<Rope> {
        // Rust is unlikely to break the property that this hash is strongly collision-resistant
    ///
        let edit_iv = Interval::new_closed_open(start, end);
    // Of course, all these are identical for ASCII. For now we use UTF-8 code units
        }
    #[test]
    }
        // spam cmd+z until the available undo history is exhausted
        self.find.iter().map(|find| {
    }
    out
            'O' => SampleEventType::ObjectSnapshot,
    #[test]
}
categories_from_constant_array!(4);

    ///
        };
    ///
///
        // position after 'f' in four
        if self.find.is_empty() {
        // likely be improved.
///
        impl From<&'static[&'static str; $num_args]> for CategoriesT {
                    undo_group: i+1,
    /// Constructs a Duration sample.  For use via xi_trace::closure.
                (!region.is_upstream() && c == start_pos) ||

    fn deletes_from_union_for_index(&self, rev_index: usize) -> Cow<Subset> {
    /// Offset of the point where the drag started.
use linewrap;


    #[bench]
/// xi_trace::trace_payload("my event", &["rpc", "response"], json!({"key": "value"}));
        }
        let mut engine = Engine::empty();
        self.invalidate_selection(text);
    }
        Self {
impl NodeInfo for RopeInfo {
            // I think it's a precondition that this will never be called
            Merge(1,2),
    fn rearrange_1() {
            Merge(1,2),
    /// The revision history of the document
                                           client, iv, new_len, px),
pub fn samples_len() -> usize {
            let next_chunk = self.inner.next();
    /// selection edges).
        let (metadata_name, sort_index) = meta.consume();
        let a = Rope::from("a\nb\nc");
    DurationBegin,
            "ops": ops,
                false => x.start
                return offset;
///
    }
        let s = String::deserialize(deserializer)?;

    // TODO: have `base_rev` be an index so that it can be used maximally efficiently with the
    }
    /// Selects the next find match.
        }
    identity_op,
        let d = engine.delta_rev_head(after_first_edit);
    /// Time complexity: O(log n)
            Assert(1, "ab".to_owned()),
}
    }
categories_from_constant_array!(2);
    current: Option<&'a str>,
        false

            self.lc_shadow.edit(start, end, new_end - start);
    }
        }
        assert_eq!(1, expand_by[0].0.priority);
        assert_eq!(snapshot[5].name, "b");
    }
        let prev_from_union = self.deletes_from_cur_union_for_index(ix);
                                  style_spans, &plan, pristine);
        for &(trans_priority, ref trans_inserts) in &expand_by {
        assert_eq!("1234567", String::from(r));
    }
///
// limitations under the License.

    /// If `line` is equal to one plus the current number of lines,
                let previous_name = thread_names.insert(sample.tid, thread_name.clone());
    /// Return the line number corresponding to the byte index `offset`.
pub fn count_newlines(s: &str) -> usize {
    #[test]
/// equivalent performance to a regular trace.  A string that needs to be copied
                }
    fn next_grapheme_offset_with_ris_of_leaf_boundaries() {
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("--j") },
        assert_eq!(snapshot[1].args.as_ref().unwrap().metadata_name.as_ref().is_some(), true);
pub type RopeDelta = Delta<RopeInfo>;
            self.current = None;
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("-p-") },

    }
    }
            AssertMaxUndoSoFar(0,3),

}
                offset = prev;
        let height = self.line_of_offset(text, text.len()) + 1;
        MergeTestState::new(3).run_script(&script[..]);
        d_builder.replace(Interval::new_closed_open(42, 45), Rope::from("GI"));
        assert_eq!("a", String::from(&tombstones_2));

                    let mut selection = self.selection.clone();
impl Metric<RopeInfo> for Utf16CodeUnitsMetric {
}

        //println!("push_maybe_split [{}] [{}] {:?}", self, other, iv);
    fn deletes_from_cur_union_for_index(&self, rev_index: usize) -> Cow<Subset> {
        for &(sel_start, sel_end) in sel {
/// assert_eq!("llo wo", String::from(c));
        if let Some(lines) = lines {

        let inserts_3 = parse_subset_list("
                return false;
        debug_subsets(&rebased_inserts);
/// # Arguments
        /// Used to store a reversible difference between the old
        }
                } else if click_count == Some(3) {
    fn add(self, rhs: Rope) -> Rope {
        b.iter(|| black_box(trace.instant("nothing", &["benchmark"])));
        self.lc_shadow.partial_invalidate(first_line, last_line, invalid);
mod sys_tid;
            Assert(0, "adc".to_owned()),

            },

                         first_line: usize, last_line: usize, pristine: bool) {
    }

        }

pub struct Engine {
                let mut el = serializer.serialize_tuple_variant("DeltaElement",
    pub fn select_next_occurrence(&mut self, text: &Rope, reverse: bool, wrapped: bool,
    min: usize,
    }
        assert_eq!(None, b.prev_codepoint_offset(0));
        }
        if self.replace_changed {
            // I think it's a precondition that this will never be called
                tid));
        self.start_drag(offset, region.start, region.end);
use std::cmp::{min,max};
pub fn enable_tracing() {
    /// Invalidate the current selection. Note that we could be even more

    All,
                priority: new_priority,
    fn offset_of_line_small() {
        assert_eq!(snapshot[3].name, "b");
        assert_eq!(trace.get_samples_count(), 5);
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    }
        if let Some(exe_name) = exe_name() {
        assert_eq!(trace.get_samples_count(), 1);


                                                          new_len, col),
        }
#[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
                         hls: &[(usize, usize)],
    #[test]
    #[cfg(feature = "benchmarks")]
        },
            Replace { chars, preserve_case } =>
        /// Groups related edits together so that they are undone and re-done
                    "something", &["benchmark"],
    }

    }
            GestureType::PointSelect => {

    fn utf16_code_units_metric() {
///
            let sel = {
        }
        assert_eq!(Some(0), b.prev_codepoint_offset(2));
                if s.ends_with('\n') {
// you may not use this file except in compliance with the License.

        } else {
                    if let Some(new_gc_dels) = new_gc_dels {
        engine.edit_rev(1, 1, first_rev, build_delta_1());
    /// want to set the process name (if provided then sets the thread name).

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                            selection.delete_range(last_selection.min(), last_selection.max(), false);

    TRACE.get_samples_count()
    }
    /// Invalidates the styles of the given range (start and end are offsets within
    /// Get the Subset to delete from the current union string in order to obtain a revision's content
    }
            Edit { ref inserts, .. } => inserts.count(CountMatcher::Zero),
                    None => return if result.is_empty() { None } else { Some(result) }
        let is_base = base_revs.contains(&rev.rev_id);
        let delta = Delta::synthesize(&self.tombstones,
        engine.edit_rev(0, 2, first_rev, build_delta_2());
}
        sel.collapse();
            engine.undo(undos.clone());
                    if s.ends_with('\r') {
                text.offset_of_line(line)
            "styles": styles,
    #[serde(rename = "dur")]
//! There is also a full CRDT merge operation implemented under
            Merge(3, 0), Merge(4, 1), Merge(5, 2),
        if let Some(ref mut trace) = self.trace {
        let text_a = Rope::from("zcbd");
        for seg in self.lc_shadow.iter_with_plan(plan) {
}
                // TODO could probably be more efficient by avoiding shuffling from head every time
    }
use std::string::ParseError;
/// A static string has the lowest overhead as no copies are necessary, roughly
                            deletes_bitxor.transform_shrink(&gc_dels)
        count_utf16_code_units(&s[..in_base_units])
/// the session ID component of a `RevId`
                    };
impl Serialize for Delta<RopeInfo> {
        next_boundary.unwrap_or(None)

    /// The selection state for this view. Invariant: non-empty.
            Edit { ei: 0, p: 1, u: 1, d: parse_delta("ac") },

        }
            timestamp_us: ns_to_us(start_ns),
    Bytes(usize),
    #[cfg(feature = "benchmarks")]
        let rebased_deletes_from_union = cur_deletes_from_union.transform_expand(&new_inserts);
        let subset_str = format!("{:#?}", expand_by[1].1);
    }
                            edit: Edit {
                client.replace_status(self.view_id, &json!(replace))
    {

            false
        assert_eq!(snapshot.len(), 5);
/// than a string that needs to be copied (or ~4.5x slower than a static string)
        ");
                match *other {
#[derive(Serialize, Deserialize, Clone, Debug)]
        let mut new_deletes = deletes.transform_expand(&deletes_at_rev);
        assert_eq!(Some(8), a.next_grapheme_offset(0));
    //
            let (start, end) = if offset < drag_state.offset {
                let delta = Delta::synthesize(&tombstones_here, &older_all_inserts, &cur_all_inserts);
    /// if there is not at least one edit.
        let b_delta_ops = compute_deltas(&b_revs, &text_b, &tombstones_b, &deletes_from_union_b);
    #[test]
/// samples are ordered chronologically for several reasons:
        self.drag_state = None;
                }
        }
#[derive(Debug, Default, PartialEq, Serialize, Deserialize, Clone)]
        assert_eq!(1, expand_by[0].0.priority);
                SelectionModifier::Set => self.set_selection(text, occ),
        trace.closure_payload("x", &["test"], || {
    use interval::Interval;
            utf16_size: count_utf16_code_units(s),
            thread_name: None,
    #[test]
    }
        let a_rope = Rope::from(&a);
            Rope::from(s1.clone()),
    pub(crate) fn line_col_to_offset(&self, text: &Rope, line: usize, col: usize) -> usize {
        let base: BTreeSet<RevId> = [3,5].iter().cloned().map(basic_rev).collect();

            Merge(0,1),
        let rope = Rope::from("hi\ni'm\nfour\nlines");
/// acquires a dependency on the `serde_json` crate.
        let mut cursor = Cursor::new(self, offset);
            Assert(1, "apbj".to_owned()),
        impl From<RopeDelta_> for Delta<RopeInfo> {
                            deletes_bitxor
    #[bench]
        } else {
                cursors.push(c - start_pos);
        let mut engine = Engine::new(Rope::from(TEST_STR));
    // TODO: don't construct transform if subsets are empty
    #[serde(deserialize_with = "deserialize_event_type")]
/// Stores the tracing data.
            sample.event_type = SampleEventType::DurationEnd;
        self.find.first_mut().unwrap().do_find(text, chars, case_sensitive, is_regex, whole_words);
                let tombstones_here = shuffle_tombstones(text, tombstones, deletes_from_union, &older_all_inserts);
                }
            Merge(1,0),
    fn merge_max_undo_so_far() {
        let mut cursor = Cursor::new(self, offset);
        } else {
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("-p-") },


    pub fn closure<S, C, F, R>(&self, name: S, categories: C, closure: F) -> R
                                     payload: Option<TracePayloadT>,
        Ok(Delta::from(d))
            if !s.is_empty() {
            None
        fn new(count: usize) -> MergeTestState {
                        for line_num in start_line..end_line {
                        Cow::Owned(deletes_from_union.bitxor(deletes_bitxor))
        assert_eq!(Some(1), a.prev_codepoint_offset(3));
            match seg.tactic {


        cursor.prev::<BaseMetric>()
                    RopeDeltaElement_::Insert(s) =>
}


        expand_by = next_expand_by;
/// # Performance
        let mut samples = vec![
        let utf8_offset = 9;

categories_from_constant_array!(1);
    /// this returns the offset of the end of the rope. Arguments higher
        assert_eq!(snapshot[6].name, "z");
            // trans-expand other by expanded so they have the same context
        let rope = Rope::from("hi\ni'm\nfour\nlines");
        let mut a = String::new();
    a_ids.intersection(&b_ids).cloned().collect()
        let max_undo_so_far = self.revs.last().unwrap().max_undo_so_far;
        let mut pos = self.pos();
        let first_rev = engine.get_head_rev_id().token();

            els: Vec<RopeDeltaElement_>,
            b.push_str(&c);
                engine.gc(&to_gc)
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        #------
    pub fn disabled() -> Self {
/// into an `InsertDelta`-based representation that does by working backward from the text and tombstones.
            let first_rev = engine.get_head_rev_id().token();
        }
    fn prev(s: &String, offset: usize) -> Option<usize> {
use edit_types::ViewEvent;

            Merge(0,1),
            Merge(1,0),
                Undo { toggled_groups, deletes_bitxor } => {
    }
            Edit { ei: 0, p: 1, u: 1, d: parse_delta("!-d-") },

use std::str;
}
// You may obtain a copy of the License at
        let mut engine = Engine::new(Rope::from(TEST_STR));
    use test::Bencher;
        let mut b = line_cache_shadow::Builder::new();
            let b_new = rearrange(b_to_merge, &common, other.deletes_from_union.len());
                inserts: new_inserts,
                MergeTestOp::Merge(ai, bi) => {
}
    {
        Cow::from(b)
        }
        } else {
                    let (start, end) = {
        let a = Rope::from("a\nb\nc");

impl Config {
    #[bench]
        let tombstones_a = Rope::from("a");
    fn select_region(&mut self, text: &Rope, offset: usize, region: SelRegion, multi_select: bool) {
    #[test]

            None => {
    /// The session ID used to create new `RevId`s for edits made on this device
            .find(|&(_, ref rev)| rev.rev_id == rev_id)
        loop {
        a.push('\n');
    /// Selects an entire word and supports multi selection.
        for r in &self.revs[ix + 1..] {
            }
#[inline]
    /// of individual region movements become carets.

        }
        let a = Rope::from("\n");
                    selection.add_region(SelRegion::new(sel_start, end_of_line));

        let mut engine = Engine::new(Rope::from(""));
        let snapshot = trace.samples_cloned_unsorted();
        assert_eq!(rope.line_of_offset(3), 1);
            1 if self.selection[0].is_caret() => {
    // * Code units in some encoding
        // This might be an over-specified test as it will
    fn can_fragment() -> bool { true }
    serde::Deserialize::deserialize(d).map(|ph : char| SampleEventType::from_chrome_id(ph))

    /// not account for any overhead of storing the data itself (i.e. pointer to
        last_line += if last_col > 0 { 1 } else { 0 };
        let last = max(last, 0) as usize;
        }
    text: Rope,
        (self.pid, self.timestamp_us).hash(state);
    }
        engine.edit_rev(1, edits+1, head, d1);
        engine.edit_rev(1, 2, initial_rev, d1.clone());
pub fn trace_closure_payload<S, C, P, F, R>(name: S, categories: C,
            let d = Delta::simple_edit(Interval::new_closed_open(0,0), Rope::from("b"), i);


    pub fn after_edit(&mut self, text: &Rope, last_text: &Rope,
            if let Some(prev) = text.prev_grapheme_offset(next_line_offset) {
    }
        let a2 = Rope::from("a");
    fn undo_5() {
                // sanitize input
            // shouldn't be called with this, but be defensive
    #[test]

    #[test]
        let end = time::precise_time_ns();
        let mut revs_3 = basic_insert_ops(inserts_3, 2);
            ix = sel_end as isize;
    fn next(&mut self) -> Option<Cow<'a, str>> {
    /// via process_name/thread_name respectively.
    //TODO: implement lines_raw using ranges and delete this
    }
        assert_eq!(Some(10), a.next_codepoint_offset(6));

        assert_eq!(4, a.offset_of_line(2));
        self.set_selection_raw(text, selection);
            Edit { ei: 1, p: 3, u: 3, d: parse_delta("-!") },
        let d3 = Delta::simple_edit(Interval::new_closed_open(0,0), Rope::from("b"), TEST_STR.len()+1);
                         styles: &StyleMap, style_spans: &Spans<Style>,
        let new_sel = selection_movement(movement, &self.selection,
#[inline]
/// The visual width of the buffer for the purpose of word wrapping.
            let c = region.end;
    fn next_grapheme_offset() {
        let script = vec![
        &self.selection
        let res = find_common(&a, &b);
        }
        ];
        // interior of last line should be last line
    // TODO make this faster somehow?
        ----
        let line = self.line_of_offset(text, offset);
#[cfg(feature = "json_payload")]
    #[test]
        assert_eq!(2, r.line_of_offset(r.len()));
            return 0;
    /// Returns a u64 that will be equal for equivalent revision IDs and
        let first_sample_timestamp = all_samples.front()
        self.push_str(&other[start..end]);
use xi_trace::trace_block;
    where S: Into<StrCow>, C: Into<CategoriesT>
use std::collections::hash_map::DefaultHasher;
        let deletes_from_union_a = parse_subset("-#---");
/// for single user cases, used by serde and ::empty
        if !before {
// Copyright 2016 The xi-editor Authors.
    /// Return the byte offset corresponding to the line number `line`.
            self.record(Sample::new_instant(name, categories, Some(payload.into())));
            // recurse at most once; this doesn't feel great but I wrote myself into a
        let d1 = Delta::simple_edit(Interval::new_closed_open(0,10), Rope::from(""), TEST_STR.len());
    for rev in revs.iter().rev() {

/// Disable tracing.  This clears all trace data (& frees the memory).
        let sample_name = to_cow_str(meta.sample_name());
#[inline]
    }
        selection.add_region(region);
fn find_leaf_split(s: &str, minsplit: usize) -> usize {
        } else {
        if !self.is_point_in_selection(offset) {
            self.current = next_chunk;
    fn hash<H: Hasher>(&self, state: &mut H) {
    fn bench_trace_block_disabled(b: &mut Bencher) {
#[cfg(test)]
        assert_eq!(snapshot[5].name, "z");

    }

        self.first_line = first;
        };
//! this central engine. It provides the ability to apply edits that depend on
            // TODO: finer grain invalidation for the line wrapping, needs info
    #[should_panic]
struct Revision {
                }
                    if let Some(last_selection) = self.selection.last() {

        } else {
    }
}
        match *self {
pub struct Lines<'a> {
    }
                                                    style_spans, client,
    fn test_samples_block() {
/// same priority into one transform to decrease the number of transforms that
        MergeTestState::new(3).run_script(&script[..]);
    replace: Option<Replace>,
            Merge(0,2), Merge(1, 2),
            for find in self.find.iter() {
                self.set_dirty(text);
    } else {
}
            engine.edit_rev(0, 0, first_rev, delta);
/// However, anything that can be converted into a Cow string can be passed as
enum WrapWidth {
        assert_eq!(Some(6), a.prev_codepoint_offset(10));
        ");
    fn basic_insert_ops(inserts: Vec<Subset>, priority: usize) -> Vec<Revision> {
            DeltaElement::Insert(ref node) =>
    #[test]
    fn len(&self) -> usize {
               a.lines_all().collect::<Vec<_>>());
        ");
                    self.set_selection(text, selection);
// The advantage of using a session ID over random numbers is that it can be
        self.iter_chunks(0, self.len())
    /// the pid or the tid depends on the name of the event
    pub fn lines_raw(&self, start: usize, end: usize) -> LinesRaw {
        if self.highlight_find {
        self.rev_id_counter += 1;
    #[test]
            let not_in_tombstones = self.deletes_from_union.complement();
    enum MergeTestOp {
            max_undo_so_far,
        });
        let d = engine.delta_rev_head(first_rev);

/// Returns the operations in `revs` that don't have their `rev_id` in
        self.deletes_from_union_before_index(rev_index + 1, true)
        let utf16_units = rope_with_emoji.measure::<Utf16CodeUnitsMetric>();
    fn ids_to_fake_revs(ids: &[usize]) -> Vec<Revision> {
/// However, anything that can be converted into a Cow string can be passed as
/// fn something_else_expensive() {
/// something_expensive();

    }
            }
                    if sel_end_ix > sel_start_ix {
    DynamicArray(Vec<String>),
    }
            }
/// (also known as [persistent](https://en.wikipedia.org/wiki/Persistent_data_structure))

        let head = engine.get_head_rev_id().token();
impl<T: AsRef<str>> From<T> for Rope {
        Subset::new(len)

        where S: Into<StrCow>, C:Into<CategoriesT>, P: Into<TracePayloadT>
/// let b = a.slice(1, 9);
        b.push_str(new);
        // invert the changes to deletes_from_union starting in the present and working backwards
    /// The current contents of the document as would be displayed on screen
            timestamp_us: ns_to_us(time::precise_time_ns()),
                        let mut rendered_lines = Vec::new();
        -> SampleGuard
        samples[1].pid = 2;
    ///
    assert!(a[0].rev_id == b[0].rev_id);
    CompleteDuration,
            // Do it the other way on the copy
use multiset::{Subset, CountMatcher};
//     http://www.apache.org/licenses/LICENSE-2.0
///
            // I think it's a precondition that this will never be called
    // Note: this function would need some work to handle retaining arbitrary revisions,
        assert_eq!(Some(17), a.next_grapheme_offset(9));
    (new_text, shuffle_tombstones(text,tombstones,old_deletes_from_union,new_deletes_from_union))
    wrap_col: WrapWidth,
                        &mut rest[bi - ai - 1]
    where S: serde::Serializer {

        assert_eq!(TEST_STR, String::from(engine.get_head()));
/// }

            'f' => SampleEventType::FlowEnd,
mod tests {
/// 1. Samples that span sections of code may be inserted on end instead of
        LinesRaw {
    ///
        {
    fn eq(&self, other: &CategoriesT) -> bool {
}
        self.set(new_pos);
        self.revs.reverse();
            sel.add_region(new_region);
    /// Selects all find matches.
use serde::de::{Deserialize, Deserializer};
    fn new<S, C>(trace: &'a Trace, name: S, categories: C, payload: Option<TracePayloadT>)
        // TODO: may make this method take the iv directly
        ChunkIter {


/// Create an instantaneous sample with a payload.  The type the payload

        let a = Rope::from("a\nb\n");
    // TODO: this should be a Cow
        // expand by each in expand_by
        }

}
    fn delta_rev_head() {
            GestureType::LineSelect =>
    inner: ChunkIter<'a>,
        assert_eq!("#-####-", &subset_str);
    /// the frontend.
            SampleEventType::FlowStart => 's',
        assert_eq!(Some(6), a.next_codepoint_offset(3));
    #[test]

            GestureType::WordSelect =>
        assert_eq!(snapshot[1].name, "thread_name");
            pos -= 1;
            //let prev_pos = self.cursor.pos();
///
            end -= 1;
        match self.inner.next() {
            Assert(1, "pb".to_owned()),
    pub fn prev_grapheme_offset(&self, offset: usize) -> Option<usize> {
            };
    fn from_base_units(_: &String, in_base_units: usize) -> usize {
/// None if an irrecoverable error occured.
/// * `categories` - A static array of static strings that tags the samples in

        self.set_selection(text, sel);
    fn rev_content_for_index(&self, rev_index: usize) -> Rope {
/// stored as tracing could still be enabled but set with a limit of 0.
        }
                        .map(DeltaElement::from).collect(),
    pub fn select_word(&mut self, text: &Rope, offset: usize, multi_select: bool) {
                                deletes_bitxor: new_deletes_bitxor,
            }


        trace.disable();
    }
        let mut plan = RenderPlan::create(height, self.first_line, self.height);
    ///
        assert_eq!(1, res);
    // this is the cost contributed by the timestamp to
            self.breaks = Some(linewrap::linewrap(text, wrap_col));
        }
}
                sel
                priority, undo_group, deletes,
    /// Get text of a given revision, if it can be found.
    }
    }
    // 96 bits has a 10^(-12) chance of collision with 400 million sessions and 10^(-6) with 100 billion.
    pub fn do_find(&mut self, text: &Rope, chars: String, case_sensitive: bool, is_regex: bool,
    },
    /// `deletes_from_union` by splicing a segment of `tombstones` into `text`
    }
    fn can_fragment() -> bool {
/// # Examples
        assert!(a != b);
                                priority,
    const TEST_STR: &'static str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

        //println!("{:?}", r.iter_chunks().collect::<Vec<_>>());

        self.hash(&mut hasher);
/// conforms to is currently determined by the feature this library is compiled

    }
///
    }
        MergeTestState::new(4).run_script(&script[..]);
                }
    pub(crate) fn offset_to_line_col(&self, text: &Rope, offset: usize) -> (usize, usize) {
        let mut deletes_from_union = self.deletes_from_union_before_index(first_candidate, false).into_owned();
    }
    /// You could construct the "union string" from `text`, `tombstones` and
        let deletes_at_rev = self.deletes_from_union_for_index(ix);
                let last: &mut (FullPriority, Subset) = out.last_mut().unwrap();
            },
        samples[1].timestamp_us = 5;
/// some way.
    }
    pub fn merge(&mut self, other: &Engine) {
        let correct = parse_subset_list("
    // Delta that deletes the right bits from the text
// You may obtain a copy of the License at
impl StringArrayEq<[&'static str]> for Vec<String> {
    /// The state for replacing matches for this view.
///     something_else_expensive();
        let mut cursors = Vec::new();
            let max_undo = self.max_undo_group_id();
                        let new_undone = undone_groups.symmetric_difference(toggled_groups).cloned().collect();
    #[derive(Clone, Debug)]
        let offset = self.line_col_to_offset(text, line, col);

/// ```
            SampleEventType::ObjectDestroyed => 'D',
pub type TracePayloadT = serde_json::Value;
    max_undo_so_far: usize,
    }
                WrapWidth::None => (),
        assert_eq!(0, r.line_of_offset(a.len() - 1));
            } else {
    }
pub struct RevId {
            inserts: Subset::new(0),
    }
        let old_deletes_from_union = self.deletes_from_cur_union_for_index(rev_index);
}
            deletes: Subset::new(0),
    Instant,

    #[test]
/// Valid within a session. If there's a collision the most recent matching
#[macro_use]
        None => {
            if inserts.is_empty() {
        -##-
    fn sample_name(&self) -> &'static str {
            }
    fn line_of_offset_panic() {
            self.mk_new_rev(priority, undo_group, base_rev, delta);
        engine

                        let new_deletes_bitxor = if gc_dels.is_empty() {
                            warn!("Failed to get string representation: {:?}", e);
                            if !inserts.is_empty() {
        assert_eq!(rope.line_of_offset(0), 0);
        let empty = Rope::from("");
    #[serde(rename = "ts")]
    }
    fn do_selection_for_find(&mut self, text: &Rope, case_sensitive: bool) {
            Scroll(range) => self.set_scroll(range.first, range.last),
            if let Some(replace) = self.get_replace() {

                MetadataType::ProcessName {name: exe_name},
            },
                breaks.convert_metrics::<BreaksMetric, BreaksBaseMetric>(line)
        delta.end()
                                gc_dels = gc_dels.union(deletes);
                }
            'N' => SampleEventType::ObjectCreated,
                    text.slice_to_string(region.min(), region.max())
    match memrchr(b'\n', &s.as_bytes()[minsplit - 1..splitpoint]) {
        // insert a character at the beginning
        old_deletes_from_union: &Subset, new_deletes_from_union: &Subset) -> Rope {
            Assert(1, "afc".to_owned()),
        offset
    }
                        self.revs.push(Revision {
    pub(crate) fn set_has_pending_render(&mut self, pending: bool) {
    // How should we count "column"? Valid choices include:
        -> Result<CategoriesT, D::Error>
    pub fn line_of_offset(&self, text: &Rope, offset: usize) -> usize {
        ---#-
            cursor: Cursor::new(self, start),
            FindAll => self.do_find_all(text),
                    }
/// Usage of static strings is encouraged for best performance to avoid copies.
}
        }



                selection.add_region(occurrence);
    }
        self.drag_state = Some(DragState {
    /// borrowed from the rope.
    fn bench_two_timestamps(b: &mut Bencher) {
        let mut b = TreeBuilder::new();
use movement::{Movement, region_movement, selection_movement};
    }
        let undone = self.undone_groups.contains(&undo_group);
        }
/// Base unit is utf8 code unit.
                SelectionModifier::Add => {
        engine.gc(&gc);
    {
            height: 10,
    }
        let style = style_map.merge_with_default(style);
    /// Replacement string.
    #[cfg(feature = "benchmarks")]
                    els: delta.els.drain(..)


            }
        assert_eq!(trace.is_enabled(), true);
            fn from(elem: RopeDeltaElement_) -> DeltaElement<RopeInfo> {

/// let a = Rope::from("hello ");
        let trace = Trace::enabled(Config::with_limit_count(10));
        // and it only needs to be consistent over one execution.
        self.lc_shadow = b.build();
                    }
#[cfg(all(not(feature = "dict_payload"), not(feature = "json_payload")))]
    }
    }
        self.scroll_to_cursor(text);
        let (mut l, mut offset) = self.get_leaf()?;
use std::sync::Mutex;
        }
                        Err(e) => {
                        deletes: transformed_deletes,
        self.find_changed = FindStatusChange::Matches;
    }
            // with offset == s.len(), but be defensive.
    }
/// Measured unit is utf8 code unit.
        }
fn count_utf16_code_units(s: &str) -> usize {
    /// Lines are ended with either Unix (`\n`) or MS-DOS (`\r\n`) style line endings.
            self.name == other.name &&
        let new_pos = self.pos() + (end - offset);
                Edit { ref inserts, ref deletes, ref undo_group, .. } => {
    }
    /// Sets the selection to a new value, invalidating the line cache as needed.
            "op": op,
        self.revs.iter().enumerate().rev()
    fn from(s: T) -> Rope {


            l[offset..].chars().next()
        }
        }
        ];

    // Render a single line, and advance cursors to next line.

/// Creates a duration sample that measures how long the closure took to execute.
        for u in s.chars() {

        }
/// an argument.
    TRACE.samples_cloned_unsorted()
        }

        use self::MergeTestOp::*;

///
    pub fn is_codepoint_boundary(&self, offset: usize) -> bool {
    fn test_get_samples_nested_trace() {
        let deletes_from_union = parse_subset("-#----#");
    /// revision, and so needs a way to get the deletion set before then.
            MetadataType::ThreadSortIndex {..} => "thread_sort_index",
/// ```rust

            } else {
                }
        assert_eq!("a0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", String::from(engine.get_head()));

            selection: SelRegion::caret(0).into(),
    /// This case is a regression test reproducing a panic I found while using the UI.


            pending_render: false,
                    if !inserts.is_empty() {
    pub fn block_payload<S, C, P>(&self, name: S, categories: C, payload: P)
        assert_eq!(trace.samples_cloned_unsorted().len(), 4);
        self.drag_state = Some(DragState { base_sel, offset, min, max });


            Edit { ei: 1, p: 1, u: 1, d: parse_delta("-f-") },
/// have to be considered in `rebase` substantially for normal editing
        assert_eq!(Some(9), a.prev_grapheme_offset(17));
        enum RopeDeltaElement_ {
                (c == pos && c == text.len() && self.line_of_offset(text, c) == line_num)
    fn gc_scenario(edits: usize, max_undos: usize) {
    #[cfg(all(not(feature = "dict_payload"), not(feature = "json_payload")))]
            s = &s[splitpoint..];
                    text.slice_to_string(start, end)
            // Go crazy
        let script = vec![
        for rev in self.revs[rev_index..].iter().rev() {
    fn do_drag(&mut self, text: &Rope, line: u64, col: u64, affinity: Affinity) {
        let new_head = engine.get_head_rev_id().token();
    pub fn lines_raw_all(&self) -> LinesRaw {
    #[test]
    pub fn is_point_in_selection(&self, offset: usize) -> bool {
    /// a point that should be scrolled into view.
        let mut d_builder = Builder::new(TEST_STR.len());
    fn identity() -> Self {
        assert!(r == a_rope.clone() + b_rope.clone());
        // same span exists in both sets (as when there is an active selection)
// distributed under the License is distributed on an "AS IS" BASIS,
            Merge(0,1), // the merge from the whiteboard scan
        self.samples.lock().unwrap().len()

    Undo {
        self.set_selection(text, selection);

    /// Tests that merging again when there are no new revisions does nothing
    /// The maximum number of entries the tracing data should allow.  Total
/// non-base revs, `N` being transformed non-base revs, and rearranges it:
            Some(Cow::Borrowed(mut s)) => {
        let other_subset = self.find_rev(other_rev).map(|rev_index| self.deletes_from_cur_union_for_index(rev_index));
        }

    /// Exposed for unit tests only.

    ProcessName { name: String },

        assert_eq!(b, String::from(a));
    pub fn next_codepoint_offset(&self, offset: usize) -> Option<usize> {
            'B' => SampleEventType::DurationBegin,

    }
                    // of which undos were used to compute deletes_from_union in edits may be lost.
        let trace = Trace::disabled();
        match cmd {
pub type Rope = Node<RopeInfo>;
        a.lines_all().collect::<Vec<_>>());
                el.serialize_field(end)?;
}
///
    s.serialize_char(ph.into_chrome_id())
        ];
    /// If the cursor is at the end of the leaf, advances to the next leaf.
        let mut selection = Selection::new();
    }
        let mut all_samples = self.samples.lock().unwrap();
    #[serde(rename = "ph")]
        if !selection.regions_in_range(offset, offset).is_empty() {
        debug_subsets(&rearranged_inserts);
    /// Note: unlike movement based selection, this does not scroll.
            }

        as_vec
            }

const FLAG_SELECT: u64 = 2;
                    self.do_gesture(text, line, column, GestureType::RangeSelect)
                        })
use std::borrow::Cow;
        );
    #[cfg(feature = "benchmarks")]
    move_delta.apply(tombstones)
                                  style_spans, &plan, pristine);
use bytecount;
        let (new_rev, new_text, new_tombstones, new_deletes_from_union) =
//See ../docs/MetricsAndBoundaries.md for more information.

extern crate libc;
    }
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("---j") },
        self.tombstones = new_tombstones;
categories_from_constant_array!(5);
                           pristine: bool)
            // to replicate whiteboard, z must be before a tombstone
    /// ties in edit priorities. Otherwise the results may be inconsistent.
            match *op {
impl Metric<RopeInfo> for BaseMetric {
    }
    }
    }
        first_line..(last_line + 1)
        assert_eq!(String::from(engine.get_head()), d.apply_to_string(TEST_STR));
    }
    /// Toggles a caret at the given offset.
/// A rope data structure.
    inserts: InsertDelta<RopeInfo>,
                },
    AsyncInstant,
pub fn enable_tracing_with_config(config: Config) {
            Edit { ei: 2, p: 1, u: 1, d: parse_delta("ab") },
        // since one of the two deletes was gc'd this should undo the one that wasn't
        let plan = RenderPlan::create(height, self.first_line, self.height);
        // is the new edit in an undo group that was already undone due to concurrency?
    session2: u32,
    pub fn is_enabled(&self) -> bool {
        where S: Serializer


    }
        }
        }
    fn deletes_from_union_before_index(&self, rev_index: usize, invert_undos: bool) -> Cow<Subset> {
        self.tombstones = tombstones;
pub type SessionId = (u64, u32);
    {
    base_sel: Selection,
            trace: None,
/// RFC reference : https://tools.ietf.org/html/rfc3629#section-4

        let script = vec![
            as_vec.push(Sample::new_metadata(
                              duration_ns: u64) -> Self
    pub(crate) fn do_edit(&mut self, text: &Rope, cmd: ViewEvent) {
        let text = Rope::from("13456");
            thread_name: Sample::thread_name(),
        let gc : BTreeSet<usize> = [1].iter().cloned().collect();
    out
    -> SampleGuard<'a>
        let b = a.slice(2, 4);
        let d2 = Delta::simple_edit(Interval::new_closed_open(0,0), Rope::from("a"), TEST_STR.len()+1);
                    }

#[derive(PartialEq, Debug)]
    /// When was the sample started.
                    return i + 1; // +1 since we know the one we just found doesn't have it
                    selection.add_region(occ);
    pub fn enable_config(&self, config: Config) {
            true => self.selection.clone(),
                peers.push(peer);
/// });
    }
        assert_eq!(2, b.offset_of_line(1));
                        b.add_span(seg.n, seg.our_line_num, line_cache_shadow::ALL_VALID);
        let col = col as usize;
            .map(|(i, _)| i)
                out.push(DeltaOp {
    s.into()
            }
        // the `false` below: don't invert undos since our first_candidate is based on the current undo set, not past
                    // can only happen on empty input
use selection::{Affinity, Selection, SelRegion};
/// # Arguments

extern crate lazy_static;
/// `dict_payload` then a Rust HashMap is expected while the `json_payload`
        let thread = std::thread::current();
        let mut engine = Engine::new(Rope::from(TEST_STR));
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("-p-") },
        if !self.is_enabled() {

    }
        d_builder.replace(Interval::new_closed_open(39, 42), Rope::from("DEEF"));
            sel.add_region(region);
// Copyright 2016 The xi-editor Authors.

    pub fn invalidate_styles(&mut self, text: &Rope, start: usize, end: usize) {
    fn do_set_replace(&mut self, chars: String, preserve_case: bool) {
    fn set_selection_raw(&mut self, text: &Rope, sel: Selection) {
    let del_delta = Delta::synthesize(tombstones, old_deletes_from_union, new_deletes_from_union);

    // TODO: insert from keyboard or input method shouldn't break undo group,
        }

                Some(Cow::from(s))
                s.len()
    /// The line ending is stripped from the resulting string. The final line ending
        assert_eq!(utf8_offset, 9);
                    };
            to_undo.insert(i+1);

        #------
                // This could be done by explicitly overriding, or by tweaking the
categories_from_constant_array!(9);
    lc_shadow: LineCacheShadow,
/// Computes a series of priorities and transforms for the deltas on the right
        self.text = new_text;
    lines: usize,
                    // We're super-aggressive about dropping these; after gc, the history
    pub fn render_styles(&self, client: &Client, styles: &StyleMap,
        let r = Rope::from(&a[..MAX_LEAF]);
    ///
        let mut a = Rope::from("hello world");
            } else {
    ///
    pub fn get_samples_count(&self) -> usize {
            match modify_selection {
            }
            fn from(mut delta: RopeDelta_) -> Delta<RopeInfo> {
            },
fn serialize_event_type<S>(ph: &SampleEventType, s: S) -> Result<S::Ok, S::Error>
    #[serde(skip_serializing_if = "Option::is_none")]
            Merge(5,4),

        if self.len() != other.len() {
            rendered_styles.push((iv.start() as isize) - ix);
            Edit { ei: 1, p: 3, u: 1, d: parse_delta("-c-") },
        });

//     http://www.apache.org/licenses/LICENSE-2.0
    use multiset::Subset;
        while offset < l.len() && !l.is_char_boundary(offset) {
            view_id: view_id,
        let mut leaf_offset = pos - offset;

        }
    }
            thread_name: Sample::thread_name(),
            trace.block_payload("z", &["test"], to_payload("test_get_samples_nested_trace"));
        offset
        use std::hash::{Hash, Hasher};
/// * `name` - A string that provides some meaningful name to this sample.

                }
                        Cow::Owned(deletes_from_union.transform_shrink(inserts))
        b.iter(|| black_box(trace.instant_payload(
            let right_str = self[splitpoint..].to_owned();
///
        ------#
                        let start_line = seg.our_line_num;
}
        }
/// let c = b.slice(1, 7);
            Edit { ei: 2, p: 1, u: 1, d: parse_delta("ab") },
            self.categories == other.categories &&
        samples[0].timestamp_us = 10;
}
    fn merge_associative() {
            Assert(2, "afc".to_owned()),
    // even worth the code complexity.
    utf16_count
    // Taking the complement of deletes_from_union leads to an interleaving valid for swapped text and tombstones,
    fn from_base_units(s: &String, in_base_units: usize) -> usize {
        deletes_from_union
        // creation) is:
pub enum SampleEventType {
                            max_undo_so_far: rev.max_undo_so_far,
}
            cur_len_utf16 += u.len_utf16();
        assert_eq!(snapshot[7].name, "z");
                // merge logic.
        match rev.edit {
        assert_eq!(snapshot[0].args.as_ref().unwrap().metadata_name.as_ref().is_some(), true);
            to_payload("some description of the trace"))));
    /// if it is non-empty. It needs to be a separate commit rather than just
        use self::MergeTestOp::*;
            Assert(3, "ab".to_owned()),
    #[test]
// Licensed under the Apache License, Version 2.0 (the "License");
    rev_id_counter: u32,
    pub fn new_duration_marker<S, C>(name: S,
    }
        assert_eq!(snapshot[0].args.as_ref().unwrap().metadata_name.as_ref().is_some(), true);
    let mut last_priority: Option<usize> = None;
///
            let sel_start_ix = clamp(region.min(), start_pos, pos) - start_pos;
                let mut peer = Engine::new(Rope::from(""));
/// let b = Rope::from("world");
        match *self {
    /// Get text of head revision.
        let trace = Trace::enabled(Config::default());
        ----#
        let r = Rope::from(&a[..MAX_LEAF]);
    None,
        assert_eq!(0, a.lines_all().count());
        match *self {
        let initial_rev = engine.get_head_rev_id().token();
                    let e = &mut self.peers[ei];
            edit: Undo { toggled_groups: BTreeSet::new(), deletes_bitxor: deletes_from_union.clone() },
        let subset_str = format!("{:#?}", expand_by[0].1);
impl<'a> SampleGuard<'a> {
        -##-
    type Item = &'a str;
            Merge(0,5), Merge(2,5), Merge(4,5), Merge(1,4),
            samples: Mutex::new(FixedLifoDeque::with_limit(config.max_samples())),
//! CRDT implementation techniques, because all operations are serialized in
        RevId { session1: self.session.0, session2: self.session.1, num: self.rev_id_counter }
        }
    }
#[inline]
                Ok(CategoriesT::DynamicArray(categories))
        RevId { session1: 1, session2: 0, num: i as u32 }
        let utf8_offset = 19;
    num: u32,
                        inserts: transformed_inserts,
        // since one of the two deletes was gc'd this should re-do the one that wasn't
    ///
/// patterns. Any large runs of typing in the same place by the same user (e.g
        let a = Rope::from("a\r\nb\r\nc");
    pub fn get_line_range(&self, text: &Rope, region: &SelRegion) -> Range<usize> {

/// A size, in pixel units (not display pixels).
        while let Err(incomp) = prev_boundary {
/// Returns the file name of the EXE if possible, otherwise the full path, or
                        });
            a.push('a');
    /// Returns a slice of the string from the byte range [`start`..`end`).
    fn edit_rev_concurrent() {
///
        count_newlines(&s[..in_base_units])
                let to_gc : BTreeSet<usize> = [i-max_undos].iter().cloned().collect();

    use test_helpers::{parse_subset_list, parse_subset, parse_delta, debug_subsets};
    pub fn slice(&self, start: usize, end: usize) -> Rope {

        engine.undo([1].iter().cloned().collect());
    #[cfg(feature = "benchmarks")]
                    priority, inserts, deletes,
#[cfg(feature = "json_payload")]
            Cancel => self.do_cancel(text),
    }
    ///
        Self::with_limit_count(size / size_of::<Sample>())
    }
            scroll_to: Some(0),
    let new_text = del_delta.apply(text);
    pub fn get_head(&self) -> &Rope {
    /// Return the offset of the codepoint after `offset`.

        where S: Into<StrCow>, C: Into<CategoriesT>
            inner: self.lines_raw(start, end)
#![cfg_attr(feature = "benchmarks", feature(test))]
                          payload: Option<TracePayloadT>) -> Self
/// # Performance
        (Revision {
#[derive(Copy, Clone)]
    pub fn empty() -> Engine {
impl MetadataType {
        result
        a.lines_all().collect::<Vec<_>>());
    /// This function does not perform any scrolling.
                    RopeDeltaElement_::Copy(start, end) =>
            }
    ThreadName { name: String },
        let mut b = String::new();


            'X' => SampleEventType::CompleteDuration,
        out.push(Revision {

        assert_eq!(2, a.line_of_offset(4));
        let snapshot = trace.samples_cloned_sorted();
    // TODO: maybe refactor this API to take a toggle set
        match self.breaks {
// Copyright 2018 The xi-editor Authors.
        engine.gc(&gc);

        let deletes_bitxor = self.deletes_from_union.bitxor(&deletes_from_union);
/// Revision 0 is always an Undo of the empty set of groups
                    filename.to_str().map(|s| s.to_string())
    }
/// # Examples
        assert_eq!("zcpbdj", String::from(&text_2));
        }).collect()
    // TODO: a case can be made to hang this on Cursor instead
        self.find.first_mut().unwrap().do_find(text, search_query, case_sensitive, false, true);
/// Creates a duration sample.  The sample is finalized (end_ns set) when the
    }
        }
                self.select_line(text, offset, line, true),
            deletes_from_union = match rev.edit {
                        None if cursor.pos() == text.len() => cursor.pos(),
        self.height
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("-p-") },
    }
    }
            end += 1;

        where D: Deserializer<'de>,
                        DeltaElement::Insert(Rope::from(s)),

// utility function to clamp a value within the given range

            }

                    s = &s[..s.len() - 1];
        old_deletes_from_union: &Subset, new_deletes_from_union: &Subset) -> (Rope,Rope) {
            base_rev: RevToken, delta: Delta<RopeInfo>) -> (Revision, Rope, Rope, Subset) {
    /// wherever there's a non-zero-count segment in `deletes_from_union`.
    /// If `modify` is `true`, the selections are modified, otherwise the results
    /// Return the offset of the codepoint before `offset`.
            sample.timestamp_us = ns_to_us(time::precise_time_ns());
        let trace = Trace::disabled();
        let expand_by = compute_transforms(revs);
use xi_rope::delta::Delta;
/// let a = Rope::from("hello world");
            Assert(0, "zcpbdj".to_owned()),

        }
// distributed under the License is distributed on an "AS IS" BASIS,
    fn from_str(s: &str) -> Result<Rope, Self::Err> {

    #[derive(Debug)]
        deletes_from_union


    }
        trace.closure_payload("x", &["test"], || {
                                gc_dels = gc_dels.transform_expand(inserts);


        let b = a.slice(2, 4);
fn find_base_index(a: &[Revision], b: &[Revision]) -> usize {
            self.breaks = None
    if x < min {
                c.provide_context(&pl, self.pos() - poffset);
                            None
    buffer_id: BufferId,

}
        };
        _ => 4
            pid: sys_pid::current_pid(),
//! conceptually represents the current text and all edit history for that
        Err(ref e) => {

    let inverse_tombstones_map = old_deletes_from_union.complement();
    pub fn scroll_height(&self) -> usize {
            trace.closure_payload("y", &["test"], || {
/// Base unit is utf8 code unit.
                        Ok(s) => Some(s),
        }
            breaks: None,
        if let Some(sel) = new_sel {
        } else {
    }
    /// Set the selection to a new value.
        b.iter(|| {
        assert_eq!(Some(2), b.prev_codepoint_offset(5));

            Undo { ref deletes_bitxor, .. } => deletes_bitxor.count(CountMatcher::All),
            Assert(1, "pbj".to_owned()),
            args: Some(SampleArgs {

    #[test]
/// rather than just this processe's samples).
        let correct: BTreeSet<RevId> = [0,2,4,8].iter().cloned().map(basic_rev).collect();
}

    fn to_payload(value: &'static str) -> &'static str {
    // and partly because you need to retain more undo history, to supply input to the
/// An element in a `RopeDelta`.
    /// All the selection regions other than the one being dragged.
            }
        if self.find_changed != FindStatusChange::None {
    #[test]
        let mut cur_len_utf16 = 0;
    }
    StaticArray(&'static[&'static str]),
            Sample::new_instant("local pid", &[], None),
        let mut next_boundary = c.next_boundary(&l, leaf_offset);
            //self.cursor.set(self.end);
        self.replace_changed = true;
    }
    fn undo() {
pub struct RopeInfo {
///

impl Eq for Sample {}
    #[serde(skip_serializing)]
    ///
        engine.edit_rev(1, 3, new_head_2, d3);

categories_from_constant_array!(10);
        where S: Into<StrCow>, C: Into<CategoriesT>
    #[cfg(feature = "benchmarks")]
        let mut pos = self.pos();
///
/// Measured unit is newline amount.
        assert_eq!(rope.line_of_offset(1), 0);
        assert_eq!(trace.get_samples_limit(), 20);
impl Eq for CategoriesT {}

        View {
fn to_cow_str<S>(s: S) -> StrCow where S: Into<StrCow> {
        }
    fn next_codepoint_offset_small() {
#![cfg_attr(feature = "cargo-clippy", allow(
            tid: sys_tid::current_tid().unwrap(),
        // Any edit cancels a drag. This is good behavior for edits initiated through
        };
}
    fn measure(_: &RopeInfo, len: usize) -> usize {
    }
        let mut offset = self.offset_of_line(text, line).saturating_add(col);
    #[test]

        let mut style_map = style_map.borrow_mut();
    /// Only works well for ASCII, will probably not be maintained long-term.
    }
/// # use xi_rope::Rope;
            Assert(0, "abc".to_owned()),
            let mut sel = drag_state.base_sel.clone();
    #[test]
            }
            }, to_payload("test_get_sorted_samples"));
    #[inline]
    /// is optional.
categories_from_constant_array!(3);
///
        b.push(rhs);
    }
        engine.undo([1,2].iter().cloned().collect());
        assert_eq!(Some(5), b.prev_codepoint_offset(9));
            'E' => SampleEventType::DurationEnd,
        self.lines_raw(0, self.len())
        assert_eq!(Some(3), a.next_grapheme_offset(0));

    pub fn default() -> Self {
    // this is the cost contributed by the timestamp to trace()
// ======== Merge helpers
        s.is_char_boundary(offset)

        // 1 for exe name & 1 for the thread name
    /// An iterator over the lines of a rope.
                let older_all_inserts = inserts.transform_union(&cur_all_inserts);
        value
/// typing a paragraph) will be combined into a single segment in a transform
    }
        for i in 0..edits {
impl Serialize for Rope {
        let a = Rope::from("a\u{00A1}\u{4E00}\u{1F4A9}");
        let deletes_from_union = Subset::new(0);
                self.set(pos);

        engine.edit_rev(1, 1, initial_rev, d1.clone());
                        ops.push(self.build_update_op("copy", None, seg.n));
    /// than this will panic.



    // This computes undo all the way from the beginning. An optimization would be to not
            rev_id: self.next_rev_id(),
impl serde::Serialize for CategoriesT {
        ---#--
                    }
fn compute_transforms(revs: Vec<Revision>) -> Vec<(FullPriority, Subset)> {
        let new_sel = self.selection.apply_delta(delta, true, keep_selections);
/// could trigger incorrect behavior if they collide, so u64 is safe.
        if self.breaks.is_some() {
        for sample in all_samples.iter() {
                },
        let line = line as usize;
                } else {
///
        self.select_next_occurrence(text, reverse, false, allow_same, modify_selection);
    #[cfg(feature = "json_payload")]
    }
        if self.find.is_empty() {
    }
    /// validate their input.
    // transform representing the characters added by common revisions after a point.
    }
        Ok(Rope::from(s))
// you may not use this file except in compliance with the License.
        }
    // recompute the prefix up to where the history diverges, but it's not clear that's
    /// how many times it has been deleted, so if a character is deleted twice
/// ```
            engine.undo(undos);
    for rev in revs.iter().rev() {
            Assert(1, "zapbj".to_owned()),
}
                max_undo_so_far: i,
        assert_eq!(rope.offset_of_line(1), 3);
                            ops.push(self.build_update_op("skip", None, n_skip));
                Undo { ref toggled_groups, ref deletes_bitxor } => {
    {
        self.utf16_size += other.utf16_size;
                      width_cache: &mut WidthCache, keep_selections: bool)
            AddSelectionAbove =>
        // NOTE 2: from_micros is currently in unstable so using new
        });
            lines: count_newlines(s),
/// Boundary is trailing and determined by a newline char.
        self.revs.push(new_rev);
///
                }
        let gc : BTreeSet<usize> = [1].iter().cloned().collect();
            selection.delete_range(offset, offset, true);
pub type StrCow = Cow<'static, str>;
                inserts: inserted,
        mut deletes_from_union: Subset, mut max_undo_so_far: usize) -> (Vec<Revision>, Rope, Rope, Subset) {
            }
        self.revs.last().unwrap().rev_id
        b if b < 0xe0 => 2,
impl SampleEventType {
        assert_eq!(Some(0), a.prev_grapheme_offset(3));
    }
    fn do_cancel(&mut self, text: &Rope) {
            Some(offset + len_utf8_from_first_byte(b))
        assert_eq!(utf16_units, 11);
        all_samples.reset_limit(config.max_samples());
            match exe_name.clone().file_name() {

use client::Client;
                leaf_offset = self.pos() - poffset;
    #[serde(rename = "xi_payload")]
/// std::mem::drop(trace_guard); // finalize explicitly if
    }
        -> Self
    /// set with this function, which will make the revisions they create not have colliding IDs.

}
        // stable order since the resolution of timestamps is 1us.
                let (pl, poffset) = self.prev_leaf()?;
                }
                    return None;
        // todo: this will be changed once multiple queries are supported
        let start = time::precise_time_ns();
mod sys_pid;
                        None
// Unless required by applicable law or agreed to in writing, software
    pub fn new(view_id: ViewId, buffer_id: BufferId) -> View {
        assert!(empty == empty);
            // about what wrapped.

        let trace = Trace::enabled(Config::with_limit_count(10));

}
    #[inline]
    }
    // ============== Merge script tests
        } else {
                metadata_name: None,
        self.sample_limit_count
                _ => { }
            let expand_by = compute_transforms(a_new);
        self.sample_limit_count * size_of::<Sample>()
    }
        }
            'i' => SampleEventType::Instant,
        use self::MergeTestOp::*;
                self.push_leaf(s.to_owned());
            // copy the current state
                if !region.is_caret() {
/// supported as an optimization when only one reference exists, making the
    pub height: f64,
}
    /// The maximum amount of space the tracing data will take up.  This does
    /// Sorting priority between processes/threads in the view.

    #[bench]
            write!(f, "Rope({:?})", String::from(self))
        let snapshot = trace.samples_cloned_unsorted();
                            rev_id: rev.rev_id,
                        _ => break
                        MetadataType::ThreadName { name: thread_name.to_string() },
        let mut all_samples = self.samples.lock().unwrap();
    /// Tracks whether the replacement string or replace parameters changed.
        self.set_selection_raw(text, sel.into());
    (out, text, tombstones, deletes_from_union)
        assert_eq!(rope.line_of_offset(15), 3);
#[inline]
    fn edit_rev_undo_2() {

    // partly because the reachability calculation would become more complicated (a
    }

            }
    #[inline]
        assert_eq!(trace.get_samples_count(), 0);
        text = new_text;
        info.lines
            Edit { ei: 2, p: 1, u: 1, d: parse_delta("ab") },


    }
        engine.edit_rev(1, 1, first_rev, build_delta_1());
}
            &self.deletes_from_union, &old_deletes_from_union);
    /// Updates the view after the text has been modified by the given `delta`.
        let mut thread_names: HashMap<u64, StrCow> = HashMap::new();
                }
    /// It does undos and gcs in a pattern that can actually happen when using the editor.
            SampleEventType::FlowInstant => 't',
            Edit { ei: 0, p: 1, u: 1, d: parse_delta("-d-") },
#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]


    #[test]
/// For example, take a string that starts with a 0xC2 byte.
        Self {

        assert_eq!("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", String::from(engine.get_head()));

    pub fn prev_codepoint_offset(&self, offset: usize) -> Option<usize> {
            "something", &["benchmark"],
    revs: Vec<Revision>,

    enabled: AtomicBool,
        b.iter(|| black_box(trace.block("something", &["benchmark"])));
            Assert(0, "zcbd".to_owned()),
                                undo_group,
    }
pub type RevToken = u64;
    /// Select entire buffer.
    }


        if let Some(ix) = style_map.lookup(style) {
/// returned value is dropped.  `trace_closure` may be prettier to read.
impl<'a> Iterator for ByteIter<'a> {
    pub fn set_size(&mut self, size: Size) {
    /// Tracks whether find highlights should be rendered.
    }
/// * `name` - A string that provides some meaningful name to this sample.
    fn measure(info: &RopeInfo, _: usize) -> usize {
    /// Tests that priorities are used to break ties correctly
        ---#--
        assert_eq!(correct, rebased_inserts);
        }
        let invalid = if all_caret {
impl<'de> serde::Deserialize<'de> for CategoriesT {
            Edit { ei: 0, p: 3, u: 1, d: parse_delta("---d") },
    // for simplicity.
                metadata_sort_index: None,
    pub fn disable(&self) {
impl<'de> Deserialize<'de> for Rope {
        // since character was inserted after gc, editor gcs all undone things
    }
        let mut retain_revs = BTreeSet::new();
            Sample::new_instant("remote pid", &[], None)];
                                          closure: F, payload: P)
        let line = self.line_of_offset(text, end);

        let all_samples = self.samples.lock().unwrap();
                                            closure: F, payload: P) -> R
        let end = self.sel_regions().last().unwrap().end;
                    };

/// from the new revisions on the left.
                if !inserts.is_empty() {
        where S: Into<StrCow>, C: Into<CategoriesT>, P: Into<TracePayloadT>,
///
/// ```
    }
}
    out.as_mut_slice().reverse();

    }
        }
    #[bench]

                //// if we aren't on a boundary we can't be at the end of the chunk
        b if b < 0xf0 => 3,
    {
                self.find_changed = FindStatusChange::All;
        undo_test(true, [1,2].iter().cloned().collect(), TEST_STR);
                } else {
// Licensed under the Apache License, Version 2.0 (the "License");
    pub fn next_grapheme_offset(&self, offset: usize) -> Option<usize> {
        self.convert_metrics::<LinesMetric, BaseMetric>(line)
    pub fn with_limit_count(limit: usize) -> Self {
                continue;
                              payload: Option<TracePayloadT>,
        assert_eq!(vec![a.as_str(), b.as_str()], r.lines_raw_all().collect::<Vec<_>>());
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    pub fn set_session_id(&mut self, session: SessionId) {
                rev_id: basic_rev(i),
        MergeTestState::new(3).run_script(&script[..]);
            Some(region) => {
    use rope::{Rope, RopeInfo};
            s.as_bytes()[offset - 1] == b'\n'
lazy_static! { static ref TRACE : Trace = Trace::disabled(); }
                    // we don't want new revisions before this to be transformed after us
                        }
//
    /// Returns the largest undo group ID used so far
        -#-

        for region in self.selection.iter() {
use std::borrow::Cow;
    Matches

    /// Selects the given region and supports multi selection.
        }
    pub fn do_move(&mut self, text: &Rope, movement: Movement, modify: bool) {
pub fn trace_closure<S, C, F, R>(name: S, categories: C, closure: F) -> R
        let replacement = match self.selection.last() {
            Assert(1, "ab".to_owned()),
        a.lines_all().collect::<Vec<_>>());

                l = pl;
        let revs = basic_insert_ops(inserts, 1);
                },


        }
    ObjectSnapshot,
    /// The line number is 0-based, thus this is equivalent to the count of newlines
    cursor: Cursor<'a, RopeInfo>,
    fn scroll_to_cursor(&mut self, text: &Rope) {
}
        } else {

        });
    #[test]
    TRACE.enable();
    pub chars: String,

        self.find.clear();
                    }

            revs: vec![rev],
    }
        assert_eq!(snapshot[4].name, "z");
                        }
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        if f.alternate() {
}
                    } else if !inserts.is_empty() {
        let tombstones = Rope::from("27");
///
}
        MergeTestState::new(3).run_script(&script[..]);
            sample_limit_count: limit
            rebased_deletes_from_union.union(to_delete)
        let mut line_num = 0;  // tracks old line cache

/// xi_trace::trace_closure("something_else_expensive", &["rpc", "response"], || {
        }
    pub fn get_caret_offset(&self) -> Option<usize> {
                            (inserts.transform_shrink(&gc_dels),
    pub pid: u64,
///
            return None;
            line_cache_shadow::CURSOR_VALID
    pub fn toggle_sel(&mut self, text: &Rope, offset: usize) {
    #[test]
            Some(right_str)
        assert!(leaf.is_char_boundary(end));
    fn merge_simple_delete_2() {
        min
        self.find_changed = FindStatusChange::None;
                            }
    fn next_rev_id(&self) -> RevId {
    /// should be constructed via SampleGuard.
            CategoriesT::StaticArray(ref self_arr) => {
                self.drag_state = None;
pub fn trace<S, C>(name: S, categories: C)
                WrapWidth::Width(px) =>
            Assert(0, "ab".to_owned()),
        assert_eq!(vec!["a", "b", "c"], a.lines_all().collect::<Vec<_>>());
        }, to_payload("test_get_samples_nested_trace"));

                    base_len: delta.base_len
        };
    } else if x < max {

        let trace = Trace::enabled(Config::with_limit_count(20));
    #[test]
        engine.edit_rev(0, 1, first_rev, build_delta_1());
    /// the other doesn't make it re-appear.
        engine.edit_rev(1, 2, new_head, d2); // note this is based on d1 before, not the undo
        assert_eq!(trace.get_samples_limit(), 0);
struct DeltaOp {
            rendered_styles.push((sel_start as isize) - ix);
use interval::Interval;
/// # Returns
            deletes_from_union,
///
    }
        ---#--
    /// Time complexity: O(log n)
    /// after an edit is applied, to allow batching with any plugin updates.
            Assert(0, "acrbd".to_owned()),
        assert_eq!("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", String::from(engine.get_head()));
                }
    }
        &self.text
                selections.push((sel_start_ix, sel_end_ix));
            match rev.edit {

        #[derive(Serialize, Deserialize)]
    /// The state for finding text for this view.
}
    #[inline]

    fn cmp(&self, other: &Sample) -> cmp::Ordering {
    /// Imagine a "union string" that contained all the characters ever
    type Output = Rope;
        if self.idx <= self.current.unwrap().len() {
    }
            None
        let rope = Rope::from("hi\ni'm\nfour\nlines");
        assert_eq!(0, b.offset_of_line(0));
        assert_eq!(String::from(&r).lines().collect::<Vec<_>>(),
    fn thread_name() -> Option<StrCow> {
    FlowEnd,
    /// An arbitrary payload to associate with the sample.  The type is
    pub fn max_samples(&self) -> usize {
        /// Used to order concurrent inserts, for example auto-indentation

            SampleEventType::DurationEnd => 'E',
    fn push_maybe_split(&mut self, other: &String, iv: Interval) -> Option<String> {
            pid: sys_pid::current_pid(),
            rendered_styles.push(0);
    }
                return None;
            let dels_from_tombstones = gc_dels.transform_shrink(&not_in_tombstones);
            args: Some(SampleArgs {
            }),
    pub fn iter_chunks_all(&self) -> ChunkIter {
            }
        }
            x.next_occurrence(text, reverse, wrapped, &self.selection)
        }
    edit: Contents,
    }
    //     old_deletes_from_union, new_deletes_from_union, text, new_text, tombstones);
                            rev_id: rev.rev_id,
        for mut find in self.find.iter_mut() {
    /// should be as unlikely to collide as two random u64s.
        let b = a.slice(1, 10);
        let rope = Rope::from("a\u{00A1}\u{4E00}\u{1F4A9}");
    }
                }

                        line_num = seg.their_line_num + seg.n;
        Self {
    }
    #[allow(dead_code)]
                    sel.add_region(region);
    fn next(s: &String, offset: usize) -> Option<usize> {
    fn edit_rev_undo_3() {
        assert_eq!(vec![&a[..line_len], &b[..line_len]], r.lines_all().collect::<Vec<_>>());
    fn append_large() {
        self.set_dirty(text);
        }, new_text, new_tombstones, new_deletes_from_union)
    }
        let mut engine = Engine::new(Rope::from(TEST_STR));
        engine.edit_rev(1, 1, first_rev, build_delta_1());
        }).unwrap_or(text.len());
    height: usize,
        for _ in 0..line_len {
                    rev_id: rev.rev_id,
    fn replace_small() {
    fn prev(s: &String, offset: usize) -> Option<usize> {
    #[should_panic]

        }
        }
    /// Sets the selection to a new value, without invalidating.
        engine.edit_rev(1, 1, first_rev, build_delta_1());
        trace.closure_payload("y", &["test"], || {},


                            if !deletes.is_empty() {
        }
    }



            Merge(0,2), Merge(1, 2),
#[cfg(test)]
    pub fn join(&self, sep: &str) -> String {
                    // are empty, could send update removing the cursor.
        let utf16_units = rope_with_emoji.convert_metrics::<BaseMetric, Utf16CodeUnitsMetric>(utf8_offset);
            Merge(0,2), Merge(1, 2),
        trace.instant("2", &["test"]);
categories_from_constant_array!(0);
        b.iter(|| black_box(trace.closure("something", &["benchmark"], || {})));
            self.truncate(splitpoint);
            trace.closure_payload("y", &["test"], || {

        let initial_rev = engine.get_head_rev_id().token();
            rendered_styles.push(iv.end() as isize - iv.start() as isize);
    fn do_gesture(&mut self, text: &Rope, line: u64, col: u64, ty: GestureType) {

use std::cmp::{min,max};
            Assert(2, "ab".to_owned()),
        }
                let offset = self.selection[0].start;

        let mut d = TracePayloadT::with_capacity(1);
            },
        let mut d_builder = Builder::new(TEST_STR.len());
                    to_payload(("some payload for the block"))));

                    self.fragment = &self.fragment[i + 1 ..];
                l = nl;
/// See `trace_block` for how the block works and `trace_payload` for a
            }

    ThreadSortIndex { sort_index: i32 },
    }
        // insert `edits` letter "b"s in separate undo groups
}
                },
        //} else {
/// let mut a = Rope::from("hello world");
        assert_eq!(trace.is_enabled(), true);
                    } else {
            }

            SampleEventType::DurationBegin => 'B',
        /// The subset of the characters of the union string from after this
    pub fn iter_chunks(&self, start: usize, end: usize) -> ChunkIter {
            Token::Str("a\u{00A1}\u{4E00}\u{1F4A9}"),
            }
    let b_ids: BTreeSet<RevId> = b.iter().map(|r| r.rev_id).collect();
        where S: Into<StrCow>, C: Into<CategoriesT>

    TRACE.disable();
    }
        let d = RopeDelta_::deserialize(deserializer)?;
                                gc_dels = gc_dels.transform_union(inserts);

            size: Size::default(),
                    deletes: deletes.clone(),
                edit: Contents::Edit {
    }
        let (mut last_line, last_col) = self.offset_to_line_col(text, end);
        d_builder.replace(Interval::new_closed_open(59, 60), Rope::from("HI"));
        };
    // * grapheme clusters
                    let (start, end) = {
        match symbol {
            self.pid == other.pid &&
        let d1 = Delta::simple_edit(Interval::new_closed_open(0,10), Rope::from(""), TEST_STR.len());
#[derive(Serialize, Deserialize, Debug, Clone)]
        if self.is_enabled() {
    rev_id: RevId,
    FlowInstant,
                   text: &Rope, start_of_line: &mut Cursor<RopeInfo>,
        MergeTestState::new(3).run_script(&script[..]);
                find_leaf_split_for_bulk(s)

                        } else {
            Edit { ei: 0, p: 3, u: 1, d: parse_delta("---d") },
}
        // +2 for exe & thread name.
                } else {
                self.do_gesture(text, line, col, ty),
        d.insert(StrCow::from("test"), StrCow::from(value));
        (Revision {
use std::mem::size_of;
                Some(Cow::from(s))
                    CategoriesT::StaticArray(ref other_arr) => self_arr.eq(other_arr),
                let (nl, noffset) = self.next_leaf()?;
    /// state and new offset.
            }
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("-p-") },
            Edit { ei: 2, p: 1, u: 1, d: parse_delta("!!") },
}
    #[test]
                }
//! text.
            }
            let head = engine.get_head_rev_id().token();
                            if !inserts.is_empty() {
/// with.  By default, the type is string-like just like name.  If compiled with
                        hls.push((sel_start_ix, sel_end_ix));
categories_from_constant_array!(6);


        let result = closure();
                        let (inserts, deletes) = if gc_dels.is_empty() {
        }
/// let result = xi_trace::trace_closure("something_expensive", &["rpc", "request"], || {

                len += 1;
                (drag_state.max, min(offset, drag_state.min))
                true => x.end,
                }
            if sel_end_ix > sel_start_ix {
                payload: payload,
/// Represents a transform from one rope to another.
    /// in the range of 511-1024 bytes.

    fn test_cross_process_samples() {
        let a = Rope::from("a\nb\nc");
        //};
            let new_end = self.line_of_offset(text, iv.start() + new_len) + 1;
                            }
        if self.len() <= MAX_LEAF {
    /// Highlights are only rendered when search dialog is open.
    /// List of categories the event applies to.
    pub fn delta_rev_head(&self, base_rev: RevToken) -> Delta<RopeInfo> {
    /// Get previous codepoint before cursor position, and advance cursor backwards.
    fn compute_transforms_1() {
            event_type: SampleEventType::Metadata,
        Ok(b.build())
        undo_test(true, [2].iter().cloned().collect(), "0123456789abcDEEFghijklmnopqr999stuvz");

                        let n_skip = seg.their_line_num - line_num;
                None => {
/// for strings, specialized for incremental editing operations. Most operations
            let after = full_priority >= trans_priority;  // should never be ==
        if self.cursor.pos() >= self.end {
    find_leaf_split(s, MIN_LEAF)
        assert_eq!(None, a.prev_grapheme_offset(0));
            Some(offset + len_utf8_from_first_byte(b))
        for _i in 0..50 {
        engine.edit_rev(1, 2, initial_rev, d1.clone());
            utf16_count += 1;
    #[test]
    }
        let len = match first_rev.edit {
                    let (start, end) = self.peers.split_at_mut(ai);
                metadata_name: None,

    }
{
            };
            args: Some(SampleArgs {
/// it would be easy enough to modify to use `Arc` instead of `Rc` if that were

            self.event_type == other.event_type &&
categories_from_constant_array!(8);
            }

//
use internal::find::FindStatus;
            trace.block_payload("z", &["test"], to_payload("test_get_sorted_samples"));
impl Hash for Sample {
    #[test]
    fn build_update_op(&self, op: &str, lines: Option<Vec<Value>>, n: usize) -> Value {
                self.select_line(text, offset, line, false),

            AssertMaxUndoSoFar(1,3),
    }
        assert_eq!(samples[1].name, "local pid");
        let mut end = leaf.len().min(offset.saturating_add(chunk_size));
                let mut cursor = Cursor::new(&text, region.min());
            }
}

            find_changed: FindStatusChange::None,
            self.find.push(Find::new());
// limitations under the License.
        assert_eq!(Some(s1.len() * 3), a.prev_grapheme_offset(s1.len() * 3 + 4));
                    match full_path_str {
    tombstones: Rope,

            duration_us: None,
///     0
        let old_revs = std::mem::replace(&mut self.revs, Vec::new());
    /// We attempt to reduce duplicate renders by setting a small timeout
        let mut b = String::new();
        b.add_span(height, 0, 0);
    TRACE.closure(name, categories, closure)
        let expanded_deletes_from_union = deletes_from_union.transform_expand(&inserted);
        // A with ring, hangul, regional indicator "US"
                where E: serde::de::Error
    }
        if offset == 0 {
use std::cmp;
            if let Edit { ref inserts, .. } = rev.edit {
                    if retain_revs.contains(&rev.rev_id) || !gc_groups.contains(&undo_group) {
    }

    }
                        &mut start[bi]
            trace: Some(&trace),
    ///
        let first_rev = &self.revs.first().unwrap();
            .map(|pos| offset + pos + 1)
}
                    self.do_gesture(text, line, column, GestureType::WordSelect)
    fn arr_eq(&self, other: &Rhs) -> bool;
        let mut offset = 0;
    /// # Panics
    }
}
        engine.undo([1].iter().cloned().collect());
        b.push('\n');

                Some(i) => {
                payload: None,
}
/// some way.

            max_undo_so_far: std::cmp::max(undo_group, head_rev.max_undo_so_far),

    fn to_payload(value: &'static str) -> TracePayloadT {
                name, categories, payload, SampleEventType::DurationBegin)),
    pub fn prev_grapheme(&mut self) -> Option<usize> {
        // NOTE: 1 us sleeps are inserted as the first line of a closure to
    // TODO: replace this with a version of `lines` that accepts a range
                MergeTestOp::AssertAll(ref correct) => {
            None
        }
                                inserts,
        assert_eq!(String::from(&a).lines().collect::<Vec<_>>(),

            }
            inner: self.iter_chunks(start, end),
            Assert(0, "acpbdj".to_owned()),
    }
        Edit { ei: usize, p: usize, u: usize, d: Delta<RopeInfo> },
        use self::MergeTestOp::*;
    undone_groups: BTreeSet<usize>,  // set of undo_group id's

    }
            Edit { ei: 1, p: 3, u: 1, d: parse_delta("--efg!") },
        }
/// assert_eq!("ello wor", String::from(&b));
        let mut b = TreeBuilder::new();
    TRACE.instant(name, categories);
///
    /// Returns the visible line number containing the given offset.
        assert_eq!(None, a.next_grapheme_offset(17));
            Assert(1, "acb".to_owned()),
    fn compute_undo(&self, groups: &BTreeSet<usize>) -> (Revision, Subset) {
        let selection = SelRegion::new(0, text.len()).into();
            text: Rope::default(),
    }
        if self.current.is_none() {
                    }

        }
    #[inline]
                serializer.serialize_newtype_variant("DeltaElement", 1,
        assert!(r != b_rope + a_rope);
    /// Select the next occurrence relative to the last cursor. `reverse` determines whether the
                    }
        // TODO(vlovich): optimize this path to use the Complete event type
            }
        self.enabled.load(AtomicOrdering::Relaxed)
        let mut cursor = Cursor::new(self, offset);
fn shuffle(text: &Rope, tombstones: &Rope,
        in_base_units
            // I think it's a precondition that this will never be called
                    } else {
            ix = iv.end() as isize;
    }
            Assert(0, "acbd".to_owned()),
                        self.revs.push(Revision {
            {
    }
}

        let tombstones_b = Rope::from("a");
            'D' => SampleEventType::ObjectDestroyed,
            }, to_payload("test_get_samples_nested_trace"));
    type Item = Cow<'a, str>;
                        deletes_from_union = deletes_from_union.transform_expand(inserts);
            Assert(1, "apb".to_owned()),
        let text_with_inserts = text_ins_delta.apply(&self.text);
/// Returns all the samples collected so far.  There is no guarantee that the
                let (pl, poffset) = self.prev_leaf()?;
                    let sel_end_ix = clamp(region.max(), start_pos, pos) - start_pos;


    /// None of the find parameters or number of matches changed.
///
    impl MergeTestState {
    }
            .map(|pos| pos + 1)
        b.iter(|| black_box(trace.instant("something", &["benchmark"])));
    pub fn next_grapheme(&mut self) -> Option<usize> {

    fn into_chrome_id(&self) -> char {
    /// Get the line range of a selected region.
                        let _ = s.pop();
            self.deletes_from_union = self.deletes_from_union.transform_shrink(&gc_dels);
            }),

            Assert(0, "ab".to_owned()),
        }

fn rebase(mut expand_by: Vec<(FullPriority, Subset)>, b_new: Vec<DeltaOp>, mut text: Rope, mut tombstones: Rope,
        Self {
            DeltaElement::Copy(ref start, ref end) => {
/// are shared.
                return false;
        match *self {
            Some(offset - len)
        engine.undo([1].iter().cloned().collect());
    /// I have a scanned whiteboard diagram of doing this merge by hand, good for reference
// you may not use this file except in compliance with the License.
            let mut len = 1;
    DurationEnd,
        }
        };
        self.push_subseq(&mut b, self_iv.suffix(edit_iv));
                last_priority = Some(priority);
        info.utf16_size
        for i in 0..5_000 {
impl Leaf for String {
    #[test]
    }

    fn gc_3() {
            next_expand_by.push((trans_priority, new_trans_inserts));
                    assert_eq!(correct, &String::from(e.get_head()), "for peer {}", ei);
            self.start_drag(offset, offset, offset);
    //
        self.scroll_to_cursor(text);
        ).min_by_key(|x| {
            wrap_col: WrapWidth::None,
            }
            }
                self.add_selection_by_movement(text, Movement::Up),
    fn mk_new_rev(&self, new_priority: usize, undo_group: usize,
            edit: Contents::Edit {
    }
            Merge(1,0),
            }


    }

    fn find_base_1() {
#[inline]
        where S: Into<StrCow>, C: Into<CategoriesT>, P: Into<TracePayloadT>
            let common = find_common(a_to_merge, b_to_merge);
use fixed_lifo_deque::FixedLifoDeque;
/// ```
    }
            Edit { ei: 0, p: 1, u: 1, d: parse_delta("abc") },
        CategoriesT::DynamicArray(c)
            let b = s.as_bytes()[offset];
    }
                }
/// Is tracing enabled.  Technically doesn't guarantee any samples will be
        Merge(usize, usize),

/// }
}
use width_cache::WidthCache;
}
        let base_subset = self.find_rev(base_rev).map(|rev_index| self.deletes_from_cur_union_for_index(rev_index));
        RopeInfo {
    pub fn sel_regions(&self) -> &[SelRegion] {
        // This might be an over-specified test as it will
        priority: usize,
    pub fn instant_payload<S, C, P>(&self, name: S, categories: C, payload: P)
            warn!("Failed to get path to current exe: {:?}", e);
            }
                        ops.push(self.build_update_op("ins", Some(rendered_lines), seg.n));

        let mut selection = Selection::new();
                        sample.tid));
                CategoriesT::StaticArray(c)
        if let Some((l, offset)) = self.get_leaf() {


        assert_eq!(String::from(&a).lines().collect::<Vec<_>>(),
            Merge(1,0), Merge(2,0),
    #[bench]
        Delta::synthesize(&old_tombstones, &prev_from_union, &self.deletes_from_union)

        if !self.lc_shadow.needs_render(plan) { return; }

        if initial_contents.len() > 0 {

            SampleEventType::FlowEnd => 'f',
        let r = r + Rope::from(&b[MIN_LEAF..]);
use tree::{Leaf, Node, NodeInfo, Metric, TreeBuilder, Cursor};
        self.find_rev_token(rev).map(|rev_index| self.rev_content_for_index(rev_index))
//
        // +2 for exe & thread name
                    }
        if line > max_line {
        let d1 = Delta::simple_edit(Interval::new_closed_open(0,10), Rope::from(""), TEST_STR.len());

        rope.line_of_offset(20);
        self.set_selection(text, SelRegion::caret(offset));
    fn consume(self) -> (Option<String>, Option<i32>) {
    fn bench_get_pid(b: &mut Bencher) {
    /// The thread the sample was captured on.  Omitted for Metadata events that
use xi_rope::spans::Spans;
    fn build_delta_2() -> Delta<RopeInfo> {
    #[serde(default = "initial_revision_counter", skip_serializing)]
    /// The maximum number of samples that should be stored.
            undo_group: 0,
            assert_eq!(Some(i / 8 * 8 + 8), a.next_grapheme_offset(i));
    pub metadata_name: Option<StrCow>,
        assert_eq!(None, a.prev_grapheme_offset(0));
impl Engine {
            .find(|&(_, ref rev)| rev.rev_id.token() == rev_token)
            trace.record(sample);
    TRACE.block_payload(name, categories, payload)

    /// Width in bytes (utf-8 code units).
            Assert(2, "b".to_owned()),
                        if n_skip > 0 {
        self.find_changed = FindStatusChange::Matches;
    }
        while offset < l.len() && !l.is_char_boundary(offset) {
        let trace = Trace::disabled();
/// The `dict_payload` or `json_payload` feature makes this ~1.3-~1.5x slower.
                }
use std;
        MergeTestState::new(2).run_script(&script[..]);

            }
        }
    fn merge_idempotent() {
                return None;
        update
        let d1 = Delta::simple_edit(Interval::new_closed_open(0,0), Rope::from("c"), TEST_STR.len());
            AssertMaxUndoSoFar(1,3),
use serde::ser::{Serialize, Serializer, SerializeStruct, SerializeTupleVariant};
            MetadataType::ProcessLabels {..} => (None, None)
        assert_eq!(samples[0].name, "remote pid");
        // 1 MB
                            edit: Undo {
    }
                MergeTestOp::Assert(ei, ref correct) => {
    pub fn start_drag(&mut self, offset: usize, min: usize, max: usize) {
        let utf8_offset = rope_with_emoji.convert_metrics::<Utf16CodeUnitsMetric, BaseMetric>(utf16_units);
        {
        }
        for chunk in self.iter_chunks(start, end) {
    /// Get next codepoint after cursor position, and advance cursor.
/// The payload associated with any sample is by default a string but may be
        assert_eq!(soln, String::from(engine.get_head()));
        assert_eq!(snapshot[0].name, "process_name");
            assert_eq!(Some((i - 1) / 8 * 8), a.prev_grapheme_offset(i));
        self.select_region(text, offset, SelRegion::new(start, end), multi_select);
        -> Self
            Assert(5, "zacpb".to_owned()),
    /// Collapse all selections in this view into a single caret
            rendered_styles.push((sel_start as isize) - ix);
    }
            cur_len_utf8 += u.len_utf8();
    fn to_payload(value: &'static str) -> TracePayloadT {
            return self.len();
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    pub(crate) fn has_pending_render(&self) -> bool {


/// Stores the relevant data about a sample for later serialization.
                if (flags & FLAG_SELECT) != 0 {
        fn run_script(&mut self, script: &[MergeTestOp]) {
//
            Edit { ei: 0, p: 3, u: 1, d: parse_delta("-c-") },
///
        let mut leaf_offset = pos - offset;
            end,
        let first_rev = engine.get_head_rev_id().token();
    /// What kind of sample this is.
        --#--
        let new_head = engine.get_head_rev_id().token();
        let region = SelRegion::caret(offset);
    pub fn with_limit_bytes(size: usize) -> Self {
            Some(region) => {
    #[test]

        };
        }


            sample: None,
                        deletes_from_union = deletes_from_union.union(deletes);
    /// Returns a tuple of a new `Revision` representing the edit based on the
use xi_rope::tree::Cursor;
    idx: usize,
        }
        }
}
    use std::collections::BTreeSet;
        };
}
    }
        }
        assert_eq!(String::from(&a).lines().collect::<Vec<_>>(),
            &rebased_deletes_from_union, &new_deletes_from_union);
pub fn trace_block<'a, S, C>(name: S, categories: C) -> SampleGuard<'a>
mod fixed_lifo_deque;

    }
    pub(crate) fn get_replace(&self) -> Option<Replace> {
            Edit { ei: 2, p: 1, u: 1, d: parse_delta("!-") },
/// Tracing is disabled initially on program launch.
        self.deletes_from_union = new_deletes_from_union;
    #[test]
            rev_id: RevId { session1: 0, session2: 0, num: 0 },
        let first = max(first, 0) as usize;
    pub fn render_if_dirty(&mut self, text: &Rope, client: &Client,
                    priority, undo_group,
        assert_eq!(rope.offset_of_line(0), 0);
use rope::{Rope, RopeInfo};
        }
        if let Some(occ) = closest_occurrence {
        #------
        let new_sel = self.drag_state.as_ref().map(|drag_state| {
        rope.offset_of_line(5);
    {
    // println!("shuffle: old={:?} new={:?} old_text={:?} new_text={:?} old_tombstones={:?}",
            let b = s.as_bytes()[offset];



/// associated performance hit across the board for turning it on).
    /// indicates a search for the next occurrence past the end of the file.
                    // TODO: update (rather than re-render) in cases of text valid
    // A colliding session will break merge invariants and the document will start crashing Xi.
        self.prev::<BaseMetric>();
        // we add the special find highlights (1) and selection (0) styles first.
        for op in &delta_ops {
            Assert(2, "ab".to_owned()),
                        b.add_span(seg.n, seg.our_line_num, line_cache_shadow::ALL_VALID);
    }
pub type TracePayloadT = std::collections::HashMap<StrCow, StrCow>;
            Merge(0,2),
                        first_sample_timestamp,
            match c.edit {
    #[test]
    }
/// feature makes the payload a `serde_json::Value` (additionally the library
/// Create a `Rope` from a `String`:
}
    pub fn token(&self) -> RevToken {
    None,
                        priority, undo_group,
                    let head = e.get_head_rev_id().token();
        result
///

            let inserted = inserts.inserted_subset();
    pub fn push_str(&mut self, mut s: &str) {
/// # Arguments
    }
        };
                        if !last_selection.is_caret() {
    }
            utf16_size: 0,
        use self::ViewEvent::*;
        });
    /// The return type is a `Cow<str>`, and in most cases the lines are slices borrowed
pub struct ChunkIter<'a> {
                },
        trace.instant("3", &["test"]);
                    if !deletes.is_empty() {
#[inline]
        });



/// Enable tracing with the default configuration.  See Config::default.
impl PartialEq for CategoriesT {
use std::ops::Range;
        assert_eq!(None, a.next_codepoint_offset(10));
}
        while !s.is_empty() {
        let utf8_offset = rope.convert_metrics::<Utf16CodeUnitsMetric, BaseMetric>(utf16_units);
        Some(self.cmp(other))
    #[test]
    fn merge_whiteboard() {
    first_line: usize,
        }
    }
                              to_payload("test_get_samples"));
    }
    /// they would be if they hadn't been deleted.
use delta::{Delta, DeltaElement};

    pub fn new_instant<S, C>(name: S, categories: C,
        d_builder.delete(Interval::new_closed_open(10, 36));
                self.do_set_replace(chars, preserve_case),

            } else if incomp == GraphemeIncomplete::PrevChunk {


        if wrap_col > 0 {
        // TODO: refine for upstream (caret appears on prev line)
        let cur_deletes_from_union = &self.deletes_from_union;

    fn eq_med() {
        assert_eq!(vec!["a\n", "b\n", "c"], a.lines_raw_all().collect::<Vec<_>>());
                    }
// Unless required by applicable law or agreed to in writing, software
        let script = vec![


fn ns_to_us(ns: u64) -> u64 {
        assert_eq!("0123456789abcDEEFghijklmnopqr999stuvz", String::from(engine.get_head()));
                        ops.push(self.build_update_op("copy", None, seg.n));

            // d-expand by other
            // selection with interior
    #[test]
impl RevId {

    fn lines_small() {
        }
        let gc : BTreeSet<usize> = [1].iter().cloned().collect();
        ");
                    Some(Contents::Edit {
pub struct Utf16CodeUnitsMetric(usize);
const MIN_LEAF: usize = 511;
    // TODO: switch to a persistent Set representation to avoid O(n) copying
    }
/// conversions in both directions.
        let trace = Trace::enabled(Config::with_limit_count(10));
        if offset == 0 {

        assert_eq!("#---#--", &subset_str);
                } else {
            find: Vec::new(),

// ======== Generic helpers
            None

        true
    /// When merging between multiple concurrently-editing sessions, each session should have a unique ID
        assert!(r.clone().slice(a.len(), r.len()) == b_rope);
    inner: LinesRaw<'a>
            inserts = inserts.transform_expand(trans_inserts, after);
    /// Only number of matches changed
        let mut revs_2 = basic_insert_ops(inserts_2, 4);

    #[serde(rename = "cat")]
    /// Starts a drag operation.
        if !self.is_enabled() {
enum Contents {
            MetadataType::ProcessName {..} => "process_name",
}
                },
            Assert(1, "abefg".to_owned()),
    }
    #[bench]
    /// from the rope.
        MergeTestState::new(2).run_script(&script[..]);

        struct CategoriesTVisitor;
    }
                    if seg.validity == line_cache_shadow::ALL_VALID {
            if let Some(pos) = pos { start_of_line.set(pos) }
            base_rev: RevToken, delta: Delta<RopeInfo>) {
            Assert(1, "arpb".to_owned()),
    /// Tracks whether there has been changes in find results or find parameters.
    pub fn enable(&self) {
    fn is_boundary(s: &String, offset: usize) -> bool {
        guard
    session_id: SessionId,
/// 2. Performance optimizations might have per-thread buffers.  Keeping all
                                  _allow_same: bool, modify_selection: &SelectionModifier) {
                let prio = FullPriority { priority, session_id: r.rev_id.session_id() };


        let mut result = json!({
        let (start, end) = {
        }

            Merge(0,2), Merge(1, 2),
        }
                self.highlight_find = visible;
    // * Unicode codepoints
    #[test]
/// a.edit_str(1, 9, "era");
            Edit { ei: 0, p: 1, u: 1, d: parse_delta("ab") },
    // encourage callers to use Cursor instead?
        let revs = basic_insert_ops(inserts, 1);
        let iv = Interval::new_closed_open(start, end);
    {
        for &(sel_start, sel_end) in hls {
/// ```
        let style_spans = style_spans.subseq(Interval::new_closed_open(start, end));
                if is_base {
        // rebase insertions on text and apply
/// See `trace_payload` for a more complete discussion.
            Merge(0,2), Merge(1, 2),
    /// Determines the current number of find results and search parameters to send them to
        engine.undo([3].iter().cloned().collect());
    }
    }
            let start = self.line_of_offset(last_text, iv.start());
        let mut sel = Selection::new();
}
//
                        let n_skip = seg.their_line_num - line_num;
    {

            SampleGuard::new_disabled()
        for i in (s1.len() * 3 + 1)..(s1.len() * 3 + 4) {
            CategoriesT::DynamicArray(ref self_arr) => {
            enabled: AtomicBool::new(true),

        let mut cursor = Cursor::new(self, offset);
            AssertAll("ac".to_owned()),
/// However, anything that can be converted into a Cow string can be passed as
        -----#

    fn bench_trace_closure_payload(b: &mut Bencher) {
    out
                        DeltaElement::Copy(start, end),
 //additional cursor features
        let utf16_units = rope.convert_metrics::<BaseMetric, Utf16CodeUnitsMetric>(utf8_offset);
                }
    pub fn get_samples_limit(&self) -> usize {
        self.revs.append(&mut new_revs);
}
        }

    out.as_mut_slice().reverse();

            self.tid == other.tid &&
            Assert(2, "ab".to_owned()),
            trace.instant_payload("a", &["test"], to_payload("test_get_sorted_samples"));
            Copy(usize, usize),
}
        let inserts_1 = parse_subset_list("
    // TODO(vlovich): Replace all of this with serde flatten + rename once
                }
            let _ = trace.block("test_samples_block", &["test"]);
            Edit { ei: 0, p: 3, u: 1, d: parse_delta("---d") },
pub fn samples_cloned_sorted() -> Vec<Sample> {
/// assert_eq!("herald", String::from(a));
        }
                    let b = if bi < ai {

fn exe_name() -> Option<String> {
                            ops.push(self.build_update_op("skip", None, n_skip));
            Merge(0,2), Merge(1, 2),
        b.iter(|| black_box(time::precise_time_ns()));
// limitations under the License.
            let sel_end_ix = clamp(region.max(), start_pos, pos) - start_pos;
    pub timestamp_us: u64,
/// as opposed to thousands of revisions.
        let d = engine.delta_rev_head(first_rev);
            SampleEventType::AsyncStart => 'b',

    ///

    #[cfg(feature = "benchmarks")]
            write!(f, "{}", String::from(self))


    }
        let r = r + Rope::from(String::from(&a[MAX_LEAF..]) + &b[..MIN_LEAF]);
        delta.serialize_field("els", &self.els)?;
            trace.closure("test_samples_closure", &["test"], || {});
    fn test_disable_drops_all_samples() {
        assert!(a != empty);
    }
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
            .map_or(0, |ref s| s.timestamp_us);
        }
            self.record(Sample::new_instant(name, categories, None));
/// ```rust
            }
impl PartialEq for Sample {
                    let after = new_full_priority >= full_priority;  // should never be ==
        engine.edit_rev(1, 1, first_rev, d1.clone());
    /// ancestor in order to be mergeable.
            Merge(1,2),
        // We add selection after find because we want it to be preferred if the

}
            result["cursor"] = json!(cursors);

        }
        let ix = style_map.add(style);
            // Do the merge one direction
        if self.is_enabled() {
        ]);
        if s.len() <= MAX_LEAF {
//!
    /// Invalidates front-end's entire line cache, forcing a full render at the next
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Move the selection by the given movement. Return value is the offset of
        let trace = Trace::disabled();
#[cfg(feature = "benchmarks")]
                last.1 = last.1.transform_union(&inserts);
))]
        let a = Rope::from("\n");
#[derive(Serialize, Deserialize, Debug)]
        }
            event_type: event_type,
    fn delta_rev_head_2() {
        Self {
        self.lc_shadow = b.build();
/// ```

    {
            name: sample_name,
    }

                    ops.push(self.build_update_op("invalidate", None, seg.n));
        self.buffer_id
        let mut selections = Vec::new();
    {

                self.set_selection_raw(text, selection);

/// discussion on payload.
}
/// configured via the `dict_payload` or `json_payload` features (there is an
                RenderTactic::Render => {
        }
        let first_rev = engine.get_head_rev_id().token();
        let trace = Trace::enabled(Config::default());
}
        #[derive(Serialize, Deserialize)]
                None => {
        }

    (1, 0)
        assert_eq!(utf16_units, 20);
    }
/// ```
    fn prev_codepoint_offset_small() {

// limitations under the License.
            } else {
impl FromStr for Rope {
    /// of the chunks is indeterminate but for large strings will generally be
    fn bench_trace_closure_disabled(b: &mut Bencher) {
    }
    use serde_test::{Token, assert_tokens};
    }
    ///
    }
        
            'e' => SampleEventType::AsyncEnd,
    #[cfg(feature = "benchmarks")]

        MergeTestState::new(6).run_script(&script[..]);

            Assert(0, "b".to_owned()),
    }
        if self.len() != other.len() {
        assert_eq!(Some(1), a.next_codepoint_offset(0));
        });
                Contents::Edit {inserts, ..} => inserts,
categories_from_constant_array!(7);
    /// Time complexity: O(log n)
}
///
        // move deleted or undone-inserted things from text to tombstones
    #[serde(serialize_with = "serialize_event_type")]
impl<'a> Iterator for Lines<'a> {
    }
        let toggled_groups = self.undone_groups.symmetric_difference(&groups).cloned().collect();
                peer.set_session_id(((i*1000) as u64, 0));
                                deletes.transform_shrink(&gc_dels))
        match self {
            word_cursor.select_word()
                                                     "insert", node)
    }
        assert_eq!(0, a.line_of_offset(0));
    if a.is_empty() {
                    union_ins_delta = union_ins_delta.transform_expand(inserts, after);

    pub fn select_all(&mut self, text: &Rope) {
            self.args == other.args

/// that sorted would be prohibitively expensive.
        engine.edit_rev(1, 1, initial_rev, d1.clone());
    /// Does a drag gesture, setting the selection from a combination of the drag
        // additional tests for line indexing
        assert_eq!(snapshot[8].name, "x");
        let mut b = String::new();
            Edit { ei: 3, p: 7, u: 1, d: parse_delta("z--") },
    }

    #[serde(skip_serializing_if = "Option::is_none")]
    }
    breaks: Option<Breaks>,
            Edit { ei: 2, p: 4, u: 1, d: parse_delta("-r-") },
                while cursor.pos() < region.max() {
            while !s.is_char_boundary(offset - len) {
    use super::*;
            let deletes = Subset::new(inserts.len());
    }
        }
        assert_eq!(Some(9), a.next_grapheme_offset(3));
    }
    /// This method is responsible for updating the cursors, and also for
fn initial_revision_counter() -> u32 {
    /// Like the scanned whiteboard diagram I have, but without deleting 'a'
use find::Find;
        if self.scroll_to.is_none() && wrap {
        impl From<RopeDeltaElement_> for DeltaElement<RopeInfo> {
        ];
    }
                    if s.ends_with('\r') {
/// ```
    }
        let d1 = Delta::simple_edit(Interval::new_closed_open(0,0), Rope::from("a"), TEST_STR.len());
            Assert(0, "acbd".to_owned()),
        self.undone_groups = groups;
        impl<'de> Visitor<'de> for CategoriesTVisitor {
            b.push('b');
            if cur_len_utf16 >= in_measured_units {

            Some(ref breaks) => {
    }
        let next_line_offset = self.offset_of_line(text, line + 1);

    let mut out = Vec::with_capacity(b_new.len());

                        undone_groups = Cow::Owned(new_undone);
        samples[0].pid = 1;
    assert!(!a.is_empty() && !b.is_empty());
                    // TODO: in the case where it's ALL_VALID & !CURSOR_VALID, and cursors
    #[test]
        }
pub struct Replace {
//! An engine for handling edits (possibly from async sources) and undo. It


// distributed under the License is distributed on an "AS IS" BASIS,
    TRACE.block(name, categories)
        }
            black_box(time::precise_time_ns());
        where S: serde::Serializer
        assert_eq!(snapshot[2].name, "x");
    }
        }
    fn find_rev_token(&self, rev_token: RevToken) -> Option<usize> {
            samples: Mutex::new(FixedLifoDeque::new())
            let mut len = 1;

        ];
    /// the heap, counters, etc); just the data itself.
        let mut rendered_styles = Vec::new();
        // set last selection or word under current cursor as search query
    }
        -> R
        let d2 = Delta::simple_edit(Interval::new_closed_open(0,0), Rope::from("a"), TEST_STR.len()+1);
    /// New offset to be scrolled into position after an edit.
            } else {
            line_cache_shadow::CURSOR_VALID | line_cache_shadow::STYLES_VALID
        let utf8_offset = rope_with_emoji.convert_metrics::<Utf16CodeUnitsMetric, BaseMetric>(utf16_units);
enum MetadataType {
                el.serialize_field(start)?;

        assert_eq!(None, a.next_grapheme_offset(s1.len() * 3 + 4));
            trace.instant("c", &["test"]);
                    if retain_revs.contains(&rev.rev_id) {
                                     categories: C,

    #[serde(rename = "sort_index")]
    /// An iterator over the raw lines. The lines, except the last, include the

///
        let mut c = GraphemeCursor::new(pos, self.total_len(), true);

        let end = time::precise_time_ns();


        trace.record(guard.sample.as_ref().unwrap().clone());
fn cow_append<'a>(a: Cow<'a, str>, b: &'a str) -> Cow<'a, str> {
    pub fn slice_to_string(&self, start: usize, end: usize) -> String {
    #[cfg(feature = "benchmarks")]
}
        engine.edit_rev(1, 1, first_rev, d1);
    fn merge_simple_delete_1() {
/// {
                selection.add_region(SelRegion::caret(region.max()));
        assert_eq!(String::from(engine.get_head()), d.apply_to_string("0123456789abcDEEFghijklmnopqr999stuvz"));
                            }
    let move_delta = Delta::synthesize(text, &inverse_tombstones_map, &new_deletes_from_union.complement());
        // it will be immediately transform_expanded by inserts if it is an Edit, so length must be before
        assert_eq!(vec!["a\n", "b\n"], a.lines_raw_all().collect::<Vec<_>>());
            Assert(0, "ab".to_owned()),
    /// `text`. The count of a character in `deletes_from_union` represents
        (line, offset - self.offset_of_line(text, line))
    }
    fn gc_6() {
          F: FnOnce() -> R
                    }
    }
                formatter.write_str("comma-separated strings")
    fn lines_raw_small() {
        let mut deletes_from_union = self.deletes_from_union_for_index(rev_index);
            update["lines"] = json!(lines);
        d_builder.build()
        ---#--
/// ```

            for rev in &self.revs {
impl Add<Rope> for Rope {
        ids.iter().cloned().map(|i| {
        let mut soln = String::from("h");
    fn basic_rev(i: usize) -> RevId {
        len
//! A rope data structure with a line count metric and (soon) other useful
        }
    }
    }
pub fn trace_payload<S, C, P>(name: S, categories: C, payload: P)
    }
        assert_eq!(snapshot.len(), 9);
                                deletes,

        }

        // if self.end is inside this chunk, verify that it is a codepoint boundary
    }
            self.next::<BaseMetric>();
        assert_eq!(trace.get_samples_limit(), 11);
    fn prev(s: &String, offset: usize) -> Option<usize> {
        *self = b.build();
    }
            };
        let mut result = String::new();
    for op in b_new {

        let script = vec![

                        Some(end) if end >= region.max() => max(0, region.max() - 1),

        // set last selection or word under current cursor as replacement string
        let (leaf, pos) = cursor.get_leaf().unwrap();
    fn arr_eq(&self, other: &Vec<String>) -> bool {

        // selection state, and for scrolling it into view if needed. This choice can
            SampleEventType::ObjectCreated => 'N',
///
        let a: Vec<Revision> = ids_to_fake_revs(&[0,2,4,6,8,10,12]);
        let rebased_inserts: Vec<Subset> = revs.into_iter().map(|c| {
            Merge(2,0),
            sample: Some(Sample::new_duration_marker(
        }
fn clamp(x: usize, min: usize, max: usize) -> usize {
                match elem {
        let script = vec![
                }
        b.push(self);
        let first_line = self.line_of_offset(text, start);
            let mut sample = self.sample.take().unwrap();
            if let GraphemeIncomplete::PreContext(_) = incomp {

        assert_eq!(5, a.offset_of_line(3));
    }
    }
    }
    /// Returns the regions of the current selection.
                for &region in rest {
    #[test]
{
    struct MergeTestState {
        gc_scenario(4,3);
                let (last, rest) = self.sel_regions().split_last().unwrap();
}
        }, to_payload("test_get_sorted_samples"));
/// assert!("hello world" == String::from(a + b));
    }
        }
        self.find_changed = FindStatusChange::All;
// Unless required by applicable law or agreed to in writing, software
        // NOTE: we derive to an interim representation and then convert
pub fn samples_cloned_unsorted() -> Vec<Sample> {
        assert_eq!("a0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", String::from(engine.get_head()));


            Merge(0,1),
        self.drag_state = None;
        //for next line
    {
            client.scroll_to(self.view_id, line, col);
        undo_test(false, [1].iter().cloned().collect(), "0!3456789abcdefGIjklmnopqr888stuvwHIyz");
    pub fn prev_codepoint(&mut self) -> Option<char> {
            }
    pub name: StrCow,
    new_without_default_derive,

impl Ord for Sample {
    pub fn next_utf8_chunk_in_leaf(&mut self, chunk_size: usize) -> &'a str {
    #[test]

        deserializer.deserialize_str(CategoriesTVisitor)
    }
    #[cfg(feature = "benchmarks")]
            rendered_styles.push(style_id as isize);
                    linewrap::rewrap_width(breaks, text, width_cache,
            if self.line_of_offset(text, offset) <= line {
            black_box(trace.block_payload(

            // cursor
    ProcessSortIndex { sort_index: i32 },
    /// This function will panic if `offset > self.len()`. Callers are expected to

                Rope::from(String::from(s1.clone()) + "\u{1f1fa}"),
            }
            }

    }
}
        cursor.next::<BaseMetric>()
                            (inserts, deletes)
    fn do_selection_for_replace(&mut self, text: &Rope) {
    /// Generate line breaks based on width measurement. Currently batch-mode,
use std::collections::HashMap;
        for i in 0..self.len() {
        };
        for i in 0..self.len() {
                        let mut word_cursor = WordCursor::new(text, region.max());
        let mut end = start_pos + len;
    ///
        let trace = Trace::enabled(Config::default());
    /// Selects a specific range (eg. when the user performs SHIFT + click).


        thread.name().map(|ref s| to_cow_str(s.to_string()))
    FlowStart,
            AssertMaxUndoSoFar(1,1),
    }
    /// For safety, this will panic if any revisions have already been added to the Engine.
        let head_rev = &self.revs.last().unwrap();
        assert_eq!(trace.samples_cloned_unsorted().len(), 3);
            }
        self.lines(0, self.len())
        let (leaf, offset) = match self.get_leaf() {
                    Some(chunk) => self.fragment = chunk,
            find.unset();
    /// Find the first revision that could be affected by toggling a set of undo groups
        let len = min(self.end - self.cursor.pos(), leaf.len() - start_pos);
    offset: usize,
///
                            max_undo_so_far: rev.max_undo_so_far,
        // original values this test was created/found with in the UI:
    {
    /// The `pristine` argument indicates whether or not the buffer has
/// A guard that when dropped will update the Sample with the timestamp & then
                    })

                } else {
        let rearranged_inserts: Vec<Subset> = rearranged.into_iter().map(|c| {
    /// Time complexity: technically O(n log n), but the constant factor is so
        ix
    /// next occurrence before (`true`) or after (`false`) the last cursor is selected. `wrapped`
        };
        assert_eq!(snapshot[1].name, "thread_name");

use xi_rope::interval::Interval;
    bytecount::count(s.as_bytes(), b'\n')
//
    }
/// record it.
            }
        let d1 = Delta::simple_edit(Interval::new_closed_open(0,0), Rope::from("h"), engine.get_head().len());
                rev_id: basic_rev(i+1),
    AsyncEnd,
        let (new_text, new_tombstones) =
            Some((l, off)) => (l, off),
        let (mut l, mut offset) = self.get_leaf()?;

    /// Splits current selections into lines.
    fn drop(&mut self) {
    fn add_selection_by_movement(&mut self, text: &Rope, movement: Movement) {
/// bytes required to represent the codepoint.
            // On the next step we want things in expand_by to have op in the context
        let inserts = parse_subset_list("
    /// This is a `Subset` of the "union string" representing the characters
                break;
    pub fn do_find_all(&mut self, text: &Rope) {
              F: FnOnce() -> R
        engine.edit_rev(1, 1, first_rev, build_delta_1());
        // todo: this will be changed once multiple queries are supported
    fn test_get_sorted_samples() {
        engine.edit_rev(1, 1, first_rev, d1.clone());
/// between two utf8 code units that form a code point is considered invalid.
    #[test]
        let rev = Revision {

}
        ]);
    fn test_trace_disabled() {
                } else if click_count == Some(2) {
            SampleGuard::new(&self, name, categories, Some(payload.into()))
        let full_priority = FullPriority { priority, session_id: rev_id.session_id() };

        }
    pub fn samples_cloned_sorted(&self) -> Vec<Sample> {
    }
        assert_eq!(1, a.line_of_offset(3));
                        b.add_span(seg.n, 0, 0);
            Token::String("a\u{00A1}\u{4E00}\u{1F4A9}"),
    #[cfg(feature = "benchmarks")]
        deletes: Subset,
extern crate serde_derive;
    /// assumed to be forward).
        trace.closure_payload("x", &["test"], || (),
            // corner and I'm lazy -cmyr
    }
            'n' => SampleEventType::AsyncInstant,
}
    /// Get the contents of the document at a given revision number
        let result = closure();
        engine.undo([1,2].iter().cloned().collect());
        engine.edit_rev(0, 2, first_rev, build_delta_2());
/// With `json_payload` feature:
    find_changed: FindStatusChange,
        }
            GestureType::MultiLineSelect =>
        assert_eq!(snapshot[0].name, "process_name");
                }
            "text": &l_str,

                    }
    /// current head, a new text `Rope`, a new tombstones `Rope` and a new `deletes_from_union`.
            );
    }
    #[test]
        // todo: the correct Find instance needs to be updated with the new parameters
    selection: Selection,
impl<'a> Drop for SampleGuard<'a> {
        // x, a, y, b, z, c
    }
            .map_or_else(|| sys_tid::current_tid().unwrap(), |ref s| s.tid);
    samples: Mutex<FixedLifoDeque<Sample>>,
                }
                    let full_priority = FullPriority { priority, session_id: r.rev_id.session_id() };
use std::fmt;
/// # Returns
            if let Edit { priority, ref inserts, .. } = r.edit {


/// beginning.
        let rope_with_emoji = Rope::from("hi\ni'm\n😀 four\nlines");
#[derive(Clone, Debug, PartialEq, Eq)]
        Engine {
                        deletes_from_union = deletes_from_union.transform_union(inserts);
    fn invalidate_selection(&mut self, text: &Rope) {
    /// won't be exceeded by the underlying storage itself (i.e. rounds down).
    /// The semantics are intended to match `str::lines()`.
            Assert(0, "acbd".to_owned()),
    }
            's' => SampleEventType::FlowStart,
        assert_eq!(String::from(&a).lines().collect::<Vec<_>>(),
            't' => SampleEventType::FlowInstant,
/// Boundary is atomic and determined by codepoint boundary.

        assert_eq!(String::from(engine.get_head()), d.apply_to_string(TEST_STR));
}
    }
        assert_eq!("-#-----", format!("{:#?}", deletes_from_union_2));
    // Thus, it's easiest to defer gc to when all plugins quiesce, but it's certainly
        ");
{
            }

        assert!(a == a2);
mod tests {
        let a = Rope::from("");
        // TODO: simplify this through the use of scopeguard crate
    fn get_or_def_style_id(&self, client: &Client, style_map: &StyleMap,
            Edit { ei: 2, p: 1, u: 1, d: parse_delta("ab") },
    /// Callers are expected to validate their input.
{
    fn undo_4() {
            Some(ref breaks) => {
struct DragState {
            type Value = CategoriesT;
            event_type: SampleEventType::CompleteDuration,
        assert_eq!(snapshot[2].name, "x");
    }
/// See `trace_closure` for how the closure works and `trace_payload` for a
    /// The empty string will yield a single empty slice. In all other cases, the
    /// that are currently deleted, and thus in `tombstones` rather than
            Merge(0,2), Merge(1, 2), Merge(3, 2),
    }

        assert_eq!(trace.samples_cloned_unsorted().len(), 0);
/// ```
    // maybe explore grabbing leaf? would require api in tree
macro_rules! categories_from_constant_array {


    }
        let new_head_2 = engine.get_head_rev_id().token();

    }
                        deletes_from_union
        assert_eq!(Some(9), b.next_codepoint_offset(5));
            }),
    fn bench_trace_closure(b: &mut Bencher) {
            Find { chars, case_sensitive, regex, whole_words } =>
        if !new_inserts.is_empty() {
        self.lines += other.lines;
use interval::Interval;
        let trace = Trace::enabled(Config::with_limit_count(10));
                // Deprecated (kept for client compatibility):
        let b: Vec<Revision> = ids_to_fake_revs(&[0,1,2,4,5,8,9]);
                    } else {
    pub fn undo(&mut self, groups: BTreeSet<usize>) {
    priority: usize,
            undone_groups: BTreeSet::new(),
            thread_name: Sample::thread_name(),
                    e.edit_rev(p, u, head, delta.clone());
        /// should go before typed text.
            max_undo_so_far: 0,
    // head revision, a token or a revision ID. Efficiency loss of token is negligible but unfortunate.
        let last_line = self.line_of_offset(text, self.selection.last().unwrap().max()) + 1;
/// the `base_revs`. This allows the rest of the merge to operate on only

    /// recomputing line wraps.
        let first_rev = engine.get_head_rev_id().token();
}
        if all_samples.is_empty() {
}
    where S: Into<StrCow>, C: Into<CategoriesT>, P: Into<TracePayloadT>,
impl Trace {

        let a = Rope::from("a\nb\n");
            match c.edit {
            enabled: AtomicBool::new(false),
        let tid = all_samples.front()


    // possible to fix it so that's not necessary.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
/// Indicates what changed in the find state.
                        }
                            }
        b.iter(|| {
            SampleEventType::ObjectSnapshot => 'O',

    }
            base_len: usize
    {
    pub(crate) fn rewrap(&mut self, text: &Rope, wrap_col: usize) {
            },
            Assert(1, "ab".to_owned()),
            match memchr(b'\n', self.fragment.as_bytes()) {



        // a, b, y, z, c, x
                    s = inserts.transform_union(&s);


    use delta::{Builder, Delta};
    fn to_base_units(s: &String, in_measured_units: usize) -> usize {
        assert_eq!(2, expand_by.len());
            args: Some(SampleArgs {
    /// Get revision id of head revision.
            edit: Undo { toggled_groups, deletes_bitxor }

pub type RopeDeltaElement = DeltaElement<RopeInfo>;
/// overhead tracing routine available.
        undo_test(false, [1,2].iter().cloned().collect(), TEST_STR);
    }
        let styles = self.render_styles(client, styles, start_pos, pos,
    /// This is a regression test to ensure that session IDs are used to break
/// # Arguments
        where S: Serializer
}
            rendered_styles.push(sel_end as isize - sel_start as isize);
            }
                _ => panic!("to_base_units called with arg too large")
    use super::*;
            ix = sel_end as isize;

        ]);
            for (i,rev) in self.revs.iter().enumerate().rev() {
            Assert(1, "arpbzj".to_owned()),

        /// together. For example, an auto-indent insertion would be un-done
        let trace = Trace::enabled(Config::default());

    /// The return type is a `Cow<str>`, and in most cases the lines are slices
///
                first_sample_timestamp,
                    }
        // rebase the deletion to be after the inserts instead of directly on the head union

}

// You may obtain a copy of the License at
    #[inline]
        let contents = Contents::Edit {
            duration_us: None,
    /// part of the initial contents since any two `Engine`s need a common
            }
    #[test]

/// * `name` - A string that provides some meaningful name to this sample.
    /// Generally racy since the underlying storage might be mutated in a separate thread.
use serde_json::Value;
        in_measured_units
// See the License for the specific language governing permissions and
        client.update_view(self.view_id, &params);

            let horiz = None;
    }
pub fn is_enabled() -> bool {
        } else {
        self.breaks = Some(linewrap::linewrap_width(text, width_cache,
            edit: Edit {
    fn test_samples_pulse() {
/// # Examples
                    s = s.transform_shrink(&transformed_inserts);

            Edit { ei: 0, p: 3, u: 1, d: parse_delta("-c-") },
                                                        style_spans, line_num);

pub struct LinesMetric(usize);  // number of lines
        Some(pos) => minsplit + pos,
        assert_eq!(output, String::from(engine.get_head()));
    fn from(c: Vec<String>) -> CategoriesT {
    }
            "pristine": pristine,
    /// Each chunk is a `&str` slice borrowed from the rope's storage. The size
/// implementation as efficient as a mutable version.
    ///
            if let Edit { ref undo_group, ref inserts, ref deletes, .. } = rev.edit {
}
    where S: Into<StrCow>, C: Into<CategoriesT>, P: Into<TracePayloadT>
        assert_tokens(&rope, &[
#[macro_use]
            self.idx = 0;
        }
    }
            let delta = Delta::simple_edit(Interval::new_closed_closed(0,0), initial_contents, 0);
            let end = self.line_of_offset(last_text, iv.end()) + 1;

        assert_eq!(snapshot[7].name, "c");

//! under `Engine::edit_rev`, which is considerably simpler than the usual

            new_deletes = new_deletes.transform_expand(&new_inserts);
            None => return "",

            Assert(0, "cbd".to_owned()),

    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
/// monotonically increasing single global integer (when tracing) per creation
        let trace = Trace::enabled(Config::with_limit_count(10));

        cursor.next_grapheme()
    fn bench_trace_instant_with_payload(b: &mut Bencher) {
}
}
    sample: Option<Sample>,
/// });
            Edit { ei: 0, p: 1, u: 2, d: parse_delta("!-") },
        leaf.as_bytes()[pos]
        if !cursors.is_empty() {
    fn is_boundary(s: &String, offset: usize) -> bool {
        serializer.serialize_str(&String::from(self))
use self::Contents::*;
        self.timestamp_us.cmp(&other.timestamp_us)
    }
            priority: 0,
    /// Note: `edit` and `edit_str` may be merged, using traits.
/// ```
            sel.add_region(
        let (first_line, _) = self.offset_to_line_col(text, region.min());

    }
/// When compiling with `dict_payload` or `json_payload`, this is ~2.1x slower
//! `Engine::merge`, which is more powerful but considerably more complex.
                    a.merge(b);
        let (ins_delta, deletes) = delta.factor();

    /// If the cursor is at the end of the rope, returns the empty string.
                    return Some(result);
        engine.undo([1,3].iter().cloned().collect());
        // rather than emitting an explicit start/stop to reduce the size of
    {
/// The result of the closure.
    pub fn max_undo_group_id(&self) -> usize {
    pub fn instant<S, C>(&self, name: S, categories: C)
                    let transformed_inserts = inserts.transform_expand(&s);
        let subset_str = format!("{:#?}", expand_by[0].1);
        AssertAll(String),
        self.cursor.next_leaf();
            rendered_styles.push(sel_end as isize - sel_start as isize);
                            rendered_lines.push(line);
                match *other {
    ///
    /// tiny it is effectively O(n). This iterator does not allocate.
    TRACE.instant_payload(name, categories, payload);
    /// Update front-end with any changes to view since the last time sent.
                edit: contents.clone()
    drag_state: Option<DragState>,
        }
            SelectionIntoLines => self.do_split_selection_into_lines(text),


        --#--
    }
    }
            lc_shadow: LineCacheShadow::default(),
        match self.selection.len() {
}
                    let transformed_deletes = deletes.transform_expand(&s);
/// Find a set of revisions common to both lists
        trace.closure_payload("z", &["test"], || {},
        d_builder.replace(Interval::new_closed_open(54, 54), Rope::from("888"));

        }
            Some(b)
            Assert(0, "ab".to_owned()),
pub struct BaseMetric(());
        b.iter(|| black_box(trace.block("something", &["benchmark"])));

                //self.cursor.next::<BaseMetric>().unwrap() - prev_pos
        let mut prev_boundary = c.prev_boundary(&l, leaf_offset);
                            let line = self.render_line(client, styles, text,
            Revision {
                WrapWidth::Bytes(col) => linewrap::rewrap(breaks, text, iv,
    pub(crate) fn record(&self, sample: Sample) {
            Edit { ei: 0, p: 3, u: 1, d: parse_delta("-c-") },
                undo_group,
                SelRegion::new(start, end)
    fn undo_3() {
        }
        for rev in &self.revs[rev_index + 1..] {
    pub fn closure_payload<S, C, P, F, R>(&self, name: S, categories: C,
    where S: Into<StrCow>, C: Into<CategoriesT>, P: Into<TracePayloadT>

        }
            return false;

        self.samples.lock().unwrap().limit()
//!

        inserts.into_iter().enumerate().map(|(i, inserts)| {
///
    /// inserted, including the ones that were later deleted, in the locations

            Edit { ei: 0, p: 1, u: 1, d: parse_delta("ab") },
        AssertMaxUndoSoFar(usize, usize),
    // TODO: maybe switch to using a revision index for `base_rev` once we disable GC
        for find in &mut self.find {
        d_builder.build()
        assert_eq!(snapshot[4].name, "y");

        b.iter(|| sys_pid::current_pid());
        let (mut new_revs, text, tombstones, deletes_from_union) = {
            fn from(c: &'static[&'static str; $num_args]) -> CategoriesT {
}
            let new_trans_inserts = trans_inserts.transform_expand(&inserted);
    fn next(s: &String, offset: usize) -> Option<usize> {
    #[test]

    fn test_ser_de() {

        self.wrap_col = WrapWidth::Width(self.size.width);
            offset -= 1;
//
            while !s.is_char_boundary(splitpoint) {
                        if n_skip > 0 {
    pub fn set_selection<S: Into<Selection>>(&mut self, text: &Rope, sel: S) {
        if offset == s.len() {
impl Metric<RopeInfo> for LinesMetric {
                if groups.contains(undo_group) {
use line_cache_shadow::{self, LineCacheShadow, RenderPlan, RenderTactic};
//! which is sufficient for asynchronous plugins that can only have one
/// Contains replacement string and replace options.
    type Item = u8;
    pub thread_name: Option<StrCow>,
        if !selection.is_empty() { // todo: invalidate so that nothing selected accidentally replaced
#[inline]
    }
        let end = self.line_col_to_offset(text, line + 1, 0);
//
    #[test]
        undo_test(true, [1].iter().cloned().collect(), "0!3456789abcdefGIjklmnopqr888stuvwHIyz");
        b.push_str(s);

    TRACE.closure_payload(name, categories, closure, payload)
    // reachability calculation.
                    } else {
        assert_eq!(Some(3), a.prev_codepoint_offset(6));
        self.revs.last().unwrap().max_undo_so_far
    ///
        let params = json!({
    /// Merge the new content from another Engine into this one with a CRDT merge
        }
        assert!(a.slice(0, 0) == empty);
        assert_eq!(snapshot[1].args.as_ref().unwrap().metadata_name.as_ref().is_some(), true);
        struct RopeDelta_ {
    pub fn byte_at(&self, offset: usize) -> u8 {
        let mut samples = self.samples_cloned_unsorted();
    /// Determines whether the offset is in any selection (counting carets and
            a = a + Rope::from(&c);
        let start = time::precise_time_ns();
        #------
        }).collect()
        if self.find.is_empty() {
                name, categories, None, start, end - start));
    }

                   r.lines_all().collect::<Vec<_>>());

    }
        memrchr(b'\n', &s.as_bytes()[..offset])
            rendered_styles.push(1);
        let base_sel = Selection::new();
    }
        // creation) is:
            Merge(2,1),

    /// and currently in a debugging state.
                    self.do_gesture(text, line, column, GestureType::PointSelect)

        let _t = trace_block("View::wrap_width", &["core"]);
        } else {
    #[test]

// See the License for the specific language governing permissions and
                     modify_selection: &SelectionModifier) {
        }
#[derive(Clone, Copy)]
            if Some(priority) == last_priority {
//
    fn arr_eq(&self, other: &[&'static str]) -> bool {
#[derive(Clone, Debug)]
            out.push(Revision { edit, rev_id: rev.rev_id, max_undo_so_far: rev.max_undo_so_far });
    }
        memchr(b'\n', &s.as_bytes()[offset..])
            tid: sys_tid::current_tid().unwrap(),
        let guard = Self {
        assert_eq!(0, b.line_of_offset(1));
    ProcessLabels { labels: String },
        assert_eq!(2, a.line_of_offset(5));
            std::thread::sleep(std::time::Duration::new(0, 1000));
    // * Actual measurement in text layout

/// .n..n...nn..  -> ........NNNN -> returns vec![N,N,N,N]
}
        plan.request_lines(first_line, last_line);
            }
            Edit { ei: 0, p: 1, u: 1, d: parse_delta("!-") },
    }
    }
    #[test]
pub struct Size {
    #[test]
            let c = i.to_string() + "\n";
///
            AddSelectionBelow =>
        assert_eq!(trace.get_samples_count(), 0);

        engine.undo([1].iter().cloned().collect());
/// See `trace_payload` for a more complete discussion.
                    deletes_from_union = Cow::Owned(deletes_from_union.transform_union(inserts));
                        }

    #[test]
                         style_spans: &Spans<Style>) -> Vec<isize>
                trace.instant_payload("b", &["test"], to_payload("test_get_samples_nested_trace"));
    }
        assert_eq!(TEST_STR, String::from(engine.get_head()));
        let text_b = Rope::from("zpbj");
    // of the union string length *before* the first revision.

    fn from_chrome_id(symbol: char) -> Self {

        let inserts = parse_subset_list("
    fn do_split_selection_into_lines(&mut self, text: &Rope) {
                        let mut word_cursor = WordCursor::new(text, region.max());
    session1: u64,
use xi_rope::rope::{Rope, LinesMetric, RopeInfo};
                RenderTactic::Discard => {
    /// storage allocated will be limit * size_of<Sample>
                    CategoriesT::DynamicArray(ref other_arr) => self_arr.arr_eq(other_arr),
    }

                    CategoriesT::DynamicArray(ref other_arr) => self_arr.eq(other_arr),
        if let Some(new_scroll_pos) = self.scroll_to.take() {
                    } else {
        if !gc_dels.is_empty() {
        assert_eq!(2, expand_by[1].0.priority);
    pub categories: Option<CategoriesT>,
            Insert(String),
            SampleEventType::AsyncEnd => 'e',
        let a: Vec<Revision> = ids_to_fake_revs(&[0,2,4,6,8,10,12]);
trait StringArrayEq<Rhs: ?Sized = Self> {
            first_line: 0,
        let b = Rope::from("b");
            Assert(1, "abc".to_owned()),
            // which we can do with another peer that inserts before a and merges.

#[inline]
                   soft_breaks: Option<&mut Cursor<BreaksInfo>>,
    /// Garbage collection means undo can sometimes need to replay the very first
        while let Err(incomp) = next_boundary {
    use engine::*;
                len += 1;
                metadata_sort_index: None,
    pending_render: bool,
            if !selection.is_empty() {

        // find the lowest toggled undo group number
            None => None
                        let end_line = start_line + seg.n;
        }
            return;
                let line = line.min(text.measure::<LinesMetric>() + 1);
    pub fn new_duration<S, C>(name: S,
            SampleGuard::new(&self, name, categories, None)

///

    /// The default amount of storage to allocate for tracing.  Currently 1 MB.
    AsyncStart,
            ),
        let mut hasher = DefaultHasher::new();
    /// The name of the event to be shown.
            let to_delete = if undone { &new_inserts } else { &new_deletes };
        self.selection = sel;
        assert_eq!(0, a.line_of_offset(1));
        let res = find_base_index(&a, &b);
        assert_eq!(String::from(&a).lines().collect::<Vec<_>>(),
        // We somewhat arbitrarily choose the last region for setting the old-style
        assert_eq!(1, self.revs.len(), "Revisions were added to an Engine before set_session_id, these may collide.");
    let a_ids: BTreeSet<RevId> = a.iter().map(|r| r.rev_id).collect();
// Additional APIs custom to strings
        }
        assert_eq!(Some(5), b.next_codepoint_offset(2));
#[derive(Clone, Copy)]
                self.select_word(text, offset, true)
        ");
        let (revs, text_2, tombstones_2, deletes_from_union_2) =

        cursor.prev_grapheme()
    pub tid: u64,
        where S: Into<StrCow>, C: Into<CategoriesT>

// See the License for the specific language governing permissions and
fn rearrange(revs: &[Revision], base_revs: &BTreeSet<RevId>, head_len: usize) -> Vec<Revision> {
            }
            Rope::concat(
        }

    }

/// 3. You may not care about them always being sorted if you're merging samples
            // with offset == s.len(), but be defensive.

                metadata_sort_index: None,

            find.find_status(matches_only)
pub struct Trace {
            Assert(1, "zpbj".to_owned()),


    pub fn is_equivalent_revision(&self, base_rev: RevId, other_rev: RevId) -> bool {
    #[test]
                SelectionModifier::AddRemovingCurrent => {
    pub preserve_case: bool
        self.len()
            FindNext { wrap_around, allow_same, modify_selection } =>
            Assert(1, "arpbj".to_owned()),
        if offset == 0 {
                    for (ei, e) in self.peers.iter().enumerate() {
        assert_eq!(utf8_offset, 19);
        }
                    result = cow_append(result, &self.fragment[.. i + 1]);
                        let un_deleted = deletes_from_union.subtract(deletes);
impl TreeBuilder<RopeInfo> {
use std::str::FromStr;
        self.height = last - first;

        /// Just the `symmetric_difference` (XOR) of the two sets.
/// discussion on payload.
            }
/// (such as insert, delete, substring) are O(log n). This module provides an immutable
                pos = leaf_offset + nl.len();
            find.update_highlights(text, delta);
    type Err = ParseError;
    /// Exposed for unit tests.


    #[inline]

            self.unset_find();
    // TODO: does Cow really help much here? It certainly won't after making Subsets a rope.
                Contents::Undo { .. } => panic!(),
                    .with_affinity(affinity)
            CategoriesT::DynamicArray(ref vec) => vec.join(sep),
        }
            FindPrevious { wrap_around, allow_same, modify_selection } =>
        #---
                    let sel_start = cursor.pos();
            None
        let b_rope = Rope::from(&b);
        d_builder.delete(Interval::new_closed_open(58, 61));
        }
            client.find_status(self.view_id, &json!(self.find_status(matches_only)));
        self.text = new_text;
                        line_num = seg.their_line_num + seg.n;
    /// The name to associate with the pid/tid.  Whether it's associated with
            retain_revs.insert(last.rev_id);
///
            pos
    let mut utf16_count = 0;
        let mut b = line_cache_shadow::Builder::new();
/// # Examples
fn default_session() -> (u64,u32) {
        use self::MergeTestOp::*;
// should make this generic, but most leaf types aren't going to be sliceable
        });

    #[cfg(feature = "benchmarks")]
        let new_full_priority = FullPriority { priority: new_priority, session_id: self.session };
pub enum CategoriesT {
    fn test_get_samples() {
    #[bench]
        match self.breaks {

}
        }

            rebase(expand_by, b_delta_ops, text_a, tombstones_a, deletes_from_union_a, 0);
            //if self.cursor.is_boundary::<BaseMetric>() {
impl Sample {
            Merge(0,1), Merge(1,0), Merge(0,1), Merge(1,0),
        let first_rev = engine.get_head_rev_id().token();
            SelectAll => self.select_all(text),
    }
        for _ in 0..line_len {
#[macro_use]

impl<'a> Iterator for ChunkIter<'a> {
        self.set_selection_raw(text, selection);
            self.next()
///
        result
    for &b in s.as_bytes() {
#[inline]
            Merge(2,3),

    }
        assert_eq!(2, a.offset_of_line(1));
impl<'a> From<&'a Rope> for String {
        assert_eq!(Some(2), b.next_codepoint_offset(0));
                    self.do_gesture(text, line, column, GestureType::LineSelect)
                    as_vec.push(Sample::new_metadata(
        let chars_left = (edits-max_undos)+1;
/// an argument.
    #[test]
impl From<Vec<String>> for CategoriesT {
pub fn disable_tracing() {
                        Some(end) => max(0, end - 1),

        // TODO: want to let caller retain more rev_id's.
    }
            timestamp_us: ns_to_us(time::precise_time_ns()),

        -##-
            let new_region = region_movement(movement, region, self,
impl<'a> Cursor<'a, RopeInfo> {
    // https://github.com/serde-rs/serde/issues/1189 is fixed.
    }
            timestamp_us: ns_to_us(timestamp_ns),
}
    }
        b.iter(|| black_box(sys_tid::current_tid()));
        hasher.finish()
                    };
                self.do_selection_for_find(text, case_sensitive),
        }
        d_builder.replace(Interval::new_closed_open(1, 3), Rope::from("!"));
/// version of Ropes, and if there are many copies of similar strings, the common parts
        self.enable_config(Config::default());
        let line_len = MAX_LEAF + MIN_LEAF - 1;
// Low level functions
    #[serde(rename = "name")]
    /// Determine whether `offset` lies on a codepoint boundary.
                    self.set_selection(text, selection);
    pub duration_us: Option<u64>,
                (drag_state.min, max(offset, drag_state.max))
    }
        selection.add_region(region);
            },
    }

        engine.edit_rev(1, 2, first_rev, d1.clone());

    pub fn set_dirty(&mut self, text: &Rope) {
/// Conceptually, see the diagram below, with `.` being base revs and `n` being
    /// the text).
        } else {
                    "something", &["benchmark"], || {},
        for _ in 0..(edits-max_undos) {
    #[cfg(feature = "benchmarks")]
        assert_eq!(vec![""], a.lines_all().collect::<Vec<_>>());
            let mut peers = Vec::with_capacity(count);
    }
        let expand_by = compute_transforms(revs);
        assert_eq!(trace.is_enabled(), true);
        self.set_dirty(text);
/// some way.
        if let Some(last) = self.revs.last() {
                    inserts: ins,
            Assert(0, "zacpb".to_owned()),
        self.set_selection(text, sel);
            Merge(0,1),
/// an argument.
        engine.undo([].iter().cloned().collect());
        let deletes_from_union_b = parse_subset("-#---");
            Assert(0, "adfc".to_owned()),
            MetadataType::ProcessSortIndex {..} => "process_sort_index",
        assert_eq!(correct, rearranged_inserts);
            next_boundary = c.next_boundary(&l, leaf_offset);
    // revision might hold content from an undo group that would otherwise be gc'ed),
        self.join(",").serialize(serializer)
/// revisions on top of the revisions represented by `expand_by`.
        let mut a = Rope::from("");

/// However, anything that can be converted into a Cow string can be passed as
                        Some(gc_dels.transform_shrink(&inserts))
        client.def_style(&style.to_json(ix));
            None => text.line_of_offset(offset)
    fn rebase_1() {
    #[serde(default = "default_session", skip_serializing)]
    }
fn compute_deltas(revs: &[Revision], text: &Rope, tombstones: &Rope, deletes_from_union: &Subset) -> Vec<DeltaOp> {
/// }
            assert_eq!(Some(s1.len() * 3 + 4), a.next_grapheme_offset(i));
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            self.set_dirty(text);
    }
            SampleEventType::Instant => 'i',
    }
        ");
    }
        // even though the actual sampling order (from timestamp of
            max: offset,

//! a previously committed version of the text rather than the current text,

        assert_eq!("-###-", &subset_str);
    fn gc() {
    pub(crate) fn unset_find(&mut self) {
        Self {
#[derive(Clone, Debug, PartialEq, Eq)]
    /// point. Used to optimize undo to not look further back.

    /// Returns the largest chunk of valid utf-8 with length <= chunk_size.
            match memchr(b'\n', &s.as_bytes()[offset..]) {
        assert_eq!(Some(3), a.prev_grapheme_offset(9));
    fn gc_5() {
    /// The maximum number of bytes the tracing data should take up.  This limit
                Delta {
        self.text = text;
}
        self.selection = sel;
        if self.is_enabled() {
    where S: Into<StrCow>, C: Into<CategoriesT>, F: FnOnce() -> R
            false => Selection::new(),
        ---#--
            splitpoint

        let height = self.line_of_offset(text, text.len()) + 1;

            for (i, op) in script.iter().enumerate() {
        self.deletes_from_union = new_deletes_from_union;
        // rebase delta to be on the base_rev union instead of the text
            CategoriesT::StaticArray(ref arr) => arr.join(sep),
/// Move sections from text to tombstones and vice versa based on a new and old set of deletions.
    #[test]
}
            pos += 1;

    fn send_update_for_plan(&mut self, text: &Rope, client: &Client,
    pub fn lines_all(&self) -> Lines {
            self.idx += 1;

            lines: 0,
        b.build()
        ##
            SampleEventType::CompleteDuration => 'X',
                    let end_of_line = match cursor.next::<LinesMetric>() {
#[macro_use]
    }
/// ```rust,ignore
            while !s.is_char_boundary(offset - len) {
    }
    #[test]
}
            match self.wrap_col {
            }
            }
                        b.add_span(seg.n, seg.our_line_num, line_cache_shadow::ALL_VALID);

            Assert(0, "ab".to_owned()),
                    if !retain_revs.contains(&rev.rev_id) && gc_groups.contains(undo_group) {
    }
        }
}
use unicode_segmentation::GraphemeIncomplete;
        self.push_subseq(&mut b, self_iv.prefix(edit_iv));
    /// The process the sample was captured in.
    }
/// Usage of static strings is encouraged for best performance to avoid copies.
            Gesture { line, col, ty } =>
        let mut gc_dels = self.empty_subset_before_first_rev();
        }
                deletes: new_deletes,
        where S: Into<StrCow>, C: Into<CategoriesT>, F: FnOnce() -> R
            duration_us: Some(ns_to_us(duration_ns)),
use rpc::{GestureType, MouseAction, SelectionModifier};

        let max_line = self.measure::<LinesMetric>() + 1;
type StyleMap = RefCell<ThemeStyleMap>;
        assert_eq!(snapshot[4].name, "y");
    1
            self.shrink_to_fit();
            categories: Some(categories.into()),
            }
        let a = Rope::concat(
        let new_deletes_from_union = {
        }
            MetadataType::ThreadName {name} => (Some(name), None),
        }
                    }

    #[allow(dead_code)]
        assert_eq!(snapshot[1].name, "thread_name");
enum FindStatusChange {
    }
            let splitpoint = find_leaf_split_for_merge(self);
        result
extern crate log;
///     something_expensive()
    }
/// Find an index before which everything is the same
        let offset = self.line_col_to_offset(text, line as usize, col as usize);
        all_samples.reset_limit(0);
                let (ins, _) = delta.factor();
        let DeltaOp { rev_id, priority, undo_group, mut inserts, mut deletes } = op;
    fn undo_2() {
            }
            trace.instant_payload("c", &["test"], to_payload("test_get_samples_nested_trace"));
            assert_eq!(Some(s1.len() * 3), a.prev_grapheme_offset(i));

        let first_rev = engine.get_head_rev_id().token();

        gc_scenario(35,20);
        let mut undone_groups = Cow::Borrowed(&self.undone_groups);
        let height = self.line_of_offset(text, text.len()) + 1;

        samples.sort();
{
        let d3 = Delta::simple_edit(Interval::new_closed_open(0,0), Rope::from("b"), TEST_STR.len()+1);
impl View {

        let mut engine = Engine::new(Rope::from(TEST_STR));
                if previous_name.is_none() || previous_name.unwrap() != *thread_name {

            self.record(Sample::new_duration(
/// Transform `revs`, which doesn't include information on the actual content of the operations,
    }
}
    /// The largest undo group number of any edit in the history up to this
        }
            HighlightFind { visible } => {

        b.set_dirty(true);
            } else if incomp == GraphemeIncomplete::NextChunk {
                    let full_path = exe_name.into_os_string();
    fn merge_insert_only_whiteboard() {
    fn next(s: &String, offset: usize) -> Option<usize> {
        let line_len = MAX_LEAF + MIN_LEAF - 1;
        assert_eq!(trace.get_samples_count(), 0);
                return;

struct FullPriority {
            Merge(0,1),
    }
        self.len() >= MIN_LEAF
/// xi_trace::trace("something happened", &["rpc", "response"]);
            tid: tid,
    }
                self.run_op(op);
    fn compute_transforms_2() {
                std::thread::sleep(std::time::Duration::new(0, 1000));
        --#--
    fn from(r: &Rope) -> String {

        // only one of the deletes was gc'd, the other should still be in effect
        let mut selection = match multi_select {
}
        let b: Vec<Revision> = ids_to_fake_revs(&[0,1,2,4,5,8,9]);
                              start_ns: u64,
}
        engine.edit_rev(1, 2, first_rev, d1.clone());
                        let mut soft_breaks = self.breaks.as_ref().map(|breaks|

        let a = Rope::from("a\nb\nc");
/// Returns all the samples collected so far ordered chronologically by
                payload: payload,
    // There will probably never be a document with more than 4 billion edits
        assert!(r != a_rope);


            let b_deltas = compute_deltas(&b_new, &other.text, &other.tombstones, &other.deletes_from_union);
                let categories = v.split(",").map(|s| s.to_string()).collect();
    pub fn block<S, C>(&self, name: S, categories: C) -> SampleGuard
        let first_rev = engine.get_head_rev_id().token();
        assert_eq!(snapshot[8].name, "c");
        assert_eq!(None, a.prev_codepoint_offset(0));
        #------
    }
            None
            Edit { ei: 2, p: 4, u: 1, d: parse_delta("---z") },
                    None
            self.set_selection(text, sel);
        self.enabled.store(false, AtomicOrdering::Relaxed);
                MergeTestOp::AssertMaxUndoSoFar(ei, correct) => {

    }
    ///
        let new_deletes_from_union = expanded_deletes_from_union.union(&deletes);
    // callers should be encouraged to use cursor instead
        if line < self.first_line {
        }
        match ty {

    /// terminating newline.
        --#-
                    };
use memchr::{memrchr, memchr};
    // allowing us to use the same method to insert the text into the tombstones.
        // if we have active find highlights, we don't collapse selections

        println!("{:#?}", delta_ops);
            for i in 0..count {
    }
    // TODO find the maximum base revision.
            SelectionForReplace => self.do_selection_for_replace(text),

    let mut s = Subset::new(head_len);
    /// No wrapping in effect.
            _ => return
        // Note: for committing plugin edits, we probably want to know the priority
    fn line_offsets() {
                            styles: &StyleMap, style_spans: &Spans<Style>,
    }
        }
/// * `categories` - A static array of static strings that tags the samples in
    where S: Into<StrCow>, C: Into<CategoriesT>
        // insert character at end, when this test was added, it panic'd here
        assert_tokens(&rope, &[
    /// This function will panic if `line > self.measure::<LinesMetric>() + 1`.
    /// # Panics
    /// 96 bits which is more than sufficient for this to never happen.
                    text.slice_to_string(start, end)
                              to_payload("test_get_samples"));
        assert_eq!(1, expand_by.len());
    ObjectCreated,
                max_undo_so_far: i+1,
            let pos = bc.next::<BreaksMetric>();
// Copyright 2016 The xi-editor Authors.
}

                    b.add_span(seg.n, 0, 0);
    }
            }
        Lines {

            GotoLine { line } => self.goto_line(text, line),
    /// method to be fast even when the selection is large.
///
                    to_payload(("some description of the closure")))));
            Some(Cow::Owned(mut s)) => {
        undo_group: usize,
/// first can make it ~1.7x slower than a regular trace.
    /// fine-grained in the case of multiple cursors, but we also want this

        let trace = Trace::enabled(Config::with_limit_count(11));
    fn find_rev(&self, rev_id: RevId) -> Option<usize> {
            Edit { ei: 1, p: 5, u: 1, d: parse_delta("---j") },
        } else {
        let utf8_offset = 13;
        let inserts = parse_subset_list("
    TRACE.samples_cloned_sorted()
    pub payload: Option<TracePayloadT>,
        let (mut last_line, last_col) = self.offset_to_line_col(text, region.max());
    /// Edit the string, replacing the byte range [`start`..`end`] with `new`.

    #[test]

    inner: ChunkIter<'a>,
        } else if line == max_line {
    }
                Some(filename) => {
/// Enable tracing with a specific configuration. Tracing is disabled initially
            fragment: ""
        let text_ins_delta = union_ins_delta.transform_shrink(cur_deletes_from_union);
extern crate time;
        let mut ops = Vec::new();
        // clamp to end of line
            // Snap to grapheme cluster boundary
        self.enabled.store(true, AtomicOrdering::Relaxed);
    /// Constructs a Begin or End sample.  Should not be used directly.  Instead
        // TODO: simplify this through the use of scopeguard crate
        let d2 = Delta::simple_edit(Interval::new_closed_open(chars_left, chars_left), Rope::from("f"), engine.get_head().len());
                }
            return ix;
    fn gc_4() {
                    if !inserts.is_empty() {
impl From<Rope> for String {
                self.select_word(text, offset, false),
        assert_eq!(0, a.lines_raw_all().count());
    }
        assert_eq!(snapshot[2].name, "a");
        }, deletes_from_union)
impl PartialOrd for Sample {
    }
            }

        }
        if offset >= next_line_offset {
    }
            None
/// Internally, the implementation uses reference counting (not thread safe, though
/// See `trace_payload` for a more complete discussion.
    pub fn get_head_rev_id(&self) -> RevId {
}
                self.do_find_next(text, true, wrap_around, allow_same, &modify_selection),
    pub fn new_disabled() -> Self {
        let mut b = TreeBuilder::new();
            offset,
        assert_eq!("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", String::from(engine.get_head()));

        max
            session: default_session(),
            return self.revs.len();
                cur_all_inserts = older_all_inserts;

        let (new_rev, new_deletes_from_union) = self.compute_undo(&groups);
                        },
        }
            let matches_only = self.find_changed == FindStatusChange::Matches;
                name, categories, Some(payload.into()), start, end - start));
        if let Some((l, offset)) = self.get_leaf() {
    /// Start of the region selected when drag was started (region is
    }
    }

                    }
            // The deletes are already after our inserts, but we need to include the other inserts

            tombstones: Rope::default(),
            Move(movement) => self.do_move(text, movement, false),
//
            Assert(0, "acpbdj".to_owned()),
    pub(crate) fn get_view_id(&self) -> ViewId {
        while !leaf.is_char_boundary(end) {
    // if this was a tuple field instead of two fields, alignment padding would add 8 more bytes.

///
{
        }
