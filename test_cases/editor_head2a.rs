// Copyright 2016 Google Inc. All rights reserved.
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

use std::borrow::Cow;
use std::cmp::{min, max};
use std::fs::File;
use std::io::{Read, Write};
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use serde_json::Value;

use xi_rope::rope::{LinesMetric, Rope};
use xi_rope::interval::Interval;
use xi_rope::delta::{Delta, Transformer};
use xi_rope::tree::Cursor;
use xi_rope::engine::Engine;
use xi_rope::spans::SpansBuilder;
use view::{Style, View};

use tabs::TabCtx;
use rpc::EditCommand;
use run_plugin::{start_plugin, PluginRef};

const FLAG_SELECT: u64 = 2;

const MAX_UNDOS: usize = 20;

const TAB_SIZE: usize = 4;

// Maximum returned result from plugin get_data RPC.
const MAX_SIZE_LIMIT: usize = 1024 * 1024;

pub struct Editor<W: Write> {
    text: Rope,
    view: View,

    engine: Engine,
    last_rev_id: usize,
    undo_group_id: usize,
    live_undos: Vec<usize>, //  undo groups that may still be toggled
    cur_undo: usize, // index to live_undos, ones after this are undone
    undos: BTreeSet<usize>, // undo groups that are undone
    gc_undos: BTreeSet<usize>, // undo groups that are no longer live and should be gc'ed

    this_edit_type: EditType,
    last_edit_type: EditType,

    // update to cursor, to be committed atomically with delta
    // TODO: use for all cursor motion?
