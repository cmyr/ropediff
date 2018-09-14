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
use std::path::{Path, PathBuf};
use std::io::Write;
use std::collections::BTreeSet;
use serde_json::Value;

use xi_rope::rope::{LinesMetric, Rope, RopeInfo};
use xi_rope::interval::Interval;
use xi_rope::delta::{self, Delta, Transformer};
use xi_rope::engine::{Engine, RevId, RevToken};
use xi_rope::spans::SpansBuilder;
use xi_rpc::RemoteError;

use view::View;
use word_boundaries::WordCursor;
use movement::{Movement, region_movement};
use selection::{Affinity, Selection, SelRegion};

use tabs::{BufferIdentifier, ViewIdentifier, DocumentCtx};
use rpc::{self, GestureType};
use syntax::SyntaxDefinition;
use plugins::rpc::{PluginUpdate, PluginEdit, ScopeSpan, PluginBufferInfo,
ClientPluginInfo};
use plugins::{PluginPid, Command};
use layers::Scopes;
use config::{BufferConfig, Table};


#[cfg(not(target_os = "fuchsia"))]
pub struct SyncStore;
#[cfg(target_os = "fuchsia")]
use fuchsia::sync::SyncStore;

const FLAG_SELECT: u64 = 2;

// TODO This could go much higher without issue but while developing it is
// better to keep it low to expose bugs in the GC during casual testing.
const MAX_UNDOS: usize = 20;

// Maximum returned result from plugin get_data RPC.
const MAX_SIZE_LIMIT: usize = 1024 * 1024;

pub struct Editor {
    text: Rope,
