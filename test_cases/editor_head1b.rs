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
