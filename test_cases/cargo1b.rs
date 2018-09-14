[package]
authors = ["Raph Levien <raph@google.com>"]
description = "A generic rope data structure built on top of B-Trees."
license = "Apache-2.0"
name = "xi-rope"
repository = "https://github.com/google/xi-editor"
version = "0.2.0"

[dependencies]
bytecount = "1.3.1"
memchr = "2.0"
suffix = "1.0"
serde = "1.0"
serde_derive = "1.0"

[dev-dependencies]
serde_test = "^1.0"
serde_json = "1.0"
tempdir = "0.3.7"
tempfile = "69"

[features]
avx-accel = ["bytecount/avx-accel"]
simd-accel = ["bytecount/simd-accel"]
