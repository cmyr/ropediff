extern crate xi_rope;
extern crate serde;
extern crate serde_json;
extern crate atty;
extern crate crossbeam;
extern crate suffix;
extern crate memchr;

use std::time::{Duration, Instant};
use std::fmt;
use std::path::{Path, PathBuf};
use xi_rope::Rope;
use xi_rope::rope::{RopeDelta, RopeInfo};
use xi_rope::delta::DeltaElement;

mod diff_play;
use diff_play::*;
use xi_rope::diff_utils::RopeScanner;

struct PrettyDelta<'a> {
    delta: &'a RopeDelta,
    base: &'a Rope,
}

const GREEN_BOLD: &str = "\x1B[37;1m";
const OTHER_COLORS: &[&str] = &[
    "\x1B[33m", // yellow
    "\x1B[34m", // blue
    "\x1B[35m", // purple
    "\x1B[36m", // cyan
];
const END: &str = "\x1B[0m";

impl<'a> fmt::Display for PrettyDelta<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let mut color_idx = 0;
        for elem in &self.delta.els {
            match *elem {
                DeltaElement::Copy(beg, end) => {
                    color_idx = (color_idx + 1) % OTHER_COLORS.len();
                    let s = self.base.slice_to_string(beg, end);
                    if atty::is(atty::Stream::Stdin) {
                        write!(f, "{}{}{}", OTHER_COLORS[color_idx], &s, END)?;
                    } else {
                        write!(f, "{}", s)?;
                    }
                }
                DeltaElement::Insert(ref n) => {
                    if atty::is(atty::Stream::Stdin) {
                        write!(f, "{}{}{}", GREEN_BOLD, String::from(n), END)?;
                    } else {
                        write!(f, "{}", String::from(n))?;

                    }
                }
            }
        }
        Ok(())
    }
}

fn pretty_print(delta: &RopeDelta, base: &Rope) -> String {
    let p = PrettyDelta { delta, base };
    p.to_string()
}

fn main() -> Result<(), ::std::io::Error> {
    let mut args = ::std::env::args();
    args.next(); // discard first arg
    let (file1, file2) = match (args.next(), args.next()) {
        (Some(one), Some(two)) => (one, two),
        _ => return big_test(),
    };

    let verbose = args.next().as_ref().map(|s| s.as_str()) == Some("-v");

    //let one = ::std::fs::read_to_string(file1)?;
    //let two = ::std::fs::read_to_string(file2)?;
    let _r = test_all(&file1, &file2, verbose);

    //let delta = SmallTricksyDiff::compute_delta(&one, &two);
    //let result = delta.apply(&one);
    //eprintln!("{}", pretty_print(&delta, &one));
    //eprintln!("{:?}", &delta);
    //println!("{}", String::from(result));
    //::std::process::exit(0);

    Ok(())
}

fn big_test() -> ::std::io::Result<()> {

    static TEST_PAIRS: &[(&str, &str)] = &[
        ("metrics1a.md", "metrics1b.md"),
        ("metrics1b.md", "metrics1a.md"),
        ("cargo1a.rs", "cargo1b.rs"),
        ("cargo1b.rs", "cargo1a.rs"),
        ("editor_head1a.rs", "editor_head1b.rs"),
        ("editor_head1b.rs", "editor_head1a.rs"),
        ("editor_head2a.rs", "editor_head2b.rs"),
        ("editor_head2b.rs", "editor_head2a.rs"),
        ("view_small1a.rs", "view_small1b.rs"),
        ("view_small1b.rs", "view_small1a.rs"),
        ("editor1a.rs", "editor1b.rs"),
        ("editor1b.rs", "editor1a.rs"),
        ("editor1c.rs", "editor1d.rs"),
        ("editor1d.rs", "editor1c.rs"),
        ("irssi1a.py", "irssi1b.py"),
        ("irssi1b.py", "irssi1a.py"),
        ("evc1a.swift", "evc1b.swift"),
        ("evc1b.swift", "evc1a.swift"),
        ("doc_controller_head1a.swift", "doc_controller_head1b.swift"),
        ("doc_controller_head1b.swift", "doc_controller_head1a.swift"),
        ("doc_controller1a.swift", "doc_controller1b.swift"),
        ("doc_controller1b.swift", "doc_controller1a.swift"),
        ("rope1a.rs", "rope1b.rs"),
        ("rope1b.rs", "rope1a.rs"),
        ("view.rs", "rope1b.rs"),
        ("rope1b.rs", "view.rs"),
        ("5klines.rs", "5klines.rs"),
        ("5klines.rs", "5klines_small_change.rs"),
        ("5klines_small_change.rs", "5klines.rs"),
        ("5klines.rs", "5klines_shuffled.rs"),
        ("5klines_shuffled.rs", "5klines.rs"),
        ("100k_lines_change.rs", "100k_lines_change.rs"),
        ("100k_lines_change.rs", "100k_lines_shuffled.rs"),
        ("100k_lines_shuffled.rs", "100k_lines_change.rs"),
    ];

    for (one, two) in TEST_PAIRS {
        test_all(one, two, false)?;
    }
    Ok(())
}

use std::process::{Command, Stdio};

#[allow(dead_code)]
fn test_find_start(one: &str, two: &str) {
    let one = Rope::from(one);
    let two = Rope::from(two);

    let start = Instant::now();
    let expected = find_diff_start(&one, &two);
    let elapsed = start.elapsed();

    let start = Instant::now();
    let mut scanner = RopeScanner::new(&one, &two);
    let result = scanner.find_min_diff_range();
    let elapsed2 = start.elapsed();

    eprintln!("baseline: {:>8}ns {:?}", elapsed.subsec_nanos(), expected);
    eprintln!("scannerz: {:>8}ns {:?}", elapsed2.subsec_nanos(), result);
    //assert_eq!(expected, result);
}

fn test_all(one: &str, two: &str, verbose: bool) -> ::std::io::Result<()> {
    let test_dir: PathBuf = ::std::env::var("DIFF_TEST_DIR")
        .expect("DIFF_TEST_DIR msut be set").into();

    assert!(test_dir.exists());
    let one_p = if Path::new(one).exists() { PathBuf::from(one) } else { test_dir.join(one) };
    let two_p = if Path::new(two).exists() { PathBuf::from(two) } else { test_dir.join(two) };

    assert!(one_p.exists());
    assert!(two_p.exists());


    let diff_cmd = Command::new("diff")
        .args(&[&one_p, &two_p])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let diff_stat = Command::new("diffstat")
        .stdin(diff_cmd.stdout.unwrap())
        .output()
        .unwrap();

    let base_len = one_p.metadata().map(|m| m.len()).unwrap();

    eprintln!("\n#### testing {} v {} ####\nbase len: {} bytes\n{}", one, two, base_len,
             String::from_utf8(diff_stat.stdout).unwrap());

    let one_str = ::std::fs::read_to_string(&one_p)?;
    let two_str = ::std::fs::read_to_string(&two_p)?;

    //test_find_start(&one_str, &two_str);

    // let's get a baseline
    //let result = run_diff(SmallSlowDiff, &one_str, &two_str, 16);
    //print_result("quadratic-16", &result, 16, verbose);

    for size in [32_usize].iter() {
        let size = *size;

        //let result = run_diff(SmallSlowDiff, &one_str, &two_str, size);
        //print_result(&format!("quadratic-{}", size), &result, size, verbose);

        let result = run_diff(FastHashDiffSimpler, &one_str, &two_str, size);
        print_result(&format!("fast hash naive-{}", size), &result, size, verbose);

        let result = run_diff(FastHashDiff, &one_str, &two_str, size);
        print_result(&format!("fast hash-{}", size), &result, size, verbose);

        let result = run_diff(MockParallelHashDiff, &one_str, &two_str, size);
        print_result(&format!("mock parallel hash-{}", size), &result, size, verbose);

        let result = run_diff(FinalDiff, &one_str, &two_str, size);
        print_result(&format!("final-{}", size), &result, size, verbose);

        //let result = run_diff(ParallelHashDiff, &one_str, &two_str, size);
        //print_result(&format!("parallel hash-{}", size), &result, size, verbose);

        let result = run_diff(SmallDiff, &one_str, &two_str, size);
        print_result(&format!("hash naive-{}", size), &result, size, verbose);

        let result = run_diff(SmallTricksyDiff, &one_str, &two_str, size);
        print_result(&format!("hash-opt-{}", size), &result, size, verbose);

        let result = run_diff(SuffixDiff, &one_str, &two_str, size);
        print_result(&format!("suffix-{}", size), &result, size, verbose);

        //let result = run_diff(SuffixDiffOpt, &one_str, &two_str, size);
        //print_result(&format!("suffix-opt-{}", size), &result, size, verbose);

    }
    Ok(())
}

fn run_diff<D, A, B>(_: D, one: A, two: B, size: usize) -> RunResult
    where D: Diff<RopeInfo>, A: AsRef<str>, B: AsRef<str>
{
    let one = Rope::from(one);
    let two = Rope::from(two);

    let start = Instant::now();
    let delta = D::compute_delta(&one, &two, size);
    let elapsed = start.elapsed();

    let result = delta.apply(&one);
    if String::from(&result) != String::from(&two) {
        eprintln!("DIFF_FAILED");
        //print_chunks(&one);
        //eprintln!("{:?}", &delta);
        //eprintln!("{}", pretty_print(&delta, &one));
        //print!("{}", String::from(&result));
        //println!("{}", String::from(&two));
        //::std::process::exit(1);
    } else {
        //eprintln!("{:?}", &delta);
        //delta.clone().factor();
    }

    let value = serde_json::to_string(&delta).unwrap();
    let worst_case = worst_case_delta_len(String::from(&two));

    RunResult {
        delta,
        duration: elapsed,
        byte_size: value.len(),
        worst_case,
    }
}

fn print_chunks(rope: &Rope) {
    for (i, s) in rope.iter_chunks_all().enumerate() {
        let color_idx = i % OTHER_COLORS.len();
        eprint!("{}{}{}", OTHER_COLORS[color_idx], &s, END);
    }
    eprintln!("####### end chunks #########")
}

    fn worst_case_delta_len<R: AsRef<str>>(s: R) -> usize {
        let s = s.as_ref();
    let d = RopeDelta {
        els: vec![DeltaElement::Insert(s.into())],
        base_len: s.len()
    };
    serde_json::to_string(&d).unwrap().len()
}


struct RunResult {
    delta: RopeDelta,
    duration: Duration,
    byte_size: usize,
    worst_case: usize,
}

impl fmt::Display for RunResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let byte_perc = 100.0 * (self.byte_size as f64 / self.worst_case as f64);
        write!(f, "{:>8}b {:>6.2}% {:>8}μ", self.byte_size, byte_perc,
        self.duration.subsec_micros())
    }
}

fn print_result(name: &str, result: &RunResult, _size: usize, verbose: bool) {
    //let verbose = true;
    eprintln!("{:24} {}", name, result);
    if verbose { eprintln!("{:?}", result.delta); }
}

