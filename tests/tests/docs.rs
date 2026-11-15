//! Structural checks on docs/implementation-status.md.
//!
//! The rules are a port of OpenGrace's test_docs_structure.cpp, which carries
//! the provenance of the defects that produced them. Every rule below was
//! written after a defect that reached a commit in OpenGrace, and not one of
//! them was caught by the compiler or by any other suite. The rules run
//! against the real document and against eight fixtures, one per rule; a
//! fixture the rules fail to catch fails this test, so the test cannot
//! silently become useless.
//!
//!   R1  The decision numbering must be strictly ascending.
//!   R2  The decision numbering must be contiguous from one.
//!   R3  No level-2 heading inside the decision span.
//!   R4  Every level-2 section has content.
//!   R5  No heading inside a fenced block.
//!   R6  Every "decision N" reference resolves.
//!   R7  At most one blank line at the end of the file.
//!   R8  The verification section holds a fenced command block.

use std::fs;

fn is_heading(line: &str) -> bool {
    line.starts_with("## ")
}

fn is_fence(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("```")
}

// A decision opens with "N. **". Returns None when the line is not one.
fn decision_number(line: &str) -> Option<u32> {
    let mut chars = line.chars().peekable();
    let mut value: u32 = 0;
    let mut digits = 0;
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            value = value * 10 + (c as u32 - '0' as u32);
            digits += 1;
            chars.next();
        } else {
            break;
        }
    }
    if digits == 0 {
        return None;
    }
    if chars.next() != Some('.') {
        return None;
    }
    if chars.next() != Some(' ') {
        return None;
    }
    if chars.next() != Some('*') {
        return None;
    }
    if chars.next() != Some('*') {
        return None;
    }
    Some(value)
}

fn check_document(text: &str) -> Vec<(String, String)> {
    let mut lines: Vec<&str> = text.split('\n').collect();
    // A file ending in a newline yields a trailing empty element from split,
    // which is kept so that R7 sees the trailing newline explicitly.
    if let Some(last) = lines.last() {
        if last.is_empty() {
            // keep
        }
    }
    let _ = &mut lines;

    let mut failures: Vec<(String, String)> = Vec::new();

    let headings: Vec<usize> = (0..lines.len())
        .filter(|&i| is_heading(lines[i]))
        .collect();

    // Decisions are counted only inside "## Open decisions".
    let mut decisions_begin = lines.len();
    let mut decisions_end = lines.len();
    for (h, &hi) in headings.iter().enumerate() {
        if lines[hi] != "## Open decisions" {
            continue;
        }
        decisions_begin = hi + 1;
        decisions_end = if h + 1 < headings.len() {
            headings[h + 1]
        } else {
            lines.len()
        };
        break;
    }

    let mut decisions: Vec<(usize, u32)> = Vec::new();
    for i in decisions_begin..decisions_end {
        if let Some(number) = decision_number(lines[i]) {
            decisions.push((i, number));
        }
    }

    // A second, unrestricted scan, used only to find the span of decision-like
    // lines for R3.
    let any_decision_line: Vec<usize> = (0..lines.len())
        .filter(|&i| decision_number(lines[i]).is_some())
        .collect();

    if decisions.is_empty() {
        failures.push(("R1".to_string(), "no decisions found".to_string()));
        return failures;
    }

    let numbers: Vec<u32> = decisions.iter().map(|&(_, n)| n).collect();

    // R1: strictly ascending.
    for i in 1..numbers.len() {
        if numbers[i] <= numbers[i - 1] {
            failures.push((
                "R1".to_string(),
                format!(
                    "decision {} follows {} at line {}",
                    numbers[i],
                    numbers[i - 1],
                    decisions[i].0 + 1
                ),
            ));
            break;
        }
    }

    // R2: contiguous from one.
    for (i, &n) in numbers.iter().enumerate() {
        let want = (i + 1) as u32;
        if n != want {
            failures.push((
                "R2".to_string(),
                format!(
                    "expected decision {} at line {}, found {}",
                    want,
                    decisions[i].0 + 1,
                    n
                ),
            ));
            break;
        }
    }

    // R3: no level-2 heading inside the decision span.
    if !any_decision_line.is_empty() {
        let first = any_decision_line[0];
        let last = any_decision_line[any_decision_line.len() - 1];
        for &hi in headings.iter() {
            if hi > first && hi < last {
                failures.push((
                    "R3".to_string(),
                    format!(
                        "heading \"{}\" sits inside the decisions at line {}",
                        lines[hi],
                        hi + 1
                    ),
                ));
                break;
            }
        }
    }

    // R4: every level-2 section has content.
    for (h, &hi) in headings.iter().enumerate() {
        let end = if h + 1 < headings.len() {
            headings[h + 1]
        } else {
            lines.len()
        };
        let mut content = false;
        for i in (hi + 1)..end {
            if !lines[i].trim().is_empty() {
                content = true;
                break;
            }
        }
        if !content {
            failures.push((
                "R4".to_string(),
                format!(
                    "section \"{}\" is empty at line {}",
                    lines[hi],
                    hi + 1
                ),
            ));
            break;
        }
    }

    // R5: no heading inside a fenced block.
    {
        let mut fenced = false;
        for i in 0..lines.len() {
            if is_fence(lines[i]) {
                fenced = !fenced;
                continue;
            }
            if fenced && is_heading(lines[i]) {
                failures.push((
                    "R5".to_string(),
                    format!(
                        "heading \"{}\" is inside a fenced block at line {}",
                        lines[i],
                        i + 1
                    ),
                ));
                break;
            }
        }
    }

    // R6: every "decision N" reference resolves.
    for i in 0..lines.len() {
        let line = lines[i];
        let mut at = 0;
        while let Some(found) = find_at(line, "decision", at) {
            let bytes = line.as_bytes();
            let mut j = found + 8;
            if j < line.len() && bytes[j] == b's' {
                j += 1;
            }
            if j < line.len() && bytes[j] == b' ' {
                j += 1;
                let begin = j;
                while j < line.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                if j > begin {
                    if let Ok(referenced) = line[begin..j].parse::<u32>() {
                        if !numbers.contains(&referenced) {
                            failures.push((
                                "R6".to_string(),
                                format!(
                                    "line {} references decision {}, \
                                     which does not exist",
                                    i + 1,
                                    referenced
                                ),
                            ));
                        }
                    }
                }
            }
            at = found + 1;
        }
    }

    // R7: at most one blank line at the end of the file.
    {
        let mut last = lines.len();
        while last > 0 && lines[last - 1].is_empty() {
            last -= 1;
        }
        let trailing = lines.len() - last;
        if trailing > 1 {
            failures.push((
                "R7".to_string(),
                format!("{} blank lines at end of file", trailing),
            ));
        }
    }

    // R8: the verification section holds a fenced command block.
    for (h, &hi) in headings.iter().enumerate() {
        if lines[hi] != "## Verification commands" {
            continue;
        }
        let end = if h + 1 < headings.len() {
            headings[h + 1]
        } else {
            lines.len()
        };
        let mut fenced = false;
        for i in (hi + 1)..end {
            if is_fence(lines[i]) {
                fenced = true;
                break;
            }
        }
        if !fenced {
            failures.push((
                "R8".to_string(),
                "the verification section has no fenced command block"
                    .to_string(),
            ));
        }
    }

    failures
}

fn find_at(haystack: &str, needle: &str, from: usize) -> Option<usize> {
    if from >= haystack.len() {
        return None;
    }
    haystack[from..].find(needle).map(|p| p + from)
}

fn read_file(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_default()
}

const GOOD: &str = "# Status\n\n## Open decisions\n\n1. **First?** Yes.\n\n\
2. **Second?** No. See decision 1.\n\n## Verification commands\n\n```sh\n\
cmake -S . -B build\n```\n";

struct Fixture {
    rule: &'static str,
    what: &'static str,
    text: &'static str,
}

// One fixture per rule, each reduced from the defect that actually occurred.
const FIXTURES: &[Fixture] = &[
    Fixture {
        rule: "R1",
        what: "a bold numbered list counted as decisions",
        text: "# Status\n\n## Open decisions\n\n1. **First?** Yes.\n\n\
34. **Not a decision.** Items:\n\n    1. **Alpha.** Text.\n\n\
2. **Second?** No.\n\n## Verification commands\n\n```sh\n\
cmake -S . -B build\n```\n",
    },
    Fixture {
        rule: "R2",
        what: "a decision number reused",
        text: "# Status\n\n## Open decisions\n\n1. **First?** Yes.\n\n\
1. **Also first?** No.\n\n## Verification commands\n\n```sh\n\
cmake -S . -B build\n```\n",
    },
    Fixture {
        rule: "R3",
        what: "a heading inside the decision span",
        text: "# Status\n\n## Open decisions\n\n1. **First?** Yes.\n\n\
## A section that got absorbed\n\nmore.\n\n2. **Second?** No.\n\n\
## Verification commands\n\n```sh\ncmake -S . -B build\n```\n",
    },
    Fixture {
        rule: "R4",
        what: "a section left empty",
        text: "# Status\n\n## Open decisions\n\n1. **First?** Yes.\n\n\
## Verification commands\n",
    },
    Fixture {
        rule: "R5",
        what: "a fenced block that swallowed a heading",
        text: "# Status\n\n## Open decisions\n\n1. **First?** Yes.\n\n```sh\n\
cmake -S . -B build\n## Verification commands\n```\n",
    },
    Fixture {
        rule: "R6",
        what: "a reference to a renumbered decision",
        text: "# Status\n\n## Open decisions\n\n1. **First?** Yes.\n\n\
2. **Second?** No. See decision 25.\n\n## Verification commands\n\n```sh\n\
cmake -S . -B build\n```\n",
    },
    Fixture {
        rule: "R7",
        what: "blank lines at the end of the file",
        text: "# Status\n\n## Open decisions\n\n1. **First?** Yes.\n\n\
## Verification commands\n\n```sh\ncmake -S . -B build\n```\n\n\n",
    },
    Fixture {
        rule: "R8",
        what: "prose where the commands belong",
        text: "# Status\n\n## Open decisions\n\n1. **First?** Yes.\n\n\
## Verification commands\n\n    **Updated in 6e: the chain changed.**\n",
    },
];

#[test]
fn docs_rules_accept_a_sound_document() {
    let good = check_document(GOOD);
    assert!(
        good.is_empty(),
        "the rules must accept a sound document: {:?}",
        good
    );
}

#[test]
fn docs_rules_catch_every_fixture() {
    for fixture in FIXTURES {
        let found = check_document(fixture.text);
        let caught = found.iter().any(|(rule, _)| rule == fixture.rule);
        if !caught {
            println!("  not caught: {} ({})", fixture.rule, fixture.what);
        }
        assert!(caught, "fixture {} was not caught", fixture.rule);
    }
}

#[test]
fn docs_rules_pass_on_the_real_document() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../docs/implementation-status.md"
    );
    let text = read_file(path);
    assert!(!text.is_empty(), "the implementation status is empty");
    let failures = check_document(&text);
    for (rule, message) in &failures {
        println!("  {}: {}", rule, message);
    }
    assert!(failures.is_empty(), "the real document failed the rules");
}
