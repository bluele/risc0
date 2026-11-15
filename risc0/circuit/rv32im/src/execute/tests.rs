// Copyright 2025 RISC Zero, Inc.
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

use risc0_binfmt::MemoryImage;
use risc0_zkp::core::digest::Digest;
use test_log::test;

use crate::{TerminateState, MAX_INSN_CYCLES};

use super::{testutil, DEFAULT_SEGMENT_LIMIT_PO2};

#[test]
fn basic() {
    let program = testutil::kernel::basic();
    let expected_cycles = program.size_in_words();
    let mut image = MemoryImage::new_kernel(program);
    let pre_image_id = image.image_id();

    println!("image_id: {pre_image_id}");

    let session = testutil::execute(
        image,
        DEFAULT_SEGMENT_LIMIT_PO2,
        MAX_INSN_CYCLES,
        testutil::DEFAULT_SESSION_LIMIT,
        &testutil::NullSyscall,
        None,
    )
    .unwrap();

    let segments = session.segments;
    assert_eq!(segments.len(), 1);
    let segment = segments.first().unwrap();
    assert_eq!(segment.claim.pre_state, pre_image_id);
    assert_ne!(segment.claim.post_state, pre_image_id);
    assert_eq!(segment.claim.input, Digest::ZERO);
    assert_eq!(segment.claim.output, Some(Digest::ZERO));
    assert_eq!(
        segment.claim.terminate_state,
        Some(TerminateState::default())
    );
    assert!(segment.read_record.is_empty());
    assert!(segment.write_record.is_empty());
    assert_eq!(segment.suspend_cycle, expected_cycles as u32 + 1);
}

#[test]
fn system_split() {
    let program = testutil::kernel::simple_loop(2000);
    let mut image = MemoryImage::new_kernel(program);
    let pre_image_id = image.image_id();

    let session = testutil::execute(
        image,
        testutil::MIN_CYCLES_PO2,
        100,
        testutil::DEFAULT_SESSION_LIMIT,
        &testutil::NullSyscall,
        None,
    )
    .unwrap();

    let segments = session.segments;
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].claim.pre_state, pre_image_id);
    assert_ne!(segments[0].claim.post_state, pre_image_id);
    assert_eq!(segments[0].claim.input, Digest::ZERO);
    assert_eq!(segments[0].claim.output, None);
    assert_eq!(segments[0].claim.terminate_state, None);

    assert_eq!(segments[1].claim.pre_state, segments[0].claim.post_state);
    assert_ne!(segments[1].claim.post_state, segments[1].claim.pre_state);
    assert_eq!(segments[1].claim.input, Digest::ZERO);
    assert_eq!(segments[1].claim.output, Some(Digest::ZERO));
    assert_eq!(
        segments[1].claim.terminate_state,
        Some(TerminateState::default())
    );

    assert!(segments[0].read_record.is_empty());
    assert!(segments[0].write_record.is_empty());
}

/// Requires an image to take one segment and to terminate with zeros.
fn terminates(name: &str, image: MemoryImage) {
    let session = testutil::run(image).unwrap_or_else(|err| panic!("{name}: {err:#}"));
    assert_eq!(session.segments.len(), 1, "{name}");
    assert_eq!(
        session.segments[0].claim.terminate_state,
        Some(TerminateState::default()),
        "{name}"
    );
}

/// Requires the executor to refuse an image, and to say `message` of it.
fn refuses(name: &str, image: MemoryImage, message: &str) {
    match testutil::run(image) {
        Ok(_) => panic!("{name}: the executor accepted the program"),
        Err(error) => assert!(format!("{error:#}").contains(message), "{name}: {error:#}"),
    }
}

#[test]
fn poseidon2() {
    for (buf_out, state) in testutil::poseidon2::places() {
        let program = testutil::poseidon2::calls(buf_out, state);
        terminates(&format!("{buf_out:#x}"), MemoryImage::new_kernel(program));
    }
}

/// Each kernel checks the memory that its call leaves before it terminates.
#[test]
fn bigint() {
    for (name, call) in testutil::bigint::accepted() {
        terminates(name, MemoryImage::new_kernel(call.program()));
    }
    for (name, call, message) in testutil::bigint::refused() {
        refuses(name, MemoryImage::new_kernel(call.program()), message);
    }
}

/// The circuit does not clear bit 0 of the target of a `jalr`, as the ISA does. A target that is
/// not word aligned traps, and these kernels handle no trap.
#[test]
fn jalr() {
    terminates("0", MemoryImage::new_kernel(testutil::jalr(0)));
    for low in 1..4 {
        let kernel = MemoryImage::new_kernel(testutil::jalr(low));
        refuses(&low.to_string(), kernel, "InstructionMisaligned");
    }
}

/// The circuit reads the four addresses of the sha2 ecall as addresses of words. Each kernel
/// checks the state that its call leaves before it terminates.
#[test]
fn sha2() {
    for (name, call) in testutil::sha2::accepted() {
        terminates(name, MemoryImage::new_kernel(call.program()));
    }
    for (name, call, message) in testutil::sha2::refused() {
        refuses(name, MemoryImage::new_kernel(call.program()), message);
    }
}

/// The circuit takes one read of a word in a cycle, and the cycle of an instruction reads the
/// instruction, its source registers and the word of a load or of a store.
#[test]
fn reads_a_word_once() {
    for (name, image) in testutil::access::accepted() {
        terminates(name, image);
    }
    for (name, image) in testutil::access::refused() {
        refuses(name, image, "instruction reads a word twice");
    }
}

/// The circuit has no cycle for a trap, and requires of every instruction a pc that is word
/// aligned.
#[test]
fn traps_and_returns() {
    use testutil::{access::user_with, traps};

    // A program that does not trap, under a kernel that handles every trap.
    terminates("one ecall", user_with(traps::ecalls(), true, 0));
    for (name, image, message) in traps::refused() {
        refuses(name, image, message);
    }
}
