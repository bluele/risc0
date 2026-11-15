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

use std::rc::Rc;

use risc0_binfmt::{MemoryImage, Program};
use risc0_zkp::field::Elem;
use test_log::test;

use crate::{
    execute::{
        testutil::{self, NullSyscall, DEFAULT_SESSION_LIMIT},
        DEFAULT_SEGMENT_LIMIT_PO2,
    },
    prove::{hal::StepMode, segment_prover, witgen::WitnessGenerator, PreflightResults},
    zirgen::circuit::{ExtVal, LAYOUT_GLOBAL, REGCOUNT_DATA},
    MAX_INSN_CYCLES,
};

fn run_preflight(program: Program) {
    let image = MemoryImage::new_kernel(program);
    let result = testutil::execute(
        image,
        DEFAULT_SEGMENT_LIMIT_PO2,
        MAX_INSN_CYCLES,
        DEFAULT_SESSION_LIMIT,
        &NullSyscall,
        None,
    )
    .unwrap();
    let segments = result.segments;
    let segment = segments.first().unwrap();

    let mut rng = rand::rng();
    let rand_z = ExtVal::random(&mut rng);

    segment.preflight(rand_z).unwrap();
}

#[test]
fn basic() {
    run_preflight(testutil::kernel::basic());
}

#[test]
fn simple_loop() {
    run_preflight(testutil::kernel::simple_loop(500000));
}

fn fwd_rev_ab_test(program: Program) {
    let image = MemoryImage::new_kernel(program);

    let session = testutil::execute(
        image,
        DEFAULT_SEGMENT_LIMIT_PO2,
        MAX_INSN_CYCLES,
        testutil::DEFAULT_SESSION_LIMIT,
        &testutil::NullSyscall,
        None,
    )
    .unwrap();

    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda")] {
            use risc0_zkp::hal::cuda::CudaHalPoseidon2;
            use crate::prove::hal::cuda::CudaCircuitHalPoseidon2;
            let hal = Rc::new(CudaHalPoseidon2::new());
            let circuit_hal = CudaCircuitHalPoseidon2::new(hal.clone());
        // } else if #[cfg(any(all(target_os = "macos", target_arch = "aarch64"), target_os = "ios"))] {
        //     use risc0_zkp::hal::metal::MetalHalSha256;
        //     use crate::prove::hal::metal::MetalCircuitHal;
        //     let hal = Rc::new(MetalHalSha256::new());
        //     let circuit_hal = MetalCircuitHal::new(hal.clone());
        } else {
            let suite = risc0_zkp::core::hash::poseidon2::Poseidon2HashSuite::new_suite();
            let hal = Rc::new(risc0_zkp::hal::cpu::CpuHal::new(suite));
            let circuit_hal = crate::prove::hal::cpu::CpuCircuitHal;
        }
    }

    let mut rng = rand::rng();
    let rand_z = ExtVal::random(&mut rng);

    let segments = session.segments;
    for segment in segments {
        tracing::debug!("fwd");

        let preflight_results = PreflightResults::new(&segment, rand_z).unwrap();

        let fwd_witgen = WitnessGenerator::new(
            hal.as_ref(),
            &circuit_hal,
            preflight_results.clone(),
            StepMode::SeqForward,
        )
        .unwrap();
        tracing::debug!("rev");
        let rev_witgen = WitnessGenerator::new(
            hal.as_ref(),
            &circuit_hal,
            preflight_results.clone(),
            StepMode::SeqReverse,
        )
        .unwrap();
        let cycles = 1 << segment.po2;
        let fwd_vec = fwd_witgen.data.to_vec();
        let rev_vec = rev_witgen.data.to_vec();
        for row in 0..cycles {
            let fwd_row = &fwd_vec[row * REGCOUNT_DATA..row * REGCOUNT_DATA + REGCOUNT_DATA];
            let rev_row = &rev_vec[row * REGCOUNT_DATA..row * REGCOUNT_DATA + REGCOUNT_DATA];
            assert_eq!(fwd_row, rev_row, "cycle: {row}");
        }
    }
}

#[test]
fn fwd_rev_ab_basic() {
    fwd_rev_ab_test(testutil::kernel::basic());
}

#[test]
fn fwd_rev_ab_split() {
    fwd_rev_ab_test(testutil::kernel::simple_loop(2000));
}

/// The `totalCycles` global must equal the number of rows in the trace: the circuit requires the
/// row preceding each first-cycle row (circularly) to have cycle number `totalCycles - 1`. A
/// prover that claims a different trace length (e.g. half of it, as if the trace held two
/// executions) must fail witness generation.
///
/// The tampered cases use the sequential step modes: the parallel mode does not unwind cleanly
/// on a constraint failure (worker threads keep running while the error propagates).
#[test]
fn total_cycles_mismatch_fails() {
    let image = MemoryImage::new_kernel(testutil::kernel::basic());
    let result = testutil::execute(
        image,
        DEFAULT_SEGMENT_LIMIT_PO2,
        MAX_INSN_CYCLES,
        DEFAULT_SESSION_LIMIT,
        &NullSyscall,
        None,
    )
    .unwrap();
    let segment = result.segments.first().unwrap();
    let prover = segment_prover().unwrap();
    let offset = LAYOUT_GLOBAL.total_cycles._super.offset;
    let total_cycles = 1u32 << segment.po2;

    // Sanity check: the honest value proves and verifies.
    let results = prover.preflight(segment).unwrap();
    assert_eq!(results.global[offset], total_cycles.into());
    let seal = prover.prove_core(results).unwrap();
    crate::verify(&seal).unwrap();

    let suite = risc0_zkp::core::hash::poseidon2::Poseidon2HashSuite::new_suite();
    let hal = risc0_zkp::hal::cpu::CpuHal::new(suite);
    let circuit_hal = crate::prove::hal::cpu::CpuCircuitHal;
    let mut rng = rand::rng();
    let rand_z = ExtVal::random(&mut rng);

    // Claiming a shorter trace (two executions of half the length) or a longer one must fail.
    for (claimed, mode) in [
        (total_cycles / 2, StepMode::SeqForward),
        (total_cycles / 2, StepMode::SeqReverse),
        (total_cycles * 2, StepMode::SeqForward),
    ] {
        let mut results = PreflightResults::new(segment, rand_z).unwrap();
        results.global[offset] = claimed.into();
        match WitnessGenerator::new(&hal, &circuit_hal, results, mode) {
            Ok(_) => panic!("witgen accepted totalCycles = {claimed} ({mode:?})"),
            Err(err) => tracing::debug!("expected failure ({claimed}, {mode:?}): {err:#}"),
        }
    }
}

use crate::prove::tests::proves;

/// Runs a segment through the two phases of the generated circuit: `WitnessGenerator::new`, which
/// reads and writes memory and decides where a bigint program ends, and
/// `WitnessGenerator::accum`, which checks the equations at the point that `mix` gives. Both take
/// the memory of the segment from the preflight, which runs the code of the executor, so that
/// only a proof holds it to the circuit.
fn phases(segment: &crate::execute::Segment) -> Result<(), String> {
    use crate::zirgen::circuit::{Val, REGCOUNT_MIX};

    let suite = risc0_zkp::core::hash::poseidon2::Poseidon2HashSuite::new_suite();
    let hal = risc0_zkp::hal::cpu::CpuHal::new(suite);
    let circuit_hal = crate::prove::hal::cpu::CpuCircuitHal;
    let mut rng = rand::rng();
    let results = PreflightResults::new(segment, ExtVal::random(&mut rng))
        .map_err(|err| format!("the preflight: {err:#}"))?;
    let witgen = WitnessGenerator::new(&hal, &circuit_hal, results, StepMode::SeqForward)
        .map_err(|err| format!("the execution phase: {err:#}"))?;
    let mix: Vec<Val> = (0..REGCOUNT_MIX).map(|_| Val::random(&mut rng)).collect();
    witgen
        .accum(&hal, &circuit_hal, &mix)
        .map_err(|err| format!("the accumulation phase: {err:#}"))?;
    Ok(())
}

/// Changes up to three things in a bigint call that the circuit proves: a field of an instruction
/// of the verify program, the order or the number of its instructions, the size that the blob
/// declares, a register or the mode. The two phases of the circuit have to accept each changed
/// call that the executor accepts, and the prover proves one in 32 of those.
///
/// The changes are the same on every run. `BIGINT_CHANGES` sets how many calls are changed, for
/// a longer run than the default.
#[test]
fn bigint_changes_that_the_executor_accepts_hold_in_the_circuit() {
    use rand::{rngs::StdRng, Rng as _, SeedableRng as _};

    use crate::execute::{
        platform::MACHINE_REGS_ADDR,
        testutil::bigint::{verify_addr, Call, OFFSETS},
    };

    let changes = std::env::var("BIGINT_CHANGES").map_or(512, |count| count.parse().unwrap());
    // Registers that the calls use, and addresses at the ends of what the circuit allows.
    const REGS: [u32; 10] = [0, 2, 10, 11, 12, 13, 14, 15, 28, 31];
    const ADDRS: [u32; 14] = [
        0x0000_0000,
        0x0000_fff0,
        0x0001_0000,
        0x0030_0000,
        0x0030_0002,
        0x0030_0010,
        0xbffe_fff0,
        0xbffe_fff4,
        0xbfff_0000,
        0xbfff_fff0,
        0xc000_0000,
        0xfffe_fff0,
        0xffff_0000,
        0xffff_fff0,
    ];

    fn change(call: &mut Call, rng: &mut StdRng) -> String {
        // The instructions that the blob has, which an earlier change may have made fewer or more
        // than it declares.
        let start = 4 + call.blob[0] as usize;
        let len = (call.blob[1] as usize).min(call.blob.len() - start);
        let index = start + rng.random_range(0..len.max(1));
        match rng.random_range(0..10) {
            0..=4 if len > 0 => {
                let (name, shift, bits, value) = match rng.random_range(0..5) {
                    0 => ("mem_op", 28, 4, rng.random_range(0..4)),
                    1 => ("poly_op", 24, 4, rng.random_range(0..8)),
                    2 => ("coeff", 21, 3, rng.random_range(0..8)),
                    3 => ("reg", 16, 5, REGS[rng.random_range(0..REGS.len())]),
                    _ => ("offset", 0, 16, OFFSETS[rng.random_range(0..OFFSETS.len())]),
                };
                let mask = ((1u32 << bits) - 1) << shift;
                call.blob[index] = (call.blob[index] & !mask) | (value << shift);
                format!("{name} of instruction {} = {value}", index - start)
            }
            5 if len > 1 => {
                let other = start + rng.random_range(0..len);
                call.blob.swap(index, other);
                format!("swap instructions {} and {}", index - start, other - start)
            }
            6 if len > 1 => {
                call.blob.remove(index);
                call.blob[1] -= 1;
                format!("remove instruction {}", index - start)
            }
            7 if len > 0 => {
                let size = [call.blob[1] - 1, call.blob[1] + 1][rng.random_range(0..2)];
                call.blob[1] = size;
                format!("declare {size} instructions")
            }
            8 if !call.regs.is_empty() => {
                let reg = rng.random_range(0..call.regs.len());
                call.regs[reg].1 = match rng.random_range(0..4) {
                    // The words around the instructions, which a cycle reads one of.
                    0 => verify_addr(call).wrapping_add_signed(4 * rng.random_range(-4..=8)),
                    // The registers of the machine, which a cycle reads one of.
                    1 => MACHINE_REGS_ADDR.0 + 4 * rng.random_range(0..32),
                    _ => ADDRS[rng.random_range(0..ADDRS.len())],
                };
                format!("x{} = {:#010x}", call.regs[reg].0, call.regs[reg].1)
            }
            _ => {
                call.mode ^= 1;
                format!("mode {}", call.mode)
            }
        }
    }

    let prover = segment_prover().unwrap();
    let mut rng = StdRng::seed_from_u64(0x6269_6769_6e74);

    // The kernels do not check the memory that a changed call leaves.
    let calls: Vec<(&str, Call)> = testutil::bigint::accepted();
    let mut accepted = 0;
    for round in 0..changes {
        let (name, mut call) = calls[round % calls.len()].clone();
        call.expect.clear();
        let what: Vec<String> = (0..rng.random_range(1..=3))
            .map(|_| change(&mut call, &mut rng))
            .collect();
        let what = format!("{name}: {} (call {round})", what.join(", "));

        // The host program panics on a division by zero, which a changed register can make of a
        // modulus. That ends the execution as an error does.
        let execute = || testutil::run(MemoryImage::new_kernel(call.program()));
        let Ok(Ok(session)) = std::panic::catch_unwind(execute) else {
            continue;
        };
        tracing::debug!("accepted: {what}");
        accepted += 1;

        let segment = session.segments.first().unwrap();
        phases(segment).unwrap_or_else(|err| panic!("{what}: {err}"));
        if accepted % 32 == 0 {
            proves(prover.as_ref(), segment).unwrap_or_else(|err| panic!("{what}: {err:#}"));
        }
    }
    // The changes exercise the circuit, not only the refusals of the executor.
    tracing::info!("the executor accepted {accepted} of {changes} changed calls");
    assert!(accepted >= changes / 16, "{accepted}");
}

/// Requires each of `count` bigint calls that the executor accepts to hold in the circuit.
///
/// The executor first runs kernels that make up to `together` of the calls in turn, and refuses
/// a kernel at the first call that it refuses: the calls before it are accepted, and the calls
/// after it go to another kernel. `together` is 1 for calls that do not leave each other alone.
/// Kernels of up to `batch` accepted calls then go to the circuit. The executor has to accept
/// each, and the two [phases] have to accept it. The prover proves kernels of one accepted call
/// in 64, or in the number of `BIGINT_PROOFS`, or of every accepted call if those make one
/// kernel. `call` gives the call of an index for a block of a kernel, and `name` says what the
/// call is. Returns how many calls the executor accepts.
fn bigint_calls_hold(
    count: u64,
    together: u64,
    batch: usize,
    call: impl Fn(u64, u32) -> testutil::bigint::Call + Sync,
    name: impl Fn(u64) -> String + Sync,
) -> usize {
    use std::{
        cell::{Cell, RefCell},
        panic::{catch_unwind, AssertUnwindSafe},
    };

    use rayon::prelude::*;

    use crate::{
        execute::{Executor, Segment},
        trace::{TraceCallback, TraceEvent},
    };

    let kernel = |indexes: &[u64]| {
        let calls: Vec<_> = (0..)
            .zip(indexes)
            .map(|(block, index)| call(*index, block))
            .collect();
        let (program, ecalls) = testutil::bigint::batch(&calls);
        (MemoryImage::new_kernel(program), ecalls)
    };
    // The first call of a kernel that the executor refuses, if it refuses one. The executor
    // says where each instruction starts, and it refuses a call within the `ecall` of the call.
    // The host program panics where an address wraps, which ends the execution as an error does.
    let refused = |indexes: &[u64]| -> Option<usize> {
        let (image, ecalls) = kernel(indexes);
        let pc = Rc::new(Cell::new(0));
        let start = pc.clone();
        let tracker = move |event: TraceEvent| {
            if let TraceEvent::InstructionStart { pc, .. } = event {
                start.set(pc);
            }
            Ok(())
        };
        let trace: Vec<Rc<RefCell<dyn TraceCallback>>> = vec![Rc::new(RefCell::new(tracker))];
        let run = || {
            Executor::new(image, &NullSyscall, None, trace, None).run(
                DEFAULT_SEGMENT_LIMIT_PO2,
                MAX_INSN_CYCLES,
                DEFAULT_SESSION_LIMIT,
                |_| Ok(()),
            )
        };
        if let Ok(Ok(_)) = catch_unwind(AssertUnwindSafe(run)) {
            return None;
        }
        let call = ecalls.iter().position(|ecall| *ecall == pc.get());
        Some(call.expect("the executor refuses a kernel outside its calls"))
    };
    // Each thread takes a span of calls.
    const SPAN: u64 = 512;
    let spans: Vec<u64> = (0..count).step_by(SPAN as usize).collect();
    let accepted: Vec<u64> = spans
        .par_iter()
        .flat_map_iter(|start| {
            let end = count.min(start + SPAN);
            let (mut next, mut accepted) = (*start, vec![]);
            while next < end {
                let indexes: Vec<u64> = (next..end.min(next + together)).collect();
                // The calls that the executor accepts, and where the calls after them begin.
                let (through, past) = match refused(&indexes) {
                    Some(call) => (call, call + 1),
                    None => (indexes.len(), indexes.len()),
                };
                accepted.extend(&indexes[..through]);
                next += past as u64;
            }
            accepted
        })
        .collect();

    // The fewest calls that the halving of a kernel finds to fail as the kernel does.
    fn narrow(indexes: &[u64], fails: &dyn Fn(&[u64]) -> bool) -> Vec<u64> {
        if indexes.len() > 1 {
            let (left, right) = indexes.split_at(indexes.len() / 2);
            if let Some(half) = [left, right].into_iter().find(|half| fails(half)) {
                return narrow(half, fails);
            }
        }
        indexes.to_vec()
    }
    let fail = |indexes: &[u64], error: String, fails: &dyn Fn(&[u64]) -> bool| -> ! {
        eprintln!("a kernel of {} calls: {error}", indexes.len());
        let names: Vec<String> = narrow(indexes, fails).into_iter().map(&name).collect();
        panic!("{error}\n{}", names.join("\n"));
    };

    let run = |indexes: &[u64]| -> Result<Vec<Segment>, String> {
        let session = testutil::run(kernel(indexes).0);
        Ok(session
            .map_err(|err| format!("the executor: {err:#}"))?
            .segments)
    };
    let holds = |indexes: &[u64]| run(indexes)?.iter().try_for_each(phases);
    let kernels: Vec<&[u64]> = accepted.chunks(batch).collect();
    kernels.par_iter().for_each(|indexes| {
        if let Err(error) = holds(indexes) {
            fail(indexes, error, &|indexes| holds(indexes).is_err());
        }
    });

    let prover = segment_prover().unwrap();
    let proven = |indexes: &[u64]| -> Result<(), String> {
        run(indexes)?.iter().try_for_each(|segment| {
            proves(prover.as_ref(), segment).map_err(|err| format!("the proof: {err:#}"))
        })
    };
    let proofs = std::env::var("BIGINT_PROOFS").map_or(64, |n| n.parse().unwrap());
    let every = if accepted.len() > batch { proofs } else { 1 };
    let some: Vec<u64> = accepted.iter().copied().step_by(every).collect();
    let proved: Vec<&[u64]> = some.chunks(batch).collect();
    for indexes in &proved {
        if let Err(error) = proven(indexes) {
            fail(indexes, error, &|indexes| proven(indexes).is_err());
        }
    }

    tracing::info!(
        "{} of {count} calls accepted, in {} kernels, {} proven",
        accepted.len(),
        kernels.len(),
        proved.len()
    );
    accepted.len()
}

/// Every short verify program that the executor accepts has to hold in the circuit: the
/// programs of up to three of the instructions of `testutil::bigint::short_instructions` that
/// end with their last instruction. `BIGINT_PROGRAM_LENGTH` sets another longest program, for a
/// longer run than the default.
#[test]
fn bigint_short_programs_that_the_executor_accepts_hold_in_the_circuit() {
    use crate::execute::{
        bigint::verify::Instruction,
        testutil::bigint::{short, Short, SHORT_ACCEPTED},
    };

    let longest = std::env::var("BIGINT_PROGRAM_LENGTH").map_or(3, |n| n.parse().unwrap());
    for len in 1..=longest {
        let programs = Short::new(len as u32);
        let accepted = bigint_calls_hold(
            programs.count(),
            32,
            256,
            |index, block| short(&programs.program(index), block),
            |index| {
                let decode = |insn: &u32| Instruction::decode(*insn).unwrap();
                let program: Vec<_> = programs.program(index).iter().map(decode).collect();
                format!("{program:?}")
            },
        );
        if let Some(expected) = SHORT_ACCEPTED.get(len - 1) {
            assert_eq!(accepted, *expected, "{len}");
        }
    }
}

/// Every value of each field of one verify instruction: the program of the instruction and an
/// end has to hold in the circuit if the executor accepts it. By default the instructions are
/// those of `testutil::bigint::operation_instructions`, and a user runs each first. With
/// `BIGINT_FIELDS` those of `operand_instructions` join them, the machine runs each too, and
/// each also comes after another instruction.
#[test]
fn bigint_instructions_that_the_executor_accepts_hold_in_the_circuit() {
    use crate::execute::{
        bigint::verify::Instruction,
        testutil::bigint::{field, operand_instructions, operation_instructions},
    };

    let mut insns = operation_instructions();
    let mut cases = vec![(0, false)];
    if std::env::var_os("BIGINT_FIELDS").is_some() {
        insns.extend(operand_instructions());
        cases.extend([(0, true), (1, false), (1, true)]);
    }
    for (mode, after) in cases {
        // A call may write where another call reads, so each has a kernel of its own.
        let accepted = bigint_calls_hold(
            insns.len() as u64,
            1,
            1,
            |index, _| field(insns[index as usize], mode, after),
            |index| match Instruction::decode(insns[index as usize]) {
                Ok(insn) => format!("{insn:?}"),
                Err(_) => format!("{:#010x}", insns[index as usize]),
            },
        );
        assert!(accepted > 0, "mode {mode}, after another {after}");
    }
}
