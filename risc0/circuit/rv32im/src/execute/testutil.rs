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

use std::collections::BTreeMap;

use anyhow::{bail, Result};
use risc0_binfmt::{MemoryImage, Program};
use risc0_core::scope;
use risc0_zkp::{
    core::{digest::Digest, log2_ceil},
    MAX_CYCLES_PO2,
};

use super::{
    executor::CycleLimit, pager::RESERVED_PAGING_CYCLES, platform::*, syscall::Syscall, Executor,
    SimpleSession, SyscallContext,
};

pub const DEFAULT_SESSION_LIMIT: CycleLimit = CycleLimit::Hard(1 << 24);
pub const MIN_CYCLES_PO2: usize = log2_ceil(RESERVED_CYCLES + RESERVED_PAGING_CYCLES as usize);

#[derive(Default)]
pub struct NullSyscall;

impl Syscall for NullSyscall {
    fn host_read(&self, _ctx: &mut impl SyscallContext, _fd: u32, buf: &mut [u8]) -> Result<u32> {
        for (i, byte) in buf.iter_mut().enumerate() {
            *byte = i as u8;
        }
        Ok(buf.len() as u32)
    }

    fn host_write(&self, _ctx: &mut impl SyscallContext, _fd: u32, _buf: &[u8]) -> Result<u32> {
        unimplemented!()
    }
}

pub fn execute<S: Syscall>(
    image: MemoryImage,
    segment_limit_po2: usize,
    max_insn_cycles: usize,
    max_cycles: CycleLimit,
    syscall_handler: &S,
    input_digest: Option<Digest>,
) -> Result<SimpleSession> {
    scope!("execute");

    if !(MIN_CYCLES_PO2..=MAX_CYCLES_PO2).contains(&segment_limit_po2) {
        bail!("Invalid segment_limit_po2: {segment_limit_po2}");
    }

    let mut segments = Vec::new();
    let trace = Vec::new();
    let result = Executor::new(image, syscall_handler, input_digest, trace, None).run(
        segment_limit_po2,
        max_insn_cycles,
        max_cycles,
        |segment| {
            tracing::trace!("{segment:#?}");
            segments.push(segment);
            Ok(())
        },
    )?;

    Ok(SimpleSession { segments, result })
}

pub mod user {
    use super::*;

    pub fn basic() -> Program {
        let mut asm = Assembler::new();
        asm.li(REG_A1, 0x4000_0000);
        asm.ecall();
        asm.program()
    }

    pub fn simple_loop(count: u32) -> Program {
        let mut asm = Assembler::new();
        asm.addi(REG_A4, REG_ZERO, 0);
        asm.li(REG_A5, count);
        // loop:
        asm.addi(REG_A4, REG_A4, 1);
        asm.blt(REG_A4, REG_A5, -4 /*loop: */);
        asm.lui(REG_A1, 0x1000);
        asm.ecall();
        asm.program()
    }
}

pub mod kernel {
    use super::*;

    pub fn basic() -> Program {
        let mut asm = Assembler::new();
        asm.host_terminate(0, 0);
        asm.program()
    }

    pub fn simple_loop(count: u32) -> Program {
        let mut asm = Assembler::new();
        asm.addi(REG_A4, REG_ZERO, 0);
        asm.li(REG_A5, count);
        // loop:
        asm.addi(REG_A4, REG_A4, 1);
        asm.blt(REG_A4, REG_A5, -4 /*loop: */);
        asm.host_terminate(0, 0);
        asm.program()
    }

    pub fn multi_read() -> Program {
        const LENGTHS: &[u32] = &[0, 1, 2, 3, 4, 5, 7, 13, 19, 40, 101];

        let ptr = 0x0050_0000;

        let mut asm = Assembler::new();
        asm.li(REG_T0, ptr);
        // Try all 4 alignments
        for i in 0..4 {
            // Try a variety of size
            for &len in LENGTHS {
                asm.host_ecall_read(0, ptr + i, len);
                for k in 0..len {
                    asm.lb(REG_T1, REG_T0, i + k);
                    asm.li(REG_T2, k);
                    asm.beq(REG_T1, REG_T2, 8);
                    asm.die();
                }
            }
        }

        asm.host_terminate(0, 0);

        asm.program()
    }
}

#[allow(unused)]
mod consts {
    pub(crate) const OP_BASE: u32 = 0b0110011;
    pub(crate) const OP_IMM: u32 = 0b0010011;
    pub(crate) const OP_LOAD: u32 = 0b0000011;
    pub(crate) const OP_STORE: u32 = 0b0100011;
    pub(crate) const OP_BRANCH: u32 = 0b1100011;
    pub(crate) const OP_JAL: u32 = 0b1101111;
    pub(crate) const OP_JALR: u32 = 0b1100111;
    pub(crate) const OP_LUI: u32 = 0b0110111;
    pub(crate) const OP_AUIPC: u32 = 0b0010111;
    pub(crate) const OP_ENV: u32 = 0b1110011;

    pub(crate) const FUNCT3_EQ: u32 = 0x0;
    pub(crate) const FUNCT3_NE: u32 = 0x1;
    pub(crate) const FUNCT3_LT: u32 = 0x4;
    pub(crate) const FUNCT3_GE: u32 = 0x5;
    pub(crate) const FUNCT3_LTU: u32 = 0x6;
    pub(crate) const FUNCT3_GEU: u32 = 0x7;

    pub(crate) const FUNCT3_BYTE: u32 = 0x0;
    pub(crate) const FUNCT3_HALF: u32 = 0x1;
    pub(crate) const FUNCT3_WORD: u32 = 0x2;
    pub(crate) const FUNCT3_BYTEU: u32 = 0x4;
    pub(crate) const FUNCT3_HALFU: u32 = 0x5;
}

use consts::*;

struct Assembler {
    text: Vec<u32>,
    data: BTreeMap<u32, u32>,
}

#[allow(dead_code)]
impl Assembler {
    pub fn new() -> Self {
        Self {
            text: vec![],
            data: BTreeMap::new(),
        }
    }

    pub fn program(&self) -> Program {
        let entry = USER_START_ADDR + WORD_SIZE;
        let entry = entry.0;
        let mut pc = entry;

        let mut image: BTreeMap<_, _> = self
            .text
            .iter()
            .map(|instr| {
                let result = (pc, *instr);
                pc += WORD_SIZE as u32;
                result
            })
            .collect();

        image.extend(self.data.iter());

        Program::new_from_entry_and_image(entry, image)
    }

    pub fn word(&mut self, addr: u32, word: u32) {
        self.data.insert(addr, word);
    }

    pub fn add(&mut self, rd: usize, rs1: usize, rs2: usize) {
        self.text.push(insn_r(
            0x00, rs2 as u32, rs1 as u32, 0x0, rd as u32, OP_BASE,
        ));
    }

    pub fn addi(&mut self, rd: usize, rs1: usize, imm: u32) {
        self.text
            .push(insn_i(imm, rs1 as u32, 0x0, rd as u32, OP_IMM));
    }

    pub fn blt(&mut self, rs1: usize, rs2: usize, offset: i32) {
        self.text.push(insn_b(
            offset as u32,
            rs2 as u32,
            rs1 as u32,
            FUNCT3_LT,
            OP_BRANCH,
        ));
    }

    pub fn beq(&mut self, rs1: usize, rs2: usize, offset: i32) {
        self.text.push(insn_b(
            offset as u32,
            rs2 as u32,
            rs1 as u32,
            FUNCT3_EQ,
            OP_BRANCH,
        ));
    }

    pub fn bne(&mut self, rs1: usize, rs2: usize, offset: i32) {
        self.text.push(insn_b(
            offset as u32,
            rs2 as u32,
            rs1 as u32,
            FUNCT3_NE,
            OP_BRANCH,
        ));
    }

    pub fn ecall(&mut self) {
        self.text.push(insn_i(0x0, 0x0, 0x0, 0x0, OP_ENV));
    }

    pub fn lui(&mut self, rd: usize, imm: u32) {
        self.text.push(insn_u(imm, rd as u32, OP_LUI));
    }

    pub fn li(&mut self, rd: usize, imm: u32) {
        if imm < (1 << 11) {
            self.addi(rd, REG_ZERO, imm);
        } else {
            // sign extend low 12 bits
            let low = ((imm as i32) << 20) >> 20;
            // upper 20 bits
            let high = (imm as i32 - low) >> 12;

            self.lui(rd, high as u32);
            self.addi(rd, rd, low as u32);
        }
    }

    pub fn lb(&mut self, rd: usize, rs1: usize, imm: u32) {
        self.text
            .push(insn_i(imm, rs1 as u32, FUNCT3_BYTE, rd as u32, OP_LOAD));
    }

    pub fn lw(&mut self, rd: usize, rs1: usize, imm: u32) {
        self.text
            .push(insn_i(imm, rs1 as u32, FUNCT3_WORD, rd as u32, OP_LOAD));
    }

    pub fn host_ecall_read(&mut self, fd: u32, ptr: u32, len: u32) {
        self.li(REG_A7, HOST_ECALL_READ);
        self.li(REG_A0, fd);
        self.li(REG_A1, ptr);
        self.li(REG_A2, len);
        self.ecall();
    }

    pub fn host_terminate(&mut self, a0: u32, a1: u32) {
        self.li(REG_A7, HOST_ECALL_TERMINATE);
        self.li(REG_A0, a0);
        self.li(REG_A1, a1);
        self.ecall();
    }

    pub fn die(&mut self) {
        self.text.push(fence());
    }
}

// 31        25 | 24  20 | 19  15 | 14  12 | 11        7 | 6    0 |
//    funct7    |   rs2  |   rs1  | funct3 |      rd     | opcode |
fn insn_r(funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

// 31                 20 | 19  15 | 14  12 | 11        7 | 6    0 |
//     imm[11:0]         |   rs1  | funct3 |      rd     | opcode |
fn insn_i(imm: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    (imm << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

// 31        25 | 24  20 | 19  15 | 14  12 | 11        7 | 6    0 |
// imm[12|10:5] |   rs2  |   rs1  | funct3 | imm[4:1|11] | opcode |
fn insn_b(imm: u32, rs2: u32, rs1: u32, funct3: u32, opcode: u32) -> u32 {
    let imm_12 = (imm >> 12) & 0b1;
    let imm_10_5 = (imm >> 5) & 0b111111;
    let imm_11 = (imm >> 11) & 0b1;
    let imm_4_1 = (imm >> 1) & 0b1111;
    (((imm_12 << 6) | imm_10_5) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (((imm_4_1 << 1) | imm_11) << 7)
        | opcode
}

// 31                                   12 | 11        7 | 6    0 |
//    imm[31:12]                           |      rd     | opcode |
fn insn_u(imm: u32, rd: u32, opcode: u32) -> u32 {
    (imm << 12) | (rd << 7) | opcode
}

fn fence() -> u32 {
    insn_i(0, 0, 0, 0, 0b0001111)
}

/// Runs an image with the limits that the tests of this crate use.
#[cfg(test)]
pub(crate) fn run(image: MemoryImage) -> Result<SimpleSession> {
    execute(
        image,
        super::DEFAULT_SEGMENT_LIMIT_PO2,
        crate::MAX_INSN_CYCLES,
        DEFAULT_SESSION_LIMIT,
        &NullSyscall,
        None,
    )
}

/// The word of the register `reg` of the machine.
#[cfg(test)]
pub(crate) const fn machine_reg(reg: usize) -> u32 {
    MACHINE_REGS_ADDR.0 + (reg * WORD_SIZE) as u32
}

#[cfg(test)]
impl Assembler {
    /// The address of the next instruction, for a program that runs where [Self::program] puts
    /// it.
    fn here(&self) -> u32 {
        USER_START_ADDR.0 + ((1 + self.text.len()) * WORD_SIZE) as u32
    }

    /// [Self::li] in two instructions whatever the value, neither of which reads the register
    /// zero.
    fn li2(&mut self, rd: usize, imm: u32) {
        let low = ((imm as i32) << 20) >> 20;
        let high = (imm.wrapping_sub(low as u32)) >> 12;
        self.lui(rd, high);
        self.addi(rd, rd, low as u32 & 0xfff);
    }

    fn jalr(&mut self, rd: usize, rs1: usize, imm: u32) {
        self.text
            .push(insn_i(imm, rs1 as u32, 0x0, rd as u32, OP_JALR));
    }

    fn host_ecall_poseidon2(
        &mut self,
        state_addr: u32,
        buf_in_addr: u32,
        buf_out_addr: u32,
        bits_count: u32,
    ) {
        self.li(REG_A7, HOST_ECALL_POSEIDON2);
        self.li(REG_A0, state_addr);
        self.li(REG_A1, buf_in_addr);
        self.li(REG_A2, buf_out_addr);
        self.li(REG_A3, bits_count);
        self.ecall();
    }
}

/// A kernel that jumps with `jalr` to the instruction after it, with `low` added to the address
/// in the register.
#[cfg(test)]
pub(crate) fn jalr(low: u32) -> Program {
    let mut asm = Assembler::new();
    // The entry, two words of `li` and the `jalr`.
    let target = USER_START_ADDR.0 + (4 * WORD_SIZE) as u32;
    asm.li(REG_T0, target + low);
    asm.jalr(REG_RA, REG_T0, 0);
    asm.host_terminate(0, 0);
    asm.program()
}

/// Kernels that call the poseidon2 ecall with byte addresses, as the circuit reads them.
#[cfg(test)]
pub(crate) mod poseidon2 {
    use super::*;

    /// Where the output and the state of [calls] lie: in user memory, and where else a kernel may
    /// have them, as its addresses are not held to user memory: registers of the machine, kernel
    /// memory, and the 64 KiB below user memory.
    pub(crate) fn places() -> [(u32, u32); 4] {
        let registers = machine_reg(REG_S2);
        let kernel = KERNEL_START_ADDR.0 + 0x0010_0000;
        [
            (0x0030_0000, 0x0040_0000),
            (registers, kernel),
            (kernel, registers),
            (0x0100, 0x0200),
        ]
    }

    /// A kernel with the output of its calls at `buf_out` and its state at `state`, neither of
    /// them a register that it sets for a call. A call with `PFLAG_CHECK_OUT` fails unless the
    /// ecall computes the digest at its output address.
    pub(crate) fn calls(buf_out: u32, state: u32) -> Program {
        // The digest of the field elements 0..16 without state
        // (`execute::poseidon2::tests::takes_byte_addresses`).
        const DIGEST: [u32; 8] = [
            1749308481, 879447913, 499502012, 1842374203, 1869354733, 71489094, 19273002, 690566044,
        ];
        const BUF_IN: u32 = 0x0010_0000;
        const DIGEST_ADDR: u32 = 0x0020_0000;

        let mut asm = Assembler::new();
        for i in 0..16 {
            asm.word(BUF_IN + i * WORD_SIZE as u32, i);
        }
        for (i, word) in (0..).zip(DIGEST) {
            asm.word(DIGEST_ADDR + i * WORD_SIZE as u32, word);
        }
        for i in 0..8 {
            asm.word(state + i * WORD_SIZE as u32, 100 + i);
        }

        // Check the known digest, then write it and check what was written.
        asm.host_ecall_poseidon2(0, BUF_IN, DIGEST_ADDR, PFLAG_IS_ELEM | PFLAG_CHECK_OUT | 1);
        asm.host_ecall_poseidon2(0, BUF_IN, buf_out, PFLAG_IS_ELEM | 1);
        asm.host_ecall_poseidon2(0, BUF_IN, buf_out, PFLAG_IS_ELEM | PFLAG_CHECK_OUT | 1);
        // Absorb two blocks of bytes into the state in memory.
        asm.host_ecall_poseidon2(state, BUF_IN, buf_out, 2);
        asm.host_terminate(0, 0);

        asm.program()
    }
}

/// Kernels that make one bigint ecall, and the blobs they make it with.
#[cfg(test)]
pub(crate) mod bigint {
    use std::io::Cursor;

    use malachite::Natural;

    use super::*;
    use crate::execute::bibc::{self, OpCode};

    // The memory operations of a verify instruction.
    pub(crate) const READ: u32 = 0;
    pub(crate) const WRITE: u32 = 1;
    pub(crate) const CHECK: u32 = 2;

    // The polynomial operations of a verify instruction.
    pub(crate) const RESET: u32 = 0;
    pub(crate) const SHIFT: u32 = 1;
    pub(crate) const SET_TERM: u32 = 2;
    pub(crate) const ADD_TOTAL: u32 = 3;
    pub(crate) const CARRY1: u32 = 4;
    pub(crate) const CARRY2: u32 = 5;
    pub(crate) const EQ_ZERO: u32 = 6;

    /// One word of a verify program.
    pub(crate) fn insn(poly_op: u32, mem_op: u32, coeff: i32, reg: usize, offset: u32) -> u32 {
        (mem_op << 28)
            | (poly_op << 24)
            | (((coeff + 4) as u32) << 21)
            | ((reg as u32) << 16)
            | offset
    }

    /// The instruction that ends the programs of the bigint2 library.
    pub(crate) fn end() -> u32 {
        insn(RESET, CHECK, 0, REG_ZERO, 0)
    }

    pub(crate) const BLOB: u32 = 0x0010_0000;
    /// Sixteen bytes that stay zero.
    pub(crate) const ZERO: u32 = 0x0020_0000;
    pub(crate) const OUT_A: u32 = 0x0030_0000;
    pub(crate) const OUT_B: u32 = 0x0030_0010;
    /// What the first word of `OUT_A` and of `OUT_B` holds before the call.
    pub(crate) const SENTINEL: u32 = 0x5a5a_5a5a;

    /// A bigint blob whose host program only stores constants.
    #[derive(Clone, Default)]
    pub(crate) struct Blob {
        /// The 16 bytes the host program stores as witness, by register and offset.
        pub stores: Vec<(usize, u32, u128)>,
        pub verify: Vec<u32>,
        /// The size of the verify program that the blob declares, if not its length.
        pub verify_size: Option<u32>,
    }

    impl Blob {
        /// The host program, in the bytecode of `bibc`: for each store, a constant of two 64-bit
        /// words and the store of it.
        fn nondet(&self) -> Vec<u32> {
            let count = self.stores.len() as u32;
            let mut bytes = b"bibc".to_vec();
            // version, inputs, types, constants, operations
            for field in [1, 0, 1, 2 * count, 2 * count] {
                bytes.extend(field.to_le_bytes());
            }
            // The only type: 16 coefficients, each at most 255 and none negative.
            for field in [16u64, 255, 0, 0] {
                bytes.extend(field.to_le_bytes());
            }
            for (_, _, value) in &self.stores {
                bytes.extend(value.to_le_bytes());
            }
            for (i, (reg, offset, _)) in (0u64..).zip(&self.stores) {
                let constant = 0x2 | ((2 * i) << 16) | (2 << 40);
                let target = ((*reg as u64) << 16) | *offset as u64;
                let store = 0x4 | (target << 16) | ((2 * i) << 40);
                bytes.extend(constant.to_le_bytes());
                bytes.extend(store.to_le_bytes());
            }
            words(&bytes)
        }

        /// The blob as a guest holds it: the three sizes and a zero, then the three parts.
        pub(crate) fn words(&self) -> Vec<u32> {
            let nondet = self.nondet();
            let verify_size = self.verify_size.unwrap_or(self.verify.len() as u32);
            let mut blob = vec![nondet.len() as u32, verify_size, 0, 0];
            blob.extend(nondet);
            blob.extend(&self.verify);
            blob
        }
    }

    pub(crate) fn words(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(WORD_SIZE)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }

    /// A kernel that makes one bigint ecall and then requires some words of memory.
    #[derive(Clone, Default)]
    pub(crate) struct Call {
        pub blob: Vec<u32>,
        pub mode: u32,
        /// The registers the programs of the blob use.
        pub regs: Vec<(usize, u32)>,
        /// Words of memory before the call.
        pub memory: Vec<(u32, u32)>,
        /// Words of memory after the call. The kernel fails on another value.
        pub expect: Vec<(u32, u32)>,
    }

    impl Call {
        pub(crate) fn program(&self) -> Program {
            batch(std::slice::from_ref(self)).0
        }
    }

    /// The bytes that a call of [batch] has for its blob and for what its programs name.
    pub(crate) const BLOCK: u32 = 0x100;

    /// A kernel that makes the calls in turn and then terminates, with the blob of each call a
    /// block past the blob of the call before it, and the address of the `ecall` of each call.
    /// The kernel loads a register with `lui` and an `addi` to the register itself, which does
    /// not read the register zero: a call of the machine can write the word of that register.
    pub(crate) fn batch(calls: &[Call]) -> (Program, Vec<u32>) {
        let blobs = || (BLOB..).step_by(BLOCK as usize).zip(calls);
        let mut asm = Assembler::new();
        let mut ecalls = vec![];
        for (blob, call) in blobs() {
            for (addr, word) in (blob..).step_by(WORD_SIZE).zip(&call.blob) {
                asm.word(addr, *word);
            }
            for (addr, word) in &call.memory {
                asm.word(*addr, *word);
            }
        }
        for (blob, call) in blobs() {
            let nondet = blob + 4 * WORD_SIZE as u32;
            let verify = nondet + call.blob[0] * WORD_SIZE as u32;
            let consts = verify + call.blob[1] * WORD_SIZE as u32;
            let ecall = [
                (REG_A0, blob),
                (REG_T0, call.mode),
                (REG_T1, nondet),
                (REG_T2, verify),
                (REG_T3, consts),
            ];
            for (reg, value) in ecall.iter().chain(&call.regs) {
                asm.li2(*reg, *value);
            }
            asm.li2(REG_A7, HOST_ECALL_BIGINT);
            ecalls.push(asm.here());
            asm.ecall();

            for (addr, word) in &call.expect {
                asm.li2(REG_T4, *addr);
                asm.lw(REG_T5, REG_T4, 0);
                asm.li2(REG_T6, *word);
                asm.beq(REG_T5, REG_T6, 8);
                asm.die();
            }
        }
        asm.li2(REG_A7, HOST_ECALL_TERMINATE);
        asm.li2(REG_A0, 0);
        asm.li2(REG_A1, 0);
        asm.ecall();
        (asm.program(), ecalls)
    }

    /// A call with the registers and the memory of the programs below: `a1` and `a2` point to
    /// `OUT_A` and `OUT_B`, which hold the sentinel, and `a3` points to `ZERO`.
    pub(crate) fn call(blob: &Blob) -> Call {
        Call {
            blob: blob.words(),
            mode: 0,
            regs: vec![(REG_A1, OUT_A), (REG_A2, OUT_B), (REG_A3, ZERO)],
            memory: vec![(OUT_A, SENTINEL), (OUT_B, SENTINEL)],
            expect: vec![],
        }
    }

    /// The host stores 1 for `OUT_A` and 2 for `OUT_B`. The verify program writes `OUT_A`, reads
    /// it back and requires the two to cancel. It never names `OUT_B`.
    pub(crate) fn valid() -> Blob {
        Blob {
            stores: vec![(REG_A1, 0, 1), (REG_A2, 0, 2)],
            verify: vec![
                insn(ADD_TOTAL, WRITE, 1, REG_A1, 0),
                insn(ADD_TOTAL, READ, -1, REG_A1, 0),
                insn(EQ_ZERO, READ, 0, REG_A3, 0),
                end(),
            ],
            ..Default::default()
        }
    }

    /// [valid], but the second instruction reads `ZERO`, so that the total stays 1.
    pub(crate) fn invalid_eqz() -> Blob {
        let mut blob = valid();
        blob.verify[1] = insn(ADD_TOTAL, READ, -1, REG_A3, 0);
        blob
    }

    /// `OUT_A` holds the sentinel. The verify program reads it, writes 1 over it and reads that
    /// back, then cancels the sentinel against `OUT_B`: it holds only if each read sees the
    /// writes before it and none after it.
    pub(crate) fn read_write_read() -> Blob {
        Blob {
            stores: vec![(REG_A1, 0, 1)],
            verify: vec![
                insn(ADD_TOTAL, READ, 1, REG_A1, 0),
                insn(ADD_TOTAL, WRITE, 1, REG_A1, 0),
                insn(ADD_TOTAL, READ, -1, REG_A1, 0),
                insn(ADD_TOTAL, READ, -1, REG_A2, 0),
                insn(EQ_ZERO, READ, 0, REG_A3, 0),
                end(),
            ],
            ..Default::default()
        }
    }

    /// `OUT_B` holds 16 and the host stores 256 for `OUT_A`. The verify program requires
    /// `sign * (16 * 16 - 256) == 0`, which holds for the numbers and not for their bytes: the
    /// equation takes a carry of `sign`.
    pub(crate) fn carry(sign: i32) -> Call {
        let blob = Blob {
            stores: vec![(REG_A1, 0, 256)],
            verify: vec![
                insn(SET_TERM, READ, 0, REG_A2, 0),
                insn(ADD_TOTAL, READ, sign, REG_A2, 0),
                insn(ADD_TOTAL, WRITE, -sign, REG_A1, 0),
                insn(CARRY1, CHECK, 0, REG_ZERO, 0),
                insn(CARRY2, CHECK, 0, REG_ZERO, 0),
                insn(EQ_ZERO, CHECK, 0, REG_ZERO, 0),
                end(),
            ],
            ..Default::default()
        };
        Call {
            memory: vec![(OUT_A, SENTINEL), (OUT_B, 16)],
            expect: vec![(OUT_A, 256)],
            ..call(&blob)
        }
    }

    /// A value of 32 bytes: the verify program reads its two halves, writes the two halves of the
    /// witness elsewhere and requires the two values to be equal.
    pub(crate) fn wide() -> Call {
        const FROM: u32 = 0x0030_0100;
        const TO: u32 = 0x0030_0200;
        let blob = Blob {
            stores: vec![(REG_A4, 0, 7), (REG_A4, 1, 9)],
            verify: vec![
                insn(SHIFT, READ, 0, REG_A5, 1),
                insn(ADD_TOTAL, READ, 1, REG_A5, 0),
                insn(SHIFT, WRITE, 0, REG_A4, 1),
                insn(ADD_TOTAL, WRITE, -1, REG_A4, 0),
                insn(EQ_ZERO, READ, 0, REG_A3, 0),
                end(),
            ],
            ..Default::default()
        };
        let mut call = call(&blob);
        call.regs.extend([(REG_A4, TO), (REG_A5, FROM)]);
        call.memory.extend([(FROM, 7), (FROM + 16, 9)]);
        call.expect = vec![(TO, 7), (TO + 16, 9)];
        call
    }

    /// The first instruction writes the four words after it, which are not instructions before
    /// that: the program holds only if it runs the words as written.
    pub(crate) fn rewrites_itself() -> Call {
        const BAD: u32 = 0xffff_ffff;
        let after = [insn(EQ_ZERO, READ, 0, REG_A3, 0), end(), BAD, BAD];
        let value = (0..).zip(after).fold(0u128, |value, (i, word)| {
            value | (u128::from(word) << (32 * i))
        });
        let blob = Blob {
            stores: vec![(REG_A4, 0, value)],
            verify: vec![insn(ADD_TOTAL, WRITE, 0, REG_A4, 0), BAD, BAD, BAD, BAD],
            ..Default::default()
        };
        let mut call = call(&blob);
        let second = BLOB + (4 + call.blob[0] + 1) * WORD_SIZE as u32;
        call.regs.push((REG_A4, second));
        call
    }

    /// Reads the 16 bytes at `addr` and requires nothing of them.
    pub(crate) fn reads(mode: u32, addr: u32) -> Call {
        reads_from(mode, addr, 0)
    }

    /// Reads the 16 bytes that are `offset` times 16 bytes past `base`.
    pub(crate) fn reads_from(mode: u32, base: u32, offset: u32) -> Call {
        let blob = Blob {
            verify: vec![insn(SET_TERM, READ, 0, REG_A5, offset), end()],
            ..Default::default()
        };
        let mut call = call(&blob);
        call.mode = mode;
        call.regs.push((REG_A5, base));
        call
    }

    /// Takes the carries of an empty total at `offset`, as the programs of the bigint2 library
    /// take theirs: with a `Check` that names the register zero.
    pub(crate) fn carries_at(offset: u32) -> Call {
        call(&Blob {
            verify: vec![
                insn(CARRY1, CHECK, 0, REG_ZERO, offset),
                insn(EQ_ZERO, CHECK, 0, REG_ZERO, offset),
                end(),
            ],
            ..Default::default()
        })
    }

    /// Writes 3 to the 16 bytes at `addr`, which the instruction names by `reg`: `a5` holds
    /// `addr`, and the host stores the witness for it.
    pub(crate) fn writes(mode: u32, reg: usize, addr: u32) -> Call {
        let blob = Blob {
            stores: vec![(REG_A5, 0, 3)],
            verify: vec![insn(SET_TERM, WRITE, 0, reg, 0), end()],
            ..Default::default()
        };
        let mut call = call(&blob);
        call.mode = mode;
        call.regs.push((REG_A5, addr));
        call
    }

    /// The word of the register `a5` of the machine.
    pub(crate) const A5_ADDR: u32 = machine_reg(REG_A5);

    /// The address of the first instruction of the verify program in the blob of `call`.
    pub(crate) fn verify_addr(call: &Call) -> u32 {
        BLOB + (4 + call.blob[0]) * WORD_SIZE as u32
    }

    /// Reads 16 bytes that begin `words` words past the instruction that reads them, or before
    /// it. The instruction is among them for -3 to 0.
    pub(crate) fn reads_near_itself(words: i32) -> Call {
        let mut call = reads(0, 0);
        let addr = verify_addr(&call).wrapping_add_signed(words * WORD_SIZE as i32);
        call.regs.last_mut().unwrap().1 = addr;
        // The words past the blob.
        let end = BLOB + (call.blob.len() * WORD_SIZE) as u32;
        call.memory.push((end, 0));
        call
    }

    /// The first instruction writes the 16 bytes that begin with itself, and the second ends the
    /// program as it did before.
    pub(crate) fn writes_itself() -> Call {
        let after = u128::from(end()) | (u128::from(end()) << 32);
        let blob = Blob {
            stores: vec![(REG_A4, 0, after)],
            verify: vec![insn(SET_TERM, WRITE, 0, REG_A4, 0), end()],
            ..Default::default()
        };
        let mut call = call(&blob);
        call.regs.push((REG_A4, verify_addr(&call)));
        call
    }

    /// The machine reads 16 bytes of its registers, from the register `first` on. The
    /// instruction names `a5`, which holds the address of those bytes.
    pub(crate) fn reads_registers(first: usize) -> Call {
        reads(1, machine_reg(first))
    }

    /// A verify program in the registers `t3` and `t4` of the machine: two resets, of which the
    /// first names the register `reg`. The blob declares two instructions, and `t2` names the
    /// registers instead of them.
    pub(crate) fn in_registers(reg: usize) -> Call {
        let blob = Blob {
            verify: vec![end(), end()],
            ..Default::default()
        };
        let mut call = call(&blob);
        call.mode = 1;
        call.regs.extend([
            (REG_T2, machine_reg(REG_T3)),
            (REG_T3, insn(RESET, CHECK, 0, reg, 0)),
            (REG_T4, end()),
        ]);
        call
    }

    macro_rules! blobs {
        ($($dir:literal: [$($name:literal),* $(,)?]),* $(,)?) => {
            [$($((
                $name,
                include_bytes!(concat!("../../../../", $dir, "/", $name, ".blob")).as_slice(),
            )),*),*]
        };
    }

    /// The programs that the bigint2 library embeds, and the two that the v1compat kernel runs
    /// for the older ecall.
    pub(crate) const EMBEDDED: [(&str, &[u8]); 23] = blobs! {
        "bigint2/src/ec": ["ec_add_256", "ec_add_384", "ec_double_256", "ec_double_384"],
        "bigint2/src/field": [
            "extfield_deg2_add_256", "extfield_deg2_add_384", "extfield_deg2_mul_256",
            "extfield_deg2_sub_256", "extfield_deg2_sub_384", "extfield_deg4_mul_256",
            "extfield_xxone_mul_256", "extfield_xxone_mul_384", "modadd_256", "modadd_384",
            "modinv_256", "modinv_384", "modmul_256", "modmul_384", "modmul_4096", "modsub_256",
            "modsub_384",
        ],
        "zkos/v1compat/src/bigint_v1compat": ["modmul_256", "mul_256"],
    };

    /// Blobs that the bigint2 library has and does not use.
    pub(crate) const UNUSED: [&str; 3] = ["extfieldadd_256", "extfieldmul_256", "extfieldsub_256"];

    fn embedded(name: &str) -> &'static [u8] {
        let (_, bytes) = EMBEDDED.iter().find(|(blob, _)| *blob == name).unwrap();
        bytes
    }

    /// The prime of secp256k1 for 32 bytes and of BLS12-381 for 48. For other sizes a number
    /// that is not a prime, which only an inverse needs.
    pub(crate) fn prime(bytes: u32) -> Natural {
        match bytes {
            32 => Natural::from_limbs_asc(&[
                0xffff_fc2f,
                0xffff_fffe,
                0xffff_ffff,
                0xffff_ffff,
                0xffff_ffff,
                0xffff_ffff,
                0xffff_ffff,
                0xffff_ffff,
            ]),
            48 => Natural::from_limbs_asc(&[
                0xffff_aaab,
                0xb9fe_ffff,
                0xb153_ffff,
                0x1eab_fffe,
                0xf6b0_f624,
                0x6730_d2a0,
                0xf385_12bf,
                0x6477_4b84,
                0x434b_acd7,
                0x4b1b_a7b6,
                0x397f_e69a,
                0x1a01_11ea,
            ]),
            _ => {
                let mut limbs = vec![0; bytes as usize / WORD_SIZE];
                limbs[0] = 1;
                *limbs.last_mut().unwrap() = 0x8000_0000;
                Natural::from_limbs_asc(&limbs)
            }
        }
    }

    /// A call of a blob of the bigint2 library. Each register that the host program names gets
    /// 64 KiB of its own. `values` gives what the host program loads, by the index of the load
    /// among its operations. A load that it does not name gets a prime if the host program
    /// divides by it, the square of that prime if it is twice as long, and else a small number,
    /// which the prime reduces.
    pub(crate) fn library(bytes: &[u8], values: &[(usize, Natural)]) -> Call {
        let blob = words(bytes);
        let nondet = &bytes[4 * WORD_SIZE..][..blob[0] as usize * WORD_SIZE];
        let program = bibc::Program::decode(&mut Cursor::new(nondet)).unwrap();
        let size =
            |op: &bibc::Op| (program.types[op.result_type].coeffs as u32).next_multiple_of(16);

        let divisors: Vec<usize> = program
            .ops
            .iter()
            .filter(|op| matches!(op.code, OpCode::Rem | OpCode::Quo | OpCode::Inv))
            .map(|op| op.b)
            .filter(|index| matches!(program.ops[*index].code, OpCode::Load))
            .collect();
        let field = divisors.first().map(|index| size(&program.ops[*index]));

        let mut call = Call {
            blob,
            ..Default::default()
        };
        for (index, op) in program.ops.iter().enumerate() {
            if !matches!(op.code, OpCode::Load | OpCode::Store) {
                continue;
            }
            let base = 0x0100_0000 + op.arena() * 0x1_0000;
            if !call.regs.contains(&(op.arena() as usize, base)) {
                call.regs.push((op.arena() as usize, base));
            }
            if !matches!(op.code, OpCode::Load) {
                continue;
            }
            let named = values.iter().find(|(load, _)| *load == index);
            let value = if let Some((_, value)) = named {
                value.clone()
            } else if divisors.contains(&index) {
                prime(size(op))
            } else if let Some(field) = field.filter(|field| size(op) == 2 * field) {
                prime(field) * prime(field)
            } else {
                Natural::from(index as u32 + 2)
            };
            let mut limbs = value.to_limbs_asc();
            limbs.resize(size(op) as usize / WORD_SIZE, 0);
            for (i, limb) in (0u32..).zip(limbs) {
                let addr = base + (op.offset() * 16) + i * WORD_SIZE as u32;
                call.memory.push((addr, limb));
            }
        }
        call
    }

    /// Each embedded program on operands that the circuit proves it for. The v1compat kernel
    /// runs its two in machine mode.
    pub(crate) fn library_calls() -> Vec<(&'static str, Call)> {
        let mut calls: Vec<_> = EMBEDDED
            .iter()
            .map(|(name, bytes)| (*name, library(bytes, &[])))
            .collect();
        for (_, call) in &mut calls[21..] {
            call.mode = 1;
        }
        calls
    }

    /// The 256-bit modular subtraction of the bigint2 library, on operands of one word.
    pub(crate) fn modsub_256(lhs: u32, rhs: u32, modulus: u32) -> Call {
        let [lhs, rhs, modulus] = [lhs, rhs, modulus].map(Natural::from);
        library(embedded("modsub_256"), &[(0, lhs), (1, rhs), (2, modulus)])
    }

    /// Where [modsub_256] has its result written: the 64 KiB of `a4`.
    pub(crate) const MODSUB_RESULT: u32 = 0x0100_0000 + 14 * 0x1_0000;

    /// Calls of the embedded programs that subtract, on operands that the circuit cannot prove
    /// them for. Each host program adds a multiple of the modulus to a difference and divides
    /// the sum. Here the sum is negative: the division rounds towards zero and the host program
    /// stores the absolute values of the quotient and the remainder, which do not satisfy the
    /// verify program. The last call overlaps two operands instead.
    pub(crate) fn divergent() -> Vec<(&'static str, Call)> {
        let n = |value: u32| Natural::from(value);
        let mut calls = vec![];
        for (bits, bytes) in [("256", 32), ("384", 48)] {
            let p = prime(bytes);
            let ones = Natural::from_limbs_asc(&vec![u32::MAX; bytes as usize / WORD_SIZE]);
            let mut push = |name: &str, values: &[(usize, Natural)]| {
                let name = format!("{name}_{bits}");
                let (name, bytes) = EMBEDDED.iter().find(|(blob, _)| *blob == name).unwrap();
                calls.push((*name, library(bytes, values)));
            };

            // rhs > lhs + modulus. The host program stores 2 as the result, and the difference
            // of 0 and 5 modulo 3 is 1.
            push("modsub", &[(0, n(0)), (1, n(5)), (2, n(3))]);
            // The same of the first coefficients of the two operands.
            push("extfield_deg2_sub", &[(0, n(0)), (1, &p + n(1))]);
            // a1 * b1 > a0 * b0 + modsqr
            push(
                "extfield_xxone_mul",
                &[(0, n(0)), (1, n(0)), (2, ones.clone()), (3, ones)],
            );
            // y1 > y2 + p
            push("ec_add", &[(0, n(1)), (1, &p + n(5)), (2, n(2)), (3, n(0))]);
        }
        // 2 * y * lambda > 2 * p^2 + 3 * x^2 + a, for a prime that is small for its 32 bytes and
        // for the prime of BLS12-381.
        let small = (Natural::from(1u32) << 127u32) - n(1);
        let y = (Natural::from(1u32) << 255u32) + n(12345);
        calls.push((
            "ec_double_256",
            library(
                embedded("ec_double_256"),
                &[(1, n(1)), (2, y), (3, small), (4, n(0))],
            ),
        ));
        let ones = Natural::from_limbs_asc(&[u32::MAX; 12]);
        calls.push((
            "ec_double_384",
            library(
                embedded("ec_double_384"),
                &[(1, n(1)), (2, ones), (4, n(0))],
            ),
        ));

        // Operands that the modulus reduces, with a number that is not its square where the
        // host program expects the square: 0 * 0 - 1 * 1 + 0 is negative.
        calls.push((
            "extfield_xxone_mul_256 without the square",
            library(
                embedded("extfield_xxone_mul_256"),
                &[
                    (0, n(0)),
                    (1, n(0)),
                    (2, n(1)),
                    (3, n(1)),
                    (4, n(7)),
                    (5, n(0)),
                ],
            ),
        ));

        // The verify program writes the quotient to the scratch before it reads `lhs`. With
        // `lhs` in the scratch it reads the quotient, and the host program has divided the
        // `lhs` that was there before.
        let mut alias = modsub_256(10, 3, 7);
        let scratch = alias.regs.iter().find(|(reg, _)| *reg == REG_SP).unwrap().1;
        for (reg, base) in &mut alias.regs {
            if *reg == REG_A1 {
                let lhs = std::mem::replace(base, scratch);
                for (addr, _) in &mut alias.memory {
                    if *addr & !0xffff == lhs {
                        *addr = *addr - lhs + scratch;
                    }
                }
            }
        }
        calls.push(("modsub_256 with lhs in the scratch", alias));
        calls
    }

    fn expecting(mut call: Call, expect: &[(u32, u32)]) -> Call {
        call.expect = expect.to_vec();
        call
    }

    /// Calls that the circuit can prove: programs of these tests, where each kernel requires the
    /// memory that its call leaves, and then every embedded program.
    pub(crate) fn accepted() -> Vec<(&'static str, Call)> {
        let twice = Blob {
            verify: vec![end(), end()],
            ..Default::default()
        };
        // The program ends before the size that its blob declares.
        let mut early = valid();
        early.verify.extend([0xffff_ffff; 2]);
        vec![
            (
                "valid",
                expecting(call(&valid()), &[(OUT_A, 1), (OUT_B, SENTINEL)]),
            ),
            (
                "read, write, read",
                expecting(call(&read_write_read()), &[(OUT_A, 1)]),
            ),
            ("wide", wide()),
            ("carry of 1", carry(1)),
            ("carry of -1", carry(-1)),
            ("rewrites itself", rewrites_itself()),
            ("reset twice", call(&twice)),
            ("ends early", expecting(call(&early), &[(OUT_A, 1)])),
            // The ends of what the circuit lets a user and the machine read and write.
            ("reads the first user bytes", reads(0, 0x0001_0000)),
            ("reads the last user bytes", reads(0, 0xbffe_fff0)),
            (
                "writes the last user bytes",
                expecting(writes(0, REG_A5, 0xbffe_fff0), &[(0xbffe_fff0, 3)]),
            ),
            ("reads the first kernel bytes", reads(1, 0xc000_0000)),
            // A `Check` has no address, and its offset is still added to its register: the
            // circuit leaves one carry bit for the sum with the low half of the register.
            ("carries past the total", carries_at(3)),
            ("the largest offset of a check", carries_at(0x1fff)),
            (
                "the largest offset of a read",
                reads_from(0, 0x0030_fff0, 0x1000),
            ),
            // The circuit reads a word once in a cycle. A read comes as near to its own
            // instruction and to its own register as the word before and the word after, and
            // an instruction in the register file names another register than itself.
            ("reads the words before itself", reads_near_itself(-4)),
            ("reads the words after itself", reads_near_itself(1)),
            (
                "reads the registers before its own",
                reads_registers(REG_A5 - 4),
            ),
            (
                "reads the registers after its own",
                reads_registers(REG_A5 + 1),
            ),
            ("a program in the registers", in_registers(REG_ZERO)),
            // A write comes after the reads of its cycle: an instruction may write the word
            // that it is, and the register that it names.
            ("writes its own instruction", writes_itself()),
            (
                "writes its own register",
                expecting(writes(1, REG_A5, A5_ADDR), &[(A5_ADDR, 3)]),
            ),
            // The operands of the library's own test: 7 is not below the modulus.
            (
                "modsub_256(4, 7, 5)",
                expecting(modsub_256(4, 7, 5), &[(MODSUB_RESULT, 2)]),
            ),
        ]
        .into_iter()
        .chain(library_calls())
        .collect()
    }

    /// Calls that the executor refuses, and what it says of each. The circuit refuses them too,
    /// apart from three where the executor refuses more than the circuit: a program that is
    /// longer than the size its blob declares, a write that the host has no witness for, and a
    /// read whose first word alone lies where the circuit allows it.
    pub(crate) fn refused() -> Vec<(&'static str, Call, &'static str)> {
        const ADDRESS: &str = "Invalid bigint address";
        const TWICE: &str = "bigint instruction reads a word twice";
        let reset = Blob {
            verify: vec![end()],
            ..Default::default()
        };
        let short = Blob {
            verify_size: Some(3),
            ..valid()
        };
        let mut unwitnessed = valid();
        unwitnessed.stores.clear();
        // The cycle of the ecall reads the mode and the address of the verify program.
        let mut nowhere = call(&valid());
        nowhere.regs.push((REG_T2, 0));
        let mut neither = call(&valid());
        neither.mode = 2;
        vec![
            (
                "a verify program at the null address",
                nowhere,
                "Invalid bigint verify program address",
            ),
            (
                "a mode that is neither user nor machine",
                neither,
                "Invalid mode for bigint ecall",
            ),
            (
                "the equation does not hold",
                call(&invalid_eqz()),
                "Invalid eqz in bigint program",
            ),
            (
                "only a reset",
                call(&reset),
                "bigint verify program does not end within its size",
            ),
            (
                "longer than its size",
                call(&short),
                "bigint verify program does not end within its size",
            ),
            (
                "a write without a witness",
                call(&unwitnessed),
                "Missing bigint witness",
            ),
            ("a read of the null address", reads(0, 0), ADDRESS),
            ("a read of the zero page", reads(0, 0x0000_fff0), ADDRESS),
            // The circuit looks at the first word and accepts this one.
            (
                "a read that ends past the user bytes",
                reads(0, 0xbffe_fff4),
                ADDRESS,
            ),
            ("a read below the kernel", reads(0, 0xbfff_0000), ADDRESS),
            (
                "a read below the kernel by the machine",
                reads(1, 0xbfff_0000),
                ADDRESS,
            ),
            (
                "a read of the kernel by a user",
                reads(0, 0xc000_0000),
                ADDRESS,
            ),
            (
                "a write to the null address",
                writes(0, REG_ZERO, 0x0030_0000),
                ADDRESS,
            ),
            (
                "a write below the kernel by the machine",
                writes(1, REG_A5, 0xbfff_0000),
                ADDRESS,
            ),
            (
                "an offset of a check past the carry bit",
                carries_at(0x2000),
                "Invalid bigint offset",
            ),
            (
                "an offset of a read past the carry bit",
                reads_from(0, 0x0030_fff0, 0x1001),
                "Invalid bigint offset",
            ),
            ("a read of its own instruction", reads_near_itself(0), TWICE),
            (
                "a read that ends with its own instruction",
                reads_near_itself(-3),
                TWICE,
            ),
            ("a read of its own register", reads_registers(REG_A5), TWICE),
            (
                "a read that ends with its own register",
                reads_registers(REG_A5 - 3),
                TWICE,
            ),
            (
                "an instruction that is its own register",
                in_registers(REG_T3),
                TWICE,
            ),
        ]
        .into_iter()
        .chain(
            divergent()
                .into_iter()
                .map(|(name, call)| (name, call, "bad carry")),
        )
        .collect()
    }
}
