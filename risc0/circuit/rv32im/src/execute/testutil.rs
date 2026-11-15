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
