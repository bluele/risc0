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

    /// One word of the program as it is.
    fn raw(&mut self, word: u32) {
        self.text.push(word);
    }

    /// [Self::li] in two instructions whatever the value, neither of which reads the register
    /// zero.
    fn li2(&mut self, rd: usize, imm: u32) {
        let low = ((imm as i32) << 20) >> 20;
        let high = (imm.wrapping_sub(low as u32)) >> 12;
        self.lui(rd, high);
        self.addi(rd, rd, low as u32 & 0xfff);
    }

    fn load(&mut self, funct3: u32, rd: usize, rs1: usize, imm: u32) {
        self.text
            .push(insn_i(imm, rs1 as u32, funct3, rd as u32, OP_LOAD));
    }

    fn store(&mut self, funct3: u32, rs2: usize, rs1: usize, imm: u32) {
        self.text
            .push(insn_s(imm, rs2 as u32, rs1 as u32, funct3, OP_STORE));
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

// 31        25 | 24  20 | 19  15 | 14  12 | 11        7 | 6    0 |
//   imm[11:5]  |   rs2  |   rs1  | funct3 |  imm[4:0]   | opcode |
#[cfg(test)]
fn insn_s(imm: u32, rs2: u32, rs1: u32, funct3: u32, opcode: u32) -> u32 {
    (((imm >> 5) & 0x7f) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | ((imm & 0x1f) << 7)
        | opcode
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

/// Programs whose instructions reach the words that their own cycles read, the instruction itself
/// and the registers of the machine, and user programs under a kernel that enters them.
#[cfg(test)]
pub(crate) mod access {
    use super::*;

    pub(crate) const BYTE: u32 = FUNCT3_BYTE;
    pub(crate) const HALF: u32 = FUNCT3_HALF;
    pub(crate) const WORD: u32 = FUNCT3_WORD;
    pub(crate) const BYTEU: u32 = FUNCT3_BYTEU;
    pub(crate) const HALFU: u32 = FUNCT3_HALFU;

    const NOP: u32 = 0x0000_0013;

    /// Loads with the instruction of `funct3` what lies `offset` bytes from the load itself:
    /// the nop before it (-4 to -1), the load (0 to 3) or the nop after it (4 to 7).
    pub(crate) fn loads_near_itself(funct3: u32, offset: i32) -> Program {
        let mut asm = Assembler::new();
        let load = asm.here() + (3 * WORD_SIZE) as u32;
        asm.li2(REG_T0, load.wrapping_add_signed(offset));
        asm.raw(NOP);
        asm.load(funct3, REG_A0, REG_T0, 0);
        asm.raw(NOP);
        asm.host_terminate(0, 0);
        asm.program()
    }

    /// Stores with the instruction of `funct3` to what lies `offset` bytes from the store itself:
    /// the nop before it (-4 to -1), the store (0 to 3) or the nop after it (4 to 7). It stores
    /// the bytes that are there, so that the program stays as it is.
    pub(crate) fn stores_near_itself(funct3: u32, offset: i32) -> Program {
        let mut asm = Assembler::new();
        let store = asm.here() + (5 * WORD_SIZE) as u32;
        let target = store.wrapping_add_signed(offset);
        let insn = insn_s(0, REG_T1 as u32, REG_T0 as u32, funct3, OP_STORE);
        let there = if target & !3 == store { insn } else { NOP };
        asm.li2(REG_T0, target);
        asm.li2(REG_T1, there >> (8 * (target & 3)));
        asm.raw(NOP);
        asm.raw(insn);
        asm.raw(NOP);
        asm.host_terminate(0, 0);
        asm.program()
    }

    /// The machine loads the word of its register `reg`, whose address `t0` holds.
    pub(crate) fn loads_register(reg: usize) -> Program {
        let mut asm = Assembler::new();
        asm.li2(REG_T0, machine_reg(reg));
        asm.load(WORD, REG_A0, REG_T0, 0);
        asm.host_terminate(0, 0);
        asm.program()
    }

    /// The machine stores `t1` to the word of its register `reg`, whose address `t0` holds.
    pub(crate) fn stores_register(reg: usize) -> Program {
        let mut asm = Assembler::new();
        asm.li2(REG_T0, machine_reg(reg));
        asm.li2(REG_T1, 7);
        asm.store(WORD, REG_T1, REG_T0, 0);
        asm.host_terminate(0, 0);
        asm.program()
    }

    /// The machine runs two instructions out of registers: one that adds zero to its register
    /// `reg`, and a return. They are in `t1` and `t2` of the machine, or of the user file if
    /// `user` is set.
    pub(crate) fn runs_registers(user: bool, reg: usize) -> Program {
        let base = if user {
            USER_REGS_ADDR
        } else {
            MACHINE_REGS_ADDR
        };
        let at = |reg: usize| base.0 + (reg * WORD_SIZE) as u32;
        let add = insn_i(0, reg as u32, 0, reg as u32, OP_IMM);
        let ret = insn_i(0, REG_RA as u32, 0, REG_ZERO as u32, OP_JALR);

        let mut asm = Assembler::new();
        if user {
            // The machine writes the two instructions to the registers of the user.
            asm.li2(REG_T0, at(REG_T1));
            asm.li2(REG_T4, add);
            asm.store(WORD, REG_T4, REG_T0, 0);
            asm.li2(REG_T4, ret);
            asm.store(WORD, REG_T4, REG_T0, WORD_SIZE as u32);
        } else {
            asm.li2(REG_T1, add);
            asm.li2(REG_T2, ret);
        }
        asm.li2(REG_T0, at(REG_T1));
        asm.jalr(REG_RA, REG_T0, 0);
        asm.host_terminate(0, 0);
        asm.program()
    }

    /// The image of a user program and of a kernel that enters it. The kernel lies at the start
    /// of kernel memory, and terminates when the user program makes an ecall.
    pub(crate) fn user(program: Program) -> MemoryImage {
        user_with(program, false, 0)
    }

    /// [user] with a kernel that enters the program `skew` bytes past its entry, and that names
    /// its end as the handler of every trap if `handles` is set.
    pub(crate) fn user_with(program: Program, handles: bool, skew: u32) -> MemoryImage {
        under(program, kernel_words(handles, skew))
    }

    /// The image of a user program and of a kernel of these words.
    pub(crate) fn under(program: Program, kernel: BTreeMap<u32, u32>) -> MemoryImage {
        let kernel = Program::new_from_entry_and_image(KERNEL_START_ADDR.0, kernel);
        MemoryImage::with_kernel(program, kernel)
    }

    /// The words of the kernel of [user_with].
    pub(crate) fn kernel_words(handles: bool, skew: u32) -> BTreeMap<u32, u32> {
        let mut kernel = Assembler::new();
        kernel.li2(REG_T0, MEPC_ADDR.0);
        // An mret goes on at the word after the address that it finds, and the programs of
        // these tests begin at the second word of user memory.
        kernel.li2(REG_T1, USER_START_ADDR.0 + skew);
        kernel.store(WORD, REG_T1, REG_T0, 0);
        kernel.raw(0x3020_0073); // mret
        let handler = KERNEL_START_ADDR.0 + (kernel.text.len() * WORD_SIZE) as u32;
        kernel.host_terminate(0, 0);

        let mut image: BTreeMap<u32, u32> = (0u32..)
            .zip(&kernel.text)
            .map(|(i, insn)| (KERNEL_START_ADDR.0 + i * WORD_SIZE as u32, *insn))
            .collect();
        image.insert(ECALL_DISPATCH_ADDR.0, handler);
        if handles {
            // The causes of `Exception` are numbered from 0 to 9, the last that of an ecall
            // without a handler.
            for cause in 0..10 {
                image.insert(TRAP_DISPATCH_ADDR.0 + cause * WORD_SIZE as u32, handler);
            }
        }
        image
    }

    /// Programs that the circuit proves: each instruction reads the words of its cycle once.
    pub(crate) fn accepted() -> Vec<(&'static str, MemoryImage)> {
        let kernel = MemoryImage::new_kernel;
        vec![
            (
                "loads the word before it",
                kernel(loads_near_itself(WORD, -4)),
            ),
            (
                "loads the word after it",
                kernel(loads_near_itself(WORD, 4)),
            ),
            (
                "loads the byte before it",
                kernel(loads_near_itself(BYTE, -1)),
            ),
            (
                "loads the byte after it",
                kernel(loads_near_itself(BYTEU, 4)),
            ),
            (
                "stores to the word before it",
                kernel(stores_near_itself(WORD, -4)),
            ),
            (
                "stores to the word after it",
                kernel(stores_near_itself(WORD, 4)),
            ),
            (
                "stores to the byte after it",
                kernel(stores_near_itself(BYTE, 4)),
            ),
            (
                "a user program loads the word before it",
                user(loads_near_itself(WORD, -4)),
            ),
            (
                "a user program loads the word after it",
                user(loads_near_itself(WORD, 4)),
            ),
            (
                "a user program stores to the word after it",
                user(stores_near_itself(WORD, 4)),
            ),
            // The load names `t0`, and the store `t0` and `t1`.
            ("loads another register", kernel(loads_register(REG_T1))),
            (
                "stores to another register",
                kernel(stores_register(REG_T2)),
            ),
            // The instruction in the register `t1` names `t3`.
            (
                "runs registers that name others",
                kernel(runs_registers(false, REG_T3)),
            ),
            (
                "runs registers of the user",
                kernel(runs_registers(true, REG_T1)),
            ),
        ]
    }

    /// Programs that the executor refuses, as the circuit does: an instruction would read a
    /// word of its cycle twice.
    pub(crate) fn refused() -> Vec<(&'static str, MemoryImage)> {
        let kernel = MemoryImage::new_kernel;
        vec![
            ("loads its own word", kernel(loads_near_itself(WORD, 0))),
            ("loads its first byte", kernel(loads_near_itself(BYTE, 0))),
            ("loads its last byte", kernel(loads_near_itself(BYTEU, 3))),
            ("loads its second half", kernel(loads_near_itself(HALF, 2))),
            ("loads its first half", kernel(loads_near_itself(HALFU, 0))),
            (
                "stores to its own word",
                kernel(stores_near_itself(WORD, 0)),
            ),
            (
                "stores to a byte of itself",
                kernel(stores_near_itself(BYTE, 2)),
            ),
            (
                "stores to a half of itself",
                kernel(stores_near_itself(HALF, 0)),
            ),
            (
                "a user program loads its own word",
                user(loads_near_itself(WORD, 0)),
            ),
            (
                "a user program loads a byte of itself",
                user(loads_near_itself(BYTEU, 1)),
            ),
            (
                "a user program stores to its own word",
                user(stores_near_itself(WORD, 0)),
            ),
            (
                "a user program stores to a byte of itself",
                user(stores_near_itself(BYTE, 3)),
            ),
            (
                "loads the register of its address",
                kernel(loads_register(REG_T0)),
            ),
            (
                "stores to the register of its address",
                kernel(stores_register(REG_T0)),
            ),
            (
                "stores to the register that it stores",
                kernel(stores_register(REG_T1)),
            ),
            (
                "runs a register that names itself",
                kernel(runs_registers(false, REG_T1)),
            ),
        ]
    }
}

/// Words of `jalr`, `ecall` and `mret` that the circuit does not take, as the ISA reserves them:
/// a `jalr` whose funct3 is not zero, and an `ecall` or an `mret` with another field that is not
/// zero.
#[cfg(test)]
pub(crate) mod reserved {
    use super::{access::user, *};

    const ECALL: u32 = 0x0000_0073;
    const MRET: u32 = 0x3020_0073;

    /// A kernel that jumps to the instruction after a `jalr` of `funct3`.
    fn jalr(funct3: u32) -> Program {
        let mut asm = Assembler::new();
        let next = asm.here() + (3 * WORD_SIZE) as u32;
        asm.li2(REG_T0, next);
        asm.raw(insn_i(0, REG_T0 as u32, funct3, REG_RA as u32, OP_JALR));
        asm.host_terminate(0, 0);
        asm.program()
    }

    /// A kernel that terminates with `ecall` as the word of its ecall.
    fn terminates_with(ecall: u32) -> Program {
        let mut asm = Assembler::new();
        asm.li(REG_A7, HOST_ECALL_TERMINATE);
        asm.li(REG_A0, 0);
        asm.li(REG_A1, 0);
        asm.raw(ecall);
        asm.program()
    }

    /// A kernel whose first instruction is `word`.
    fn begins_with(word: u32) -> Program {
        let mut asm = Assembler::new();
        asm.raw(word);
        asm.host_terminate(0, 0);
        asm.program()
    }

    /// A user program of one ecall, of the word `ecall`.
    fn user_ecall(ecall: u32) -> Program {
        let mut asm = Assembler::new();
        asm.raw(ecall);
        asm.program()
    }

    /// The canonical words, which the executor runs.
    pub(crate) fn accepted() -> Vec<(String, MemoryImage)> {
        vec![
            (
                "a jalr of funct3 0".into(),
                MemoryImage::new_kernel(jalr(0)),
            ),
            (
                "an ecall".into(),
                MemoryImage::new_kernel(terminates_with(ECALL)),
            ),
            // The kernel of the user program enters it with an mret.
            ("a user ecall".into(), user(user_ecall(ECALL))),
        ]
    }

    /// The reserved words, which the executor refuses as illegal instructions.
    pub(crate) fn refused() -> Vec<(String, MemoryImage)> {
        let mut images = Vec::new();
        for funct3 in 1..8 {
            let image = MemoryImage::new_kernel(jalr(funct3));
            images.push((format!("a jalr of funct3 {funct3}"), image));
        }
        let ecall = |rs1: u32, rd: u32| insn_i(0, rs1, 0, rd, OP_ENV);
        for (rs1, rd) in [(0, 1), (0, 16), (0, 31), (1, 0), (31, 0), (1, 1)] {
            let word = ecall(rs1, rd);
            let image = MemoryImage::new_kernel(terminates_with(word));
            images.push((format!("an ecall {word:#010x}"), image));
            images.push((format!("a user ecall {word:#010x}"), user(user_ecall(word))));
        }
        // The rd, rs1 and rs2 fields of an mret.
        for word in [
            MRET | (1 << 7),
            MRET | (1 << 15),
            MRET & !(0x1f << 20),
            MRET | (1 << 20),
        ] {
            let image = MemoryImage::new_kernel(begins_with(word));
            images.push((format!("an mret {word:#010x}"), image));
        }
        images
    }
}

/// User programs that trap, and kernels that handle the traps or enter a user program off its
/// words.
#[cfg(test)]
pub(crate) mod traps {
    use super::{
        access::{kernel_words, under, user, user_with, WORD},
        *,
    };

    /// A user program of one ecall.
    pub(crate) fn ecalls() -> Program {
        let mut asm = Assembler::new();
        asm.ecall();
        asm.program()
    }

    /// A user program that does what traps, in the order of the names in [refused].
    pub(crate) fn traps(cause: u32) -> Program {
        let mut asm = Assembler::new();
        match cause {
            0 => {
                let target = asm.here() + (3 * WORD_SIZE) as u32 + 2;
                asm.li2(REG_T0, target);
                asm.jalr(REG_RA, REG_T0, 0);
            }
            1 => {
                asm.li2(REG_T0, KERNEL_START_ADDR.0 + 0x1000);
                asm.jalr(REG_RA, REG_T0, 0);
            }
            2 => asm.raw(0xffff_ffff),
            3 => asm.raw(0x0010_0073),
            4 => {
                asm.li2(REG_T0, asm.here() + 2);
                asm.load(WORD, REG_A0, REG_T0, 0);
            }
            5 => {
                asm.li2(REG_T0, KERNEL_START_ADDR.0);
                asm.load(WORD, REG_A0, REG_T0, 0);
            }
            6 => {
                asm.li2(REG_T0, 0x0020_0002);
                asm.store(WORD, REG_T0, REG_T0, 0);
            }
            7 => {
                asm.li2(REG_T0, KERNEL_START_ADDR.0);
                asm.store(WORD, REG_T0, REG_T0, 0);
            }
            _ => unimplemented!(),
        }
        asm.host_terminate(0, 0);
        asm.program()
    }

    /// User programs that the executor refuses, and what it says of each. The circuit cannot
    /// prove the step from a trap to its handler, nor an instruction at a pc that is not word
    /// aligned.
    pub(crate) fn refused() -> Vec<(&'static str, MemoryImage, &'static str)> {
        const HANDLED: &str = "Trap to a handler of the kernel";
        let names = [
            "a jump to an address that is not word aligned",
            "a jump into kernel memory",
            "a word that is no instruction",
            "ebreak",
            "a load of a word from an address that is not word aligned",
            "a load from kernel memory",
            "a store of a word to an address that is not word aligned",
            "a store to kernel memory",
        ];
        // The executor does not take a handler in user memory for the handler of an ecall.
        let mut kernel = kernel_words(true, 0);
        kernel.insert(ECALL_DISPATCH_ADDR.0, USER_START_ADDR.0 + WORD_SIZE as u32);
        let mut programs: Vec<_> = (0u32..)
            .zip(names)
            .map(|(cause, name)| (name, user_with(traps(cause), true, 0), HANDLED))
            .collect();
        programs.extend([
            (
                "an ecall whose handler is in user memory",
                under(ecalls(), kernel),
                HANDLED,
            ),
            // Nothing handles the trap, which upstream refuses too.
            (
                "a word that is no instruction, under a kernel without a handler",
                user(traps(2)),
                "Invalid trap address",
            ),
            (
                "entered two bytes past its entry",
                user_with(ecalls(), true, 2),
                "mret to an unaligned address",
            ),
            (
                "entered one byte past its entry",
                user_with(ecalls(), true, 1),
                "mret to an unaligned address",
            ),
        ]);
        programs
    }
}

/// Kernels that make one sha2 ecall.
#[cfg(test)]
pub(crate) mod sha2 {
    use risc0_zkp::core::{
        digest::{Digest, DIGEST_WORDS},
        hash::sha::{Impl, Sha256 as _, BLOCK_WORDS, SHA256_INIT},
    };

    use super::*;

    pub(crate) const STATE_IN: u32 = 0x0020_0000;
    pub(crate) const STATE_OUT: u32 = 0x0020_0100;
    pub(crate) const DATA: u32 = 0x0030_0000;
    pub(crate) const K: u32 = 0x0040_0000;
    /// Words of kernel memory that nothing else in these programs touches.
    pub(crate) const KERNEL_MEMORY: u32 = KERNEL_START_ADDR.0 + 0x0010_0000;
    /// The last words of memory that hold a state.
    pub(crate) const LAST_WORDS: u32 = 0u32.wrapping_sub((DIGEST_WORDS * WORD_SIZE) as u32);

    const ROUND_CONSTANTS: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    /// The registers of a call: the state that it begins with, where its state goes, its
    /// blocks, their number, and the round constants.
    #[derive(Clone, Copy)]
    pub(crate) struct Call {
        pub state_in: u32,
        pub state_out: u32,
        pub data: u32,
        pub count: u32,
        pub k: u32,
    }

    impl Default for Call {
        fn default() -> Self {
            Self {
                state_in: STATE_IN,
                state_out: STATE_OUT,
                data: DATA,
                count: 1,
                k: K,
            }
        }
    }

    impl Call {
        /// A kernel with the initial state of SHA-256 at `STATE_IN`, two blocks at `DATA` and
        /// the round constants at `K`. After the call it requires, where the state goes, the
        /// state that the compression function gives for the words that the call names.
        pub(crate) fn program(&self) -> Program {
            let mut memory = BTreeMap::new();
            for (i, word) in (0u32..).zip(SHA256_INIT.as_words()) {
                memory.insert(STATE_IN + i * WORD_SIZE as u32, *word);
            }
            for i in 0..2 * BLOCK_WORDS as u32 {
                memory.insert(DATA + i * WORD_SIZE as u32, 0x0101_0101 * (i + 1));
            }
            for (i, word) in (0u32..).zip(ROUND_CONSTANTS) {
                memory.insert(K + i * WORD_SIZE as u32, word);
            }

            // The state that the call leaves.
            let word = |addr: u32| memory.get(&addr).copied().unwrap_or(0);
            let digest = |addr: u32| -> Digest {
                Digest::new(core::array::from_fn(|i| {
                    word(addr + (i * WORD_SIZE) as u32)
                }))
            };
            let mut state = digest(self.state_in);
            for block in 0..self.count {
                let block = self.data + block * (BLOCK_WORDS * WORD_SIZE) as u32;
                let halves = [block, block + (DIGEST_WORDS * WORD_SIZE) as u32].map(digest);
                state = *Impl::compress(&state, &halves[0], &halves[1]);
            }

            let mut asm = Assembler::new();
            for (addr, word) in &memory {
                asm.word(*addr, *word);
            }
            asm.li(REG_A7, HOST_ECALL_SHA2);
            asm.li(REG_A0, self.state_in);
            asm.li(REG_A1, self.state_out);
            asm.li(REG_A2, self.data);
            asm.li(REG_A3, self.count);
            asm.li(REG_A4, self.k);
            asm.ecall();
            for (i, word) in (0u32..).zip(state.as_words()) {
                asm.li(REG_T4, self.state_out.wrapping_add(i * WORD_SIZE as u32));
                asm.lw(REG_T5, REG_T4, 0);
                asm.li(REG_T6, *word);
                asm.beq(REG_T5, REG_T6, 8);
                asm.die();
            }
            asm.host_terminate(0, 0);
            asm.program()
        }
    }

    /// Calls that the circuit proves.
    pub(crate) fn accepted() -> Vec<(&'static str, Call)> {
        let call = Call::default();
        vec![
            ("one block", call),
            ("no block", Call { count: 0, ..call }),
            ("two blocks", Call { count: 2, ..call }),
            (
                "the state goes where it comes from",
                Call {
                    state_out: STATE_IN,
                    ..call
                },
            ),
            // A cycle reads a round constant and a word of a block.
            (
                "the blocks begin a word past the round constants",
                Call {
                    data: K + WORD_SIZE as u32,
                    ..call
                },
            ),
            // A kernel makes the call, and its addresses are not held to user memory.
            (
                "the state goes to kernel memory",
                Call {
                    state_out: KERNEL_MEMORY,
                    ..call
                },
            ),
            (
                "the state goes to registers of the machine",
                Call {
                    state_out: machine_reg(REG_S2),
                    ..call
                },
            ),
            // The cycles after the first do not read the registers again.
            (
                "the state goes to the registers of the call",
                Call {
                    state_out: machine_reg(REG_A0),
                    ..call
                },
            ),
            (
                "the state goes to the last words of memory",
                Call {
                    state_out: LAST_WORDS,
                    ..call
                },
            ),
        ]
    }

    /// Calls that the executor refuses, and what it says of each.
    pub(crate) fn refused() -> Vec<(&'static str, Call, &'static str)> {
        const UNALIGNED: &str = "is an unaligned address";
        const TWICE: &str = "sha2 ecall reads a word twice";
        let call = Call::default();
        vec![
            (
                "the state comes from an address that is not word aligned",
                Call {
                    state_in: STATE_IN + 1,
                    ..call
                },
                UNALIGNED,
            ),
            (
                "the state goes to an address that is not word aligned",
                Call {
                    state_out: STATE_OUT + 2,
                    ..call
                },
                UNALIGNED,
            ),
            (
                "the blocks are not word aligned",
                Call {
                    data: DATA + 3,
                    ..call
                },
                UNALIGNED,
            ),
            (
                "the round constants are not word aligned",
                Call { k: K + 1, ..call },
                UNALIGNED,
            ),
            (
                "the first block is the round constants",
                Call { data: K, ..call },
                TWICE,
            ),
            (
                "the second block is the round constants",
                Call {
                    data: K - (BLOCK_WORDS * WORD_SIZE) as u32,
                    count: 2,
                    ..call
                },
                TWICE,
            ),
            // Memory has no word past its end, which upstream refuses too.
            (
                "the state goes past the end of memory",
                Call {
                    state_out: LAST_WORDS + (DIGEST_WORDS / 2 * WORD_SIZE) as u32,
                    ..call
                },
                "Invalid store address",
            ),
        ]
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

    /// Four ends of a program, as the 16 bytes that a write puts over four instructions.
    fn ends() -> u128 {
        (0..4).fold(0, |value, i| value | (u128::from(end()) << (32 * i)))
    }

    /// The instructions of the short programs. Each operation on the polynomials comes with a
    /// read of `a1`, of `a3` and of `a4` and with a write of `a4`, an `AddTotal` with the
    /// coefficients 1 and -1. Each operation that a `Check` has comes with a `Check`. The last
    /// is a write of `a5`.
    pub(crate) fn short_instructions() -> Vec<u32> {
        let mut insns = vec![];
        for poly_op in RESET..=EQ_ZERO {
            let coeffs: &[i32] = if poly_op == ADD_TOTAL { &[1, -1] } else { &[0] };
            for &coeff in coeffs {
                for reg in [REG_A1, REG_A3, REG_A4] {
                    insns.push(insn(poly_op, READ, coeff, reg, 0));
                }
                insns.push(insn(poly_op, WRITE, coeff, REG_A4, 0));
            }
        }
        for poly_op in [RESET, SHIFT, CARRY1, CARRY2, EQ_ZERO] {
            insns.push(insn(poly_op, CHECK, 0, REG_ZERO, 0));
        }
        insns.push(insn(SET_TERM, WRITE, 0, REG_A5, 0));
        insns
    }

    /// The programs of a number of [short_instructions] that end with their last instruction
    /// and not before it. A reset ends a program unless it is the first instruction, so the
    /// last instruction is a reset, and no instruction between the first and the last is one.
    /// A program of one instruction does not end.
    pub(crate) struct Short {
        len: u32,
        first: Vec<u32>,
        between: Vec<u32>,
        last: Vec<u32>,
    }

    impl Short {
        pub(crate) fn new(len: u32) -> Self {
            let first = short_instructions();
            let (last, between): (Vec<u32>, Vec<u32>) =
                first.iter().partition(|insn| (**insn >> 24) & 0xf == RESET);
            Self {
                len,
                first,
                between,
                last,
            }
        }

        pub(crate) fn count(&self) -> u64 {
            let [first, between, last] =
                [&self.first, &self.between, &self.last].map(|insns| insns.len() as u64);
            match self.len {
                0 => 0,
                1 => first,
                len => first * between.pow(len - 2) * last,
            }
        }

        pub(crate) fn program(&self, mut index: u64) -> Vec<u32> {
            let mut digit = |insns: &[u32]| {
                let insn = insns[(index % insns.len() as u64) as usize];
                index /= insns.len() as u64;
                insn
            };
            let mut program = vec![digit(&self.first)];
            for _ in 2..self.len {
                program.push(digit(&self.between));
            }
            if self.len > 1 {
                program.push(digit(&self.last));
            }
            program
        }
    }

    /// How many of the programs of one instruction, of two and so on the executor accepts.
    pub(crate) const SHORT_ACCEPTED: [usize; 4] = [0, 180, 5340, 154260];

    /// A call of a program of [short_instructions] with the `block`th blob of a [batch]. The
    /// block holds what the program names after the blob: the number 1 for `a1`, zeros for
    /// `a3`, and zeros for `a4`, where the host stores 1. `a5` points to the program, where the
    /// host stores four ends of a program.
    pub(crate) fn short(verify: &[u32], block: u32) -> Call {
        const OPERANDS: u32 = 0xc0;
        let blob = Blob {
            stores: vec![(REG_A4, 0, 1), (REG_A5, 0, ends())],
            verify: verify.to_vec(),
            ..Default::default()
        }
        .words();
        assert!(blob.len() * WORD_SIZE <= OPERANDS as usize);
        let base = BLOB + block * BLOCK;
        let program = base + (4 + blob[0]) * WORD_SIZE as u32;
        let [one, zero, out] = [0, 16, 32].map(|at| base + OPERANDS + at);
        Call {
            blob,
            mode: 0,
            regs: vec![
                (REG_A1, one),
                (REG_A3, zero),
                (REG_A4, out),
                (REG_A5, program),
            ],
            memory: vec![(one, 1)],
            expect: vec![],
        }
    }

    /// Offsets at the ends of what the circuit adds to a register.
    #[cfg(feature = "prove")]
    pub(crate) const OFFSETS: [u32; 11] = [
        0, 1, 2, 3, 4, 0x0fff, 0x1000, 0x1001, 0x1fff, 0x2000, 0xffff,
    ];

    /// One instruction for every operation on memory and on the polynomials that the encoding
    /// has room for, the circuit having the operation or not, with every coefficient.
    #[cfg(feature = "prove")]
    pub(crate) fn operation_instructions() -> Vec<u32> {
        let mut insns = vec![];
        for mem_op in 0..16 {
            for poly_op in 0..16 {
                for coeff in -4..4 {
                    insns.push(insn(poly_op, mem_op, coeff, REG_A4, 0));
                }
            }
        }
        insns
    }

    /// One instruction for each pair of operations that the circuit has, with every register
    /// and with offsets at the ends of what the circuit adds to a register.
    #[cfg(feature = "prove")]
    pub(crate) fn operand_instructions() -> Vec<u32> {
        let mut insns = vec![];
        for mem_op in [READ, WRITE, CHECK] {
            for poly_op in RESET..=EQ_ZERO {
                for reg in 0..REG_MAX {
                    for offset in OFFSETS {
                        insns.push(insn(poly_op, mem_op, 1, reg, offset));
                    }
                }
            }
        }
        insns
    }

    /// A call of the program of `insn` and an end, which comes `after` a read that adds 1 to
    /// the total or as the first instruction. The registers that the ecall does not set point
    /// to the ends of what the circuit lets a user and the machine read and write, to the
    /// program and to the registers themselves. If the instruction writes, the host stores a
    /// witness where it writes: four ends of a program for the program, and else 1.
    #[cfg(feature = "prove")]
    pub(crate) fn field(insn: u32, mode: u32, after: bool) -> Call {
        let (mem_op, reg, offset) = (insn >> 28, (insn >> 16) as usize & 0x1f, insn & 0xffff);
        let mut blob = Blob {
            // A host program of the same size for every instruction.
            stores: vec![(REG_A4, 0, 1)],
            verify: vec![insn, end()],
            ..Default::default()
        };
        if after {
            blob.verify
                .insert(0, self::insn(ADD_TOTAL, READ, 1, REG_A1, 0));
        }
        let program = BLOB + (4 + blob.words()[0]) * WORD_SIZE as u32;
        let [one, zero, out] = [0, 16, 32].map(|at| BLOB + 0xc0 + at);
        let regs = vec![
            (REG_RA, 0x0000_fff0),
            (REG_SP, 0x0001_0000),
            (REG_GP, 0xbffe_fff0),
            (REG_TP, 0xbffe_fff4),
            (REG_S0, 0xbfff_0000),
            (REG_S1, 0xbfff_fff0),
            (REG_A1, one),
            (REG_A2, 0x0030_0002),
            (REG_A3, zero),
            (REG_A4, out),
            (REG_A5, program),
            (REG_A6, program + 16),
            (REG_S2, 0xc000_0000),
            (REG_S3, 0xfffe_fff0),
            (REG_S4, machine_reg(REG_ZERO)),
            (REG_S5, machine_reg(REG_S5)),
            (REG_S6, 0xffff_fff0),
            (REG_S7, 0x0030_fff0),
            (REG_S8, USER_REGS_ADDR.0),
            (REG_S9, KERNEL_END_ADDR.0),
            (REG_S10, MEPC_ADDR.0),
            (REG_S11, 0xfeff_fff0),
            (REG_T4, program - 16),
            (REG_T5, SAFE_WRITE_ADDR.0),
            (REG_T6, GLOBAL_OUTPUT_ADDR.0),
        ];
        if mem_op == WRITE {
            // What the ecall sets, for a register that the list above does not have.
            let ecall = [
                (REG_ZERO, 0),
                (REG_T0, mode),
                (REG_T1, BLOB + 4 * WORD_SIZE as u32),
                (REG_T2, program),
                (REG_A0, BLOB),
                (REG_A7, HOST_ECALL_BIGINT),
            ];
            let base = regs.iter().chain(&ecall).find(|(name, _)| *name == reg);
            // The constants of the blob, which follow the program, for `t3`.
            let base = base.map_or(program + 4 * blob.verify.len() as u32, |(_, base)| *base);
            let at = base.wrapping_add(16 * offset);
            let value = if at == program { ends() } else { 1 };
            blob.stores = vec![(reg, offset, value)];
        }
        Call {
            blob: blob.words(),
            mode,
            regs,
            memory: vec![(one, 1)],
            expect: vec![],
        }
    }
}
