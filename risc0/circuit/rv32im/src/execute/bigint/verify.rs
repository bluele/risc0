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

//! The verify program of a bigint ecall.
//!
//! A bigint ecall names two programs. The host runs the first to compute a witness, and the
//! circuit runs the second to check it. This module runs the second by the rules of the circuit,
//! for the executor and for the prover's preflight alike. The executor is to complete a call
//! only if the circuit can prove it, and fails the execution where a rule refuses.
//!
//! The circuit is the reference: `BigIntEcall`, `BigIntStep`, `BigIntAddr` and `BigIntAccum` in
//! `inst_bigint.zir`. Some rules here refuse more than the circuit does.

use anyhow::{anyhow, bail, ensure, Result};
use derive_more::Debug;
use num_derive::FromPrimitive;
use num_traits::FromPrimitive as _;
use risc0_binfmt::WordAddr;

use super::{BigIntBytes, BigIntExec, BIGINT_WIDTH_BYTES, BIGINT_WIDTH_WORDS};
use crate::execute::{
    platform::*,
    r0vm::{LoadOp, Risc0Context},
};

/// The most coefficients a polynomial of a verify program may have.
///
/// This and [MAX_WORK] bound the host's work for one call, which the cycles of the call do not:
/// a multiplication is one cycle whatever the sizes of its factors. Of the programs of the
/// bigint2 library and of the v1compat kernel, the largest needs 1552 coefficients
/// (`modmul_4096`).
const MAX_COEFFS: usize = 4096;

/// The most coefficient products the multiplications of one verify program may take.
///
/// Of the same programs, the largest takes 795,136 (`modmul_4096`).
const MAX_WORK: u64 = 1 << 23;

/// The three carry bytes of a coefficient hold the carry plus this offset, in 22 bits.
const CARRY_OFFSET: i64 = 128 * 256 * 64;

/// The state of the bigint circuit after one cycle.
#[derive(Clone, Debug)]
pub(crate) struct BigIntState {
    pub is_ecall: bool,
    pub mode: u32,
    pub pc: WordAddr,
    pub poly_op: PolyOp,
    pub coeff: u32,
    pub bytes: BigIntBytes,
    pub next_state: CycleState,
}

#[derive(Debug)]
#[debug("{poly_op:?}({mem_op:?}, c:{coeff}, r:{reg}, o:{offset})")]
pub(crate) struct Instruction {
    pub poly_op: PolyOp,
    pub mem_op: MemoryOp,
    pub coeff: i32,
    pub reg: u32,
    pub offset: u32,
}

#[derive(Clone, Copy, Debug, FromPrimitive, PartialEq)]
pub(crate) enum PolyOp {
    Reset,
    Shift,
    SetTerm,
    AddTotal,
    Carry1,
    Carry2,
    EqZero,
}

#[derive(Clone, Copy, Debug, FromPrimitive, PartialEq)]
pub(crate) enum MemoryOp {
    Read,
    Write,
    Check,
}

impl Instruction {
    // instruction encoding:
    // 3  2   2  2    1               0
    // 1  8   4  1    6               0
    // mmmmppppcccaaaaaoooooooooooooooo
    pub fn decode(insn: u32) -> Result<Self> {
        Ok(Self {
            mem_op: MemoryOp::from_u32((insn >> 28) & 0x0f)
                .ok_or_else(|| anyhow!("Invalid mem_op in bigint program"))?,
            poly_op: PolyOp::from_u32((insn >> 24) & 0x0f)
                .ok_or_else(|| anyhow!("Invalid poly_op in bigint program"))?,
            coeff: ((insn >> 21) & 0x07) as i32 - 4,
            reg: (insn >> 16) & 0x1f,
            offset: insn & 0xffff,
        })
    }
}

fn overflow() -> anyhow::Error {
    anyhow!("Overflow in bigint program")
}

/// A polynomial with integer coefficients, lowest degree first.
///
/// The circuit evaluates these polynomials at a point of an extension of the Baby Bear field that
/// the prover does not choose. An equation between them holds there, whatever the point, if it
/// holds over the integers, so the arithmetic here is exact: an operation that overflows is an
/// error, and nothing wraps.
#[derive(Default)]
struct Poly(Vec<i64>);

impl Poly {
    fn one() -> Self {
        Self(vec![1])
    }

    fn add_scaled(&mut self, rhs: &[i64], scale: i64) -> Result<()> {
        if self.0.len() < rhs.len() {
            self.0.resize(rhs.len(), 0);
        }
        for (lhs, rhs) in self.0.iter_mut().zip(rhs) {
            *lhs = rhs
                .checked_mul(scale)
                .and_then(|rhs| lhs.checked_add(rhs))
                .ok_or_else(overflow)?;
        }
        Ok(())
    }

    fn mul(&self, rhs: &Self) -> Result<Self> {
        if self.0.is_empty() || rhs.0.is_empty() {
            return Ok(Self::default());
        }
        let mut product = vec![0i64; self.0.len() + rhs.0.len() - 1];
        for (i, lhs) in self.0.iter().enumerate() {
            for (j, rhs) in rhs.0.iter().enumerate() {
                let coeff = &mut product[i + j];
                *coeff = lhs
                    .checked_mul(*rhs)
                    .and_then(|term| coeff.checked_add(term))
                    .ok_or_else(overflow)?;
            }
        }
        Ok(Self(product))
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|coeff| *coeff == 0)
    }
}

/// The polynomials of the accumulator, as `BigIntAccum` keeps them at the point of evaluation.
struct Accum {
    poly: Poly,
    term: Poly,
    total: Poly,
    /// The carries of `total`, from the first instruction that asks for a carry until the
    /// equation is checked.
    carry: Option<Vec<i64>>,
    /// The coefficient products of the multiplications so far.
    work: u64,
}

impl Accum {
    fn new() -> Self {
        Self {
            poly: Poly::default(),
            term: Poly::one(),
            total: Poly::default(),
            carry: None,
            work: 0,
        }
    }

    fn reset(&mut self) {
        self.poly = Poly::default();
        self.term = Poly::one();
        self.total = Poly::default();
        self.carry = None;
    }

    fn check_size(&self, poly: &Poly) -> Result<()> {
        ensure!(
            poly.0.len() <= MAX_COEFFS,
            "bigint verify program is too large"
        );
        Ok(())
    }

    /// The bytes of a `Check` instruction.
    ///
    /// They are advice in the circuit. A carry instruction takes its bytes from the carries of
    /// `total`, and the equation that follows holds only for those.
    fn check_bytes(&mut self, insn: &Instruction) -> Result<BigIntBytes> {
        let mut bytes = [0u8; BIGINT_WIDTH_BYTES];
        match insn.poly_op {
            PolyOp::Reset => return Ok(bytes),
            PolyOp::Carry1 | PolyOp::Carry2 | PolyOp::Shift | PolyOp::EqZero => {}
            PolyOp::SetTerm | PolyOp::AddTotal => bail!("Invalid poly_op in bigint program"),
        }

        let carries = match &mut self.carry {
            Some(carries) => carries,
            carry => {
                let mut carries = Vec::with_capacity(self.total.0.len());
                let mut prev = 0i64;
                for coeff in &self.total.0 {
                    let sum = coeff.checked_add(prev).ok_or_else(overflow)?;
                    ensure!(sum % 256 == 0, "bad carry");
                    prev = sum / 256;
                    carries.push(prev);
                }
                carry.insert(carries)
            }
        };

        let start = insn.offset as usize * BIGINT_WIDTH_BYTES;
        for (i, byte) in bytes.iter_mut().enumerate() {
            // A coefficient past the end of `total` is zero, and so is its carry.
            let carry = carries.get(start + i).copied().unwrap_or(0);
            ensure!(
                (-CARRY_OFFSET..CARRY_OFFSET).contains(&carry),
                "bigint carry is out of range"
            );
            let value = carry + CARRY_OFFSET;
            *byte = match insn.poly_op {
                PolyOp::Carry1 => (value >> 14) as u8,
                PolyOp::Carry2 => ((value >> 8) & 0x3f) as u8,
                _ => (value & 0xff) as u8,
            };
        }
        Ok(bytes)
    }

    fn step(&mut self, insn: &Instruction, bytes: &BigIntBytes) -> Result<()> {
        let delta = bytes.map(i64::from);
        match insn.poly_op {
            PolyOp::Reset => self.reset(),
            PolyOp::Shift => {
                self.poly.add_scaled(&delta, 1)?;
                self.poly.0.splice(0..0, [0; BIGINT_WIDTH_BYTES]);
                self.check_size(&self.poly)?;
            }
            PolyOp::SetTerm => {
                self.poly.add_scaled(&delta, 1)?;
                self.term = std::mem::take(&mut self.poly);
            }
            PolyOp::AddTotal => {
                self.poly.add_scaled(&delta, 1)?;
                let poly = std::mem::take(&mut self.poly);
                let term = std::mem::replace(&mut self.term, Poly::one());

                self.work = self
                    .work
                    .saturating_add(poly.0.len() as u64 * term.0.len() as u64);
                ensure!(
                    self.work <= MAX_WORK,
                    "bigint verify program takes too much work"
                );
                let product = poly.mul(&term)?;
                self.total.add_scaled(&product.0, insn.coeff.into())?;
                self.check_size(&self.total)?;
            }
            PolyOp::Carry1 => {
                let delta = delta.map(|byte| byte - 128);
                self.poly.add_scaled(&delta, 64 * 256)?;
            }
            PolyOp::Carry2 => self.poly.add_scaled(&delta, 256)?,
            PolyOp::EqZero => {
                // total + (X - 256) * (poly + delta) == 0
                self.poly.add_scaled(&delta, 1)?;
                let mut shifted = std::mem::take(&mut self.poly);
                self.total.add_scaled(&shifted.0, -256)?;
                shifted.0.insert(0, 0);
                self.total.add_scaled(&shifted.0, 1)?;
                ensure!(self.total.is_zero(), "Invalid eqz in bigint program");
                self.reset();
            }
        }
        Ok(())
    }
}

/// Runs the verify program of a bigint ecall on the witness the host computed for it.
pub(crate) fn run(ctx: &mut impl Risc0Context, exec: &BigIntExec) -> Result<()> {
    let mut state = BigIntState {
        is_ecall: true,
        mode: exec.mode,
        pc: exec.verify_program_ptr,
        poly_op: PolyOp::Reset,
        coeff: 0,
        bytes: Default::default(),
        next_state: CycleState::BigIntStep,
    };
    let mut accum = Accum::new();

    ctx.on_bigint_cycle(CycleState::BigIntEcall, &state);
    while state.next_state == CycleState::BigIntStep {
        step(ctx, exec, &mut state, &mut accum)?;
    }
    Ok(())
}

fn step(
    ctx: &mut impl Risc0Context,
    exec: &BigIntExec,
    state: &mut BigIntState,
    accum: &mut Accum,
) -> Result<()> {
    // The circuit does not know the size the blob declares: it reads the next word of memory
    // until the program ends. Refuse to read past that size.
    let fetched = (state.pc.0 - exec.verify_program_ptr.0) as usize;
    ensure!(
        fetched < exec.verify_program_size,
        "bigint verify program does not end within its size"
    );

    // Each instruction is read from memory when it runs, as the circuit reads it, so that a
    // write of this program to a later instruction is seen.
    state.pc.inc();
    let insn = Instruction::decode(ctx.load_u32(LoadOp::Record, state.pc)?)?;

    // The circuit reads the register of every instruction and adds the offset to it, a `Check`
    // included, with room for one carry out of the low half of the register: `NormalizeU32`
    // holds only if that half and 16 times the offset add up to less than 2^17. A register that
    // is not word aligned is refused here even for a `Check`, which the circuit accepts.
    let base = ctx.load_aligned_addr_from_machine_register(LoadOp::Record, insn.reg as usize)?;
    let low = (base.baddr().0 & 0xffff) + insn.offset * (BIGINT_WIDTH_BYTES as u32);
    ensure!(low < (1 << 17), "Invalid bigint offset");

    // The circuit reads the instruction, its register and the words of a `Read` in the one cycle
    // of the instruction, and its memory argument takes one read of a word in a cycle: a read
    // has to come after the access before it (`IsForward`). The instruction may lie in the
    // register file, and a `Read` may reach its own instruction or its own register.
    const TWICE: &str = "bigint instruction reads a word twice";
    let reg_addr = MACHINE_REGS_ADDR.waddr() + insn.reg;
    let read_before = |addr: WordAddr| addr == state.pc || addr == reg_addr;
    ensure!(state.pc != reg_addr, TWICE);

    tracing::trace!("step({:?}, {insn:?}, {base:?})", state.pc);
    let bytes = match insn.mem_op {
        MemoryOp::Read => {
            let addr = operand_addr(base, insn.offset, state.mode)?;
            let mut bytes = [0u8; BIGINT_WIDTH_BYTES];
            for (i, word) in bytes.chunks_exact_mut(WORD_SIZE).enumerate() {
                ensure!(!read_before(addr + i), TWICE);
                word.copy_from_slice(&ctx.load_u32(LoadOp::Record, addr + i)?.to_le_bytes());
            }
            bytes
        }
        MemoryOp::Write => {
            // The bytes are advice in the circuit. The executor has those of the host's witness,
            // and refuses a write that the witness has nothing for.
            let addr = operand_addr(base, insn.offset, state.mode)?;
            let bytes = *exec
                .witness
                .get(&addr)
                .ok_or_else(|| anyhow!("Missing bigint witness: {addr:?}"))?;
            for (i, word) in bytes.chunks_exact(WORD_SIZE).enumerate() {
                ctx.store_u32(addr + i, u32::from_le_bytes(word.try_into().unwrap()))?;
            }
            bytes
        }
        MemoryOp::Check => accum.check_bytes(&insn)?,
    };

    accum.step(&insn, &bytes)?;

    // A `Reset` ends the program unless it is the first instruction: the circuit decides with
    // `isEcall` of the previous cycle, which the cycle of the ecall sets.
    let first = std::mem::replace(&mut state.is_ecall, false);
    state.poly_op = insn.poly_op;
    state.coeff = (insn.coeff + 4) as u32;
    state.bytes = bytes;
    state.next_state = if insn.poly_op == PolyOp::Reset && !first {
        CycleState::Decode
    } else {
        CycleState::BigIntStep
    };

    ctx.on_bigint_cycle(CycleState::BigIntStep, state);
    Ok(())
}

/// The address of the 16 bytes that a `Read` or a `Write` works on.
///
/// The circuit applies `BigIntAddr` to the sum of the register and the offset: the address of the
/// first word is at or above 0x0001_0000, it is outside the 64 KiB below 0xc000_0000, and in user
/// mode it is below them. A sum past the end of memory is refused: the circuit takes it modulo
/// 2^32, which leaves it below 0x0001_0000 under the rule of `NormalizeU32`. Beyond the circuit,
/// this requires the same of the last word as of the first.
fn operand_addr(base: WordAddr, offset: u32, mode: u32) -> Result<WordAddr> {
    let allowed = |addr: WordAddr| {
        addr >= ZERO_PAGE_END_ADDR.waddr()
            && addr < MEMORY_END_ADDR
            && !(USER_BIGINT_END_ADDR.waddr()..USER_END_ADDR.waddr()).contains(&addr)
            && (mode == 1 || addr < USER_BIGINT_END_ADDR.waddr())
    };
    let first = base + offset * BIGINT_WIDTH_WORDS as u32;
    let last = first + (BIGINT_WIDTH_WORDS as u32 - 1);
    ensure!(allowed(first) && allowed(last), "Invalid bigint address");
    Ok(first)
}

#[cfg(test)]
mod tests {
    use risc0_binfmt::ByteAddr;

    use super::*;
    use crate::execute::{
        bigint::ecall_execute,
        r0vm::TestRisc0Context,
        testutil::bigint::{self as fixture, Blob, Call, *},
    };

    fn word(addr: u32) -> WordAddr {
        ByteAddr(addr).waddr()
    }

    /// Runs `call` on a machine as the call has it when it makes its ecall. Memory that the call
    /// does not set holds zeros for 16 words from each word that it sets, and at `ZERO`.
    fn run(call: &Call) -> (Result<()>, TestRisc0Context) {
        let mut ctx = TestRisc0Context::default();
        for reg in 0..REG_MAX {
            ctx.store_register(MACHINE_REGS_ADDR.waddr(), reg, 0)
                .unwrap();
        }
        for (addr, _) in call.memory.iter().chain([&(ZERO, 0)]) {
            for i in 0..16u32 {
                ctx.store_u32(word(*addr) + i, 0).unwrap();
            }
        }
        for (i, value) in (0u32..).zip(&call.blob) {
            ctx.store_u32(word(BLOB) + i, *value).unwrap();
        }
        for (addr, value) in &call.memory {
            ctx.store_u32(word(*addr), *value).unwrap();
        }

        let nondet = BLOB + 4 * WORD_SIZE as u32;
        let verify = nondet + call.blob[0] * WORD_SIZE as u32;
        let consts = verify + call.blob[1] * WORD_SIZE as u32;
        let named = [
            (REG_A0, BLOB),
            (REG_T0, call.mode),
            (REG_T1, nondet),
            (REG_T2, verify),
            (REG_T3, consts),
        ];
        for (reg, value) in named.iter().chain(&call.regs) {
            ctx.store_register(MACHINE_REGS_ADDR.waddr(), *reg, *value)
                .unwrap();
        }
        (ecall_execute(&mut ctx), ctx)
    }

    fn error(call: &Call) -> String {
        run(call).0.unwrap_err().to_string()
    }

    #[test]
    fn decodes_the_encoding_of_the_fixtures() {
        let mem_ops = [MemoryOp::Read, MemoryOp::Write, MemoryOp::Check];
        let poly_ops = [
            PolyOp::Reset,
            PolyOp::Shift,
            PolyOp::SetTerm,
            PolyOp::AddTotal,
            PolyOp::Carry1,
            PolyOp::Carry2,
            PolyOp::EqZero,
        ];
        for (mem_op, mem) in (0..).zip(mem_ops) {
            for (poly_op, poly) in (0..).zip(poly_ops) {
                for (coeff, reg, offset) in [(-4, 0, 0), (3, 31, 0xffff), (0, 11, 2)] {
                    let insn = Instruction::decode(insn(poly_op, mem_op, coeff, reg, offset));
                    let insn = insn.unwrap();
                    assert_eq!((insn.mem_op, insn.poly_op), (mem, poly));
                    assert_eq!(
                        (insn.coeff, insn.reg, insn.offset),
                        (coeff, reg as u32, offset)
                    );
                }
            }
        }
        // Operations that the encoding has room for and the circuit does not have.
        assert!(Instruction::decode(insn(RESET, 3, 0, 0, 0)).is_err());
        assert!(Instruction::decode(insn(7, READ, 0, 0, 0)).is_err());
    }

    #[test]
    fn carry_bytes_hold_22_bits() {
        let insn = |poly_op| Instruction::decode(insn(poly_op, CHECK, 0, REG_ZERO, 0)).unwrap();
        let bytes = |carry: i64| {
            let mut accum = Accum::new();
            accum.total = Poly(vec![carry * 256]);
            [CARRY1, CARRY2, EQ_ZERO].map(|poly_op| accum.check_bytes(&insn(poly_op)).map(|b| b[0]))
        };
        let value = |bytes: [Result<u8>; 3]| {
            let [high, mid, low] = bytes.map(|byte| i64::from(byte.unwrap()));
            (high - 128) * 64 * 256 + mid * 256 + low
        };
        for carry in [0, 1, -1, 255, -256, (1 << 21) - 1, -(1 << 21)] {
            assert_eq!(value(bytes(carry)), carry);
        }
        for carry in [1 << 21, -(1 << 21) - 1, i64::MAX / 256] {
            let [high, ..] = bytes(carry);
            assert_eq!(
                high.unwrap_err().to_string(),
                "bigint carry is out of range"
            );
        }

        // A carry past the end of the total is zero.
        let mut accum = Accum::new();
        accum.total = Poly(vec![256]);
        let far = Instruction::decode(fixture::insn(CARRY1, CHECK, 0, REG_ZERO, 0xffff)).unwrap();
        assert_eq!(accum.check_bytes(&far).unwrap(), [128; 16]);
    }

    #[test]
    fn check_takes_only_carries() {
        for poly_op in [SET_TERM, ADD_TOTAL] {
            let blob = Blob {
                verify: vec![insn(poly_op, CHECK, 0, REG_ZERO, 0), end()],
                ..Default::default()
            };
            assert_eq!(error(&call(&blob)), "Invalid poly_op in bigint program");
        }
    }

    #[test]
    fn arithmetic_does_not_wrap() {
        let max = Poly(vec![i64::MAX]);
        assert_eq!(max.mul(&Poly::one()).unwrap().0, [i64::MAX]);
        assert!(max.mul(&Poly(vec![2])).is_err());
        // The second coefficient of the product is the sum of two terms that each fit.
        assert!(Poly(vec![1 << 62, 1 << 62]).mul(&Poly(vec![1, 1])).is_err());

        let mut sum = Poly(vec![i64::MAX - 1]);
        sum.add_scaled(&[1], 1).unwrap();
        assert!(sum.add_scaled(&[1], 1).is_err());
        assert!(sum.add_scaled(&[1 << 62], 2).is_err());
        assert_eq!(sum.0, [i64::MAX]);

        // The carries add up without wrapping too.
        let mut accum = Accum::new();
        accum.total = Poly(vec![i64::MAX - 255, i64::MAX]);
        let insn = Instruction::decode(insn(CARRY1, CHECK, 0, REG_ZERO, 0)).unwrap();
        assert_eq!(
            accum.check_bytes(&insn).unwrap_err().to_string(),
            "Overflow in bigint program"
        );
    }

    #[test]
    fn reads_and_writes_where_the_circuit_allows() {
        const USER: u32 = 0;
        const MACHINE: u32 = 1;
        for (mode, base, offset, expected) in [
            // The zero page ends at 0x0001_0000.
            (USER, 0, 0, None),
            (USER, 0x0000_fff0, 0, None),
            (USER, 0x0000_fff0, 1, Some(0x0001_0000)),
            (USER, 0x0001_0000, 0, Some(0x0001_0000)),
            (MACHINE, 0x0000_fff0, 0, None),
            // The 64 KiB below the kernel are closed to both modes, and all 16 bytes have to
            // stay below them: the circuit looks at the first word only.
            (USER, 0xbffe_fff0, 0, Some(0xbffe_fff0)),
            (USER, 0xbffe_fff4, 0, None),
            (USER, 0xbffe_0000, 0x0fff, Some(0xbffe_fff0)),
            (USER, 0xbffe_0000, 0x1000, None),
            (MACHINE, 0xbffe_fff0, 0, Some(0xbffe_fff0)),
            (MACHINE, 0xbfff_0000, 0, None),
            (MACHINE, 0xbfff_fff0, 0, None),
            (MACHINE, 0xbfff_fff4, 0, None),
            // Above them only the machine.
            (USER, 0xc000_0000, 0, None),
            (MACHINE, 0xc000_0000, 0, Some(0xc000_0000)),
            (MACHINE, 0xffff_fff0, 0, Some(0xffff_fff0)),
            // These sums are past the end of memory.
            (MACHINE, 0xffff_fff0, 1, None),
            (MACHINE, 0xffff_fff0, 0x1001, None),
        ] {
            let actual = operand_addr(word(base), offset, mode).map(|addr| addr.baddr().0);
            assert_eq!(actual.ok(), expected, "{mode} {base:#010x} {offset:#x}");
        }
    }

    #[test]
    fn bounds_the_work_of_a_program() {
        // A shift makes the polynomial 16 coefficients longer.
        let shifts = |count| vec![insn(SHIFT, READ, 0, REG_A3, 0); count];
        let program = |mut verify: Vec<u32>| {
            verify.push(end());
            call(&Blob {
                verify,
                ..Default::default()
            })
        };

        run(&program(shifts(MAX_COEFFS / 16 - 1))).0.unwrap();
        assert_eq!(
            error(&program(shifts(MAX_COEFFS / 16))),
            "bigint verify program is too large"
        );

        // Two products of two polynomials of 2048 coefficients take all the work there is.
        let mut verify = vec![];
        for _ in 0..2 {
            verify.extend(shifts(127));
            verify.push(insn(SET_TERM, READ, 0, REG_A3, 0));
            verify.extend(shifts(127));
            verify.push(insn(ADD_TOTAL, READ, 1, REG_A3, 0));
        }
        assert_eq!(2 * 2048 * 2048, MAX_WORK);
        run(&program(verify.clone())).0.unwrap();
        verify.push(insn(ADD_TOTAL, READ, 1, REG_A3, 0));
        assert_eq!(
            error(&program(verify)),
            "bigint verify program takes too much work"
        );
    }

    /// The tests take the embedded programs from `EMBEDDED`, so that a blob of the tree that the
    /// list does not have fails here.
    #[test]
    fn lists_the_blobs_of_the_tree() {
        use std::path::Path;

        fn blobs(dir: &Path, found: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    blobs(&path, found);
                } else if path.extension().is_some_and(|ext| ext == "blob") {
                    found.push(path.file_stem().unwrap().to_str().unwrap().to_string());
                }
            }
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut found = vec![];
        blobs(&root.join("bigint2/src"), &mut found);
        blobs(&root.join("zkos/v1compat/src"), &mut found);
        found.sort();

        let embedded = EMBEDDED.iter().map(|(name, _)| *name);
        let mut listed: Vec<_> = embedded.chain(UNUSED).collect();
        listed.sort();
        assert_eq!(found, listed);
    }

    /// Each embedded program takes the cycle of the ecall and one for each instruction of the
    /// size that its blob declares, and runs within the limits. The largest is `modmul_4096`,
    /// with the numbers that the comments on the limits give.
    #[test]
    fn accepts_the_library_programs() {
        let mut most = (0, 0);
        for (name, call) in library_calls() {
            let (result, ctx) = run(&call);
            result.unwrap_or_else(|err| panic!("{name}: {err}"));
            assert_eq!(ctx.bigint_cycles, call.blob[1] as usize + 1, "{name}");

            // The sizes of the polynomials of a program do not depend on its bytes. With these
            // bytes every equation holds.
            let verify = &call.blob[4 + call.blob[0] as usize..][..call.blob[1] as usize];
            let mut accum = Accum::new();
            let mut coeffs = 0;
            for word in verify {
                let insn = Instruction::decode(*word).unwrap();
                let byte = if insn.poly_op == PolyOp::Carry1 {
                    128
                } else {
                    0
                };
                accum.step(&insn, &[byte; BIGINT_WIDTH_BYTES]).unwrap();
                coeffs = coeffs.max(accum.poly.0.len()).max(accum.total.0.len());
            }
            most = (most.0.max(coeffs), most.1.max(accum.work));
        }
        assert_eq!(most, (1552, 795_136));
    }
}
