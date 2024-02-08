use crate::inst::Inst;
use crate::optimizer::SelectInst;
use smallvec::SmallVec;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

pub type BasicBlockMap = HashMap<i64, SmallVec<SelectInst, 32>>;

#[derive(Debug, Clone)]
pub struct BranchAnalyzer {
    pub basic_blocks: BasicBlockMap,
}

impl BranchAnalyzer {
    #[inline]
    pub fn new() -> Self {
        Self {
            basic_blocks: HashMap::new(),
        }
    }

    pub fn perform_link(&mut self, insts: &[SelectInst], base_pc: i64) {
        let bb = match self.basic_blocks.entry(base_pc) {
            Entry::Occupied(_) => {
                return;
            }
            Entry::Vacant(e) => e.insert(SmallVec::new()),
        };

        let mut n = base_pc;

        loop {
            let Some(&inst) = insts.get((n / 4) as usize) else {
                break;
            };

            bb.push(inst);

            match inst {
                SelectInst::Raw(inst) => {
                    match inst {
                        Inst::B { label } => {
                            let b = n.checked_add(label).unwrap();
                            self.perform_link(insts, b);

                            break;
                        }
                        Inst::Tbz { .. } | Inst::Tbnz { .. } |
                        Inst::Br { .. }
                        | Inst::Blr { .. }
                        | Inst::Bl { .. } // this may be a function call, so we skip.
                        | Inst::Ret { .. }
                        | Inst::Udf { .. }
                        | Inst::Svc { .. } => break,
                        _ => (),
                    }

                    n += 4;
                }
                SelectInst::Optimized { amount, .. } => {
                    n += amount as i64 * 4;
                }
            }
        }
    }

    #[inline]
    pub fn finalize(self) -> BasicBlockMap {
        self.basic_blocks
    }
}

#[cfg(test)]
mod tests {
    use crate::block_br::BranchAnalyzer;
    use crate::inst::Inst;
    use crate::optimizer::SelectInst;

    #[test]
    fn test_link() {
        let mut b = BranchAnalyzer::new();

        let insts = [
            SelectInst::Raw(Inst::B { label: 4 }),
            SelectInst::Raw(Inst::Nop),
            SelectInst::Raw(Inst::Nop),
            SelectInst::Raw(Inst::B { label: 8 }),
            SelectInst::Raw(Inst::Nop),
            SelectInst::Raw(Inst::Nop),
            SelectInst::Raw(Inst::Nop),
            SelectInst::Raw(Inst::Nop),
        ];

        b.perform_link(&insts, 0);

        println!("{:?}", b);
    }
}
