use std::cell::RefCell;

use parus::either::Either;
use parus::lexer::lexer::{self, Epsilon, Grammar, Lexer, Symbol};
use parus::parser::ll::LLParser;
use parus::parser::parser::{NonEpsTreeNode, Parser, TreeNode};

// Simple LL(1) grammar for arythmetic expressions.

// NT = {E, E', F, F', L}
// T = {+ - * / ( )}

// Productions:
// E -> F E'
// E'-> + F E'
//      - F E'
//      eps

// F -> L F'
// F'-> * L F'
//      / L F'
//      eps

// L -> int
//      (E)

/// Describes operation.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

/// Describes brackets.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug)]
enum Bracket {
    Open,
    Close,
}

/// Describes arithemetic grammar.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug)]
enum ArithmNode {
    // Non-terminals.
    E,
    EStroke,
    F,
    FStroke,
    L,

    // Terminals.
    Int(i32),
    Op(Op),
    Bracket(Bracket),
}

impl Symbol for ArithmNode {
    fn is_terminal(&self) -> bool {
        matches!(self, Self::Int(_) | Self::Op(_) | Self::Bracket(_))
    }

    fn start_non_terminal() -> Self {
        Self::E
    }

    fn is_accept(&self, oth: &Self) -> bool {
        match self {
            Self::Int(_) => matches!(oth, Self::Int(_)),
            Self::Op(lhs) => match oth {
                Self::Op(rhs) => lhs == rhs,
                _ => false,
            },
            Self::Bracket(lhs) => match oth {
                Self::Bracket(rhs) => lhs == rhs,
                _ => false,
            },
            _ => false,
        }
    }
}

struct ArithmGrammar {}

impl ArithmGrammar {
    const PRODUCTIONS: [&'static [Either<&'static [ArithmNode], lexer::Epsilon>]; 5] = [
        &[
            // E -> F E'
            Either::Left(&[ArithmNode::F, ArithmNode::EStroke]),
        ],
        &[
            // E'-> + F E'
            Either::Left(&[ArithmNode::Op(Op::Add), ArithmNode::F, ArithmNode::EStroke]),
            // E'-> - F E'
            Either::Left(&[ArithmNode::Op(Op::Sub), ArithmNode::F, ArithmNode::EStroke]),
            // E'-> eps
            Either::Right(Epsilon {}),
        ],
        &[
            // F -> L F'
            Either::Left(&[ArithmNode::L, ArithmNode::FStroke]),
        ],
        &[
            // F'-> * L F'
            Either::Left(&[ArithmNode::Op(Op::Mul), ArithmNode::L, ArithmNode::FStroke]),
            // F'-> / L F'
            Either::Left(&[ArithmNode::Op(Op::Div), ArithmNode::L, ArithmNode::FStroke]),
            // F'->  eps
            Either::Right(Epsilon {}),
        ],
        &[
            // L -> int
            Either::Left(&[ArithmNode::Int(0)]),
            // L -> (E)
            Either::Left(&[
                ArithmNode::Bracket(Bracket::Open),
                ArithmNode::E,
                ArithmNode::Bracket(Bracket::Close),
            ]),
        ],
    ];
}

impl Grammar<ArithmNode> for ArithmGrammar {
    fn get_productions(&self, symbol: &ArithmNode) -> &[Either<&[ArithmNode], lexer::Epsilon>] {
        Self::PRODUCTIONS[match symbol {
            ArithmNode::E => 0,
            ArithmNode::EStroke => 1,
            ArithmNode::F => 2,
            ArithmNode::FStroke => 3,
            ArithmNode::L => 4,
            _ => panic!("only non-terminals have productions"),
        }]
    }
}

struct BytesArithmLexer {
    bytes: Vec<u8>,
    pos: usize,
    next_pos: RefCell<usize>,
    cur: RefCell<Option<ArithmNode>>,
}

impl BytesArithmLexer {
    fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            bytes: bytes,
            pos: 0,
            next_pos: RefCell::new(0),
            cur: RefCell::new(None),
        }
    }
}

impl Lexer<ArithmNode> for BytesArithmLexer {
    fn cur(&self) -> Option<ArithmNode> {
        if self.pos == self.bytes.len() {
            return None;
        }
        if let Some(ok) = self.cur.borrow().as_ref() {
            return Some(ok.clone());
        }
        let mut next_pos = self.pos;
        if self.bytes[self.pos].is_ascii_digit() {
            while next_pos < self.bytes.len() && self.bytes[next_pos].is_ascii_digit() {
                next_pos += 1;
            }
            let int = std::str::from_utf8(&self.bytes[self.pos..next_pos])
                .expect("got utf8 string")
                .parse::<i32>()
                .expect("got correct i32");

            self.cur.replace(Some(ArithmNode::Int(int)));
        } else {
            next_pos = self.pos + 1;
            self.cur.replace(Some(match self.bytes[self.pos] {
                b'+' => ArithmNode::Op(Op::Add),
                b'-' => ArithmNode::Op(Op::Sub),
                b'*' => ArithmNode::Op(Op::Mul),
                b'/' => ArithmNode::Op(Op::Div),
                b'(' => ArithmNode::Bracket(Bracket::Open),
                b')' => ArithmNode::Bracket(Bracket::Close),
                _ => panic!("unexpected symbol: {:?}", self.bytes[self.pos]),
            }));
        }
        self.next_pos.replace(next_pos);
        self.cur.borrow().clone()
    }

    fn shift(&mut self) {
        if self.pos == self.bytes.len() {
            return;
        }
        if self.cur.borrow().is_none() {
            self.cur();
        }
        self.pos = *self.next_pos.borrow();
        self.cur.replace(None);
    }
}

type NonEpsNode = NonEpsTreeNode<ArithmNode>;

// Sugar to produce trees in tests.
macro_rules! make_node {
    ($vertex: expr, $children: expr) => {
        Box::new(TreeNode::NonEps(NonEpsNode {
            vertex: $vertex,
            children: $children,
        }))
    };
}

macro_rules! eps_node {
    () => {
        Box::new(TreeNode::Eps)
    };
}

use ArithmNode::*;

#[test]
fn test_parse_simple_expr() {
    let expr = b"2+3";
    let grammar = ArithmGrammar {};
    let mut lexer = BytesArithmLexer::from_bytes(expr.into());
    let parser: LLParser<ArithmNode, ArithmGrammar> = LLParser::new(grammar);
    let tree = parser.parse(&mut lexer as &mut dyn Lexer<ArithmNode>);
    //     -  E -
    //    /        \
    //   F       --  E'--
    //  /\      /     /  \
    // L F'    +    F   E'
    // |  \         /\   \
    // 2  eps      /  \   eps
    //             L  F'
    //             |   \
    //             3   eps
    assert_eq!(
        tree,
        Some(make_node!(
            E,
            Some(vec![
                make_node!(
                    F,
                    Some(vec![
                        make_node!(L, Some(vec![make_node!(Int(2), None)])),
                        make_node!(FStroke, Some(vec![eps_node!()])),
                    ])
                ),
                make_node!(
                    EStroke,
                    Some(vec![
                        make_node!(Op(Op::Add), None),
                        make_node!(
                            F,
                            Some(vec![
                                make_node!(L, Some(vec![make_node!(Int(3), None)])),
                                make_node!(FStroke, Some(vec![eps_node!()])),
                            ])
                        ),
                        make_node!(EStroke, Some(vec![eps_node!(),]))
                    ])
                ),
            ])
        ))
    );
}
