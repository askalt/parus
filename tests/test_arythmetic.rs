use std::cell::RefCell;

use core::fmt::Debug;
use either::Either;
use parus::grammar::grammar::{Epsilon, Grammar, RandomGrammarIterator, Symbol};
use parus::lexer::lexer::Lexer;
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
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

impl Debug for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => write!(f, "+"),
            Self::Sub => write!(f, "-"),
            Self::Mul => write!(f, "*"),
            Self::Div => write!(f, "/"),
        }
    }
}

/// Describes brackets.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
enum Bracket {
    Open,
    Close,
}

impl Debug for Bracket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open => write!(f, "("),
            Self::Close => write!(f, ")"),
        }
    }
}

/// Describes arithemetic grammar.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
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

impl Debug for ArithmNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::E => write!(f, "E"),
            Self::EStroke => write!(f, "E'"),
            Self::F => write!(f, "F"),
            Self::FStroke => write!(f, "F'"),
            Self::L => write!(f, "L"),
            Self::Int(int) => write!(f, "{}", int),
            Self::Op(op) => write!(f, "{:?}", op),
            Self::Bracket(bracket) => write!(f, "{:?}", bracket),
        }
    }
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
    const PRODUCTIONS: [&'static [Either<&'static [ArithmNode], Epsilon>]; 5] = [
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
    fn get_productions(&self, symbol: &ArithmNode) -> &[Either<&[ArithmNode], Epsilon>] {
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

use ArithmNode::*;
mod util;

/// Calculates expression by the parsing tree using stack.
fn calculate_stack(u: &Box<TreeNode<ArithmNode>>, stack: &mut Vec<i32>) {
    assert!(!u.is_eps());
    let u = u.as_non_eps().unwrap();
    let ch = u.children.as_ref().unwrap();
    match u.vertex {
        E => {
            // E -> F E'
            calculate_stack(&ch[0], stack);
            calculate_stack(&ch[1], stack);
        }
        EStroke => {
            if ch[0].is_eps() {
                // E' -> eps
                return;
            }
            // We know that rule if E' -> (+|-) F E'
            // We need to calculate F and then do the operation.
            calculate_stack(&ch[1], stack);
            let op = ch[0].as_non_eps().unwrap();
            let f = stack.pop().unwrap();
            let s = stack.pop().unwrap();
            let res = match op.vertex {
                Op(Op::Add) => s + f,
                Op(Op::Sub) => s - f,
                _ => panic!("wrong tree"),
            };
            stack.push(res);
            // Now we can calculate E'.
            calculate_stack(&ch[2], stack);
        }
        F => {
            // F -> L F'
            calculate_stack(&ch[0], stack);
            calculate_stack(&ch[1], stack);
        }
        FStroke => {
            if ch[0].is_eps() {
                // F' -> eps
                return;
            }
            // We know that rule if F' -> (*|/) L F'
            // We need to calculate L and then do the operation.
            calculate_stack(&ch[1], stack);
            let op = ch[0].as_non_eps().unwrap();
            let f = stack.pop().unwrap();
            let s = stack.pop().unwrap();
            let res = match op.vertex {
                Op(Op::Mul) => s * f,
                Op(Op::Div) => {
                    if f == 0 {
                        panic!("div by 0")
                    } else {
                        s / f
                    }
                }
                _ => panic!("wrong tree"),
            };
            stack.push(res);
            // Now we can calculate F'.
            calculate_stack(&ch[2], stack);
        }
        L => match ch[0].as_non_eps().unwrap().vertex {
            Int(x) => {
                stack.push(x);
            }
            Bracket(_) => {
                calculate_stack(&ch[1], stack);
            }
            _ => panic!("wrong tree"),
        },
        _ => todo!(),
    }
}

/// Calculates the expression value by the parsing tree.
fn calculate(u: Box<TreeNode<ArithmNode>>) -> i32 {
    let mut stack = Vec::new();
    calculate_stack(&u, &mut stack);
    assert!(stack.len() == 1);
    stack.pop().unwrap()
}

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
    assert_eq!(calculate(tree.unwrap()), 5);
}

#[test]
fn example_iterate_arithmetic_grammar() {
    let expected = vec![
        "0", "0*0", "0/0", "0*0*0", "0*0/0", "0/0*0", "0/0/0", "0+0", "0-0", "(0)",
    ];

    let grammar = ArithmGrammar {};
    let actual: Vec<_> = grammar
        .into_iterator()
        .take(expected.len())
        .map(|str| {
            str.iter()
                .map(|it| format!("{:?}", it))
                .collect::<Vec<String>>()
                .join("")
        })
        .collect();

    assert_eq!(actual, expected);
}

fn calculate_helper(expr: &[u8]) -> i32 {
    let grammar = ArithmGrammar {};
    let mut lexer = BytesArithmLexer::from_bytes(expr.into());
    let parser: LLParser<ArithmNode, ArithmGrammar> = LLParser::new(grammar);
    let tree = parser.parse(&mut lexer as &mut dyn Lexer<ArithmNode>);
    assert!(tree.is_some());
    calculate(tree.unwrap())
}

#[test]
fn test_expressions() {
    assert_eq!(calculate_helper(b"2+2"), 4);
    assert_eq!(calculate_helper(b"(2-3)*5"), -5);
    assert_eq!(calculate_helper(b"1-2-3"), -4);
    assert_eq!(calculate_helper(b"150*(13+4)"), 2550);
    assert_eq!(
        calculate_helper(b"1*2*3+4*(45-3)+42*42*(10-(16*3))"),
        -66858
    );
}

#[test]
fn example_random_expressions() {
    let grammar = ArithmGrammar {};
    let iterator = RandomGrammarIterator::new(grammar, 30, 60);
    let actual: Vec<_> = iterator
        .take(10)
        .map(|str| {
            str.iter()
                .map(|it| format!("{:?}", it))
                .collect::<Vec<String>>()
                .join("")
        })
        .collect();
    println!("{:?}", actual.iter().map(|it| it.len()).collect::<Vec<_>>());
    println!("{}", actual.join("\n"));
}
