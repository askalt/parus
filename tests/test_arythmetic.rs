use std::cell::RefCell;

use core::fmt::Debug;
use either::Either;
use parus::grammar::grammar::{Epsilon, Grammar, RandomGrammarIterator, Symbol};
use parus::lexer::lexer::Lexer;
use parus::parser::ll::LLParser;
use parus::parser::parser::{NonEpsTreeNode, Parser, TreeNode};
use std::time::Instant;

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
                .ok()?
                .parse::<i32>()
                .ok()?;

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
fn calculate_stack(u: &Box<TreeNode<ArithmNode>>, stack: &mut Vec<i32>) -> Option<()> {
    assert!(!u.is_eps());
    let u = u.as_non_eps().unwrap();
    let ch = u.children.as_ref().unwrap();
    match u.vertex {
        E => {
            // E -> F E'
            calculate_stack(&ch[0], stack)?;
            calculate_stack(&ch[1], stack)?;
        }
        EStroke => {
            if ch[0].is_eps() {
                // E' -> eps
                return Some(());
            }
            // We know that rule if E' -> (+|-) F E'
            // We need to calculate F and then do the operation.
            calculate_stack(&ch[1], stack)?;
            let op = ch[0].as_non_eps()?;
            let f = stack.pop()?;
            let s = stack.pop()?;
            let res = match op.vertex {
                Op(Op::Add) => s.checked_add(f),
                Op(Op::Sub) => s.checked_sub(f),
                _ => panic!("wrong tree"),
            }?;
            stack.push(res);
            // Now we can calculate E'.
            calculate_stack(&ch[2], stack)?;
        }
        F => {
            // F -> L F'
            calculate_stack(&ch[0], stack)?;
            calculate_stack(&ch[1], stack)?;
        }
        FStroke => {
            if ch[0].is_eps() {
                // F' -> eps
                return Some(());
            }
            // We know that rule if F' -> (*|/) L F'
            // We need to calculate L and then do the operation.
            calculate_stack(&ch[1], stack)?;
            let op = ch[0].as_non_eps()?;
            let f = stack.pop()?;
            let s = stack.pop()?;
            let res = match op.vertex {
                Op(Op::Mul) => s.checked_mul(f),
                Op(Op::Div) => s.checked_div(f),
                _ => None,
            }?;
            stack.push(res);
            // Now we can calculate F'.
            calculate_stack(&ch[2], stack)?;
        }
        L => match ch[0].as_non_eps().unwrap().vertex {
            Int(x) => {
                stack.push(x);
            }
            Bracket(_) => {
                calculate_stack(&ch[1], stack)?;
            }
            _ => {
                return None;
            }
        },
        _ => {
            return None;
        }
    };
    Some(())
}

/// Calculates the expression value by the parsing tree.
fn calculate(u: Box<TreeNode<ArithmNode>>) -> Option<i32> {
    let mut stack = Vec::new();
    calculate_stack(&u, &mut stack)?;
    if stack.len() != 1 {
        return None;
    }
    stack.pop()
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
    assert_eq!(calculate(tree.unwrap()), Some(5));
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

fn calculate_helper(expr: &[u8]) -> Option<i32> {
    let grammar = ArithmGrammar {};
    let mut lexer = BytesArithmLexer::from_bytes(expr.into());
    let parser: LLParser<ArithmNode, ArithmGrammar> = LLParser::new(grammar);
    let tree = parser.parse(&mut lexer as &mut dyn Lexer<ArithmNode>)?;
    calculate(tree)
}

#[test]
fn test_expressions() {
    assert_eq!(calculate_helper(b"2+2"), Some(4));
    assert_eq!(calculate_helper(b"(2-3)*5"), Some(-5));
    assert_eq!(calculate_helper(b"1-2-3"), Some(-4));
    assert_eq!(calculate_helper(b"150*(13+4)"), Some(2550));
    assert_eq!(
        calculate_helper(b"1*2*3+4*(45-3)+42*42*(10-(16*3))"),
        Some(-66858)
    );
    assert_eq!(calculate_helper(b"1/0"), None);
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

/// Calculates an expression in the stupid way.
/// O(n^2).
/// This methods does not build the parsing tree explicitly and for now, ofcourse, wins from
/// parsing with crate LLParser.
fn stupid_calculate(expr: &[u8]) -> Option<i32> {
    let l = expr.len();
    if l == 0 {
        return Some(0);
    }
    let mut pos = l as i32 - 1;
    let mut bal = 0;
    let mut mul_div_pos = None;
    while pos >= 0 {
        let c = expr[pos as usize];
        match c {
            b'(' => bal -= 1,
            b')' => bal += 1,
            b'+' | b'-' => {
                if bal == 0 {
                    let (le, ri) = expr.split_at(pos as usize);
                    if ri.len() < 2 {
                        return None;
                    }
                    let left_res = stupid_calculate(le)?;
                    let right_res = stupid_calculate(&ri[1..])?;
                    return if c == b'+' {
                        left_res.checked_add(right_res)
                    } else {
                        left_res.checked_sub(right_res)
                    };
                }
            }
            b'*' | b'/' => {
                if bal == 0 && mul_div_pos.is_none() {
                    mul_div_pos.replace(pos);
                }
            }
            _ => {}
        }
        pos -= 1;
    }
    if let Some(pos) = mul_div_pos {
        let (le, ri) = expr.split_at(pos as usize);
        if ri.len() < 2 {
            return None;
        }
        let left_res = stupid_calculate(le)?;
        let right_res = stupid_calculate(&ri[1..])?;
        return if expr[pos as usize] == b'*' {
            left_res.checked_mul(right_res)
        } else {
            left_res.checked_div(right_res)
        };
    }
    if expr[0] == b'(' {
        if *expr.last()? != b')' {
            return None;
        }
        return stupid_calculate(&expr[1..expr.len() - 1]);
    }
    if !expr[0].is_ascii_digit() {
        return None;
    }
    let mut pos: i32 = l as i32 - 1;
    while pos >= 0 && expr[pos as usize].is_ascii_digit() {
        pos -= 1;
    }
    let int = std::str::from_utf8(&expr[(pos + 1) as usize..])
        .ok()?
        .parse::<i32>()
        .ok()?;
    Some(int)
}

#[test]
fn test_stupid_calculate() {
    assert_eq!(stupid_calculate(b"2+2"), Some(4));
    assert_eq!(stupid_calculate(b"(2-3)*5"), Some(-5));
    assert_eq!(stupid_calculate(b"1-2-3"), Some(-4));
    assert_eq!(stupid_calculate(b"150*(13+4)"), Some(2550));
    assert_eq!(
        stupid_calculate(b"1*2*3+4*(45-3)+42*42*(10-(16*3))"),
        Some(-66858)
    );
    assert_eq!(stupid_calculate(b"1/0"), None);
    assert_eq!(stupid_calculate(b"4/77811*(0)"), Some(0));
}

/// Helps to generate expressions with random ints in leafs.
///
/// Describes arithemetic grammar.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
enum FullArithmNode {
    // Non-terminals.
    E,
    EStroke,
    F,
    FStroke,
    L,
    Int,
    IntRest,

    // Terminals.
    Digit(i32),
    Op(Op),
    Bracket(Bracket),
}

impl Debug for FullArithmNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::E => write!(f, "E"),
            Self::EStroke => write!(f, "E'"),
            Self::F => write!(f, "F"),
            Self::FStroke => write!(f, "F'"),
            Self::L => write!(f, "L"),
            Self::Int => write!(f, "Int"),
            Self::Digit(d) => write!(f, "{}", d),
            Self::IntRest => write!(f, "IntRest"),
            Self::Op(op) => write!(f, "{:?}", op),
            Self::Bracket(bracket) => write!(f, "{:?}", bracket),
        }
    }
}

impl Symbol for FullArithmNode {
    fn is_terminal(&self) -> bool {
        matches!(self, Self::Digit(_) | Self::Op(_) | Self::Bracket(_))
    }

    fn start_non_terminal() -> Self {
        Self::E
    }

    fn is_accept(&self, _: &Self) -> bool {
        // Stub, this grammar is not used for parsing, only for
        // generating.
        false
    }
}

struct RandomNumbersGrammar {}

impl RandomNumbersGrammar {
    const PRODUCTIONS: [&'static [Either<&'static [FullArithmNode], Epsilon>]; 7] = [
        &[
            // E -> F E'
            Either::Left(&[FullArithmNode::F, FullArithmNode::EStroke]),
        ],
        &[
            // E'-> + F E'
            Either::Left(&[
                FullArithmNode::Op(Op::Add),
                FullArithmNode::F,
                FullArithmNode::EStroke,
            ]),
            // E'-> - F E'
            Either::Left(&[
                FullArithmNode::Op(Op::Sub),
                FullArithmNode::F,
                FullArithmNode::EStroke,
            ]),
            // E'-> eps
            Either::Right(Epsilon {}),
        ],
        &[
            // F -> L F'
            Either::Left(&[FullArithmNode::L, FullArithmNode::FStroke]),
        ],
        &[
            // F'-> * L F'
            Either::Left(&[
                FullArithmNode::Op(Op::Mul),
                FullArithmNode::L,
                FullArithmNode::FStroke,
            ]),
            // F'-> / L F'
            Either::Left(&[
                FullArithmNode::Op(Op::Div),
                FullArithmNode::L,
                FullArithmNode::FStroke,
            ]),
            // F'->  eps
            Either::Right(Epsilon {}),
        ],
        &[
            // Increase probability of this.
            // L -> int
            Either::Left(&[FullArithmNode::Int]),
            // L -> int
            Either::Left(&[FullArithmNode::Int]),
            // L -> int
            Either::Left(&[FullArithmNode::Int]),
            // L -> (E)
            Either::Left(&[
                FullArithmNode::Bracket(Bracket::Open),
                FullArithmNode::E,
                FullArithmNode::Bracket(Bracket::Close),
            ]),
        ],
        &[
            // Int -> 0
            // Forbid for now.
            // Either::Left(&[FullArithmNode::Digit(0)]),

            // Int -> [!0] RestInt
            Either::Left(&[FullArithmNode::Digit(1), FullArithmNode::IntRest]),
            Either::Left(&[FullArithmNode::Digit(2), FullArithmNode::IntRest]),
            Either::Left(&[FullArithmNode::Digit(3), FullArithmNode::IntRest]),
            Either::Left(&[FullArithmNode::Digit(4), FullArithmNode::IntRest]),
            Either::Left(&[FullArithmNode::Digit(5), FullArithmNode::IntRest]),
            Either::Left(&[FullArithmNode::Digit(6), FullArithmNode::IntRest]),
            Either::Left(&[FullArithmNode::Digit(7), FullArithmNode::IntRest]),
            Either::Left(&[FullArithmNode::Digit(8), FullArithmNode::IntRest]),
            Either::Left(&[FullArithmNode::Digit(9), FullArithmNode::IntRest]),
        ],
        &[
            // ResInt -> eps
            Either::Right(Epsilon {}),
            // ResInt -> [digit] RestInt,
            Either::Left(&[FullArithmNode::Digit(0), FullArithmNode::IntRest]),
            Either::Left(&[FullArithmNode::Digit(1), FullArithmNode::IntRest]),
            Either::Left(&[FullArithmNode::Digit(2), FullArithmNode::IntRest]),
            Either::Left(&[FullArithmNode::Digit(3), FullArithmNode::IntRest]),
            Either::Left(&[FullArithmNode::Digit(4), FullArithmNode::IntRest]),
            Either::Left(&[FullArithmNode::Digit(5), FullArithmNode::IntRest]),
            Either::Left(&[FullArithmNode::Digit(6), FullArithmNode::IntRest]),
            Either::Left(&[FullArithmNode::Digit(7), FullArithmNode::IntRest]),
            Either::Left(&[FullArithmNode::Digit(8), FullArithmNode::IntRest]),
            Either::Left(&[FullArithmNode::Digit(9), FullArithmNode::IntRest]),
        ],
    ];
}

impl Grammar<FullArithmNode> for RandomNumbersGrammar {
    fn get_productions(&self, symbol: &FullArithmNode) -> &[Either<&[FullArithmNode], Epsilon>] {
        Self::PRODUCTIONS[match symbol {
            FullArithmNode::E => 0,
            FullArithmNode::EStroke => 1,
            FullArithmNode::F => 2,
            FullArithmNode::FStroke => 3,
            FullArithmNode::L => 4,
            FullArithmNode::Int => 5,
            FullArithmNode::IntRest => 6,
            _ => panic!("only non-terminals have productions"),
        }]
    }
}

fn collect_as_strings<S: Symbol>(
    iterator: impl Iterator<Item = Vec<S>>,
    number: usize,
) -> Vec<String> {
    iterator
        .take(number)
        .map(|str| {
            str.iter()
                .map(|it| format!("{:?}", it))
                .collect::<Vec<String>>()
                .join("")
        })
        .collect()
}

#[test]
fn fuzz_ll_parser_vs_stupid() {
    let grammar = RandomNumbersGrammar {};

    let iterator = RandomGrammarIterator::new(grammar, 15, 30);
    let actual = collect_as_strings(iterator, 100000);

    for str in actual {
        // println!("{}", str);
        let res1 = calculate_helper(&str.clone().into_bytes());
        let res2 = stupid_calculate(&str.clone().into_bytes());
        // println!("{}", res1.is_some());
        assert_eq!(res1, res2);
    }
}

fn benchmark_parsing_positive(
    parse: impl Fn(&[u8]) -> Option<i32>,
    min_length: usize,
    max_length: usize,
    count: usize,
) {
    let grammar = RandomNumbersGrammar {};

    let iterator = RandomGrammarIterator::new(grammar, min_length, max_length);
    let actual = collect_as_strings(iterator, count);

    let before = Instant::now();
    let mut has_some = false;
    for str in actual {
        let res1 = parse(&str.clone().into_bytes());
        if res1.is_some() {
            has_some = true;
        }
    }
    let after = Instant::now();
    println!("time: {:?}", (after - before).as_millis());
    assert_eq!(has_some, true);
}

#[test]
fn benchmark_ll() {
    benchmark_parsing_positive(calculate_helper, 50, 100, 100000);
}

#[test]
fn benchmark_stupid() {
    benchmark_parsing_positive(stupid_calculate, 50, 100, 100000);
}
